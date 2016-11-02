// Author: paulbunn@email.unc.edu (Paul Bunn)
// Last Updated: March 2015

#include "csv_utils.h"

#include "StringUtils/string_utils.h"

#include <climits>  // For INT_MIN
#include <cfloat>   // For DBL_MIN
#include <iostream> // For cout (for debugging; can remove if not needed).
#include <fstream>  // For sprintf
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace string_utils;

namespace file_reader_utils {

namespace {

bool IsMissingValue(
    const GenericDataType& type,
    const ReadCsvInput& input,
    const string& value) {
  if (input.in_na_strings_.find(value) != input.in_na_strings_.end()) {
    return true;
  }
  // Already checked the string value of this string; return false
  // if type is String.
  if (type == GenericDataType::STRING) return false;

  // All other types are numeric. See if value is any of the type-specific
  // values representing NA.
  if (type == GenericDataType::DOUBLE) {
    double d;
    if (!Stod(value, &d)) return false;
    return d == input.in_na_double_;
  }
  if (type == GenericDataType::INT) {
    int i;
    if (!Stoi(value, &i)) return false;
    return i == input.in_na_int_;
  }
  if (type == GenericDataType::INT_64) {
    int64_t i;
    if (!Stoi(value, &i)) return false;
    return i == input.in_na_int64_;
  }
  if (type == GenericDataType::UINT_64) {
    uint64_t i;
    if (!Stoi(value, &i)) return false;
    return i == input.in_na_uint64_;
  }
  // Unknown type. At any rate, we've already checked type-specific
  // NA values for each known (supported) type, so return false.
  return false;
}

bool IsInfinityValue(const string& inf_str, const string& value) {
  return value == inf_str;
}

// Determines if line_num should be skipped, based on the input parameters.
// Note that there are some ambiguities, e.g. if line_num is in both
// lines_to_keep and lines_to_skip. The caller is responsible for making
// sure this doesn't happen; but these are handled by preferentially
// taking inclusion over exclusion.
bool ShouldSkipLine(
    const string& line, const string& comment_id,
    const int line_num, const int lines_read, const bool has_header,
    const LinesToRead& line_params,
    bool* read_header_line, bool* abort) {
  const set<int>& lines_to_skip = line_params.lines_to_skip_;
  const set<int>& lines_to_keep = line_params.lines_to_keep_;
  const int range_to_keep_start = line_params.range_to_keep_start_;
  const int range_to_keep_end = line_params.range_to_keep_end_;
  const int range_to_skip_start = line_params.range_to_skip_start_;
  const int range_to_skip_end = line_params.range_to_skip_end_;
  // Abort if already read in all lines specified in lines_to_keep.
  if (!lines_to_keep.empty() && lines_read == lines_to_keep.size()) {
    *abort = true;
    return true;
  }

  // Skip empty lines.
  if (line.empty() && line_params.skip_empty_lines_) {
    return true;
  }
  string trimmed_line;
  RemoveLeadingWhitespace(line, &trimmed_line);
  if (trimmed_line.empty() && line_params.skip_whitespace_only_lines_) {
    return true;
  }

  // Skip lines that begin with comment_id.
  if (!comment_id.empty() &&
      HasPrefixString(trimmed_line, comment_id)) {
    return true;
  }

  // Skip lines that begin with 'abort_on_prefix_'.
  if (!line_params.abort_on_prefix_.empty() &&
      HasPrefixString(trimmed_line, line_params.abort_on_prefix_)) {
    *abort = true;
    return true;
  }

  // Skip header line.
  if (!(*read_header_line) && has_header) {
    *read_header_line = true;
    return true;
  }

  // Keep line if it is in lines_to_keep.
  if (lines_to_keep.find(line_num) != lines_to_keep.end()) {
    return false;
  } else if (!lines_to_keep.empty()) {
    return true;
  }

  // Keep line if it is within range specified by
  //   [range_to_keep_start, range_to_keep_end].
  if (range_to_keep_start >= 0) {
    if (range_to_keep_end >= 0) {
      if (line_num > range_to_keep_end) *abort = true;
      return !(line_num >= range_to_keep_start &&
               line_num <= range_to_keep_end);
    } else {
      return !(line_num >= range_to_keep_start);
    }
  } else if (range_to_keep_end >= 0) {
    if (line_num > range_to_keep_end) *abort = true;
    return !(line_num <= range_to_keep_end);
  }

  // Skip line if it is in lines_to_skip.
  if (lines_to_skip.find(line_num) != lines_to_skip.end()) {
    return true;
  } else if (!lines_to_skip.empty()) {
    return false;
  }

  // Skip line if it is within range specified by
  //   [range_to_skip_start, range_to_skip_end].
  if (range_to_skip_start >= 0) {
    if (range_to_skip_end >= 0) {
      return (line_num >= range_to_skip_start ||
              line_num <= range_to_skip_end);
    } else {
      return line_num >= range_to_skip_start;
    }
  } else if (range_to_skip_end >= 0) {
    return line_num <= range_to_skip_end;
  }
  
  // No filtering applies to this line_num.
  return false;
}

}  // namespace

bool CsvUtils::ReadCsv(
    const ReadCsvInput& input, ReadCsvOutput* output) {
  if (output == nullptr) return false;
  if (input.range_columns_to_read_.empty() && input.columns_to_read_.empty()) {
    output->error_msg_ += "ERROR in Reading Csv file: No columns specified for reading.\n";
    return false;
  }
  // There are two formats for the input: those where field 'columns_to_read_'
  // are used, and those where 'range_columns_to_read_' is used. This function
  // will only handle the latter case, so convert the former into the latter here.
  if (input.range_columns_to_read_.empty()) {
    ReadCsvInput new_input = input;
    vector<pair<vector<int>, GenericDataType>>& new_format_cols_to_read =
        new_input.range_columns_to_read_;
    for (pair<int, GenericDataType> item : input.columns_to_read_) {
      new_format_cols_to_read.push_back(make_pair(vector<int>(), item.second));
      vector<int>& new_range = new_format_cols_to_read.back().first;
      new_range.push_back(item.first);
    }
    return ReadCsv(new_input, output);
  }

  output->output_.clear();

  // Open File.
  ifstream file(input.filename_.c_str());
  if (!file.is_open()) {
    output->error_msg_ += "ERROR in Reading Csv: Unable to open file '" +
                          input.filename_ + "'.\n";
    return false;
  }

  // Loop through lines of file.
  bool read_header_line = false;
  bool abort = false;
  string line;
  int line_index = 1;
  int last_line_to_keep = -1;
  if (input.line_params_.range_to_keep_end_ >= 0) {
    last_line_to_keep = input.line_params_.range_to_keep_end_;
  } else if (!input.line_params_.lines_to_keep_.empty()) {
    last_line_to_keep = *((input.line_params_.lines_to_keep_.end())--);
  }
  while(getline(file, line)) {
    // Optimization to abort early if we've reached the end of the range
    // of line numbers to keep.
    if (last_line_to_keep >= 0 && line_index > last_line_to_keep) {
      break;
    }

    // Remove trailing character '13' (Carriage Return) at end of line, if present.
    if (line[line.length() - 1] == 13) {
      line = line.substr(0, line.length() - 1);
    }

    // Skip lines that shouldn't be processed (e.g. header line).
    if (ShouldSkipLine(
            line, input.comment_char_, line_index, output->output_.size(),
            input.has_header_, input.line_params_, &read_header_line, &abort)) {
      line_index++;
      if (abort) break;
      continue;
    }

    // Split line by column; initially reading it into a vector of strings.
    vector<string> columns;
    Split(line, input.delimiters_, false /* Don't Skip/Flatten Empty Columns */,
          &columns);

    // Pick out the relevant columns, and parse into the appropriate data type.
    vector<GenericDataHolder> row_values;
    for (const pair<vector<int>, GenericDataType>& itr :
         input.range_columns_to_read_) {
      const GenericDataType& type = itr.second;
      bool within_range = false;
      int start_range = -1;
      int end_range = -1;
      for (const int columns_itr : itr.first) {
        // Special parsing needed for range-notation. See comments in csv_utils.h
        // for details.
        if (columns_itr == 0) {
          if (within_range) {
            // Nothing to do, ready to proceed to loop below.
          } else {
            within_range = true;
            continue;
          }
        } else if (within_range) {
          if (start_range == -1) {
            start_range = columns_itr;
            continue;
          } else if (end_range == -1) {
            end_range = columns_itr;
            continue;
          } else {
            output->error_msg_ +=
                "ERROR reading line " + Itoa(line_index) + ": columns "
                "specifications cannot be parsed. Specifications: [" +
                Join(itr.first, ", ") + "]. See csv_utils for proper "
                "usage. Aborting.\n";
            return false;
          }
        } else {
          start_range = columns_itr;
          end_range = columns_itr;
        }
        if (start_range >= 0 && end_range == -1) {
          end_range = columns.size();
        } else if (start_range == -1 || start_range > end_range) {
          output->error_msg_ +=
              "ERROR reading line " + Itoa(line_index) + ": columns "
              "specifications cannot be parsed. Specifications: [" +
              Join(itr.first, ", ") + "]. See csv_utils for proper "
              "usage. Aborting.\n";
          return false;
        }
        for (int column = start_range; column <= end_range; column++) {
          if (columns.size() < column) {
            output->error_msg_ +=
                "ERROR in Reading Csv: Attempting to read column " +
                Itoa(column) + " of line " + Itoa(line_index) + ": '" +
                line + "', but only parsed " + Itoa(columns.size()) +
                " columns.\n";
            return false;
          }

          // Parse the column according to it's type.
          GenericDataHolder holder;
          const string& col_value = columns[column - 1];
          if (IsMissingValue(type, input, col_value)) {
            if (type == GenericDataType::STRING) holder.str_ = input.out_na_string_;
            else if (type == GenericDataType::INT) holder.int_ = input.out_na_int_;
            else if (type == GenericDataType::INT_64) holder.int_ = input.out_na_int64_;
            else if (type == GenericDataType::UINT_64) holder.int_ = input.out_na_uint64_;
            else if (type == GenericDataType::DOUBLE) holder.dbl_ = input.out_na_double_;
          } else if (IsInfinityValue(input.in_inf_, col_value)) {
            if (type == GenericDataType::INT) {
              holder.int_ = input.out_inf_int_;
            } else if (type == GenericDataType::INT_64) {
              holder.int64_ = input.out_inf_int64_;
            } else if (type == GenericDataType::UINT_64) {
              holder.uint64_ = input.out_inf_uint64_;
            } else if (type == GenericDataType::DOUBLE) {
              holder.dbl_ = input.out_inf_double_;
            } else if (type == GenericDataType::STRING) {
              // Ignore that this string happens to match the infinity string,
              // simply copy the string to holder.str_.
              holder.str_ = col_value;
            } else {
              output->error_msg_ += "ERROR in Reading Csv file: Unsupported "
                                   "data type: " + Itoa(type) + ".\n";
              return false;
            }
          } else if (type == GenericDataType::STRING) {
            holder.str_ = col_value;
          } else if (type == GenericDataType::INT) {
            if (!Stoi(col_value, &holder.int_)) {
              if (!input.allow_error_lines_) {
                output->error_msg_ +=
                    "ERROR in Reading Csv file: Attempting to read column " +
                    Itoa(column) + " of line " + Itoa(line_index) + " ('" +
                    line + "'): expected INT value, but observed '" +
                    col_value + "'.\n";
                return false;
              } else {
                output->error_line_and_column_.insert(make_pair(line_index, column - 1));
                holder.int_ = input.out_na_int_;
              }
            }
          } else if (type == GenericDataType::INT_64) {
            if (!Stoi(col_value, &holder.int64_)) {
              if (!input.allow_error_lines_) {
                output->error_msg_ +=
                    "ERROR in Reading Csv file: Attempting to read column " +
                    Itoa(column) + " of line " + Itoa(line_index) + " ('" +
                    line + "'): expected INT value, but observed '" +
                    col_value + "'.\n";
                return false;
              } else {
                output->error_line_and_column_.insert(make_pair(line_index, column - 1));
                holder.int64_ = input.out_na_int64_;
              }
            }
          } else if (type == GenericDataType::UINT_64) {
            if (!Stoi(col_value, &holder.uint64_)) {
              if (!input.allow_error_lines_) {
                output->error_msg_ +=
                    "ERROR in Reading Csv file: Attempting to read column " +
                    Itoa(column) + " of line " + Itoa(line_index) + " ('" +
                    line + "'): expected INT value, but observed '" +
                    col_value + "'.\n";
                return false;
              } else {
                output->error_line_and_column_.insert(make_pair(line_index, column - 1));
                holder.uint64_ = input.out_na_uint64_;
              }
            }
          } else if (type == GenericDataType::DOUBLE) {
            if (!Stod(col_value, &holder.dbl_)) {
              if (!input.allow_error_lines_) {
                output->error_msg_ +=
                    "ERROR in Reading Csv file: Attempting to read column " +
                    Itoa(column) + " of line " + Itoa(line_index) + " ('" +
                    line + "'): expected DOUBLE value, but observed '" +
                    col_value + "'.\n";
                return false;
              } else {
                output->error_line_and_column_.insert(make_pair(line_index, column - 1));
                holder.dbl_ = input.out_na_double_;
              }
            }
          } else {
            output->error_msg_ += "ERROR in Reading Csv file: Unsupported "
                                 "data type: " + Itoa(type) + ".\n";
            return false;
          }
          row_values.push_back(holder);
        }
        within_range = false;
        start_range = -1;
        end_range = -1;
      }
    }
    output->output_.push_back(row_values);
    line_index++;
  }
  return true;
}

}  // namespace file_reader_utils
