// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "read_table_with_header.h"

#include "FileReaderUtils/read_file_utils.h"
#include "MapUtils/map_utils.h"
#include "MathUtils/constants.h"
#include "MathUtils/data_structures.h"
#include "StringUtils/string_utils.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <fstream>   // For sprintf and fstream.
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace map_utils;
using namespace math_utils;
using namespace string_utils;
using namespace std;

namespace file_reader_utils {

namespace {

// Adds the default value for "NA" to 'values' in the indicated column.
void AddNaToken(const int current_col_index, vector<DataHolder>* values) {
  DataHolder item;
  item.name_ = NA_STRING;
  item.type_ = DataType::DATA_TYPE_STRING;

  // Add item to values.
  if (values->size() > current_col_index) {
    (*values)[current_col_index] = item;
  } else {
    values->push_back(item);
  }
}

}  // namespace

bool ReadTableWithHeader::CollapseValue(
    const DataHolder& input,
    const vector<VariableCollapseParams>& collapse_params,
    DataHolder* output, bool* was_collapsed) {
  if (output == nullptr || was_collapsed == nullptr) return false;

  // Nothing to do if no VariableCollapseParams specified, or if this variable
  // is to be collapsed by bucketing.
  if (collapse_params.empty() || collapse_params[0].num_buckets_ > 0) {
    output->value_ = input.value_;
    output->name_ = input.name_;
    output->type_ = input.type_;
    *was_collapsed = false;
    return true;
  }

  string input_str;
  double input_val = DBL_MIN;
  if (input.type_ == DataType::DATA_TYPE_NUMERIC) {
    input_val = input.value_;
    input_str = Itoa(input.value_);
  } else if (input.type_ == DataType::DATA_TYPE_STRING) {
    input_str = input.name_;
    Stod(input.name_, &input_val);
  } else {
    cout << "ERROR: Unsupported DataType for DataHolder: "
         << input.type_ << endl;
    return false;
  }

  // Walk through collapse_params, stopping at the first one for which
  // the input value matches.
  for (const VariableCollapseParams& params : collapse_params) {
    // Handle case that we're supposed to round this to the nearest multiple
    // of some fixed constant.
    if (params.round_to_nearest_ > 0.0) {
      if (input.type_ != DataType::DATA_TYPE_NUMERIC) {
        cout << "ERROR: Unable to round '" << input_str
             << "' to the nearest multiple of " << params.round_to_nearest_
             << " as it is not a numeric value." << endl;
        return false;
      }
      // No rounding to do if value is infinity or zero.
      if (input_val == 0.0 ||
          input_val == -numeric_limits<double>::infinity() ||
          input_val == numeric_limits<double>::infinity()) {
        output->value_ = input_val;
        output->name_ =
            input_val == 0.0 ? "0.0" :
            input_val == numeric_limits<double>::infinity() ? "inf" : "-inf";
      } else {
        const int dividend = input_val / params.round_to_nearest_;
        const double rounded_value =
            (abs(input_val - dividend * params.round_to_nearest_) <
             params.round_to_nearest_ / 2) ?
            params.round_to_nearest_ * dividend :
            params.round_to_nearest_ * (dividend + 1);
        output->name_ = Itoa(rounded_value);
        output->value_ = rounded_value;
      }
      output->type_ = DataType::DATA_TYPE_NUMERIC;
      *was_collapsed = true;
      return true;
    }

    // Set output (default) value in case we match this param.
    DataHolder temp;
    if (params.to_type_ == DataType::DATA_TYPE_NUMERIC) {
      temp.value_ = params.to_val_;
      temp.name_ = Itoa(params.to_val_);
    } else if (params.to_type_ == DataType::DATA_TYPE_STRING) {
      temp.name_ = params.to_str_;
    } else {
      cout << "ERROR: Unsupported DataType for VariableCollapseParams: "
           << params.to_type_ << endl;
      return false;
    }

    // Check if we match this param based on str_value.
    if (params.from_type_ == DataType::DATA_TYPE_STRING) {
      if (!params.from_str_.empty() && params.from_str_ == input_str) {
        output->value_ = temp.value_;
        output->name_ = temp.name_;
        output->type_ = params.to_type_;
        *was_collapsed = true;
        return true;
      }
    // Check if we match this param based on numeric value.
    } else if (params.from_type_ == DataType::DATA_TYPE_NUMERIC) {
      if (params.from_val_ == input_val) {
        output->value_ = temp.value_;
        output->name_ = temp.name_;
        output->type_ = params.to_type_;
        *was_collapsed = true;
        return true;
      }
    } else if (params.from_type_ == DataType::DATA_TYPE_NUMERIC_RANGE) {
      // All 'from' params have default values, which is an error.
      if (params.from_range_.first == DBL_MAX &&
          params.from_range_.second == DBL_MIN) {
        return false;
      }
      // Check if we match this param based on numeric range.
      if (input_val >= params.from_range_.first &&
          input_val <= params.from_range_.second) {
        output->value_ = temp.value_;
        output->name_ = temp.name_;
        output->type_ = params.to_type_;
        *was_collapsed = true;
        return true;
      }
    }
  }

  // No matching regions found. Return original value.
  output->value_ = input.value_;
  output->name_ = input.name_;
  output->type_ = input.type_;
  *was_collapsed = false;
  return true;
}

bool ReadTableWithHeader::AddToken(
    const string& token, const bool is_first_data,
    const int data_col_index, const int col_index_in_file,
    const set<string> na_strings,
    const vector<VariableCollapseParams>& collapse_params,
    vector<DataHolder>* values, map<int, set<string> >* nominal_columns,
    bool* new_nominal_col_found,
    bool* is_na, set<int>* na_cols_for_line,
    string* error_msg) {
  // Check if this token is the missing value (NA) string.
  if (na_strings.find(token) != na_strings.end() ||
      (na_strings.empty() && NA_STRINGS.find(token) != NA_STRINGS.end())) {
    *is_na = true;
    na_cols_for_line->insert(data_col_index);
    AddNaToken(data_col_index, values);
    return true;
  }

  // Collapse value, if appropriate.
  DataHolder orig_value;
  orig_value.name_ = token;
  // First, interpret this value as a string. Below, we will correct
  // this if the value can be interpreted as a numeric value (and the
  // column is not marked as NOMINAL).
  orig_value.type_ = DataType::DATA_TYPE_STRING;
  DataHolder item;
  bool is_collapsed = false;
  if (!CollapseValue(orig_value, collapse_params, &item, &is_collapsed)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Adding Token: Unable to collapse value.\n";
    }
    return false;
  }

  // Recheck (now that we've collapsed) if token is NA.
  const string& new_token = is_collapsed ? item.name_ : token;
  if (na_strings.find(new_token) != na_strings.end() ||
      (na_strings.empty() && NA_STRINGS.find(new_token) != NA_STRINGS.end())) {
    *is_na = true;
    na_cols_for_line->insert(data_col_index);
    AddNaToken(data_col_index, values);
    return true;
  }

  // By default, will assume current column is not nominal, unless we have
  // a list of nominal columns and this column isn't one of them.
  bool current_column_is_nominal = false;
  if (nominal_columns != nullptr) {
    map<int, set<string> >::iterator itr =
        nominal_columns->find(data_col_index);
    current_column_is_nominal = itr != nominal_columns->end();
  }
  
  // Check if this column represents a nominal variable (by seeing if column was
  // already identified as such, or if the value cannot be parsed as a double).
  const bool is_numeric = Stod(new_token, &(item.value_));
  if (is_numeric) {
    item.type_ = DataType::DATA_TYPE_NUMERIC;
  }

  if (current_column_is_nominal || !is_numeric) {
    // Set item.name_, to indicate this value is nominal (not numeric).
    item.name_ = new_token;
    item.type_ = DataType::DATA_TYPE_STRING;

    // Check if this is the first time we've identified this column as nominal.
    if (nominal_columns != nullptr &&
        nominal_columns->find(data_col_index) == nominal_columns->end()) {
      // If this is not the first time a (non-missing) data value has appeared
      // for this column, then we need to
      // update all data values found in the column (in the rows above this one)
      // to be nominal instead of numeric.
      if (!is_first_data) {
        *new_nominal_col_found = true;
      }

      // Add this column to nominal_columns.
      set<string> value_to_add;
      value_to_add.insert(new_token);
      nominal_columns->insert(make_pair(data_col_index, value_to_add));
    } else if (nominal_columns != nullptr) {
      // This column was already known to be nominal. Add the new value to the
      // set of values for this column.
      set<string>& existing_values =
          nominal_columns->find(data_col_index)->second;
      existing_values.insert(new_token);
    }
  }

  // Add item to values.
  if (values->size() > data_col_index) {
    (*values)[data_col_index] = item;
  } else {
    values->push_back(item);
  }
  return true;
}

// Returns info on the appropriate format for the input file.
string ReadTableWithHeader::PrintInputFormat() {
  return  "\nNote that input.txt should have format:\n  "
          "\tX_1\tX_2$\t...\tX_p\n"
          "\tX_1,1\tX_2,1\t...\tX_p,1\n"
          "\tX_1,2\tX_2,2\t...\tX_p,2\n"
          "\t...\t...\t...\t...\n"
          "\tX_1,n\tX_2,n\t...\tX_p,n\n"
          "Where the first row is the 'TITLE' row, with titles for "
          "each of the (in)dependent variables. Titles that have a "
          "'$' suffix are considered NOMINAL variables. The distinction "
          "between independent vs. dependent variables, as well as the "
          "model (linear formula) specifying the relationship between the "
          "variables, can be specified once input data file has been read. "
          "\n  You can use simply X_1, ..., X_p for the title names in "
          "the first row, if you don't want to explicitly name them; "
          "but the first row MUST be the 'TITLE' row (i.e. it must not "
          "contain the actual data). The next n rows should be filled "
          "with the appropriate data."
          "\n  The input/output file names cannot contain spaces.\n";
}

bool ReadTableWithHeader::ReadLine(
    const string& line, const string& delimiter,
    const int expected_num_columns, const int line_num,
    const set<string>& na_strings, const vector<VariableParams>& var_params,
    vector<DataHolder>* values, map<int, set<string> >* nominal_columns,
    set<int>* na_cols_for_line, vector<bool>* is_first_data_value_for_col,
    bool* new_nominal_col_found, string* error_msg) {
  if (values == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: NULL 'values'; Check API call to ReadLine(). Aborting.";
    }
    return false;
  }
  if (line.empty()) {
    return true;
  }

  if (expected_num_columns >= 0) {
    values->resize(expected_num_columns);
  }

  // Split line around delimiter.
  vector<string> col_strs;
  Split(line, delimiter, false /* do not collapse empty strings */, &col_strs);

  if (expected_num_columns >= 0 && col_strs.size() != expected_num_columns) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to parse line " + Itoa(line_num) +
                    ": Wrong number of entries: expected " +
                    Itoa(expected_num_columns) +
                    " entries (based on TITLE line), but found " +
                    Itoa(static_cast<int>(col_strs.size())) + ".\n";
    }
    return false;
  }
  if (col_strs.size() != var_params.size() ||
      is_first_data_value_for_col == nullptr ||
      is_first_data_value_for_col->size() != col_strs.size()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to parse line " + Itoa(line_num) +
                    ": Wrong number of entries. Col strs size: " +
                    Itoa(static_cast<int>(col_strs.size())) +
                    ", num var params: " +
                    Itoa(static_cast<int>(var_params.size())) + ".\n";
    }
    return false;
  }

  // Iterate through columns, copying values to 'values'.
  for (int i = 0; i < col_strs.size(); ++i) {
    bool is_na = false;
    const bool is_first_data = (*is_first_data_value_for_col)[i];
    if (!AddToken(
          col_strs[i], is_first_data, i, i,
          na_strings, var_params[i].collapse_params_,
          values, nominal_columns, new_nominal_col_found, &is_na,
          na_cols_for_line, error_msg)) {
      return false;
    }
    if (!is_na) (*is_first_data_value_for_col)[i] = false;
  }
  return true;
}

bool ReadTableWithHeader::ReadLine(
    const string& line, const string& delimiter,
    const set<int>& cols_to_read, const int line_num,
    const set<string>& na_strings, const vector<VariableParams>& var_params,
    vector<DataHolder>* values, map<int, set<string> >* nominal_columns,
    set<int>* na_cols_for_line, vector<bool>* is_first_data_value_for_col,
    bool* new_nominal_col_found, string* error_msg) {
  if (values == nullptr) {
    if (error_msg != nullptr) *error_msg += "Null values.\n";
    return false;
  }
  if (line.empty()) {
    return true;
  }
  if (is_first_data_value_for_col == nullptr ||
      is_first_data_value_for_col->size() != cols_to_read.size()) {
    if (error_msg != nullptr) *error_msg += "Null values.\n";
    return false;
  }


  values->clear();
  values->resize(cols_to_read.size());

  // Split line around delimiter.
  vector<string> col_strs;
  Split(line, delimiter, false /* do not collapse empty strings */, &col_strs);

  // Iterate through relevant columns, copying values to 'values'.
  const int num_cols_found = col_strs.size();
  int i = -1;
  for (const int col_index : cols_to_read) {
    ++i;
    if (col_index >= num_cols_found) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Cannot find column index " +
                      Itoa(col_index) + ", line " +
                      Itoa(line_num) + " has only " +
                      Itoa(num_cols_found) + " columns:\n" +
                      line + "\n";
      }
      return false;
    }
    const bool is_first_data_for_col_i = (*is_first_data_value_for_col)[i];
    const string& token = col_strs[col_index];
    bool is_na = false;
    if (!AddToken(
          token, is_first_data_for_col_i, i, col_index, na_strings,
          var_params[col_index].collapse_params_,
          values, nominal_columns, new_nominal_col_found,
          &is_na, na_cols_for_line, error_msg)) {
      return false;
    }
    if (!is_na) (*is_first_data_value_for_col)[i] = false;
  }

  return true;
}

bool ReadTableWithHeader::ReadDataFile(
    const FileInfo& file_info,
    const vector<string>& orig_file_header,
    const set<int>& cols_to_read,
    const vector<VariableParams>& var_params,
    vector<string>* data_values_header,
    vector<vector<DataHolder> >* data_values,
    map<int, set<string> >* nominal_columns, map<int, set<int> >* na_columns,
    string* error_msg) {
  if (data_values == nullptr || data_values_header == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Null header and/or data_values; check API for call "
                    "to ReadDataFile(). Aborting.\n";
    }
    return false;
  }

  const string& filename = file_info.name_;
  const string& delimiter = file_info.delimiter_;
  const set<string>& na_strings = file_info.na_strings_;

  // Open input file.
  ifstream input_file(filename.c_str());
  if (!input_file.is_open()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to find file '" + filename +
                    "'. Make sure that it is present "
                    "in your current directory.\n";
    }
    return false;
  }
  
  // Read Title line of input file.
  string title_line;
  if (!getline(input_file, title_line)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Empty input file '" + filename + ".\n";
    }
    return false;
  }
  RemoveWindowsTrailingCharacters(&title_line);
  vector<string> header;
  set<int> nominal_cols_in_header;
  if (!GetTitles(title_line, delimiter, &header, &nominal_cols_in_header)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Improper format in data file '" + filename +
                    ".\n" + PrintInputFormat() + "\n";
    }
    return false;
  }

  // Update nominal_columns with columns identified as being nominal (based
  // on presence of '$'-suffix) from the header line.
  // NOTE: This process is made more complicated by the fact that
  // 'nominal_columns' requires indexing w.r.t. 'data_values', *not* w.r.t.
  // the original columns in the data file.
  if (!nominal_cols_in_header.empty()) {
    if (cols_to_read.empty()) {
      for (const int nom_col : nominal_cols_in_header) {
        nominal_columns->insert(make_pair(nom_col, set<string>()));
      }
    } else {
      // First need a map from original column index to data_values column index.
      map<int, int> orig_index_to_data_values_index;
      int i = -1;
      for (const int col_to_read : cols_to_read) {
        i++;
        orig_index_to_data_values_index.insert(make_pair(col_to_read, i));
      }
      for (const int nom_col : nominal_cols_in_header) {
        if (orig_index_to_data_values_index.find(nom_col) ==
            orig_index_to_data_values_index.end()) {
          cout << "ERROR: Unexpected original index " << nom_col
               << " is not present in cols_to_read: "
               << Join(cols_to_read, ", ") << endl;
          return false;
        }
        nominal_columns->insert(make_pair(
            orig_index_to_data_values_index[nom_col], set<string>()));
      }
    }
  }

  // Set data_values header (same as orig_file_header if all columns are used;
  // otherwise, will be a subset).
  data_values_header->clear();
  if (cols_to_read.empty()) {
    *data_values_header = orig_file_header;
  } else {
    for (const int column_taken : cols_to_read) {
      data_values_header->push_back(orig_file_header[column_taken]);
    }
  }

  // Read data values from file.
  data_values->clear();
  string line;
  int data_row = 1;
  bool need_another_pass = false;
  const int num_cols_to_read =
      cols_to_read.empty() ? var_params.size() : cols_to_read.size();
  vector<bool> is_first_data_value_for_col(num_cols_to_read, true);
  while (getline(input_file, line)) {
    RemoveWindowsTrailingCharacters(&line);
    vector<DataHolder> sample_values;
    set<int> na_columns_for_line;
    bool new_nominal_col_found = false;
    if (cols_to_read.empty() &&
        !ReadLine(line, delimiter, header.size(), data_row, na_strings,
                  var_params, &sample_values, nominal_columns,
                  &na_columns_for_line, &is_first_data_value_for_col,
                  &new_nominal_col_found, error_msg)) {
      return false;
    } else if (!cols_to_read.empty() &&
               !ReadLine(line, delimiter, cols_to_read, data_row, na_strings,
                         var_params, &sample_values, nominal_columns,
                         &na_columns_for_line, &is_first_data_value_for_col,
                         &new_nominal_col_found, error_msg)) {
      return false;
    }
    need_another_pass |= new_nominal_col_found;
    if (na_columns != nullptr && !na_columns_for_line.empty()) {
      na_columns->insert(make_pair(data_values->size(), na_columns_for_line));
    }
    ++data_row;
    data_values->push_back(sample_values);
  }

  // If there were any columns that had at least one value that couldn't be
  // parsed as a numeric value, then re-label the column as NOMINAL.
  if (need_another_pass &&
      !UpdateNominalColumns(
          Keys(*nominal_columns), data_values, nominal_columns, error_msg)) {
    return false;
  }

  // We need another pass if any of the variables have collapse params
  // indicating num_buckets.
  if (!UpdateBucketColumns(cols_to_read, var_params, data_values, error_msg)) {
    return false;
  }

  return true;
}

bool ReadTableWithHeader::UpdateBucketColumns(
    const set<int>& cols_to_read,
    const vector<VariableParams>& vars_params,
    vector<vector<DataHolder>>* data_values,
    string* error_msg) {
  if (data_values == nullptr) return false;

  // Check if any variables indicate to collapse by buckets, and if
  // so, keep track of the column index (w.r.t. data_values) and the
  // number of buckets for that variable.
  map<int, int> col_index_to_num_buckets;
  for (const VariableParams& var_params : vars_params) {
    const vector<VariableCollapseParams>& collapse_params =
        var_params.collapse_params_;
    if (!collapse_params.empty() && collapse_params[0].num_buckets_ > 0) {
      int col_index_wrt_data_values;
      if (!OrigColToDataValuesCol(cols_to_read, var_params.col_.index_,
              &col_index_wrt_data_values, error_msg)) {
        return false;
      }
      col_index_to_num_buckets.insert(make_pair(
          col_index_wrt_data_values, collapse_params[0].num_buckets_));
    }
  }

  // Return if no variables need to be collapsed by bucket.
  if (col_index_to_num_buckets.empty()) return true;

  // Find min and max of each column to be collapsed.
  map<int, pair<double, double>> col_to_min_and_max;
  for (const vector<DataHolder>& data_row : *data_values) {
    for (const int col_index : Keys(col_index_to_num_buckets)) {
      if (col_index < 0 || col_index >= data_row.size()) {
        return false;
      }
      const DataHolder& current_data = data_row[col_index];
      if (current_data.type_ != DataType::DATA_TYPE_NUMERIC) {
        return false;
      }
      const double& current_value = current_data.value_;
      // Cannot use bucketing if there are infinite values.
      if (std::isinf(current_value)) {
        return false;
      }
      pair<double, double>* min_and_max = FindOrInsert(
          col_index, col_to_min_and_max, make_pair(DBL_MAX, DBL_MIN));
      if (current_value < min_and_max->first) {
        min_and_max->first = current_value;
      }
      if (current_value > min_and_max->second) {
        min_and_max->second = current_value;
      }
    }
  }

  // Collapse all indicated params into buckets.
  for (vector<DataHolder>& data_row : *data_values) {
    for (const pair<int, pair<double, double>>& col_and_min_max :
         col_to_min_and_max) {
      const int col_index = col_and_min_max.first;
      const double& min = col_and_min_max.second.first;
      const double& max = col_and_min_max.second.second;
      const double col_value = data_row[col_index].value_;
      const int num_buckets = col_index_to_num_buckets[col_index];
      if (!GetBucketValue(
             col_value, num_buckets, min, max,
             &data_row[col_index].value_, error_msg)) {
        return false;
      }
    }
  }

  return true;
}

bool ReadTableWithHeader::GetBucketValue(
    const double& orig_value, const int num_buckets,
    const double& left_end, const double& right_end,
    double* bucket_value, string* error_msg) {
  if (bucket_value == nullptr) return false;

  // Sanity-check input parameters make sense.
  if (left_end > right_end || num_buckets <= 0 ||
      orig_value < left_end || orig_value > right_end) {
    return false;
  }
  
  // If left and right endpoints are the same, all buckets are the same.
  // Return that point.
  if (left_end == right_end) {
    *bucket_value = left_end;
    return true;
  }

  const double bucket_width = (right_end - left_end) / num_buckets;
  const int bucket_index = orig_value == right_end ? 
      num_buckets - 1 : (orig_value - left_end) / bucket_width;
  // Adding 0.5 shifts the value to be the midpoint of the bucket.
  *bucket_value = left_end + (bucket_index + 0.5) * bucket_width;

  return true;
}


bool ReadTableWithHeader::UpdateNominalColumns(
      const set<int>& nominal_columns,
      vector<vector<DataHolder>>* data_values,
      map<int, set<string>>* nominal_columns_and_values,
      string* error_msg) {
  for (vector<DataHolder>& row_values : *data_values) {
    for (const int col_index : nominal_columns) {
      DataHolder& current_value = row_values[col_index];
      // Convert current_value to a nominal value, if necessary.
      if (current_value.name_.empty()) {
        current_value.name_ = GetDataHolderString(current_value, 17);
      }
      current_value.value_ = 0.0;
      current_value.type_ = DataType::DATA_TYPE_STRING;
      set<string>* values =
          FindOrInsert(col_index, *nominal_columns_and_values, set<string>());
      values->insert(current_value.name_);
    }
  }
  return true;
}

bool ReadTableWithHeader::CopyDataFromOldFormatToNew(
    const vector<string>& ids, const vector<double>& weights,
    const vector<vector<string>>& families,
    const vector<double>& dep_var,
    const MatrixXd& indep_vars,
    vector<DataRow>* data, string* error_msg) {
  // Sanity-check input.
  const int n = dep_var.size();
  if (data == nullptr || indep_vars.rows() != n ||
      (!ids.empty() && ids.size() != n) ||
      (!weights.empty() && weights.size() != n) ||
      (!families.empty() && families.size() != n)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in CopyDataFromOldFormatToNew: mismatching vector "
                    "sizes: ids (" + Itoa(static_cast<int>(ids.size())) +
                    "), weights (" + Itoa(static_cast<int>(weights.size())) +
                    "), families (" + Itoa(static_cast<int>(families.size())) +
                    "), dep var (" + Itoa(static_cast<int>(dep_var.size())) +
                    "), indep vars (" +
                    Itoa(static_cast<int>(indep_vars.rows())) +
                    ").\n";
    }
    return false;
  }

  data->clear();
  const int p = indep_vars.cols();
  for (int i = 0; i < ids.size(); ++i) {
    data->push_back(DataRow());
    DataRow& data_row = data->back();
    if (!ids.empty()) {
      data_row.id_ = ids[i];
    }
    if (!weights.empty()) {
      data_row.weight_ = weights[i];
    }
    if (!families.empty()) {
      data_row.families_.resize(families[i].size());
      copy(families[i].begin(), families[i].end(), data_row.families_.begin());
    }
    data_row.dep_var_value_ = dep_var[i];
    data_row.indep_var_values_.clear();
    for (int j = 0; j < p; ++j) {
      data_row.indep_var_values_.push_back(indep_vars(i, j));
    }
  }
  return true;
}

bool ReadTableWithHeader::CopyDataFromOldFormatToNew(
    const vector<string>& ids, const vector<double>& weights,
    const vector<vector<string>>& families,
    const vector<CensoringData>& dep_var,
    const MatrixXd& indep_vars,
    vector<DataRow>* data, string* error_msg) {
  // Sanity-check input.
  if (ids.size() != weights.size() || ids.size() != families.size() ||
      ids.size() != dep_var.size() || ids.size() != indep_vars.rows() ||
      data == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in CopyDataFromOldFormatToNew: mismatching vector "
                    "sizes: ids (" + Itoa(static_cast<int>(ids.size())) +
                    "), weights (" + Itoa(static_cast<int>(weights.size())) +
                    "), families (" + Itoa(static_cast<int>(families.size())) +
                    "), dep var (" + Itoa(static_cast<int>(dep_var.size())) +
                    "), indep vars (" +
                    Itoa(static_cast<int>(indep_vars.rows())) +
                    ").\n";
    }
    return false;
  }

  data->clear();
  const int p = indep_vars.cols();
  for (int i = 0; i < ids.size(); ++i) {
    data->push_back(DataRow());
    DataRow& data_row = data->back();
    data_row.id_ = ids[i];
    data_row.weight_ = weights[i];
    data_row.families_.resize(families[i].size());
    copy(families[i].begin(), families[i].end(), data_row.families_.begin());
    const CensoringData& censor_data = dep_var[i];
    data_row.cox_dep_var_value_.survival_time_ = censor_data.survival_time_;
    data_row.cox_dep_var_value_.censoring_time_ = censor_data.censoring_time_;
    data_row.cox_dep_var_value_.is_alive_ = censor_data.is_alive_;
    data_row.indep_var_values_.clear();
    for (int j = 0; j < p; ++j) {
      data_row.indep_var_values_.push_back(indep_vars(i, j));
    }
  }
  return true;
}

}  // namespace file_reader_utils
