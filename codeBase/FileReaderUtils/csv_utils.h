// Author: paulbunn@email.unc.edu (Paul Bunn)
// Last Updated: March 2015
//
// Description: Helper functions for reading in .csv files (or more
// generally, and file in table format).

#ifndef CSV_UTILS_H
#define CSV_UTILS_H

#include "FileReaderUtils/read_file_utils.h"
#include "StringUtils/string_utils.h"

#include <climits>  // For INT_MIN
#include <cfloat>   // For DBL_MIN
#include <set>
#include <string.h>
#include <vector>

using namespace std;
using namespace string_utils;

namespace file_reader_utils {

// An enum representing the various data types that a column may take.
enum GenericDataType {
  STRING,
  DOUBLE,
  INT,
  INT_64,
  UINT_64,
};

// Holds a data value of one of the three basic types (string, int, double).
// This generic struct allows us to construct a data structure (e.g. a vector)
// of values of different underlying types (e.g. create a vector of GenericDataHolder).
struct GenericDataHolder {
 public:
  string str_;
  double dbl_;
  int int_;
  int64_t int64_;
  uint64_t uint64_;
};

// Parameters for the lines to read.
struct LinesToRead {
  // At most one of the following (pairs of) fields should be specified:
  //  - lines_to_skip_: If non-empty, all other lines except these are read
  //  - lines_to_keep_: Only lines in this set are read
  //  - [range_to_keep_start_, range_to_keep_end_]:
  //      - If both values are negative, these are ignored
  //      - If range_to_keep_start_ < 0 and range_to_keep_end_ >= 0:
  //        Lines 0 through range_to_keep_end_ are read
  //      - If range_to_keep_start_ >= 0 and range_to_keep_end_ < 0:
  //        Lines range_to_keep_start_ through the end of the file are read
  //      - If both range_to_keep_start_ >= 0 and range_to_keep_end_ >= 0:
  //        Lines between range_to_keep_start_ and range_to_keep_end_ (inclusive)
  //        are read
  //  - [range_to_skip_start_, range_to_skip_end_]:
  //      - If both values are negative, these are ignored
  //      - If range_to_skip_start_ < 0 and range_to_skip_end_ >= 0:
  //        Lines 0 through range_to_skip_end_ are skipped
  //      - If range_to_skip_start_ >= 0 and range_to_skip_end_ < 0:
  //        Lines range_to_skip_start_ through the end of the file are skipped
  //      - If both range_to_skip_start_ >= 0 and range_to_skip_end_ >= 0:
  //        Lines between range_to_skip_start_ and range_to_skip_end_ (inclusive)
  //        are skipped
  // Note that the below fields assume C++ indexing for lines:
  // the first line has index '0'.
  set<int> lines_to_skip_;
  set<int> lines_to_keep_;
  int range_to_keep_start_;
  int range_to_keep_end_;
  int range_to_skip_start_;
  int range_to_skip_end_;

  // Abort if a line beginning with the below string is encountered (ignore this
  // field if empty).
  string abort_on_prefix_;

  // Empty-lines may be an error, a valid data row, or a line to be ignored.
  // Set this to true if in the latter case; default is true, so that unless
  // otherwise specified, ReadCsv will skip empty lines. If you want ReadCsv to
  // attempt (and likely fail) to parse emtpy lines, set below field to 'false'.
  bool skip_empty_lines_;
  // Similar to above, but ignoring whitespace (\s, \t) characters.
  bool skip_whitespace_only_lines_;

  LinesToRead() {
    abort_on_prefix_ = "";
    skip_empty_lines_ = true;
    skip_whitespace_only_lines_ = true;
    range_to_keep_start_ = -1;
    range_to_keep_end_ = -1;
    range_to_skip_start_ = -1;
    range_to_skip_end_ = -1;
  }
};

// A structure to hold input parameters to ReadCsv.
  // (these lines are ignored) and values to use for each data type when a row
  // is missing a value.
  // Also, if error_lines is NULL, then returns false whenever a column cannot
  // be parsed into the desired data type; but if non-null, then error lines,
  // together with the (first) column string that caused the problem, are put
  // into error_lines and ReadCsv still returns true (assuming no other errors).
struct ReadCsvInput {
  string filename_;
  set<string> delimiters_;
  // Lines beginning with comment_char_ are ignored.
  string comment_char_;

  bool has_header_;

  // The ReadCsvOutput structure below has a field that can store the
  // (line_index, col_index) of values it could not parse. The below field
  // dictates whether that field is used, or whether ReadCsv should just
  // return false if it encounters values it cannot parse.
  bool allow_error_lines_;

  // Special characters: to/from characters to expect/use for Missing, NA, and
  // infinity. The 'out_na_XXX' fields below specify what symbol to read into
  // the output GenericDataType when a special input character (or misssing value)
  // is encountered.
  //   - NA special in characters
  double in_na_double_;
  int in_na_int_;
  int64_t in_na_int64_;
  uint64_t in_na_uint64_;
  set<string> in_na_strings_;
  //   - NA special out characters
  double out_na_double_;
  int out_na_int_;
  int64_t out_na_int64_;
  uint64_t out_na_uint64_;
  string out_na_string_;
  //   - Infinity special characters
  string in_inf_;
  double out_inf_double_;
  int out_inf_int_;
  int64_t out_inf_int64_;
  uint64_t out_inf_uint64_;

  // Determines which lines of the file to read.
  LinesToRead line_params_;

  // Which columns to read, and their data-type. There are two fields for this:
  // columns_to_read_by_type_ and columns_to_read_. The latter simply lists
  // all the columns to read, indicating the ((1-based)column_index, GenericDataType),
  // and thus the number of columns read equals the size of this vector. The
  // former allows ranges to be specified, and is useful if the number of
  // columns is large and/or is unknown. Specifically, specify ranges by padding
  // a range with '0's (as a hack to indicate the values in between are a range)
  // where you can indicate no end to the range by having only one value in
  // between the '0'-padding. For example, if you were interested in columns:
  //   1-10, 14, 16, 18, 20-22, 24-28, and 30 - END_OF_COLUMS,
  // (say all these columns have same type; columns of other types are handled
  // similarly) then you would specify for this GenericDataType the vector:
  //   [0, 1, 10, 0, 14, 16, 18, 0, 20, 22, 0, 0, 24, 28, 0, 0, 30, 0].
  vector<pair<vector<int>, GenericDataType>> range_columns_to_read_;
  vector<pair<int, GenericDataType>> columns_to_read_;

  ReadCsvInput() {
    filename_ = "";
    delimiters_.insert("\t");
    comment_char_ = "";
    has_header_ = true;
    allow_error_lines_ = false;

    in_na_double_ = numeric_limits<double>::min();
    in_na_int_ = numeric_limits<int>::min();
    in_na_int64_ = numeric_limits<int64_t>::min();
    in_na_uint64_ = numeric_limits<uint64_t>::min();
    in_na_strings_ = NA_STRINGS;

    out_na_double_ = numeric_limits<double>::min();
    out_na_int_ = numeric_limits<int>::min();
    out_na_int64_ = numeric_limits<int64_t>::min();
    out_na_uint64_ = numeric_limits<uint64_t>::min();
    out_na_string_ = "NA";
    
    in_inf_ = "";
    out_inf_double_ = numeric_limits<double>::infinity();
    out_inf_int_ = numeric_limits<int>::max();
    out_inf_int64_ = numeric_limits<int64_t>::max();
    out_inf_uint64_ = numeric_limits<uint64_t>::max(); 
  }
};

// A structure to hold output to calls to ReadCsv.
struct ReadCsvOutput {
  // Holds the read-in data; outer-vector is indexed by (read) line index,
  // inner-vector is indexed by (read) column index.
  vector<vector<GenericDataHolder>> output_;

  // If there was an error in parsing a value, this stores the line and column
  // index of the error. Note that the line index is with respect to lines
  // that are being read, while the column index is the absolute index of
  // the column (as opposed to the index within the set of columns being read).
  set<pair<int, int>> error_line_and_column_;

  // Error Message, in case a function returned false.
  string error_msg_;

  ReadCsvOutput() {
    error_msg_ = "";
  }
};

// A class defining basic helper functions for getting string/double values
// out of the various data holder objects above.
class CsvUtils {
 public:
  // Reads filename (an input file in csv (or table) format by reading in
  // the columns specified by columns_to_read, and populating 'output' with
  // the results. For example, on input file:
  //   AGE  CITY  IQ  HEIGHT
  //   34   L.A.  125 6.1
  //   22   NYC   65  6.5
  //   9    D.C.  87  3.9
  // and columns_to_read = [([1, 3], INT), ([4], DOUBLE)]
  // output would be: [ [int_ = 34, int_ = 125, dbl_ = 6.1],
  //                    [int_ = 22, int_ = 65,  dbl_ = 6.5],
  //                    [int_ = 9,  int_ = 87,  dbl_ = 3.9] ]
  static bool ReadCsv(
      const ReadCsvInput& input, ReadCsvOutput* output);
};

}  // namespace file_reader_utils

#endif
