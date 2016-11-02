// Date: Mar 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description: Utility functions for reading in a file that has data
// organized in a table format.

#include "FileReaderUtils/read_file_utils.h"
#include "MathUtils/data_structures.h"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef READ_TABLE_WITH_HEADER_H
#define READ_TABLE_WITH_HEADER_H

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace math_utils;

namespace file_reader_utils {

class ReadTableWithHeader {
 public:
  // Returns a string describing the expected format of the
  // input file to be read.
  static string PrintInputFormat();

  // Parses the data in 'file' into 'data_values', keeping track of nominal
  // columns (and their values on each row) and rows/columns that have missing
  // (NA) values. If cols_to_read is non-empty, only read the columns specified.
  static bool ReadDataFile(
      const FileInfo& file_info,
      const vector<string>& orig_file_header,
      const set<int>& cols_to_read,
      const vector<VariableParams>& var_params,
      vector<string>* data_values_header,
      vector<vector<DataHolder> >* data_values,
      map<int, set<string> >* nominal_columns,
      map<int, set<int> >* na_columns,
      string* error_msg);
  // Same as above, pass in empty cols_to_read (read all columns).
  static bool ReadDataFile(
      const FileInfo& file_info,
      const vector<string>& orig_file_header,
      const vector<VariableParams>& var_params,
      vector<string>* data_values_header,
      vector<vector<DataHolder> >* data_values,
      map<int, set<string> >* nominal_columns,
      map<int, set<int> >* na_columns,
      string* error_msg) {
    return ReadDataFile(
        file_info, orig_file_header, set<int>(), var_params,
        data_values_header, data_values, nominal_columns, na_columns, error_msg);
  }

  // Collapses input according to collapse_params (just puts input into
  // output if no collapsing is performed).
  static bool CollapseValue(
    const DataHolder& input,
    const vector<VariableCollapseParams>& collapse_params,
    DataHolder* output, bool* was_collapsed);

  // If any of the columns have num_buckets_ set for their collapse params,
  // then collapse all values in that column to be the nearest bucket.
  // NOTE: cols_to_read is needed, as data_values may have different
  // columns (and hence indexing) than the index of the variable's col_ in
  // var_params.
  static bool UpdateBucketColumns(
      const set<int>& cols_to_read,
      const vector<VariableParams>& var_params,
      vector<vector<DataHolder>>* data_values,
      string* error_msg);

  // Given and interval [left_end, right_end] and a point 'orig_value'
  // within that interval, partitions the interval into num_buckets
  // partitions, and assigns the orig_value to the closest bucket.
  static bool GetBucketValue(
    const double& orig_value, const int num_buckets,
    const double& left_end, const double& right_end,
    double* bucket_value, string* error_msg);

 private:
  // Copies data from the old format to the new format.
  static bool CopyDataFromOldFormatToNew(
      const vector<string>& ids, const vector<double>& weights,
      const vector<vector<string>>& families,
      const vector<double>& dep_var,
      const MatrixXd& indep_vars,
      vector<DataRow>* data, string* error_msg);
  // Same as above, with API for Cox (instead of Linear/Logistic).
  static bool CopyDataFromOldFormatToNew(
      const vector<string>& ids, const vector<double>& weights,
      const vector<vector<string>>& families,
      const vector<CensoringData>& dep_var,
      const MatrixXd& indep_vars,
      vector<DataRow>* data, string* error_msg);

  // For columns in 'nominal_columns', go through the data_values, and
  // makes sure all corresponding DataHolder objects have field 'name_'
  // set (if not, convert value_ to name_).
  static bool UpdateNominalColumns(
      const set<int>& nominal_columns,
      vector<vector<DataHolder>>* data_values,
      map<int, set<string>>* nominal_columns_and_values,
      string* error_msg);

  // Adds 'token' to 'values' at position 'current_col_index' (if this index is
  // less than size of 'values', otherwise pushes a new DataHolder to back of
  // 'values'). Also, if 'token' cannot be parsed as a numeric value (or if
  // 'nominal_columns' indicates the 'current_col_index' is a nominal column),
  // then the DataHolder that gets added to values will populate the 'name_'
  // field instead of the 'value_' field, and 'nominal_colunns' will be updated
  // by adding <'current_col_index', 'token'> to it (if not already there).
  // 'line_num' is only needed to determine if this is the first row of
  // data (so that if a token cannot be parsed as a numeric value, we know
  // whether to return false or not (return false if previous rows were
  // parsed as numeric values; return true if this is the first data row, and
  // mark the column as nominal).
  static bool AddToken(
      const string& token, const bool is_first_data,
      const int data_col_index, const int col_index_in_file,
      const set<string> na_strings,
      const vector<VariableCollapseParams>& collapse_params,
      vector<DataHolder>* values, map<int, set<string> >* nominal_columns,
      bool* new_nominal_col_found,
      bool* is_na, set<int>* na_cols_for_line,
      string* error_msg);

  // If expected_num_columns >=0, then this function sets 'values' to be a
  // vector of size 'expected_num_columns', and then attempts to read this many
  // values from the given 'line'. Otherwise, reads however many items exist
  // in the line into 'values'. Also populates 'nominal_columns' with the
  // columns (and all distinct values that appeared in that column) that were
  // indicated as 'nominal' via the header file (via '$') or because there was
  // a value in that column that could not be parsed as a double.
  // Also populates 'na_cols_for_line' with the indices of columns for which
  // this line indicates 'NA'.
  // Returns true if successful, otherwise returns false (and 'error_msg'
  // contains details of the failure).
  static bool ReadLine(
      const string& line, const string& delimiter,
      const int expected_num_columns, const int line_num,
      const set<string>& na_strings, const vector<VariableParams>& var_params,
      vector<DataHolder>* values, map<int, set<string> >* nominal_columns,
      set<int>* na_cols_for_line, vector<bool>* is_first_data_value_for_col,
      bool* new_nominal_col_found, string* error_msg);
  // Same as above, but only reads the indicated columns.
  static bool ReadLine(
      const string& line, const string& delimiter,
      const set<int>& cols_to_read, const int line_num,
      const set<string>& na_strings, const vector<VariableParams>& var_params,
      vector<DataHolder>* values, map<int, set<string> >* nominal_columns,
      set<int>* na_cols_for_line, vector<bool>* is_first_data_value_for_col,
      bool* new_nominal_col_found, string* error_msg);
};

}  // namespace file_reader_utils

#endif
