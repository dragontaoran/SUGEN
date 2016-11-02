#include "read_file_utils.h"
#include "MathUtils/data_structures.h"
#include "StringUtils/string_utils.h"
#include "TestUtils/test_utils.h"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef READ_INPUT_H
#define READ_INPUT_H

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace math_utils;
using namespace string_utils;
using namespace test_utils;
using namespace std;

namespace {

static const char kSubgroupString[] = "I_Subgroup";

bool CopyVariables(
    const vector<double>& in, vector<CensoringData>* out, string* error_msg) {
  // This API needed only because Template doesn't know that this method
  // should never be called with this type for 'out'.
  // Do nothing, but record error.
  if (error_msg != nullptr) {
    *error_msg +=
        "ERROR in CopyVariables: Unexpected call with CensoringData API.\n";
  }
  return false;
}

bool CopyVariables(
    const vector<double>& in, vector<double>* out, string* error_msg) {
  if (in.empty() || out == nullptr) return true;
  out->clear();
  for (const double& value : in) {
    out->push_back(value);
  }
  return true;
}

bool CopyVariables(
    const vector<double>& in, VectorXd* out, string* error_msg) {
  if (in.empty() || out == nullptr) return true;
  out->resize(in.size());
  for (int i = 0; i < in.size(); ++i) {
    (*out)[i] = in[i];
  }
  return true;
}

bool CopyVariables(
    const vector<VectorXd>& in, vector<VectorXd>* out, string* error_msg) {
  if (in.empty() || out == nullptr) return true;
  out->clear();
  for (const VectorXd& row : in) {
    out->push_back(row);
  }
  return true;
}

bool CopyVariables(
    const vector<VectorXd>& in, MatrixXd* out, string* error_msg) {
  if (in.empty() || out == nullptr) return true;
  const VectorXd& first_row = in[0];
  out->resize(in.size(), first_row.size());
  for (int i = 0; i < in.size(); ++i) {
    out->row(i) = in[i];
  }
  return true;
}

}  // namespace

namespace file_reader_utils {

class ReadInput {
 public:
  static string GetErrorTermId();
  // Asks user to enter the desired number of data rows to simulate ("n").
  static int GetNumDataRowsPerSimulationFromUser();

  // Reads in filename, prompts user to enter model, and generates the
  // dependent/independent variables used in that model.
  // Populates dep_var and indep_vars with these values, and populates
  // legend with all of the linear terms (for the independent variables)
  // in the model. Returns true if successful, false otherwise (in which
  // case an error message containing the error is printed).
  static bool ReadFileAndGetModel(ModelAndDataParams* params);

  // Prompts user for the Linear Model to be solved. Makes sure that the model
  // specified is consistent:
  //   - The syntax is recognizable
  //   - The operations specified are supported (currently +, *, log, exp)
  //   - The NOMINAL variables never have log or exp applied to them
  // If any of the above fail, user is re-prompted to enter a valid model.
  // Populates dependent_var and model_rhs with the model; e.g. for model:
  //   Log(Y) = c_0 + c_1 * Log(X_1) * X_2 + c_2 * exp(X_2),
  // user should enter (notice contants and error term are not entered):
  //   Log(Y) = Log(X_1) * X_2 + exp(X_2)
  // and then dependent_var will represent Log(Y), and model_rhs will
  // represent Log(X_1) * X_2 + exp(X_2).
  // Also populates input_cols_used with the indices of all columns (from
  // the input data file) that are used in (either side of) the model.
  // Returns true unless an unexpected error is encountered.
  static bool GetModel(
      const ModelType& model_type,
      const vector<string>& variable_names,
      const map<int, set<string> >& nominal_columns,
      DepVarDescription* model_lhs, Expression* model_rhs,
      set<int>* input_cols_used, string* error_msg);
  // Same as above, but with no a-priori assumptions about valid variable
  // names nor which variables are nominal (only relevant when sanity checking
  // model; i.e. we won't be verifying that the variable names user specifies
  // are already present, e.g. in a data file).
  static bool GetModel(
      const ModelType& model_type,
      DepVarDescription* model_lhs, Expression* model_rhs, string* error_msg);

  // Given a model as described by dependent_var and model_rhs, populates
  // dep_var and indep_vars with the appropriate values, and prints the final
  // titles in 'legend'. For example, if model is:
  //   Log(Y) = X_1 * Log(X_2) + X_2 + X_1,
  // and X_1 is a NOMINAL variable with three possible values (so there will
  // be two corresponding indicator variables for it), then final formula is:
  //   Log(Y) = X_1,1 * Log(X_2) + X_1,2 * Log(X_2) + X_2 + X_1,1 + X_1,2
  // and legend is:
  //   [(X_1,1 * Log(X_2)), (X_1,2 * Log(X_2)), X_2, X_1,1, X_1,2]
  // Uses na_columns and input_cols_used to determine which rows from the input
  // data file to skip, based on that row having 'NA' in a relevant column.
  // Returns true unless as unexpected error is encountered.
  // NOTE: Missing values in 'data_values' should be represented by
  // NA_STRING ("PHB_NA_PHB"), *not* by one of the na strings in
  // input->file_.na_strings_ (i.e. whatever method created data_values,
  // probably read_table_with_header::ReadDataFile(), should have already
  // replaced all missing values (according to input->file_.na_strings_)
  // with NA_STRING).
  // NOTE: The is_simulated_data input controls whether the dependent variable
  // value(s) are read from (a column of) data_values, or whether they are
  // already populated in 'input' on input (for simulations, the dependent
  // variable value is computed directly via an equation involving the
  // independent variables, use-cases for simulated data need to be sure to
  // populate the dep_vars_ field of ModelAndDataParams before calling this fn.
  static bool StoreDataValuesInParams(
      const bool is_simulated_data,
      const vector<string>& data_values_header,
      const vector<vector<DataHolder> >& data_values,
      ModelAndDataParams* input);
  // Same as above, with default 'false' for is_simulated_data.
  static bool StoreDataValuesInParams(
      const vector<string>& data_values_header,
      const vector<vector<DataHolder> >& data_values,
      ModelAndDataParams* input) {
    return StoreDataValuesInParams(false, data_values_header, data_values, input);
  }

  // Fill Category (2)-(4) fields of params, by calling:
  //   - ParseModelAndDataParams() for Category (2) fields
  //   - ReadDataFile() for Category (3) fields
  //   - StoreDataValuesInParams() for Category (4) fields
  // On input, Category (1) fields should already be populated.
  static bool FillModelAndDataParams(ModelAndDataParams* params);

 private:
  // Returns true if value is any of:
  //   '1', '1.0', 'T', 'true', 'True', or 'TRUE'
  static bool IsTrueIndicator(const string& value);
  // Returns true if value is any of:
  //   '0', '0.0', 'F', 'false', 'False', or 'FALSE'
  static bool IsFalseIndicator(const string& value);

  // Prompts user with input request 'query_text'. If user-entered response
  // is parsable as an int, return that int. Otherwise, display error to
  // user and re-prompt.
  static int GetIntegerFromUser(const string& query_text);
  // Ditto above, except tries to parse response as a double.
  static double GetDoubleFromUser(const string& query_text);
  // Asks user to enter the desired number of simulations to run ("k").
  static int GetNumSimulationsFromUser();

  // Attempts to parse strata by finding the referenced strings in 'header'
  // and populates 'strata_cols' with the corresponding indices.
  // The string should have format:
  //   (STRATA_1, STRATA_2, ..., STRATA_N)
  static bool GetColumnIndicesFromString(
      const string& strata, const vector<string>& header, set<int>* strata_cols);
  // Attempts to parse subgroup by finding the referenced strings in 'header'
  // and populates 'subgroup_cols' with the corresponding indices. This is same
  // as above, except we use a vector rather than a set, in case the original
  // order of terms in subgroup string are important (for subgroup they are,
  // since there is a RHS of the --subgroup parameter, which specifies values
  // to match, in the same order as the columns were listed on the LHS).
  static bool GetColumnIndicesFromString(
      const string& subgroup, const vector<string>& header,
      vector<int>* subgroup_cols);

  // Returns appropriate text (strings) that will be used to prompt user for
  // appropriate input, depending on whether the Model is Linear/Logistic or
  // Cox Proportional Hazards (we distinguish among these cases based on
  // whether the dependent variable is a single Expression (linear/logistic)
  // or a vector of Expressions (one of each of the time variables, and one
  // for the status ("Delta") variable; for Cox).
  static string PromptUserForModelText(const ModelType& type);

  // Prompts user for the Linear Model to be solved. Makes sure that the model
  // is parsable, and if so, populates dependent_var and model_rhs with the
  // model and returns true. Returns false otherwise.
  static bool PromptForAndParseModel(
      const bool from_file, const ModelType& model_type,
      DepVarDescription* model_lhs, Expression* model_rhs, string* error_msg);

  // Takes in a string that should be a list of variables the user
  // wants to indicate as being non-numeric, and checks variable_names
  // to get the index of the column (with respect to input data) of these
  // variables.
  static bool ProcessUserEnteredNominalColumns(
      const string& user_input, const vector<string>& variable_names,
      set<int>* nominal_columns, string* error_msg);
  
  // Prompts user for desired nominal columns, displaying first the columns that
  // were already detected as nominal (based on Title line having a '$' next
  // to that variable name, and/or non-double values in the column). Updates
  // nominal_columns with the actual columns the user wants to be nominal.
  static bool GetNominalColumns(
      const vector<string>& variable_names,
      vector<vector<DataHolder> >& data_values,
      map<int, set<string> >* nominal_columns, string* error_msg);

  // Updates 'columns' to reflect the indices in 'new_columns' by:
  //  (A) For indices in 'columns' that are NOT present in 'new_columns',
  //      check to make sure all data values in the corresponding set can
  //      be parsed as a double; otherwise return false
  //  (B) For indices in 'new_columns' that are not in 'columns', add the
  //      index to 'columns', with the corresponding set being all the
  //      (distinct) data values that appear among the data values for
  //      that column
  //  Returns true as long as (A) above does not fail.
  static bool UpdateNominalColumns(
      vector<vector<DataHolder> >& data_values,
      const set<int>& new_columns,
      map<int, set<string> >* columns,
      string* error_msg);

  // Returns a string describing the expected format of the model.
  static string ExampleModel(const ModelType& type);

  // Let's consider how user specifies subgroups (via a command-line arg):
  //   --subgroup "(age, height, sex) = {(0, 5, M), (1, 4, F), ...}
  // Then the subgroup columns are the 'age', 'height', and 'sex' columns;
  // 'subgroup_cols' would then be a vector of length 3, whose first
  // coordinate represents the column index of 'age', 2nd coordinate is the
  // column index of 'height', and 3rd is column index of 'sex'.
  // The 'subgroups' input then represents the RHS of the --subgroup arg
  // above; i.e. the outer-vector has length how ever many tuples appear
  // (in example above, two 3-tuples are listed explicitly, but more are
  // implied); the inner-vectors all have length 3; for example, the first
  // element of 'subgroups' would be ["0", "5", "M"].
  // 'sample_values' has the actual data values for a given row. 
  // As output, look at the values in each of the subgroup cols, and determine
  // which subgroup (if any) the given row belongs to (i.e. the index of
  // the vector in 'subgroups' whose values exactly match this rows values
  // in each of the subgroup columns). If no such match is found,
  // subgroup_index is not changed (so caller is responsible for setting
  // a distinct value, e.g. a negative vlaue, so they can test if the value
  // changed). 
  static bool ParseSubgroup(
      const vector<string>& data_values_header,
      const vector<VariableColumn>& subgroup_cols,
      const vector<vector<string>>& subgroups,
      const vector<DataHolder>& sample_values,
      int* subgroup_index);

  // If strata_cols is non-empty, for indices in strata_cols, looks for the
  // corresponding values in sample_values, and puts them in a vector; then
  // sees if such a vector already exists in strata_names. If so (say there
  // is a match in position 'j' of strata_names), sets strata_index = j
  // and adds (row_index, j) to 'strata'. If not, define j = strata_names.size(),
  // then pushes the new string vector to the back of strata_names, and adds a
  // new strata to 'strata' by pushing back: (row_index, j); i.e. the index
  // of the strata is the index of the corresponding string in strata_names.
  static bool ParseStrata(
      const int row_index,
      const vector<string>& data_values_header,
      const set<VariableColumn>& strata_cols,
      const vector<DataHolder>& sample_values,
      int* strata_index,
      vector<vector<string>>* strata_names,
      map<int, int>* row_to_strata);

  // Uses the indicated column index to retrieve id information for this row.
  static bool ParseSampleId(
      const int orig_id_col, const set<int>& input_cols_used,
      const vector<DataHolder>& sample_values,
      vector<string>* ids, string* error_msg);

  // Uses the indicated column index to retrieve weight information for this row.
  static bool ParseSampleWeight(
      const int orig_weight_col, const set<int>& input_cols_used,
      const vector<DataHolder>& sample_values,
      vector<double>* weights, string* error_msg);

  // Uses the indicated column indices to retrieve family info for this row.
  static bool ParseSampleFamilies(
      const vector<VariableColumn>& orig_family_cols,
      const set<int>& input_cols_used,
      const vector<DataHolder>& sample_values,
      vector<vector<string>>* families, string* error_msg);

  // Reads dependent_var into dep_var; where multiple values are parsed
  // in the case the variable is of Nominal type.
  static bool ComputeRowDependentVariableValues(
      const ModelType& model_type,
      const vector<DataHolder>& sample_values,
      const map<string, int>& name_to_column,
      const map<int, set<string> >& nominal_columns,
      const DepVarDescription& dependent_var,
      DepVarHolder* dep_var, string* error_msg);

  // Use statistics (mean and standard deviation) for each linear term to
  // standardize the values in linear_terms.
  static bool StandardizeLinearTerms(
      const VariableNormalization standardize_vars,
      const set<int>& non_binary_linear_terms,
      const vector<double>& linear_terms_sums,
      const vector<double>& linear_terms_sums_squared,
      vector<tuple<bool, double, double>>* linear_terms_mean_and_std_dev,
      MatrixXd* linear_terms, string* error_msg);
};

}  // namespace file_reader_utils

#endif
