// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description:
//   Utility functions for reading in data.

#include "FileReaderUtils/read_file_structures.h"
#include "MapUtils/map_utils.h"
#include "MathUtils/data_structures.h"
#include "TestUtils/test_utils.h"

#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef READ_FILE_UTILS_H
#define READ_FILE_UTILS_H

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace map_utils;
using namespace math_utils;
using namespace test_utils;
using namespace std;

namespace file_reader_utils {

/* ================================ Constants =============================== */

extern const set<string> NA_STRINGS;

/* ============================== END Constants ============================= */

/* =============================== Functions ================================ */
// Returns just the filename portion of the filepath.
// For example: "foo/bar/zed.txt" would return "zed.txt".
extern string GetFileName(const string& filepath);
// Returns the directory of the given (full) filepath, or "." if the filepath
// contains no directory (as determined by presence of "/").
// For example, input "foo/bar/file.txt" would return "foo/bar".
extern string GetDirectory(const string& filepath);
// Returns the working directory (where the executable is running).
extern string GetWorkingDirectory();
// For given directory, find all the files in that directory. Returns false
// if the directory is not found; if directory is found but empty, returns
// true (but files will be empty). 'directory' can be empty (interpreted as "./")
// and can either contain the final "/" suffix or not.
// NOTE: This method is robust for running on windows or unix: as long as it is
// compiled on the same system that it is run.
extern bool GetFilesInDirectory(
    const bool full_path, const string& directory, vector<string>* files);
// Same as above, uses default value 'true' for full_path.
inline bool GetFilesInDirectory(const string& directory, vector<string>* files) {
  return GetFilesInDirectory(true, directory, files);
}

extern bool PrintDataToFile(
    const string& filename, const string& sep, const vector<string>& header,
    const vector<vector<DataHolder> >& data_values);
inline bool PrintDataToFile(
    const FileInfo& file_info, const vector<string>& header,
    const vector<vector<DataHolder> >& data_values) {
  return PrintDataToFile(file_info.name_, file_info.delimiter_, header, data_values);
}

extern string PrintModelAndDataParams(const ModelAndDataParams& params);

// Copies all Category 1 fields of params_one to params_two, with caveats:
//   1) Model Type (type_) and Model (model_str_) are only copied if they
//      haven't already been set in params_two.
//   2) Subgroup and Strata are not copied (since in general these will be
//      different for each model, and unlike model above (which is mandatory,
//      and therefore its absence means it hasn't been set yet), there is no way
//      to distinguish between the case that Subgroup (resp. Strata) was
//      deliberately not-set (should be empty) vs. the case that we desire
//      to copy params_one values. Rather than guess, callers must
//      explicitly copy Subgroup and Strata fields before/after calling the
//      below CopyModelAndDataParams()) function.
extern void CopyModelAndDataParams(
    const ModelAndDataParams& params_one, ModelAndDataParams* params_two);

// Returns a string representing the input VariableCollapseParams.
extern string GetVariableCollapseParamsString(
    const VariableCollapseParams& params);

// Prints subgroup and strata to the provided string.
extern bool GetModelAndDataParamsSubgroupsAndStrata(
    const ModelAndDataParams& params, string* subgroup_and_strata_str);
// Similar to above, with differen API for describing Subgroups and Strata.
extern bool GetSubgroupAndStrataDescription(
    const map<int, set<int>>& rows_in_each_strata,
    const map<int, set<int>>& subgroup_rows_per_index,
    string* subgroup_and_strata_str);

// Prints Summary Statistics about a Subgroup to the indicated file.
extern bool PrintSubgroupSynopsis(
    const vector<VariableColumn>& subgroup_cols,
    const vector<vector<string>>& subgroups,
    const map<int, set<int>>& subgroup_index_to_its_rows,
    ofstream& out_file);
// Same as above, but prints to provided string instead of output file.
extern bool PrintSubgroupSynopsis(
    const vector<VariableColumn>& subgroup_cols,
    const vector<vector<string>>& subgroups,
    const map<int, set<int>>& subgroup_index_to_its_rows,
    string* output);
// Same as above, but doesn't give the counts (number of rows) belonging
// to each subgroup (to be used e.g. for simulations, when the number
// of rows per subgroup may not be a constant across all iterations).
extern bool PrintSubgroupSynopsis(
    const vector<VariableColumn>& subgroup_cols,
    const vector<vector<string>>& subgroups,
    ofstream& out_file);

// Removes Windows formating. New lines on windows are marked by \r\n, where
// \r is Carriage Return and \n is New Line.
// Typically, when RemoveWindowsTrailingCharacters is called, it is from
// a line retrieved via 'getline', and hence the trailing \n has already
// been removed. But, we go ahead and check for it here anyway (in case
// caller did not first use 'getline').
// Thus, this functions removes \r and \r\n suffixes (\r is identified as
// char index 13, and \n is char index 10).
extern void RemoveWindowsTrailingCharacters(string* input);

// Takes in ModelAndDataParams with some fields populated, and fills in the
// fields it can. In particular:
//   Fields that should already be set on input:
//     - file_
//     - model_type_
//     - model_str_
//     - (Optional) header_
//     - subgroup_str_
//     - strata_str_
//     - collapse_params_str_
//     - time_params_str_
//     - var_norm_params_str_
//     - id_str_
//     - weight_str_
//     - family_str_
//   Fields that get populated:
//     - header_ (if it wasn't already set in input)
//     - nominal_columns_ (only columns identified as nomial from their
//       names on the header line; i.e. this fn doesn't look at actual data)
//     - model_lhs_
//     - model_rhs_
//     - input_cols_used_ (from model, subgroups/strata)
//     - subgroups_cols_ (from subgroup LHS)
//     - subgroups_ (representation of subgroup RHS)
//     - use_subgroup_as_covariate_ (from whether "I_subgroup" appears in RHS of model)
//     - strata_cols_
//     - id_col_
//     - weight_col_
//     - family_cols_
//     - var_params_
//   Fields *not* populated (must be populated later, when data is actually read,
//   e.g. via StoreDataValuesInParams()):
//     - legend_
//     - final_model_
//     - subgroup_rows_per_index_
//     - row_to_strata_
//     - ids_
//     - weights_
//     - families_
//     - nominal_columns_ (partially filled above based on header, but more
//       columns could later be added based on data values in the columns)
//     - nominal_columns_and_values_
//     - na_rows_and_columns_
//     - na_rows_skipped_
//     - dep_vars_
//     - dep_vars_cox_
//     - linear_term_values_
extern bool ParseModelAndDataParams(ModelAndDataParams* params);

// Parses the input model (string) by separating the (names of the) dependent
// and independent variables (i.e. splits the model around the "=" sign).
// The (names of the) dependent variable(s) are stored in "model_lhs",
// and the (names of the) independent variable(s) are stored in "model_rhs".
// 'header' is used to make sure all variables listed in the model appear
// in the header, as well as to indicate which column index they are (which
// is then in-turn used to populate 'input_cols_used'). Also provides basic
// sanity-checking of the RHS of the model (that it can be parsed as an
// Expression, that 'log' and 'exp' aren't applied to nominal variables, etc.
extern bool ParseModel(
    const string& model, const string& left_truncation_col_name,
    const ModelType type, const vector<string>& header,
    DepVarDescription* model_lhs, Expression* model_rhs,
    set<int>* input_cols_used, bool* use_subgroup_as_covariate,
    string* error_msg);

// Read the first line of 'file', placing each read item into 'titles'.
// Returns true if file was successfully parsed, false otherwise.
extern bool GetTitles(const string& title_line, const string& delimiter,
                      vector<string>* titles,
                      set<int>* nominal_columns);

// Reads the top (non-comment) line of the indicated file, and reads each
// column's title (columns are determined/separated by 'delimiter') into
// 'header'.
extern bool GetHeader(
    const FileInfo& file_info, vector<string>* header, string* error_msg);

// Returns a string expression for the dependent variable(s).
extern string GetDependentVarString(
    const ModelType model_type, const DepVarDescription& dependent_var);

// Prints the final model to final_model.
// Linear models have an extra error term (that was NOT specified by user)
// in the model; Logistic and Cox PH models do not.
extern void PrintFinalModel(
    const ModelType type, const DepVarDescription& model_lhs,
    const vector<string>& legend, string* final_model);

// Combines the provided Variable parameters (all keyed by variable name) into
// a single VariableParams data structure.
extern bool CombineVariableParams(
    const vector<string>& header,
    const map<string, vector<VariableCollapseParams>>& var_name_to_collapse_params,
    const map<string, TimeDependentParams>& var_name_to_time_params,
    const map<string, VariableNormalization>& var_name_to_normalization,
    vector<VariableParams>* var_params, string* error_msg);

// Makes sure that the model specified by model_lhs and model_rhs
// is consistent:
//   - The syntax is recognizable
//   - The operations specified are supported (currently +, *, log, exp)
//   - The NOMINAL variables never have log or exp applied to them
// If so, populates input_cols_used with all the columns referred to by
// model_lhs and model_rhs, and returns true. Otherwise, returns false
// and populates error_msg with the reason for failure.
extern bool SanityCheckModel(
    const ModelType& type,
    const map<string, int>& titles,
    const map<int, set<string> >& nominal_columns,
    const DepVarDescription& model_lhs, const Expression& model_rhs,
    set<int>* input_cols_used, string* error_msg);

// Makes sure that the model specified by model_lhs is consistent:
//   - The syntax is recognizable
//   - The operations specified are supported (currently +, *, log, exp)
//   - The NOMINAL variables never have log or exp applied to them
// If so, populates input_cols_used with all the columns referred to by
// model_lhs and returns true. Otherwise, returns false.
extern bool SanityCheckDependentVariable(
    const ModelType& type,
    const map<string, int>& titles,
    const map<int, set<string> >& nominal_columns,
    const DepVarDescription& model_lhs,
    set<int>* input_cols_used,
    string* error_msg);

// Makes sure that the model specified by model_rhs is consistent:
//   - The NOMINAL variables never have log or exp applied to them
// If so, populates input_cols_used with all the columns referred to by
// model_rhs and returns true. Otherwise, returns false.
extern bool SanityCheckIndependentVariables(
    const map<string, int>& variable_names_and_cols,
    const map<int, set<string> >& nominal_columns,
    const Expression& model_rhs,
    set<int>* input_cols_used,
    string* error_msg);
// Same as above, but can be used for dependent or independent variables.
extern bool SanityCheckVariables(
    const map<string, int>& variable_names_and_cols,
    const map<int, set<string> >& nominal_columns,
    const Expression& variable_expression,
    set<int>* input_cols_used,
    string* error_msg);

// Base on user-entered 'model', parses the model into the model_lhs
// and model_rhs; returns true if model could be successfully parsed,
// and false otherwise (with error_msg containing the reason for failure).
extern bool ProcessUserEnteredModel(
    const string& model, const ModelType& model_type,
    DepVarDescription* model_lhs, Expression* model_rhs, string* error_msg);

// Calls the appropriate ParseDependentTerm* function below (based on model_type)
// to parse the appropriate field of dep_term.
extern bool ParseDependentTerm(
    const string& input, const ModelType& model_type,
    DepVarDescription* dep_term, string* error_msg);
// Same as above, but for the case of a Linear Model.
extern bool ParseDependentTermLinearModel(
    const string& input,
    Expression* dep_term, vector<string>* dep_var_name, string* error_msg);
// Same as above, but for the case of a Logistic Model.
extern bool ParseDependentTermLogisticModel(
    const string& input,
    Expression* dep_term, vector<string>* dep_var_name, string* error_msg);
// Same as above, but for the case of a Cox PH Model.
extern bool ParseDependentTermCoxModel(
    const string& input,
    vector<Expression>* dep_term, vector<string>* time_vars_names,
    vector<string>* dep_vars_names, string* error_msg);
// Same as above, but for the case of a Time-Dependent NPMLE Model.
extern bool ParseDependentTermTimeDependentNpmleModel(
    const string& input,
    vector<Expression>* dep_term, vector<string>* time_vars_names,
    vector<string>* dep_vars_names, string* error_msg);
// Same as above, but for the case of a Time-Independent NPMLE Model.
extern bool ParseDependentTermTimeIndependentNpmleModel(
    const string& input,
    vector<Expression>* dep_term, vector<string>* time_vars_names,
    vector<string>* dep_vars_names, string* error_msg);

// Evaluates each Expression in linear_terms, populating linear_terms_values
// with the results.
extern bool ComputeRowCovariateValues(
    const int subgroup_index,
    const vector<vector<string>>& subgroups,
    const set<string> na_strings,
    const map<int, set<string> >& nominal_columns,
    const vector<Expression>& linear_terms,
    const map<string, int>& name_to_column,
    const vector<DataHolder>& sample_values,
    VectorXd* linear_terms_values,
    string* error_msg);

// Uses the values in 'sample_values' (which represents one row of
// data from the input file), determines what value to use for
// every variable that may appear in the model. Note that the output
// data structure might have different keys (and size) as the input, because:
//   1) If I_Subgroup is in the model RHS, then there will be extra Keys
//      for each (Subgroup) indicator variable
//   2) If there are any nominal variables, there will be (m - 1) variables
//      created from these.
// Input:
//  - subgroup_index: If I_Subgroup appears in the RHS of the model, then
//                    when this function is called, the Subgroup (index) that
//                    this row (Subject) belongs to should have already
//                    been determined; this input field stores it. If
//                    I_Subgroup does NOT appear in the model RHS, pass
//                    -1 for this field (negative values signal to ignore it)
//  - subgroups:      The out vector holds the Subgroups, so in particular
//                    subgroup_index indicates the element of 'subgroups'
//                    that this row belongs to. The inner vector is a list
//                    of the values (in each column that defines a subgroup)
//                    that define the particular Subgroup in question. The
//                    'subgroups' field is needed to expand I_Subgroups
//                    into the appropriate number of indicator variables, and
//                    then subgroup_index determines which (if any) of
//                    these indicators is '1' for this row (all others get '0').
//                    If subgroup_index is negative, this field is ignored
//                    (and in particular, this field should be an empty vector).
//  - na_strings:     If any value on this row matches a value in this set,
//                    the row is treated as NA (is_na_row is set to true)
//  - nominal_columns:The set of columns that have non-numeric values, and the
//                    set of distinct values found in such columns. For
//                    each nominal column, this function will create (m - 1)
//                    (indicator) variables, and assign value 0/1 to each
//                    based on this row's value in that column.
//  - name_to_column: Variable name to column index. Also, this determines
//                    which columns of the data row get read; i.e. columns
//                    not present in this map will not be read.
//  - sample_values:  The data values for this row
// Output:
//  - is_na_row:      If non-null, will be true if any of the values on
//                    this row were in the set 'na_strings'. If null,
//                    it is assumed all values are not NA (so 'na_strings'
//                    input will be ignored).
//  - var_values:     Map from (expanded) variable name to its value.
extern bool GetVariableValuesFromDataRow(
    const int subgroup_index,
    const vector<vector<string>>& subgroups,
    const set<string> na_strings,
    const map<int, set<string> >& nominal_columns,
    const map<string, int>& name_to_column,
    const vector<DataHolder>& sample_values,
    bool* is_na_row, map<string, double>* var_values, string* error_msg);
// Same as above, without Subgroups.
inline bool GetVariableValuesFromDataRow(
    const set<string> na_strings,
    const map<int, set<string> >& nominal_columns,
    const map<string, int>& name_to_column,
    const vector<DataHolder>& sample_values,
    bool* is_na_row, map<string, double>* var_values, string* error_msg) {
	return GetVariableValuesFromDataRow(
		-1, vector<vector<string>>(), na_strings, nominal_columns, name_to_column,
		sample_values, is_na_row, var_values, error_msg);
}
// Same as above, without Subgroup and na_strings.
inline bool GetVariableValuesFromDataRow(
    const map<int, set<string> >& nominal_columns,
    const map<string, int>& name_to_column,
    const vector<DataHolder>& sample_values,
    map<string, double>* var_values, string* error_msg) {
	return GetVariableValuesFromDataRow(
		set<string>(), nominal_columns, name_to_column, sample_values,
		nullptr, var_values, error_msg);
}
// Same as above, with slightly different API (nominal columns indexed
// by Variable Name instead of Variable Column).
inline bool GetVariableValuesFromDataRow(
    const set<string> na_strings,
    const map<string, set<string> >& nominal_variables,
    const map<string, int>& name_to_column,
    const vector<DataHolder>& sample_values,
    bool* is_na_row, map<string, double>* var_values, string* error_msg) {
  // Translate nominal_variables to be keyed by column index (use name_to_column
  // to provide the mapping).
  map<int, set<string>> nominal_columns;
  for (const pair<string, set<string>>& nominal_name_and_col : nominal_variables) {
    const string& var_name = nominal_name_and_col.first;
    const set<string>& var_values = nominal_name_and_col.second;
    map<string, int>::const_iterator name_to_col_itr = name_to_column.find(var_name);
    if (name_to_col_itr != name_to_column.end()) {
      nominal_columns.insert(make_pair(name_to_col_itr->second, var_values));
    }
  }
  return GetVariableValuesFromDataRow(
      na_strings, nominal_columns, name_to_column, sample_values,
      is_na_row, var_values, error_msg);
}
// Same as above, but without na_strings and is_na_row.
inline bool GetVariableValuesFromDataRow(
    const map<string, set<string> >& nominal_variables,
    const map<string, int>& name_to_column,
    const vector<DataHolder>& sample_values,
    map<string, double>* var_values, string* error_msg) {
  return GetVariableValuesFromDataRow(
      set<string>(), nominal_variables, name_to_column, sample_values,
      nullptr, var_values, error_msg);
}
// Same as above, but with different API. Namely, instead of mapping variable
// name to a column (and then having nominal_columns and sample_values
// indexed by this column), we use the variable name directly to index
// the nominal columns and sample values. Also, vars_in_linear_terms
// is used to determine which columns are read (ignore columns not in it).
extern bool GetVariableValuesFromDataRow(
    const set<string>& vars_in_linear_terms,
    const int subgroup_index,
    const vector<vector<string>>& subgroups,
    const set<string> na_strings,
    const map<string, set<string> >& nominal_columns,
    const map<string, DataHolder>& sample_values,
    bool* is_na_row, map<string, double>* var_values, string* error_msg);
// Same as above, without Subgroups.
inline bool GetVariableValuesFromDataRow(
    const set<string>& vars_in_linear_terms,
    const set<string> na_strings,
    const map<string, set<string> >& nominal_variables,
    const map<string, DataHolder>& sample_values,
    bool* is_na_row, map<string, double>* var_values, string* error_msg) {
  return GetVariableValuesFromDataRow(
      vars_in_linear_terms, -1, vector<vector<string>>(), na_strings,
      nominal_variables, sample_values, is_na_row, var_values, error_msg);
}
// Same as above, without na_strings.
inline bool GetVariableValuesFromDataRow(
    const set<string>& vars_in_linear_terms,
    const map<string, set<string> >& nominal_variables,
    const map<string, DataHolder>& sample_values,
    map<string, double>* var_values, string* error_msg) {
  return GetVariableValuesFromDataRow(
      vars_in_linear_terms, set<string>(), nominal_variables, sample_values,
      nullptr, var_values, error_msg);
}
// Same as above, without vars_in_linear_terms (read all columns).
// DEPRECATED: Since the same set of strings will get generated for every row,
// users should just call Keys() on 'sample_values' once (before iterating
// through data rows), and then use and API above; otherwise, wasting time.
inline bool GetVariableValuesFromDataRow(
    const map<string, set<string> >& nominal_variables,
    const map<string, DataHolder>& sample_values,
    map<string, double>* var_values, string* error_msg) {
  const set<string> col_names_to_read = Keys(sample_values);
  return GetVariableValuesFromDataRow(
      col_names_to_read, nominal_variables, sample_values, var_values, error_msg);
}

// Breaks the model RHS (represented by 'expression') into its linear terms.
// Also expands all non-numeric variables (including I_Subgroup) into the
// appropriate number of (indicator) variables. The final model RHS (now
// separated into linear terms, and fully expanded due to non-numeric
// covariate expansion) is stored on legend, with each linear term
// stored as an Expression in linear_terms.
// NOTE: is_cox determines if an extra term (not explicitly present in the
// string representation of the RHS of the model) should be added: this is
// because for linear/logistic (i.e. non-cox) models, the user must enter
// the model RHS *without* explicitly listing the constant term, and thus
// it needs to be manually added.
// NOTE: If "I_Subgroup" is *not* part of the model (i.e. does not appear
// as a variable name in 'expression), then the input 'subgroups' parameter
// is ignored, and in particular subgroup will not appear in the legend
// nor linear_terms (as desired).
// NOTE: the Expressions in linear_terms may have sub-expressions (i.e.
// fields 'subterm_one_' and 'subterm_two_') that were created on the
// heap (i.e. using 'new'); caller is responsible for deleting each
// Expression in linear_terms.
// NOTE: Each Expression of linear_terms is created on the stack (so doesn't
// require deleting); however, each subterm of each of these expressions is
// created on the heap, so any code that uses this function must explicitly
// call DeleteExpression on each of the expressions of linear_terms.
extern bool GetLegendAndLinearTerms(
    const bool has_constant_term,
    const map<string, int>& variable_name_to_column,
    const vector<vector<string>>& subgroups,
    const map<int, set<string> >& nominal_columns,
    const Expression& expression,
    vector<set<int>>* orig_linear_terms_to_legend_indices,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg);
// Same as above, doesn't populate orig_linear_terms_to_legend_indices.
inline bool GetLegendAndLinearTerms(
    const bool has_constant_term,
    const map<string, int>& variable_name_to_column,
    const vector<vector<string>>& subgroups,
    const map<int, set<string> >& nominal_columns,
    const Expression& expression,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg) {
  return GetLegendAndLinearTerms(
      has_constant_term, variable_name_to_column, subgroups, nominal_columns,
      expression, nullptr, linear_terms, legend, error_msg);
}
// Same as above, but different API.
extern bool GetLegendAndLinearTerms(
    const bool has_constant_term,
    const set<string>& variable_names,
    const vector<vector<string>>& subgroups,
    const map<string, set<string> >& nominal_variables,
    const Expression& expression,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg);
// Same as above, but for simulations (so no need for column index from a data
// file; and no nominal variables).
extern bool GetLegendAndLinearTermsForSimulation(
    const bool has_constant_term,
    const Expression& expression,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg);

// Extracts the variable names in each linear term, looks for their corresponding
// position in 'header', and updates that index in 'col_index_to_linear_terms'
// by adding the index of the linear_term to that set.
extern bool GetLinearTermsInvolvingEachColumn(
    const vector<Expression>& linear_terms,
    const vector<string>& temp_data_header,
    const vector<string>& orig_header,
    const set<int>& nominal_columns,
    vector<set<int>>* col_index_to_linear_terms);
// Same as above, with different API: Only passes in one header, and the
// nominal columns are listed w.r.t. to column name, not column index.
extern bool GetLinearTermsInvolvingEachColumn(
    const vector<Expression>& linear_terms,
    const vector<string>& header,
    const set<string>& nominal_columns,
    vector<set<int>>* col_index_to_linear_terms);

// Breaks 'expression' into its linear terms, treating each an Expression of
// its own, and putting them in 'linear_terms'.
// NOTE: This assumes the input Expression was based on the proper string
// format. For example, the expression: AGE*GENDER + AGE*HEIGHT (two linear
// terms) could also be written: "AGE * (GENDER + HEIGHT)". If written in
// the latter format, the Expression will store the "outermost" operation
// as multiplication, and thus it will be treated as a single linear term.
// NOTE: the Expressions in linear_terms may have sub-expressions (i.e.
// fields 'subterm_one_' and 'subterm_two_') that were created on the
// heap (i.e. using 'new'); caller is responsible for deleting each
// Expression in linear_terms.
// NOTE: Each Expression of linear_terms is created on the stack (so doesn't
// require deleting); however, each subterm of each of these expressions is
// created on the heap, so any code that uses this function (currently, only
// GetLegendAndLinearTerms() above calls it) must explicitly call
// DeleteExpression on each of the expressions of linear_terms.
extern bool GetLinearTermsInExpression(
    const Expression& expression,
    vector<Expression>* linear_terms, string* error_msg);

// Expands 'expression' by performing "indicator-expansion" for non-numeric
// variables. For example, if 'expression' is:
//   Log(Race * Nationality + Marital_Status)
// where Race is one of {White, Black, Asian, Other}, Nationality is one of
// {CANADA, USA, MEXICO, FRANCE, SPAIN}, and Marital_Status is one of
// {Single, Married, Divorced}, then the original expression would get expanded
// into 24 = (#Distinct_Race - 1) * (#Distinct_Nat - 1) * (#Distinct_Status - 1)
// terms, e.g. one of which is:
//   Log(I_(Race=Asian) * I_(Nat=USA) + I_(Status=Divorced))
// NOTE: Anticipated usage assumes 'expression' represents a single linear term;
// for example, contrast the two similar expressions:
//   Race + Nationality     vs.     Race * Nationality
// The former we'd want to expand into 7 = (#Distinct_Race - 1) + (#Distinct_Nat - 1)
// linear terms:
//   I_(Race=Black) + I_(Race=Asian) + I_(Race=Other) +
//   I_(Nat=USA) + I_(Nat=MEXICO) + I_(Nat=FRANCE) + I_(Nat=SPAIN)
// while the latter we'd want 12 = (#Distinct_Race - 1) * (#Distinct_Nat - 1)
// linear terms, e.g. one of which is:
//   I_(Race=Asian) * I_(Nat=USA)
// This function does the "multiplicative" version of the expansion, so the input
// expression should represent a single linear term (and then this function can
// be called on each linear term).
extern bool ExpandExpression(
    const Expression& expression,
    const map<string, int>& variable_name_to_column,
    const vector<vector<string>>& subgroups,
    const map<int, set<string> >& nominal_columns,
    const string& current_title,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg);
// Same as above, with different API.
extern bool ExpandExpression(
    const Expression& expression,
    const set<string>& variable_names,
    const vector<vector<string>>& subgroups,
    const map<string, set<string> >& nominal_variables,
    const string& current_title,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg);

// Goes through expression, adding any variable name seen (e.g. any non-empty
// field 'Expression.var_name_' for any of the (sub-)expressions) to
// 'vars_in_expression'. Checks to make sure each encountered variable name
// is either in 'var_names' or equals kSubgroupIndicator, otherwise returns false.
extern bool ExtractVariablesFromExpression(
    const Expression& expression,
    const set<string>& var_names,
    set<string>* vars_in_expression, string* error_msg);
// Same as above, but there is no check against the preliminary set 'var_names'.
inline bool ExtractVariablesFromExpression(
    const Expression& expression,
    set<string>* vars_in_expression, string* error_msg) {
  return ExtractVariablesFromExpression(
      expression, set<string>(), vars_in_expression, error_msg);
}
// Same as above, but first need to determine the Expression to parse
// (intended for parsing Model's LHS, where a priori, we don't know
// how many Expressions there are, depending on ModelType).
extern bool ExtractVariablesFromExpression(
    const ModelType& model_type,
    const DepVarDescription& model_lhs,
    const set<string>& var_names,
    set<string>* vars_in_expression, string* error_msg);
// Same as above, but there is no check against the preliminary set 'var_names'.
inline bool ExtractVariablesFromExpression(
    const ModelType& model_type,
    const DepVarDescription& model_lhs,
    set<string>* vars_in_expression, string* error_msg) {
  return ExtractVariablesFromExpression(
      model_type, model_lhs, set<string>(), vars_in_expression, error_msg);
}

// Copies orig_expression to new_expression, replacing any variable names
// that are a Key in 'non_numeric_vars_to_expansion_var' with the
// corresponding Value. Also populates 'legend' with a string representation
// of the resulting 'new_expression'.
// NOTE: Could populate model's 'legend' here, but just as easy to do
// it elsewhere (on-demand, and keeps this function cleaner) e.g. by calling
// GetExpressionString() on 'new_expression'.
extern bool ReplaceVariableInExpressionWithIndicatorExpansion(
    const Expression& orig_expression,
    const map<string, string>& non_numeric_vars_to_expansion_var,
    Expression* new_expression, string* legend, string* error_msg);

// Checks the expression to make sure that no illegal operations are
// applied to a nominal variable (e.g. Log).
extern bool SanityCheckFunctionsAppliedToNominalVariables(
    const Expression& expression, const set<string>& nominal_vars,
    string* error_msg);

// Takes the original header of column names and creates a new one,
// which has expanded nominal columns by the appropriate amount: for a nominal
// column with N distinct values appearing in it, creates N - 1 columns.
extern bool ConstructLegend(
    const vector<string>& input_header,
    const map<int, set<string>>& nominal_columns_and_values,
    vector<string>* output_legend, string* error_msg);

// The original data file may differ from the sample_values passed in to
// various ReadInput functions: columns that were present in the original
// file that are not used are omitted in sample_values. However, some
// fields of ModelAndDataParams refer to the column index w.r.t. the
// original data file.
// This function converts the original column index (w.r.t. the original
// data file) to the column index w.r.t. to sample_values.
extern bool OrigColToDataValuesCol(
    const set<int>& input_cols_used, const int orig_col_index,
    int* col_index, string* error_msg);

/* ============================= END Functions ============================== */

}  // namespace file_reader_utils

#endif
