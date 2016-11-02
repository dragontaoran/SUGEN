// Date: Dec 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description: Utility functions for regression models.

#include "MathUtils/data_structures.h"
#include "FileReaderUtils/read_file_structures.h"
#include "MathUtils/statistics_utils.h"

#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

#ifndef REGRESSION_UTILS_H
#define REGRESSION_UTILS_H

using Eigen::VectorXd;
using namespace math_utils;
using namespace file_reader_utils;

namespace regression {

// The type of regression to perform.
enum RegressionType {
  LINEAR_REGRESSION,
  LOGISTIC_REGRESSION,
  COX_REGRESSION,
};

// Same as below, but for multiple models (distinguish which model the argument
// is applied to by "*_two" suffix, which is only recognized for arguments:
//   strata_two, subgroup_two, model_two
// All other arguments are applied to both models.
extern bool ParseRegressionCommandLineArgs(
    int argc, char* argv[],
    const bool check_model_and_input_file_are_present,
    vector<ModelAndDataParams>* params,
    set<string>* unparsed_args);
// Same as above, without a holder for arguments that couldn't be passed, and
// using default 'true' for check_model_and_input_file_are_present.
inline bool ParseRegressionCommandLineArgs(
    int argc, char* argv[],
    vector<ModelAndDataParams>* params) {
  return ParseRegressionCommandLineArgs(argc, argv, true, params, nullptr);
}
// Parses frequently encountered command-line arguments into the
// corresponding (Category 1) fields of params. Specifically:
//   --in: Sets file_.name_.
//   --out: Sets outfile_.name_.
//   --sep: Sets file_.delimiter_.
//   --comment_char: Sets file_.comment_char_.
//   --inf_char: Sets file_.infinity_char_.
//   --na_strings: Sets file_.na_strings_. Format: Comma-Separated list
// The following 4 arguments have an optional "_k" suffix, which indicates the
// argument should be applied to the k^th model. Here, k is a positive integer,
// where the first model should have k = 1 (as opposed to k = 0).
//   --model[_k]: Sets model_str_ (for the k^th model).
//   --model_type[_k]: Sets model_type_ (for the k^th model).
//                     Valid values: Linear, Logistic, Cox, Right-Censored, Interval-Censored
//                     (Cox and Right-Censored are equivalent).
//   --strata[_k]: Sets strata_str_ (for the k^th model). Format:
//                   String: Comma-Separated list of columns that determine strata.
//   --subgroup[_k]: Sets subgroup_str_ (for the k^th model). Format:
//               String: See command_line_utils::ParseSubgroups().
//   --max_itr: Sets max_itr_.
//   --[no]kme: Toggles kme_type_ between LEFT_CONTINUOUS and NONE.
//   --[no]kme_for_log_rank: Toggles kme_type_for_log_rank_ between LEFT_CONTINUOUS and NONE.
//   --[no]ties_constant: Sets use_ties_constant_
//   --id_col: Sets id_str_
//   --weight_col: Sets weight_str_
//   --family_cols: Sets family_str_. Format: Comma-Separated list of family columns.
//   --left_truncation_col: Sets left_truncation_str_.
//   --collapse: Sets collapse_params_str_. Format: see comment above
//               ParseTimeDependentParams in command_line_utils.h
//   --standardization: Sets var_norm_params_str_. Format: see comment above
//               ParseVariableNormalizationParams in command_line_utils.h
//   --extrapolation: Sets time_params_str_. Format: see comment above
//               ParseTimeDependentParams in command_line_utils.h
//   --[no]std: Toggles standardize_vars_ between VAR_NORM_NONE and
//              VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY; default is the latter.
//   --std_all: Sets standardize_vars_ to VAR_NORM_STD_W_N_MINUS_ONE.
//              NOTE: If another type of VariableNormalization (than the above 3)
//              is desired, it must be explicitly set by the *_main.cpp program.
// The following set analysis_params_ fields:
//   --[no]standard: Sets analysis_params_.standard_analysis_.
//   --[no]robust: Sets analysis_params_.robust_analysis_.
//   --[no]log_rank: Sets analysis_params_.log_rank_analysis_.
//   --[no]peto: Sets analysis_params_.peto_analysis_.
//   --[no]score_method: Sets analysis_params_.score_method_analysis_.
//   --[no]score_method_width: Sets analysis_params_.score_method_width_analysis_.
//   --[no]satterthwaite: Sets analysis_params_.satterthwaite_analysis_.
// The following set print_options_ fields:
//   --[no]print_estimates: Sets print_estimates_; default true
//   --[no]print_var: Sets print_variance_; default true
//   --[no]print_robust_var: Sets print_robust_variance_; default false
//   --[no]print_se: Sets print_se_; default true
//   --[no]print_t_stat: Sets print_t_stat_; default true
//   --[no]print_p_value: Sets print_p_value_; default true
//   --[no]print_ci_width: print_ci_width_; default false
//   --[no]print_cov_matrix: Sets print_covariance_matrix_; default false
inline bool ParseRegressionCommandLineArgs(
    int argc, char* argv[],
    const bool check_model_and_input_file_are_present,
    ModelAndDataParams* params, set<string>* unparsed_args) {
  vector<ModelAndDataParams> temp;
  temp.push_back(*params);
  if (!ParseRegressionCommandLineArgs(
          argc, argv, check_model_and_input_file_are_present, &temp, unparsed_args) ||
      temp.size() != 1) {
    return false;
  }
  *params = temp[0];
  return true;
}
// Same as above, without a holder for arguments that couldn't be passed, and
// using default 'true' for check_model_and_input_file_are_present.
inline bool ParseRegressionCommandLineArgs(
    int argc, char* argv[], ModelAndDataParams* params) {
  return ParseRegressionCommandLineArgs(argc, argv, true, params, nullptr);
}

// Populates a string representation of the input LinearTerm.
// For example, if LinearTerm represents:
//   2.0*AGE*Log(HEIGHT)
// then the title will be "AGE*Log(HEIGHT)" (the constant factor
// of the linear term is ignored).
extern string GetLinearTermTitle(const LinearTerm& term);

// Nominal covariates are expanded by adding each possible nominal value as a
// subscript of the Variable name. For example if RACE is the title of a nominal
// variable, and the data includes possible values {ASIAN, AFRICAN, WHITE}, then
// two new covariates will be created with titles: "RACE_AFRICAN" and
// "RACE_WHITE" (in general, if there are M possible values for a nominal
// variable, M - 1 covariates will be created, representing indicators for the
// last M - 1 values). Things get more complicated if a linear term involves
// multiple variables. For example, for a linear term "RACE * STATUS", where
// STATUS has possible valuse {SINGLE, MARRIED, DIVORCED, UNKNOWN}, the
// following covariates will have been created:
//   RACE_AFRICAN*STATUS_MARRIED, RACE_ASIAN*STATUS_MARRIED,
//   RACE_AFRICAN*STATUS_DIVORCED, RACE_ASIAN*STATUS_DIVORCED
//   RACE_AFRICAN*STATUS_UNKNOWN, RACE_ASIAN*STATUS_UNKNOWN
// Meanwhile, the original model term would have been expressed simply as
// "RACE*STATUS".
// This function determines if the first term represents a possible (nominal
// value) expansion of the second term. Specifically, we'll assume that the
// only valid op_ for LinearTerm is MULT, and then split 'original_term'
// around the '*' sign, and search for matches within 'expanded_term'
// of all the resulting variables.
// TODO(PHB): Update this function to handle LinearTerms whose operation is
// NOT multiplication.
extern bool IsNominalExpansionString(
    const string& expanded_term, const string& original_term);

// Given an input vector associating a beta value to a term title,
// uses legend to set beta_values: for each title in legend, looks
// up the beta value in the var_name_to_beta_value mapping, and
// sets that index of beta_values to this value.
// For the most part, titles in 'legend' are looked-up as-is in
// 'var_name_to_beta_value'; the exception is that the latter
// might be from the RHS of the user-entered model, in which case
// the final covariates may not exactly match the model RHS: nominal
// columns may have been expanded into multiple covariates, and
// an indicator for subgroup on model's RHS may have been
// expanded into multiple subgroup covariates. Thus, if a given
// entry in 'legend' cannot be found as-is in 'var_name_to_beta_value',
// this function will attempt to treat the entry as a nominal covariate
// and/or subgroup covariate (by stripping appended '_N' or prefix
// 'Subgroup_', respectively).
extern bool GetBetaFromLegend(
    const map<string, double>& var_name_to_beta_value,
    const vector<string>& legend,
    VectorXd* beta_values, string* error_msg);

// Gets the names of the dependent variable(s):
//   - For LINEAR and LOGISTIC models, there is a single dependent variable (e.g. "Y")
//   - For COX models, there are 2 or 3 dependent variables (time, status) or
//     (survival time, censoring time, status). They will be stored in this order.
//   - For Interval Censored Data, this will consist of all of the dependent
//     variable (status column) name(s), followed by the name of the time column(s).
extern bool GetDependentVariableNames(
    const ModelType& model_type, const DepVarDescription& model_lhs,
    vector<string>* dep_var_names);

extern bool PrintRegressionStatistics(
    const string& command,
    const ModelAndDataParams& params, const SummaryStatistics& summary_stats);
// Same as above, no command.
inline bool PrintRegressionStatistics(
    const ModelAndDataParams& params, const SummaryStatistics& summary_stats) {
  return PrintRegressionStatistics("", params, summary_stats);
}
// Same as above, but populates the input string instead of printing to file.
extern bool PrintRegressionStatistics(
    const ModelAndDataParams& params, const SummaryStatistics& summary_stats,
    string* output);

// Print Header (Metadata) of the regression results.
extern bool PrintRegressionHeader(
    const ModelAndDataParams& params, const SummaryStatistics& summary_stats,
    string* output);


/* PHB. There are two potential use-cases here:
 *   1) For reading actual data, if the Model's LHS is an expression rather
 *      than just a single variable name, then a function for evaluating the
 *      expression to obtain the LHS (dependent variable) value is necessary.
 *   2) For simulated data, we need to generate the LHS (dependent variable) value
 *      based on the RHS values and model type.
 * Originally, the function below attempted to do both, which doesn't make any
 * sense, since the LHS values are computed (read) directly from provided values
 * in case (1), while they need to be computed via RHS (or in the case of Cox,
 * via the LHS expression for simulating time/status) values for (2). 
 * Instead, we achieve (1) automatically using EvaluateExpression(), so we don't
 * need a function for this here; and (2) belongs in SimulationUtils, and hence I
 * moved this function there, renaming it
 * ComputeDependentVariableFromSimulatedValues().
 * I've left the function here for now, as it implements parts of both (1) and
 * (2), and may be useful (once modified) in the future?...
// Given the value representing the RHS of the model, together with all of the
// sampled variable values (in case the model's LHS involves more sampled values,
// as in Logistic case, where LHS is I(Y > exp(RHS) / (1 + exp(RHS))), where Y is
// sampled from some distribution, e.g. U(0, 1)), computes the appropriate value
// for the dependent variable(s) based on the model type. Values will correspond
// to the order in "dep_var_names" that is generated via call to
// GetDependentVariableNames() above.
extern bool ComputeDependentVariableValues(
    const ModelType& model_type,
    const VectorXd& actual_beta_values,
    const Expression& model_rhs,
    const DepVarDescription& model_lhs,
    const vector<string>& header,
    const vector<vector<DataHolder>>& sampled_var_values,
    DepVarHolder* dep_vars, string* error_msg);  
*/

}  // namespace regression

#endif
