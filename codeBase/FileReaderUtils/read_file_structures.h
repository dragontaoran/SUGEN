// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description:
//   Utility structures, enums, and constants for reading in data.

#include "MathUtils/data_structures.h"
#include "MathUtils/number_comparison.h"

#include <Eigen/Dense>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef READ_FILE_STRUCTURES_H
#define READ_FILE_STRUCTURES_H

using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace math_utils;
using namespace std;

namespace file_reader_utils {

/* ================================== Enums ================================= */

// Specifies the kind of model.
enum ModelType {
	MODEL_TYPE_UNKNOWN,
	MODEL_TYPE_LINEAR,
	MODEL_TYPE_LOGISTIC,
	MODEL_TYPE_RIGHT_CENSORED_SURVIVAL,					// Cox
	MODEL_TYPE_INTERVAL_CENSORED,						// Cox
	MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED,        // Cox
	MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED,      // Cox
};

// Specifies how to normalize a variable's values (across all the values that
// appear for that variable).
enum VariableNormalization {
  // Specify the following via strings: "NONE", "None", "none".
  VAR_NORM_NONE,
  // Specify this via strings:
  //   "std", "standard", "population_std", "std_population", "std_population",
  //   "standard_population".
  VAR_NORM_STD,
  // Same as above, but only apply this if variable is non-binary. Specify via:
  //   "std_non_binary", "standard_non_binary", "pop_std_non_binary"
  VAR_NORM_STD_NON_BINARY,
  // Specify this via strings:
  //   "sample_std", "std_sample", "sample_standard", "standard_sample".
  VAR_NORM_STD_W_N_MINUS_ONE,
  // Same as above, but only apply this if variable is non-binary. Specify via:
  //   "sample_std_non_binary" or "std_sample_non_binary"
  VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY,
};

// For time-dependent variables, specifies how to evaluate the variable at
// a time that is in-between the times for which a value for that 
// variable is known.
enum InterpolationType {
  IT_UNKNOWN,
  IT_LEFT,               // Use the closest point to the left (creates Right-Continuous fn)
  IT_RIGHT,              // Use the closest point to the right (creates Left-Continuous fn)
  IT_NEAREST,            // Use the value at the *nearest* timepoint
  // Note that the following option should not be used for CATEGORICAL variables
  IT_LINEAR_INTERP,      // Use linear-interpolation between the two nearest timepoints
  IT_BASELINE_CONSTANT,  // Constant (time-independent), with the constant value
                         // equal to the first timepoint value. Specify with
                         // strings: "baseline", "constant" or "baseline_constant"
};

// Specifies how to evaluate a (time-dependent) variable at a timepoint
// that lies outside (before or after) its first/last known value.
enum ExtrapolationType {
  ET_UNKNOWN,
  ET_CONSTANT,
  ET_LEFTMOST,
  ET_RIGHTMOST,
};

/* ================================ END Enums =============================== */


/* =============================== Structures =============================== */

// A structure to hold file-level information about a file.
struct FileInfo {
  string name_;
  string delimiter_;
  string comment_char_;
  string infinity_char_;
  set<string> na_strings_;

  FileInfo() {
    name_ = "";
    delimiter_ = "\t";
    comment_char_ = "#";
    infinity_char_ = "";
    na_strings_.insert("NA");
  }

  void Copy(const FileInfo& info) {
    name_ = info.name_;
    delimiter_ = info.delimiter_;
    comment_char_ = info.comment_char_;
    infinity_char_ = info.infinity_char_;
    na_strings_ = info.na_strings_;
  }
};

// Holds the representation of the model's LHS.
struct DepVarDescription {
  // One (and only one) of the following fields should be populated,
  // which holds the dependent variable information:
  //   - LINEAR MODELS.
  Expression model_lhs_linear_;
  //   - LOGISTIC MODELS.
  Expression model_lhs_logistic_;
  // The following vector should have length 2 or 3, depending on whether the model
  // LHS was presented as (Time, Status) or (Survival Time, Censoring Time, Status).
  // NOTE: If you want to describe (SurvivalTime, CensoringTime), you must use the
  // 2nd format (with model_lhs_cox_.size() = 3), and set the Expression for Status
  // to be identically -1, which indicates to use Min(SurvivalTime, CensoringTime).
  vector<Expression> model_lhs_cox_;
  // For time-dependent interval-censored NPMLE. Vector has length 1 + K,
  // where K = number of dependent variables (so K = 1 for univariate case).
  vector<Expression> model_lhs_time_dep_npmle_;
  // For time-independent interval-censored NPMLE. Vector has length 2 * K,
  // where K = number of dependent variables (so K = 1 for univariate case).
  vector<Expression> model_lhs_time_indep_npmle_;

  // Holds the name(s) of the Time related dependent variable(s). This
  // vector either has length 0, 1, 2, or 2 * K (where K = number dep vars):
  //   - Should be length 0 for Linear, Logistic, and time-indep NPMLE models.
  //   - If length is 1, then it refers to the only Time variable. For
  //     Cox, this is either Survival Time or Censoring Time (will need
  //     to look at value of Status column to determine which it is);
  //     whereas for time-dependent NPMLE, this is the Evaluation Time column.
  //   - If length 2, this is either Interval-Censored (Left-Time, Right-Time)
  //     column names, or Cox PH (Survival Time, Censoring Time) column names.
  //   - If length 2 * K, then this is time-indep NPMLE. First two entries are
  //     (left, right) time endpoints for first dep var, etc.
  vector<string> time_vars_names_; 
  // Holds the name of the left-truncation column (if present); only used for Cox.
  string left_truncation_name_;
  // Holds the name(s) of the dependent variable(s), NOT including "Time"
  // variables (i.e. for Survival Models, only the "Status" variable names
  // are kept here). For univariate case, the following vector has length 1.
  vector<string> dep_vars_names_;

  DepVarDescription() {
    left_truncation_name_ = "";
  }
};

// Holds the values for the Dependent Variables.
// One (and only one) of the following three fields should be populated,
// which holds the dependent variable information.
struct DepVarHolder {
  // TODO(PHB): Consider using VectorXd instead of vector<double> for
  // dep_vars_linear_ and dep_vars_logistic_: most (all?) use-cases would
  // benefit from this.
  vector<double> dep_vars_linear_;
  vector<bool> dep_vars_logistic_;      // Coordinate i is true iff Y_i = 1.
  vector<CensoringData> dep_vars_cox_;
  // We do not have a field for Interval-Censored NPMLE, because DepVarHolder
  // is not used for this; instead of calling the standard method to populate
  // ModelAndDataParams Category (4) fields (read_input::StoreDataValuesInParams()),
  // Interval-Censored NPMLE uses its own special function 
  // (read_time_[in]dep_interval_censored_data::PopulateSubjectInfo()) and data
  // structure (SubjectInfo, via TimeDepIntervalCensoredData.subject_to_values_).
};

// A structure to hold information about how to collapse specific
// values for a variable into a default value (e.g. if we want to treat
// all numbers less than one the same).
// Not all fields should be set. In particular, collapsing will:
//   1) Use num_buckets_, if positive.
//   2) Use round_to_nearest_, if positive
//   3) Otherwise, exactly one of the from_* and exactly one of the to_* fields
//      should be set to non-default values.
// Also, if using num_buckets_ or round_to_nearest_, then each variable should
// have a single VariableCollapseParams object associated with it. In contrast,
// if using from/to values, a variable may have multiple VariableCollapseParams
// objects associated to it, describing what to do for different 'from' values.
// NOTE: Using num_buckets_ vs. round_to_nearest_ are in some sense equivalent,
// assuming you know the range (Min, Max) of the data. Using round_to_nearest_
// should be preferred over using num_buckets_, as the latter requires reading
// all data, finding the Min and Max, and then going though all data again
// to put it in the appropriate bucket; while round_to_nearest_ can be done in
// one pass.
struct VariableCollapseParams {
  // Partitions the range of values into 'num_buckets_' buckets, collapsing each
  // value to the closest partition point (bucket). For Numeric data types only.
  int num_buckets_;
  // Rounds each value to the nearest multiple of round_to_nearest_.
  double round_to_nearest_;

  string from_str_;
  double from_val_;
  pair<double, double> from_range_;
  DataType from_type_;

  string to_str_;
  double to_val_;
  DataType to_type_;

  VariableCollapseParams() {
    num_buckets_ = 0;
    round_to_nearest_ = 0.0;
    from_str_ = "";
    from_val_ = DBL_MAX;
    from_range_ = make_pair(DBL_MAX, DBL_MIN);
    from_type_ = DataType::DATA_TYPE_UNKNOWN;
    to_str_ = "";
    to_val_ = DBL_MIN;
    to_type_ = DataType::DATA_TYPE_UNKNOWN;
  }
};

// Descriptors for the variable's column within the data file.
struct VariableColumn {
  // If non-negative, this holds the column index of the variable.
  int index_;
  // If non-empty, this holds the name (column title) of the variable.
  string name_;

  VariableColumn() {
    index_ = -1;
    name_ = "";
  }

  // So that VariableColumn can be used as the Key to a set/map.
  bool operator <(const VariableColumn& x) const {
    return std::tie(name_, index_) < std::tie(x.name_, x.index_);
  }
};

// Kinds of analysis to run (for Regression algorithms).
struct AnalysisParams {
  bool standard_analysis_;
  bool robust_analysis_;
  bool peto_analysis_;
  bool log_rank_analysis_;
  bool score_method_analysis_;
  bool score_method_width_analysis_;
  bool satterthwaite_analysis_;

  AnalysisParams() {
    standard_analysis_ = true;
    robust_analysis_ = false;
    peto_analysis_ = false;
    log_rank_analysis_ = false;
    score_method_analysis_ = false;
    score_method_width_analysis_ = false;
    satterthwaite_analysis_ = false;
  }

  void Copy(const AnalysisParams& params) {
    standard_analysis_ = params.standard_analysis_;
    robust_analysis_ = params.robust_analysis_;
    peto_analysis_ = params.peto_analysis_;
    log_rank_analysis_ = params.log_rank_analysis_;
    score_method_analysis_ = params.score_method_analysis_;
    score_method_width_analysis_ = params.score_method_width_analysis_;
    satterthwaite_analysis_ = params.satterthwaite_analysis_;
  }
};

// Which fields of SummaryStatistics to print.
struct SummaryStatisticsPrintOptions {
  bool print_estimates_;
  bool print_variance_;
  bool print_robust_variance_;
  bool print_se_;
  bool print_t_stat_;
  bool print_p_value_;
  bool print_ci_width_;
  bool print_covariance_matrix_;

  SummaryStatisticsPrintOptions() {
    print_estimates_ = true;
    print_variance_ = true;
    print_robust_variance_ = false;
    print_se_ = true;
    print_t_stat_ = true;
    print_p_value_ = true;
    print_ci_width_ = false;
    print_covariance_matrix_ = false;
  }
};

// Specifies how this variable should be evaluated for points that lie
// before (resp. after) the first (resp. last) known value for it.
struct OutsideIntervalParams {
  // If user explicitly set left ExtrapolationType. This is necessary
  // because if it turns out this variable is NOMINAL, using the default
  // of outside_right_type_ = ExtrapolationType::ET_CONSTANT and
  // default_right_val_.type_ = DATA_TYPE_NUMERIC is not appropriate.
  // So we can return an error if user explicitly set it, otherwise
  // the code should use different defaults for NOMINAL variables.
  bool left_explicitly_set_;
  // For times that are before all timepoints.
  ExtrapolationType outside_left_type_;
  DataHolder default_left_val_;

  // For times that are before all timepoints.
  ExtrapolationType outside_right_type_;
  DataHolder default_right_val_;
  bool right_explicitly_set_;

  OutsideIntervalParams () {
    left_explicitly_set_ = false;
    outside_left_type_ = ExtrapolationType::ET_LEFTMOST;
    default_left_val_.value_ = 0.0;
    default_left_val_.type_ = DataType::DATA_TYPE_NUMERIC;
    right_explicitly_set_ = false;
    outside_right_type_ = ExtrapolationType::ET_RIGHTMOST;
    //PHBoutside_right_type_ = ExtrapolationType::ET_CONSTANT;
    default_right_val_.value_ = 0.0;
    default_right_val_.type_ = DataType::DATA_TYPE_NUMERIC;
  }
};

// Specifies how this variable should be evaluated at different time-points.
struct TimeDependentParams {
  InterpolationType interp_type_;
  OutsideIntervalParams outside_params_;

  TimeDependentParams() {
    interp_type_ = InterpolationType::IT_RIGHT;
  }
};

// A structure to hold parameters for a single (in)dependent variable.
struct VariableParams {
  VariableColumn col_;
  // DEPRECATED. The norm_ field is no longer used: it was confusing to
  // implement this, since e.g. the VariableParams belong to an independent
  // variable, but normalization is more naturally applied to a linear term.
  // Instead of using the norm_ field, use ModelDataAndParams.standardize_vars_
  // field, which applies a standardization option to *all* linear terms.
  // TODO(PHB): If any use-case actually demands that standardization options
  // be applied on a per-independent variable and/or a per-linear term basis
  // (instead of one global option applied to all linear terms), then
  // un-DEPRECATE this field, and add code to use it.
  VariableNormalization norm_;
  vector<VariableCollapseParams> collapse_params_;
  TimeDependentParams time_params_;

  VariableParams() {
    norm_ = VariableNormalization::VAR_NORM_NONE;
  }
};

// A structure to hold all of a model's paramaters.
// There are 3 components/categories of the fields in this structure,
// based on how they are populated:
//   1) Direct input from the user: Either from the command-line or prompting
//      by the program. Such fields are:
//   2) Parsing user's arguments into structured fields
//   3) Reading Input data file(s) (or simulated data).
//   4) Performing data manipulation, other aggregation operations
// In terms of populating these fields:
//  - Category (1):   Populated directly by *_main.cpp file
//  - Category (2-4): Use ReadInput::FillModelAndDataParams(), to do these
//                    all at once. Or, if specialized treatment is desired:
//  - Category (2):   Use read_file_utils::ParseModelAndDataParams()
//  - Category (3):   Use ReadTableWithHeader::ReadDataFile(), or in case
//                    of simulations, simulation_utils::SimulateValues()
//  - Category (4):   Use ReadInput::StoreDataValuesInParams(), or
//                    for interval-censored survival models, use
//                    read_time_dep_interval_censored_data::PopulateSubjectInfo()
struct ModelAndDataParams {

	string error_msg_;  // Will be populated on any error.

	// ================= Category 1: User-Provided Specificiations ===============
	// Data File.
	FileInfo file_;  // Mandatory.
	// Output File.
	FileInfo outfile_;  // Mandatory.

	// Model.
	ModelType model_type_;  // Mandatory, each *_main.cpp file should fill this in
	// NOTE: The model's RHS should *not* explicitly include the Error Term (for
	// Linear models) nor the Constant Term (for Linear and Logisitc models), as
	// the structure of the code will implicitly create these as appropriate; this
	// is true even in the case of simulations (the error/constant term is
	// handled specially for simulations; but the user should not include them
	// in the specification of the model's RHS, the constant coefficient for
	// the constant term multiplier will be picked up separately via --beta
	// command-line arg).
	string model_str_;      // Mandatory.

	// Maximum number of iterations allowed in the underlying regression algorithm
	int max_itr_;

	// Analysis to run.
	AnalysisParams analysis_params_;
	// SummaryStatistics to print.
	SummaryStatisticsPrintOptions print_options_;

	// KME Multiplier.
	KaplanMeierEstimatorType kme_type_;
	KaplanMeierEstimatorType kme_type_for_log_rank_;

	// See explanation of field ties_constant_ in struct StratificationData.
	bool use_ties_constant_;

	// Subgroup: The full string following the --subgroup agrument.
	string subgroup_str_;

	// Strata: The --strata argument.
	string strata_str_;

	// Id, Weight, and Family: the --id_col, --weight_col, and --family_cols arguments.
	string id_str_;
	string weight_str_;
	string family_str_;

	// The --left_truncation_col argument.
	string left_truncation_str_;

	// Variable and Data Params.
	//   - String representation of collapse parameters; see comment above
	//     ParseCollapseParams (in command_line_utils.h) for explanation of format
	//     of this string, and use that function to parse it.
	string collapse_params_str_;  // Optional. Default (empty): No collapsing.
	//   - String representation of time parameters; see comment above
	//     ParseTimeDependentParams (in command_line_utils.h) for explanation of
	//     format of this string, and use that function to parse it.
	// TODO(PHB): This field is disabled for now. If you want to use time-dependent
	// params, use the interface of read_time_[in]dep_interval_censored_data.[h | cpp].
	string time_params_str_;  // Optional. Default (empty): Either not relevant,
							// otherwise use all default TimeDependentParams.
	//   - String representation of collapse parameters; see comment above
	//     ParseVariableNormalizationParams (in command_line_utils.h) for
	//     explanation of format of this string and use that function to parse it.
	string var_norm_params_str_;  // Optional. Default (empty): Use default NormalizationParams.
	//   - How linear terms should be standardized before computations.
	VariableNormalization standardize_vars_;   

	// ============== Category 2: Parsed User-Provided Specificiations ===========

	// Header of Input File: List of column names, indexed in the order they appear
	// in the original data file.
	vector<string> header_;  // Mandatory.
	
	// Nominal Columns.
	//   - The set of nominal columns, as detected from the variable names
	//     (those with '$' suffix) and from reading the actual data.
	//     NOTE: The indexing of the items in this set is w.r.t. the original input
	//     data (i.e. same size as header_), and they are 0-based (first column is '0').
	set<int> nominal_columns_;  // Optional. Default: Empty.

	// Used Columns: the set of columns in the data file needed for something
	// (dep var, indep var, id, subgroup, strata, etc.).
	// NOTE: The indexing of the items in this set is w.r.t. the original input
	// data (i.e. same size as header_), and they are 0-based (first column is '0').
	set<int> input_cols_used_;

	// Stores the RHS of the model. For example, for model:
	//   Y = Log(X_1) * X_2 + X_3^2 + X_1 * X_3
	// it would store everything on the RHS of the equal sign.
	// NOTE: The Error Term (for LINEAR models) and the Constant Term (for
	// LINEAR and LOGISTIC models) are *not* stored in model_rhs_.
	Expression model_rhs_;  // Mandatory.
	// Stores the LHS of the model.
	DepVarDescription model_lhs_;

	// Stores the Convergence Criteria (for the underlying regression algorithm).
	ConvergenceCriteria convergence_criteria_;

	// Variable and Data Params.
	//   - Stores the parameters for all variables, once parsed from the
	//     string representations above. This vector is indexed in accordance
	//     to header_ above (so those two fields have same size; namely, the
	//     number of columns in the original input data file).
	vector<VariableParams> var_params_;  // Mandatory.

	// Subgroup Params.
	//   - The LHS of the --subgroup argument; holds the column names and
	//     indices of the columns that determine subgroup membership; the
	//     indexing of the vector is the order of the subgroups entered by
	//     the user. The column name_ field is populated from user input;
	//     the column index_ is from reading (header line of) data file.
	//     NOTE: The column indices for each of the subgroup_cols_ is w.r.t.
	//     the original input data (i.e. same size as header_), and is 0-based.
	vector<VariableColumn> subgroup_cols_;  // Optional. Default: Empty.
	//   - The RHS (i.e. string representation) of the subgroups. Outer vector
	//     has size equal to the number of subgroups, inner-vector has size
	//     equal to the number of columns that determine subgroup membership.
	vector<vector<string>> subgroups_;  // Optional. Default: Empty.
	bool use_subgroup_as_covariate_;  // Optional. Default: 'false'.

	// Strata Params.
	//   - Holds the column names and indices of the columns that determine
	//     stratification index. The column name_ field is populated from user
	//     input; the column index_ is from reading (header line of) data file.
	//     NOTE: The column indices for each of the subgroup_cols_ is w.r.t.
	//     the original input data (i.e. same size as header_), and is 0-based.
	set<VariableColumn> strata_cols_;  // Optional. Default (empty) means no stratification.

	// Variable and Data Params.
	//   - Name and index of the id column; column name_ is populated from
	//     user input; column index_ is from reading (header line of) data file.
	//     NOTE: The column index is w.r.t. the original input data (i.e. same
	//     size as header_), and is 0-based.
	VariableColumn id_col_;  // Optional. Default: Empty.
	//   - Name and index of the family column; column name_ is populated from
	//     user input; column index_ is from reading (header line of) data file.
	//     NOTE: The column index is w.r.t. the original input data (i.e. same
	//     size as header_), and is 0-based.
	vector<VariableColumn> family_cols_;  // Optional. Default: Empty.
	//   - Name and index of the weight column; column name_ is populated from
	//     user input; column index_ is from reading (header line of) data file.
	//     NOTE: The column index is w.r.t. the original input data (i.e. same
	//     size as header_), and is 0-based.
	VariableColumn weight_col_;  // Optional. Default: Empty.

	// ============ Category 3: Fields from Input Data File ======================

	//   - The column indices of nominal columns, and the set of all distinct
	//     values encountered in each such column.
	//     NOTE: Unlike 'nominal_columns_' above, the indexing of the items
	//     (columns) in this map is w.r.t. the temporary 'data_values' object (e.g.
	//     created by ReadTableWithHeader::ReadDataFile()); they are 0-based w.r.t.
	//     those columns. In particular, this data structure differs from the
	//     rows/columns of the original data file in that only relevant columns
	//     (those involved in the Model, family, etc.) are present.
	map<int, set<string>> nominal_columns_and_values_;  // Optional. Default: Empty

	// Missing Value (NA) rows.
	//   - Indexed by input data row: which rows have missing values and the
	//     corresponding columns with missing values.
	//     NOTE: Similar to 'nominal_columns_and_values_' above, but unlike
	//     'nominal_columns_' above, the indexing of the rows
	//     (na_rows_and_columns_.first) and columns in the sets
	//     (na_rows_and_columns_.second) is w.r.t. the temporary 'data_values'
	//     object (e.g. created by ReadTableWithHeader::ReadDataFile()); they
	//     are 0-based w.r.t. those rows and columns. In particular, this data
	//     structure differs from the rows/columns of the original data file in
	//     that only relevant columns (those involved in the Model, family, etc.)
	//     are present.
	map<int, set<int> > na_rows_and_columns_;  // Optional Default: Empty.

	// ============ Category 4: Fields from manipulating/aggregating =============

	// The following two fields must wait until reading the full data file before
	// populating, as we don't know all the linear terms until we've identified
	// all nominal variables and all of their distinct values, for expanding them
	// into the appropriate number of indicator variables.
	//   - Legend: List of all the linear terms (matches user-entered model RHS,
	//             except nominal variables have been expanded to the appropriate
	//             number of indicator functions).
	vector<string> legend_;
	//   - Final Model: The LHS and RHS of the model, where the RHS has been
	//                  expanded (for nominal variables); the RHS is basically
	//                  just a '+'-prefixed concatenation of the terms in legend_;
	//                  used for printing the model in the output file(s).
	string final_model_;
	//   - Map from the original header (Column Names) to the indices w.r.t.
	//     to legend_ for which these Columns (Variables) appear. This
	//     vector has size matching header_ above, and the i^th entry lists
	//     the indices (w.r.t. legend_) in which the i^th Variable of header
	//     appears.
	vector<set<int>> header_index_to_legend_indices_;
	//   - Map from the linear terms in the original model RHS to the
	//     indices w.r.t. legend_ for which these linear terms appear.
	vector<set<int>> orig_linear_terms_to_legend_indices_;

	// Linear Term Statistics.
	//   - In same order as the linear terms (vector<Expression>) in model_rhs_.
	//     Holds the statistics (mean, std dev) for each linear term; the bool
	//     indicates whether this linear term has only bivarate values '0' and '1'.
	vector<tuple<bool, double, double>> linear_terms_mean_and_std_dev_;

	// Subgroup Params.
	//   - Map from subgroup index to the set of rows belonging to that subgroup.
	//     NOTE: The rows in the set are 1-based (as opposed to 0-based), and are
	//     indexed w.r.t. the temporary 'data_values' object (e.g. created by
	//     ReadTableWithHeader::ReadDataFile()); in particular, this may be
	//     different than e.g. linear_term_values_.rows(), since the latter
	//     may have skipped NA rows and rows that were not part of any subgroup.
	map<int, set<int>> subgroup_rows_per_index_;  // Optional. Default: Empty.

	// Strata Params.
	//   - Map from data row (after skipped, e.g. NA, rows have been removed)
	//     to the strata that that row belongs to.
	//     NOTE: The Keys (rows) are 0-based (in contrast to the
	//     'subgroup_rows_per_index_' field above, which is 1-based), and are
	//     indexed w.r.t. the temporary 'data_values' object (e.g. created by
	//     ReadTableWithHeader::ReadDataFile()); in particular, this may be
	//     different than e.g. linear_term_values_.rows(), since the latter
	//     may have skipped NA rows and rows that were not part of any subgroup.
	map<int, int> row_to_strata_;  // Optional. Default: Empty.

	//   - The id (from id_col_) of each (non-NA) row; vector is indexed
	//     corresponding to input data file (minus skipped, e.g. NA, rows).
	vector<string> ids_;  // Optional. Default: Empty.
	//   - The weight (from weight_col_) of each (non-NA) row; vector is indexed
	//     corresponding to input data file (minus skipped, e.g. NA, rows).
	vector<double> weights_;  // Optional. Default: Empty.
	//   - The famil[y | ies] this row belongs to (based on the row's value in
	//     the family_col_); vector is indexed corresponding to input data file
	//     (minus skipped, e.g. NA, rows).
	vector<vector<string>> families_;  // Optional. Default: Empty.

	// Missing Value (NA) rows.
	//   - Just a list of rows that were skipped due to missing values; this
	//     may not simply equal Keys(na_rows_and_columns_) if, e.g., none
	//     of the missing values appeared in relevant columns, or if the user
	//     specified *not* to skip rows with missing values.
	//     NOTE: The rows in the set are 1-based (as opposed to 0-based), and are
	//     indexed w.r.t. the temporary 'data_values' object (e.g. created by
	//     ReadTableWithHeader::ReadDataFile()); in particular, this may be
	//     different than e.g. linear_term_values_.rows(), since the latter
	//     may have skipped rows that had missing values, were not part of
	//     any subgroup, etc.
	set<int> na_rows_skipped_;  // Optional. Default: Empty.

	// Data Values.
	//   - Holds the values for the linear terms. The outer vector is indexed
	//     corresponding to input data file (minus skipped, e.g. NA, rows); the
	//     inner vector holds values of the linear terms, in the order that they
	//     appear in legend_.
	//     NOTE: This data structure will *not* hold (even for simulations) the
	//     values for the linear term representing the Error Term (for LINEAR
	//     models), but it *does* hold the Constant Term (for LINEAR and
	//     LOGISTIC models, for simulations and not).
	MatrixXd linear_term_values_;
	//   - Holds the values for the dependent variable(s). Depending on the
	//     model (Linear, Logistic, Cox, etc.), a unique field within DepVarHolder
	//     will be populated. Irregardless of which it is, the vector below is
	//     indexed corresponding to input data file (minus skipped, e.g. NA, rows).
	DepVarHolder dep_vars_;

	// ====================== Default Constructor ================================
	ModelAndDataParams() {
		model_type_ = ModelType::MODEL_TYPE_UNKNOWN;
		max_itr_ = 1000;
		model_str_ = "";
		subgroup_str_ = "";
		strata_str_ = "";
		id_str_ = "";
		weight_str_ = "";
		family_str_ = "";
		left_truncation_str_ = "";
		collapse_params_str_ = "";
		time_params_str_ = "";
		var_norm_params_str_ = "";
		error_msg_ = "";
		use_subgroup_as_covariate_ = false;
		standardize_vars_ =
			VariableNormalization::VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;
		kme_type_ = KaplanMeierEstimatorType::NONE;
		kme_type_for_log_rank_ = KaplanMeierEstimatorType::NONE;
		use_ties_constant_ = true;
	}

	void Copy(const ModelAndDataParams& params) {
		file_.Copy(params.file_);
		analysis_params_.Copy(params.analysis_params_);
		model_rhs_ = params.model_rhs_;
		model_lhs_ = params.model_lhs_;
		id_col_ = params.id_col_;
		weight_col_ = params.weight_col_;
		dep_vars_ = params.dep_vars_;

		error_msg_ = params.error_msg_;
		model_type_ = params.model_type_;
		model_str_ = params.model_str_;
		kme_type_ = params.kme_type_;
		kme_type_for_log_rank_ = params.kme_type_for_log_rank_;
		use_ties_constant_ = params.use_ties_constant_;
		subgroup_str_ = params.subgroup_str_;
		strata_str_ = params.strata_str_;
		id_str_ = params.id_str_;
		weight_str_ = params.weight_str_;
		family_str_ = params.family_str_;
		left_truncation_str_ = params.left_truncation_str_;
		collapse_params_str_ = params.collapse_params_str_;
		time_params_str_ = params.time_params_str_;
		var_norm_params_str_ = params.var_norm_params_str_;
		standardize_vars_ = params.standardize_vars_;
		header_ = params.header_;
		input_cols_used_ = params.input_cols_used_;
		var_params_ = params.var_params_;
		subgroup_cols_ = params.subgroup_cols_;
		subgroups_ = params.subgroups_;
		use_subgroup_as_covariate_ = params.use_subgroup_as_covariate_;
		strata_cols_ = params.strata_cols_;
		family_cols_ = params.family_cols_;
		nominal_columns_ = params.nominal_columns_;
		nominal_columns_and_values_ = params.nominal_columns_and_values_;
		na_rows_and_columns_ = params.na_rows_and_columns_;
		legend_ = params.legend_;
		final_model_ = params.final_model_;
		linear_terms_mean_and_std_dev_ = params.linear_terms_mean_and_std_dev_;
		subgroup_rows_per_index_ = params.subgroup_rows_per_index_;
		row_to_strata_ = params.row_to_strata_;
		ids_ = params.ids_;
		weights_ = params.weights_;
		families_ = params.families_;
		na_rows_skipped_ = params.na_rows_skipped_;
		linear_term_values_ = params.linear_term_values_;
	}
};

/* ============================= END Structures ============================= */

}  // namespace file_reader_utils

#endif
