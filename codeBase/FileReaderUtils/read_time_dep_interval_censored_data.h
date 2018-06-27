// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description:
//   Utility functions for reading in data that describes
//   (optionally time-dependent) interval censored data.
//   Data file is expected to have format:
//     Case 1: Time-Dependent.
//       Subject_ID  Time  Status  X_1  X_2  ...  X_n
//     Case 2: Time-Independent.
//       Time  Status  X_1  X_2 ... X_n

#include "FileReaderUtils/read_file_utils.h"
#include "MathUtils/data_structures.h"

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef READ_TIME_DEP_INTERVAL_CENSORED_DATA_H
#define READ_TIME_DEP_INTERVAL_CENSORED_DATA_H

using math_utils::DataHolder;
using namespace std;

namespace file_reader_utils {

// Forward declare, so can make it a friend class.
class ReadTimeIndepIntervalCensoredData;
  
// Holds the estimates for one of the dependent covariates for the Multivariate
// MPLE case.
struct DependentCovariateEstimates {
  VectorXd beta_;
  VectorXd lambda_;
};

enum CensoringType {
  CENSOR_TYPE_UNKNOWN,
  CENSOR_TYPE_RIGHT,
  CENSOR_TYPE_LEFT,  // Not yet supported.
  CENSOR_TYPE_INTERVAL,
};

// The following fields allow collapsing of the columns representing time;
// useful to limit the number of distinct timepoints, by merging timepoints
// that are close together. At most one of the two fields should be set;
// they are ignored if non-positive.
struct CollapseTimesParams {
  int num_buckets_;
  double round_to_nearest_;

  CollapseTimesParams() {
    num_buckets_ = 0;
    round_to_nearest_ = 0.0; 
  }
};

struct EventTimeAndCause {
  CensoringType type_;

  // The following two fields should be used if type_ == CENSOR_TYPE_INTERVAL_;
  // they represent the Left and Right times.
  double lower_;
  double upper_;

  // The following field should be used if type_ == CENSOR_TYPE_RIGHT_; it
  // holds Survival and Censoring times and Status.
  CensoringData censoring_info_;

  // For multi-cause of event, holds the cause index for this Subject
  // (a mapping between cause index and cause name is available via
  // TimeDepIntervalCensoredData.event_cause_to_index_.
  int event_cause_;

  EventTimeAndCause() {
    type_ = CensoringType::CENSOR_TYPE_UNKNOWN;
    lower_ = -1.0;
    upper_ = -1.0;
    event_cause_ = -1;
  }
};

enum TimeIndependentType {
  UNKNOWN_TIME_DEP = 0,
  TIME_DEP,
  TIME_INDEP,
};

// The data structure to cover all possible input API's to the functions in
// ReadTimeDepIntervalCensoredData.
struct TimeDepIntervalCensoredInParams {
  FileInfo file_info_;

  // Expression representing the RHS of the model(s). Vector length is the
  // number of models.
  vector<Expression> model_rhs_;

  // In case we know data is time (in)dependent, this field can be used to
  // save time/effort. Use 'UNKNOWN_TIME_DEP' if uncertain (program will
  // determine the case based on the data).
  TimeIndependentType time_indep_type_;

  // Time-independent variables can be stored in a separate data structure,
  // to avoid redundancy (storing a value for every distinct time); see
  // discussion above SubjectInfo's linear_term_values_ field.
  // Set this field to forbid use of the linear_term_values_.first,
  // which is equivalent to treating all variables as time-dependent.
  bool use_time_indep_data_structure_;

  // Data Column information (column names, indices, collapse params, etc.).
  VariableParams id_params_;
  VariableParams family_params_;
  VariableParams time_params_;
  VariableParams event_type_params_;
  VariableParams event_cause_params_;
  // Even for multiple-type, we assume a file format where there is a single
  // Status column (as opposed to a status column for each event type). In
  // the multi-event case, there is a separate "Event-Type" column, which
  // indicates which event the Status column (for the row in question)
  // corresponds to.
  VariableParams status_params_;
  // For Univariate (single dependent variable) case, the outer vector below
  // has size 1; otherwise it has size equal to the number of models.
  // The inner vector has size equal to the number of linear terms
  // in the parsed RHS of the (k^th event-type's) model (note that this may
  // not equal the final number of linear terms, as e.g. indicator covariates
  // for nominal variables have not yet been introduced).
  vector<vector<VariableParams>> indep_vars_params_;

  // In case some of the models are right-censored (as determined by the
  // Event Type column), the set of values in the Event Type column that
  // correspond to a Right-Censored event.
  set<int> right_censored_events_;
  
  // Maps Event Type name to index.
  map<string, int> event_type_to_index_;
  // Maps Event Cause name to index.
  map<string, int> event_cause_to_index_;

  VariableNormalization linear_terms_normalization_;

  TimeDepIntervalCensoredInParams() {
    use_time_indep_data_structure_ = true;
    time_indep_type_ = TimeIndependentType::UNKNOWN_TIME_DEP;
    linear_terms_normalization_ =
        VariableNormalization::VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;
  }
};

// Holds data relevant to a single Subject in the study, for all the model/event types:
//   - Times of interest:
//       - Interval-Censored: (L, U), the interval where Status switched from '0' to '1'
//       - Right-Censored: \tilde{T} = min(T, C) (minimum of survival and censoring times)
//   - Status:
//       - Interval Censored: This data is implicit through the (L, R) values
//       - Right-Censored: Status is stored in times_.censoring_info_.is_alive_.
//   - Covariates: Values of all indep variables for this subject, at all relevant times. 
//   - Family: The family (cluster) this Subject belongs to
struct SubjectInfo {
  // For clustered data, the family (a.k.a. cluster) this Subject belongs to.
  int family_index_;

  // Vector has length K = number of event types, and holds the relevant
  // times for this Subject for that event type.
  vector<EventTimeAndCause> times_;

  // Containers for Covariates.
  // DISCUSSION:
  // 0) NOTATION:
  //    Suppose there are K' models: k \in [1..K]
  //    Suppose model_k has M_k distinct time points.
  //    Suppose that for this Subject (say Subject "i"), model_k has
  //      p_k = p_ik_dep + p_ik_indep
  //    linear terms, with p_ik_dep time-dependent covariates and p_ik_indep
  //    time-independent covariates.
  // 1) NOTE: p_k (the number of linear terms/covariates in the final model_k's
  //    RHS) may not equal p'_ik (the number of independent variables) since the
  //    model RHS can be a combination of indep variables (e.g. if the model's 
  //    RHS is "AGE*HEIGHT", then p' = 2 but p = 1), and also nominal variables
  //    need to be expaneded to indicator functions (e.g. if RHS is "RACE", a
  //    non-numeric variable with D distinct values, then p' = 1 but p = D - 1).
  // 2) A covariate for a given Subject can be time-dependent or time-independent.
  //    In the former case, we must store this covariate's value at *all*
  //    time points; whereas in the former case, we just need to store a single
  //    value. Further complicating things, a given covariate may be time-
  //    independent for one Subject, but time-dependent for another.
  // 3) Unlike the time/status (dependent variables), the values of the
  //    independent variables are model independent: i.e. even if a row of data
  //    data is labelled as pertaining to a particular event-type or event-cause,
  //    the values of the independent variables (at that Observation time) will
  //    be applied to *all* event-types/causes. However, for time-dependent
  //    covariates, there *is* a dependence of the values of independent
  //    variables on the model (in terms of how the data
  //    values are stored), since the set of distinct times is different for
  //    each model.
  // Note (3) above indicated that the values of time-dependent covariates
  // (a.k.a. independent variables) must be stored in a model-dependent way,
  // whereas the values of time-independent covariates could be stored in a
  // model-independent way. However, for consistency/ease of code, we will go
  // ahead and store both (values for) the time-dependent and time-independent
  // covariates in a model-dependent fashion; i.e. each model will store the
  // corresponding values of all (time-dependent + time-independent) covariates
  // at all of the distinct time points for that model/event-type. As noted,
  // storing duplicated values for the time-independent covariates for each
  // model/event-type is extraneous, but the cost is minor since there are (at most)
  //   \sum_k p_ik_indep
  // values that are being duplicated; the expensive part is storing the
  // M_k * p_ik_dep values of the time-dependent covariates for each model,
  // but this cost is unavoidable.
  //
  // The vector has size K' = number of models (typically number of event types,
  // but for Competing Risks, may be the number of event causes). For the k^th
  // element, the VectorXd has size p_ik_indep, and the MatrixXd has Dim
  // (p_ik_dep, M_k).
  // (As discussed in point (0) above, the number of time-[in]dependent
  // covariates p_ik_dep and p_ik_indep may depend on the Subject "i", i.e. not
  // all Subjects need have the same number of time-dependent covariates.
  vector<pair<VectorXd, MatrixXd>> linear_term_values_;

  // This vector specifies which covariates are time-independent for this
  // Subject, and hence whether this covariate's values will be found in
  // the VectorXd or the MatrixXd component of linear_term_values_ above.
  // The outer vector has size K' = number of models, and the k^th element
  // has size p_k.
  vector<vector<bool>> is_time_indep_;

  SubjectInfo () {
    family_index_ = -1;
  }
};

// The data structure to cover all possible output API's to the functions in
// ReadTimeDepIntervalCensoredData.
// 
// DISCUSSION:
//   We allow the user to specify multiple models (a separate model for each
//   event type, or alternatively, for each event cause), and we also allow
//   multiple event types. The possibilities are:
//     a) Univariate: One event type, cause, and model
//     b) Multivariate: Multiple event types; each with the same model or
//        each with their own model
//     c) Competing Risks: Multiple event causes, each with the same model,
//        or each with their own model
//     d) Multiple event types *and* event causes.
//  In case (d), we allow multiple models, one per event type OR one per event
//  cause, but not one for each (event type, event cause) pair; i.e. the
//  model can depend on (at most) one of Event-Type or Event-Cause.
//  
//  NOTATION:
//    Let K = number of event-types, and K' = number of models. Typically
//    (when the model is determined by event-type), K = K'.
struct TimeDepIntervalCensoredData {
  // One set of distinct times for each event type, so vector has size
  // K = number of event types.
  vector<set<double>> distinct_times_;

  // The line indices (w.r.t. the input data file) of rows that had at
  // least one missing (NA) value in a column used by that model.
  // NOTE: The missing value can appear in one of 4 kinds of columns:
  //   1) A requisite column (Time, Subject Id, Family Id (if present),
  //      Event-Type (if present), Event-Column (if present)
  //   2) Status Column
  //   3) Independent Variable column
  //   4) Unused Column
  // Missing values in (4) don't affect anything, and they won't cause a
  // row to appear in the na_rows_ field below. Missing values in (1) will
  // cause the entire row to be skipped. Missing values as in (2) will
  // dictate the Status is ignored, but the covariate values are still used;
  // vice-versa for missing values as in (3).
  // Vector is indexed by Model, so it has size K'.
  vector<set<int>> na_rows_;
  
  // The following vector holds all of the data values and other info for
  // each Subject (and for each model/event type).
  // WARNING: All of the other vectors in the present struct are indexed
  // w.r.t. to the model index (and hence have size K' = number of models).
  // In contrast, this vector has size n = num Subjects.
  vector<SubjectInfo> subject_info_;

  // The following maps aren't needed for computations, but can be useful
  // for printing and/or debugging.
  //   - Maps Subject_ID to that Subject's corresponding index in subject_info_
  map<int, string> subject_index_to_id_;
  //   - Maps Family name to index.
  map<int, string> family_index_to_id_;
  //   - Maps Event Type name to index.
  map<int, string> event_type_index_to_name_;
  //   - Maps Event Cause name to index.
  map<int, string> event_cause_index_to_name_;

  // The linear_terms of the RHS of the model (after expansion of non-numeric
  // terms, if relevant). The outer-vector has size K' (number of models), the
  // inner vector has size p_k, i.e. the number of linear terms (covariates)
  // for the k^th model.
  vector<vector<string>> legend_;

  // The following field stores the list of non-numeric independent variables.
  // In particular, it is a map of a non-numeric variable's name to the set of
  // distinct values that appear for that variable (across all Subjects).
  map<string, set<string> > nominal_variables_;

  // The following holds the mean and standard_deviation (after having applied
  // all modifications (collapse, extrapolation, expansion of non-numeric))
  // of each linear term, indexed in the same order as (the corresponding
  // event type's) legend_.
  // The outer vector has size K' = number of models.
  // The k^th element of the inner vector has the same size as the k^th
  // legend (p_k). The first coordinate of the tuple indicates if the linear
  // term is binary (all values are 0 or 1), the second is the column mean,
  // and the third is the column std dev. Note that the 'is_binary' bit is
  // used to avoid standardizing binary variables, as standardizing them may
  // actually *increase* number of computations needed for EM convergence,
  // because it is overly sensitive to cases where most of the binary values
  // are 0 or 1); 'true' means "standardize", i.e. true = not binary.
  vector<vector<tuple<bool, double, double>>> linear_terms_mean_and_std_dev_;

  string error_msg_;

  TimeDepIntervalCensoredData() {
    error_msg_ = "";
  }
};

class ReadTimeDepIntervalCensoredData {
 friend class ReadTimeIndepIntervalCensoredData;
 public:
  // Similar to above, but first creates the TimeDepIntervalCensoredInParams
  // object 'in_params' above via the provided ModelAndDataParams 'params'.
  static bool ReadFile(const bool use_time_indep_data_structure,
                       const set<string>& right_censored_events,
                       const string& event_type_col,
                       const string& event_cause_col,
                       const vector<string>& event_types,
                       const vector<string>& event_causes,
                       const vector<ModelAndDataParams>& all_models,
                       TimeDepIntervalCensoredData* out_data);
  // Same as above, for a single model/event-type.
  static bool ReadFile(const bool use_time_indep_data_structure,
                       const set<string>& right_censored_events,
                       const string& event_type_col,
                       const string& event_cause_col,
                       const vector<string>& event_types,
                       const vector<string>& event_causes,
                       const ModelAndDataParams& params,
                       TimeDepIntervalCensoredData* out_data) {
    vector<ModelAndDataParams> temp;
    temp.push_back(params);
    return ReadFile(
        use_time_indep_data_structure, right_censored_events, event_type_col,
        event_cause_col, event_types, event_causes, temp, out_data);
  }
  // Same as above, with no "Event-Cause" column, nor
  // an explicit value for 'use_time_indep_data_structure' (default is true).
  static bool ReadFile(const string& event_type_col,
                       const ModelAndDataParams& params,
                       TimeDepIntervalCensoredData* out_data) {
    return ReadFile(true, set<string>(), event_type_col, "",
                    vector<string>(), vector<string>(), params, out_data);
  }
  // Same as above, with no "Event-Type" nor "Event-Cause" columns, nor
  // an explicit value for 'use_time_indep_data_structure' (default is true).
  static bool ReadFile(const ModelAndDataParams& params,
                       TimeDepIntervalCensoredData* out_data) {
    return ReadFile(true, set<string>(), "", "",
                    vector<string>(), vector<string>(), params, out_data);
  }

  // Checks standardization type to see if (linear term) standardization was
  // applied. If so, unstandardizes the beta and lambda estimates and variance.
  static bool UnstandardizeResults(
      const VariableNormalization std_type,
      const VectorXd& standardized_beta, const VectorXd& standardized_lambda,
      const MatrixXd& standardized_variance,
      const vector<tuple<bool, double, double>>& coordinate_mean_and_std_dev,
      VectorXd* beta, VectorXd* lambda, MatrixXd* variance, string* error_msg);
  // Same as above, but just for beta and lambda.
  static bool UnstandardizeBetaAndLambda(
      const VariableNormalization std_type,
      const VectorXd& standardized_beta, const VectorXd& standardized_lambda,
      const vector<tuple<bool, double, double>>& coordinate_mean_and_std_dev,
      VectorXd* beta, VectorXd* lambda, string* error_msg);
  // Same as above, for multivariate case.
  static bool UnstandardizeBetaAndLambda(
      const VariableNormalization std_type,
      const vector<DependentCovariateEstimates>& standardized_estimates,
      const vector<vector<tuple<bool, double, double>>>& coordinate_mean_and_std_dev,
      vector<DependentCovariateEstimates>* estimates, string* error_msg);

  // Parses command-line arguments:
  //   --model_type_k MODEL_DESCRIPTION
  //   --model_cause_k MODEL_DESCRIPTION
  //   --event_type COL_NAME
  //   --event_cause COL_NAME
  //   --event_types COMMA_SEP_LIST
  //   --event_causes COMMA_SEP_LIST
  //   --right_censored_events COMMA_SEP_LIST
  // Where MODEL_DESCRIPTION should be in quotes, and has format:
  //   (TIME, STATUS) = X1 + X2 + ...
  static bool ParseIntervalCensoredCommandLineArguments(
      int argc, char* argv[],
      string* event_type_col, string* event_cause_col,
      vector<string>* event_types, vector<string>* event_causes,
      set<string>* right_censored_events,
      vector<ModelAndDataParams>* params);

  static void PrintTimeDepIntervalCensoredData(
      const TimeDepIntervalCensoredData& data);

  // Parse the model LHS into the provided data structure. Expected format of
  // input model_lhs_str:
  //   (TIME_COL_NAME, DEP_VAR_1, ..., DEP_VAR_K)
  static bool ParseModelLhs(
      const string& model_lhs_str,
      string* time_col, vector<string>* dep_var_names);

 private:
  // Reads in the file specified by in_params, populating the relevant
  // fields in out_data. May also update in_params as well, e.g.
  // if the col_.name_ fields of the VariableParams are set (but not
  // the col_.index_ fields), then it will update col_index_ fields.
  static bool ReadFile(TimeDepIntervalCensoredInParams& in_params,
                       TimeDepIntervalCensoredData* out_data);

  // Similar to ReadFile above, but now assumes all VariableParams have the
  // col_.index_ field properly set.
  static bool ReadFile(
      const TimeDepIntervalCensoredInParams& in_params,
      const vector<VariableParams>& all_indep_var_params,
      const vector<set<int>>& cols_in_models,
      vector<vector<map<double, bool>>>* subject_to_time_and_status_by_event,
      vector<vector<map<double, vector<DataHolder>>>>* subject_to_data_by_model,
      TimeDepIntervalCensoredData* out_data);

  // Parses the relevant columns ('cols_to_read') in the data 'file' into
  // 'data_values', populates na_columns, nominal_variables, and distinct_times.
  static bool ReadInputData(
      const int num_models,
      const string& filename,
      const string& delimiter, const string& infinity_char,
      const VariableParams& id_params,
      const VariableParams& family_params,
      const VariableParams& event_type_params,
      const VariableParams& event_cause_params,
      const VariableParams& time_params,
      const VariableParams& status_params,
      const vector<VariableParams>& indep_vars_params,
      const vector<set<int>>& cols_in_models,
      const set<int>& right_censored_events,
      const set<string>& na_strings,
      const vector<double> bucketed_times,
      const map<string, int>& event_type_to_index,
      const map<string, int>& event_cause_to_index,
      map<int, string>* subject_index_to_id,
      map<int, string>* family_index_to_id,
      vector<set<int>>* na_rows,
      map<string, set<string> >* nominal_variables,
      vector<vector<map<double, bool>>>* subject_to_time_and_status_by_event,
      vector<vector<map<double, vector<DataHolder>>>>* subject_to_data_by_model,
      vector<SubjectInfo>* subject_info,
      string* error_msg);

  // Parses the 'index' entry of 'str_values' into 'parsed_value', collapsing
  // the value according to 'params' if necessary. If the value is in na_strings,
  // sets 'is_na' to true and returns true. If 'enforce_numeric', returns
  // false if string cannot be parsed as a numeric value.
  static bool ReadValue(
      const bool enforce_numeric,
      const string& col_name, const int col_index,
      const vector<string>& str_values,
      const string& infinity_char,
      const set<string>& na_strings,
      const vector<VariableCollapseParams>& params,
      map<string, set<string> >* nominal_variables,
      bool* is_na, bool* is_new_nominal_col,
      DataHolder* parsed_value, string* error_msg);

  // Reads the indicated columns into 'values'. Also populates 'nominal_variables'
  // with the columns (and all distinct values that appeared in that column)
  // thate were indicated as 'nominal' via the header file (via '$') or because
  // there was a value in that column that could not be parsed as a double.
  // Also populates 'na_cols_for_line' with the indices of columns for which
  // this line indicates 'NA'.
  // Returns true if successful, otherwise returns false (and 'error_msg'
  // contains details of the failure).  
  static bool ReadLine(
      const string& line, const string& delimiter, const string& infinity_char,
      const int line_num,
      const VariableParams& id_params,
      const VariableParams& family_params,
      const VariableParams& event_type_params,
      const VariableParams& event_cause_params,
      const VariableParams& time_params,
      const VariableParams& status_params,
      const vector<VariableParams>& indep_vars_params,
      const set<int>& right_censored_events,
      const set<string>& na_strings,
      const map<string, int>& event_type_to_index,
      const map<string, int>& event_cause_to_index,
      map<string, int>* subject_to_subject_id,
      map<string, int>* family_to_index,
      int* subject_id, bool* status, double* time,
      int* family_index, int* event_type, int* event_cause,
      set<pair<int, int>>* right_censored_special_rows_seen,
      vector<DataHolder>* indep_var_values,
      map<string, set<string> >* nominal_variables,
      bool* right_censored_special_row, bool* right_censored_row,
      bool* is_na_row, bool* is_na_status, bool* is_na_event_cause,
      bool* is_new_nominal_col, set<int>* na_indep_vars, string* error_msg);

  static bool CheckEventTypeAndCauseConsistency(
      const map<string, set<string>>& nominal_values,
      const VariableParams& event_type_params,
      const VariableParams& event_cause_params,
      const set<string>& event_types,
      const set<string>& event_causes);

  static bool SanityCheckEventTypeAndCauseConsistency(
      const int num_models,
      const string& event_type_col,
      const string& event_cause_col,
      const vector<string>& event_types,
      const vector<string>& event_causes,
      const set<string>& right_censored_events,
      set<int>* right_censored_event_indices);

  // Reads the time column, and performs bucketing/rounding.
  static bool BucketTimes(
      const FileInfo& file_info,
      const VariableParams& time_params,
      const set<string>& na_strings,
      vector<double>* bucketed_times,
      string* error_msg);

  static bool GetColumnIndex(
      const string& title_line, const string& delimiter, const string& col_name,
      int* col_index, string* error_msg);
  
  static bool GetColumnIndices(
      const string& title_line, const string& delimiter,
      const vector<string>& col_names,
      vector<int>* col_indices, string* error_msg);

  // Read Variable Names on header line, looking for Nominal variables, as
  // identified by those with a '$'-suffix.
  static bool GetNominalVariablesFromHeader(
    const string& title_line, const string& delimiter,
    map<string, set<string>>* nominal_variables, string* error_msg);

  // For VariableParams items in in_params that have the col_.name_ set
  // but not the col_.index_, reads title_line and then sets col_.index_.
  static bool FillColumnIndices(
      const string& title_line,
      TimeDepIntervalCensoredInParams* in_params, string* error_msg);

  // Determines the mean and standard deviation for each linear term.
  // DISCUSSION: The time-dependent nature of covariates makes the definition
  // of statistics (mean and standard deviation) complicated:
  //   a) The number of observations per Subject may not be equal for all
  //      Subjects; leading to potentinally skewed statistics (e.g. if one
  //      Subject has 100 examination times, everybody else has 1)
  //   b) For a fixed variable, not all values of that variable may be used.
  //      For example, consider a scenario with 2 Subjects, one has
  //      100 examination times and the other has 1 examination time. Then
  //      the set of distinct {L_i, R_i} times has size 4, which means for
  //      each covariate, (at most) only 4 of the 100 values will be used
  //      in the computations; so should all of the 100 values be used
  //      in the standardization?
  //   c) Also, some Variables may define "extrapolation" by choosing the
  //      nearest-left; others nearest-right; others baseline constant.
  //      When a linear term involves variables that use different
  //      extrapolation options, using stats from before the final values
  //      are computed is impossible
  //   d) In light of (b) and (c), it would seem like we ought to first
  //      compute the (unstandardized) values of all linear terms at
  //      all of the distinct time points, then standardize, then update
  //      the values. While this is possible, it is more computationally
  //      expensive than our actual choice (see (e) below).
  //   e) Since we are going to unstandardize estimates in the end, i.e.
  //      we're only standardizing in the first place to make sure
  //      computations don't blow-up, it really doesn't matter that the
  //      standardizing statistics (mean and standard deviation) are
  //      precise; just that they are in the right ballpark. For this
  //      reason, we will just use each Subject's baseline (first) values
  //      of each of the covariates, and evaluate each linear term using
  //      these values, finding the resulting statistics (mean and std dev).
  // Note that in terms of computation: If there are M_k distinct times, 
  // p linear terms, and n Subjects, then:
  //   For strategy (d) above: First compute the n x (p, M_k)-Matrices of values,
  //     and then perform computations, so O(n * p * M_k) computation cost. 
  //   For strategy (e) above: Computing statistics on p linear terms with
  //     O(n) values is O(n * p).
  static bool GetStatisticsForLinearTerms(
      const bool do_sample_variance,
      const int model_index, const int id_col,
      const set<string>& na_strings,
      const map<string, set<string>>& nominal_variables,
      const vector<VariableParams>& indep_vars_params_k,
      const vector<Expression>& linear_terms_k,
      const vector<string>& data_header,
      const vector<vector<map<double, vector<DataHolder>>>>& subject_to_data_by_model,
      vector<tuple<bool, double, double>>* linear_terms_mean_and_std_dev_k,
      string* error_msg);

  // Finds the (L, R) times for each interval-censored event for each Subject.
  // Also sets out_data->distinct_times_: the set of distinct times for
  // each event type.
  static bool PopulateTimes(
      const int num_events,
      const set<int>& right_censored_events,
      const vector<vector<map<double, bool>>>& subject_to_time_and_status_by_event,
      vector<SubjectInfo>* subject_info,
      vector<set<double>>* distinct_times,
      string* error_msg);

  // Computes final values for all subjects:
  //   - Final Data Structure will be a Matrix of size (p, M_k), where
  //     p is the number of linear terms, and M_k is the number of distinct
  //     time points. Each subject will have such a matrix
  //   - To compute the value of coordinate (i, j) of this matrix:
  //       a) Use the (unstandardized) values of each variable that appears
  //          in the linear term. Since variables are time-dependent, pick
  //          the value for each variable according to TimeDependentParams
  //       b) Plug in the values of the variables from (a) into the
  //          linear term expression (i.e. evaluate the expression)
  //       c) Standardize: subtract the mean and divide by standard deviation,
  //          using (mean, std_dev) values for that linear term (i.e.
  //          using linear_terms_mean_and_std_dev_)
  // NOTE: There are various options for how to extend the discrete observations
  // (covariate value vs. time) to a continuous function of time:
  //   a) For a time t that is BEFORE any observed time-point for a subject:
  //      i)  Set X(t) = 0
  //      ii) Set X(t) = X(T_1) (i.e. use the first time-point T_1 that we
  //                             have for this subject).
  //   b) For a time t that is AFTER any observed time-point for a subject:
  //      i)   Set X(t) = X(T_N) (i.e. use the last time-point T_N that we
  //                             have for this subject).
  //      ii)  Set X(t) = 0
  //   c) For time t \in (T_j, T_{j+1}):
  //      i)  Set X(t) = X(T_j)     (i.e. right-continuous curve)
  //      ii) Set X(t) = X(T_{j+1}) (i.e. left-continuous curve)
  // Not sure if it makes sense to simultaneously choose a.ii AND b.i;
  // intuitively, a.ii makes sense, so we go with that option, and therefore
  // we go with b.i. Also, (c) should be consistent with the choices made for
  // (a) and (b); so RIGHT_CONTINUOUS should mean: (a.i, b.i, c.i);
  // and LEFT_CONTINUOUS should mean: (a.ii, b.ii, c.ii).
  static bool PopulateIndependentVariableValues(
      const bool use_time_indep_data_structure,
      const int model_index,
      const VariableNormalization& standardize_linear_terms,
      const vector<tuple<bool, double, double>>& linear_terms_mean_and_std_dev_k,
      const vector<VariableParams>& indep_vars_params_k,
      const map<string, set<string>>& nominal_variables,
      const vector<Expression>& linear_terms_k,
      const set<double>& distinct_times_k,
      const vector<vector<map<double, vector<DataHolder>>>>& subject_to_data_by_model,
      vector<SubjectInfo>* subject_info,
      string* error_msg);

  // Input:
  //   - indep_var_params: Parameters for the current independent variable
  //   - times:            Set of all distinct times
  //   - time_to_values:   Key is time, Value is the pair (Status, Data). Each
  //                       subject will have their own version of this data
  //                       structure. The Keys will be the examination times
  //                       for that Subject, the corresponding Status will
  //                       be their status, and the vector<DataHolder> will
  //                       be the values of each of the indep values at
  //                       that examination time.
  //   - indep_var_index: The index (within vector<DataHolder>) of the current
  //                      independent variable
  // Output:
  //   - "row" will have size equal to the number of distinct times (times.size())
  //     It will hold the value of the current independent variable at each of
  //     those times.
  static bool GetTimeDependentValues(
      const int indep_var_index,
      const TimeDependentParams& indep_var_params,
      const set<double>& times,
      const map<double, vector<DataHolder>>& time_to_values,
      bool* is_time_indep,
      vector<DataHolder>* row, string* error_msg);

  // Evaluates the 'linear_term' by parsing the Expression that represents it,
  // plugging in values from 'var_to_value' for any variables that appear in
  // the expression. Standardizes the linear_term by applying the appropriate
  // normalization, based on the provided 'mean_and_std_dev' tuple.
  static bool EvaluateLinearTerm(
      const Expression& linear_term,
      const VariableNormalization& standardize_linear_terms,
      const tuple<bool, double, double>& mean_and_std_dev,
      const map<string, double>& var_to_value,
      const set<string>& time_indep_vars,
      double* value, bool* is_time_indep, string* error_msg);
};

}  // namespace file_reader_utils

#endif
