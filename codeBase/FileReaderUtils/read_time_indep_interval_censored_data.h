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
#include "FileReaderUtils/read_time_dep_interval_censored_data.h"
#include "MathUtils/data_structures.h"

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef READ_TIME_INDEP_INTERVAL_CENSORED_DATA_H
#define READ_TIME_INDEP_INTERVAL_CENSORED_DATA_H

using math_utils::DataHolder;
using namespace std;

namespace file_reader_utils {

// Temporary storage for a Subject's values; used only internally.
struct SubjectValues {
  vector<DataHolder> indep_var_values_;
  double lower_time_;
  double upper_time_;
  EventTimeAndCause event_info_;
};

// The data structure to cover all possible input API's to the functions in
// ReadTimeIndepIntervalCensoredData.
struct TimeIndepIntervalCensoredInParams {
  FileInfo file_info_;

  // Expression representing the RHS of the model(s). Vector length is the
  // number of models.
  vector<Expression> model_rhs_;

  // Data Column information (column names, indices, collapse params, etc.).
  VariableParams id_params_;
  VariableParams family_params_;
  VariableParams left_time_params_;
  VariableParams right_time_params_;
  VariableParams event_type_params_;
  VariableParams event_cause_params_;
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

  TimeIndepIntervalCensoredInParams() {
    linear_terms_normalization_ =
        VariableNormalization::VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;
  }
};

class ReadTimeIndepIntervalCensoredData {
 public:
  // Reads in the specified file, and populates out_data.
  static bool ReadFile(const set<string>& right_censored_events,
                       const string& event_type_col,
                       const string& event_cause_col,
                       const vector<string>& event_types,
                       const vector<string>& event_causes,
                       const vector<ModelAndDataParams>& all_models,
                       TimeDepIntervalCensoredData* out_data);
  // Same as above, for a single model/event-type.
  static bool ReadFile(const set<string>& right_censored_events,
                       const string& event_type_col,
                       const string& event_cause_col,
                       const vector<string>& event_types,
                       const vector<string>& event_causes,
                       const ModelAndDataParams& params,
                       TimeDepIntervalCensoredData* out_data) {
    vector<ModelAndDataParams> temp;
    temp.push_back(params);
    return ReadFile(
        right_censored_events, event_type_col, event_cause_col,
        event_types, event_causes, temp, out_data);
  }
  // Same as above, with no "Event-Cause" column, nor
  // an explicit value for 'use_time_indep_data_structure' (default is true).
  static bool ReadFile(const string& event_type_col,
                       const ModelAndDataParams& params,
                       TimeDepIntervalCensoredData* out_data) {
    return ReadFile(set<string>(), event_type_col, "",
                    vector<string>(), vector<string>(), params, out_data);
  }
  // Same as above, with no "Event-Type" nor "Event-Cause" columns, nor
  // an explicit value for 'use_time_indep_data_structure (default is true).
  static bool ReadFile(const ModelAndDataParams& params,
                       TimeDepIntervalCensoredData* out_data) {
    return ReadFile(set<string>(), "", "", vector<string>(), vector<string>(),
                    params, out_data);
  }

  // Parse the model LHS into the provided data structure. Expected format of
  // input model_lhs_str:
  //   ([DEP_VAR_1:] Left_Endpoint_1_Col, Right_Endpoint_1_Col, ...
  //    [DEP_VAR_K:] Left_Endpoint_K_Col, Right_Endpoint_K_Col)
  static bool ParseModelLhs(
      const string& model_lhs_str,
      vector<tuple<string, string, string>>* dep_var_name_and_endpoints);

  // Note on generalizing observations at specific time points to making
  // indep_varialbes a (continuous) function of time:
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
  static bool PopulateSubjectInfo(
      const int model_index,
      const VariableNormalization& standardize_linear_terms,
      const vector<tuple<bool, double, double>>& linear_terms_mean_and_std_dev_k,
      const vector<VariableParams>& indep_vars_params_k,
      const map<string, set<string>>& nominal_variables,
      const vector<Expression>& linear_terms_k,
      const vector<vector<vector<DataHolder>>>& subject_to_data_by_model,
      vector<SubjectInfo>* subject_info, string* error_msg);
  // Same as above, for a single dependent covariate/event.
  static bool PopulateSubjectInfo(
      const VariableNormalization& standardize_linear_terms,
      const vector<tuple<bool, double, double>>& linear_terms_mean_and_std_dev,
      const vector<VariableParams>& indep_vars_params,
      const map<string, set<string>>& nominal_variables,
      const vector<Expression>& linear_terms,
      const vector<vector<DataHolder>>& subject_to_data,
      vector<SubjectInfo>* subject_info, string* error_msg);

  // Determines the mean and standard deviation for each linear term.
  static bool GetStatisticsForTimeIndependentLinearTerms(
      const bool do_sample_variance, const int model_index,
      const map<string, set<string>>& nominal_variables,
      const vector<VariableParams>& indep_vars_params_k,
      const vector<Expression>& linear_terms_k,
      const vector<string>& data_header,
      const vector<vector<vector<DataHolder>>>& data_values,
      vector<tuple<bool, double, double>>* linear_terms_mean_and_std_dev,
      string* error_msg);

 private:
  // Reads in the file specified by in_params, populating the relevant
  // fields in out_data. May also update in_params as well, e.g.
  // if the col_.name_ fields of the VariableParams are set (but not
  // the col_.index_ fields), then it will update col_index_ fields.
  static bool ReadFile(TimeIndepIntervalCensoredInParams& in_params,
                       TimeDepIntervalCensoredData* out_data);
  // Similar to above, but now assumes all VariableParams have the col_.index_
  // field properly set.
  // Important: We assume that file is already oriented to start reading the
  // *DATA* rows, i.e. that if the file has Title Line and/or other MetaData/
  // Comment Lines, that file is already positioned past all of these, on the
  // first row (line) of actual data.
  static bool ReadFile(
      const TimeIndepIntervalCensoredInParams& in_params,
      const vector<VariableParams>& all_indep_var_params,
      const vector<set<int>>& cols_in_models,
      vector<vector<vector<DataHolder>>>* subject_to_data_by_model,
      TimeDepIntervalCensoredData* out_data);

  // Parses the relevant columns ('cols_to_read') in the data 'file' into
  // 'data_values', populates na_columns, nominal_variables, and distinct_times.
  static bool ReadInputData(
      const int num_models,
      const string& filename,
      const string& delimiter, const string& infinity_char,
      const VariableParams& left_time_params,
      const VariableParams& right_time_params,
      const VariableParams& family_params,
      const VariableParams& event_type_params,
      const VariableParams& event_cause_params,
      const vector<VariableParams>& indep_vars_params,
      const vector<set<int>>& cols_in_models,
      const set<int>& right_censored_events,
      const set<string>& na_strings,
      const vector<double> bucketed_end_times,
      const vector<double> bucketed_right_times,
      const map<string, int>& event_type_to_index,
      const map<string, int>& event_cause_to_index,
      map<int, string>* family_index_to_id,
      vector<set<int>>* na_rows,
      map<string, set<string> >* nominal_variables,
      vector<set<double>>* distinct_times,
      vector<vector<vector<DataHolder>>>* subject_to_data_by_model,
      vector<SubjectInfo>* subject_info,
      string* error_msg);

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
      const VariableParams& left_time_params,
      const VariableParams& right_time_params,
      const VariableParams& family_params,
      const VariableParams& event_type_params,
      const VariableParams& event_cause_params,
      const vector<VariableParams>& indep_vars_params,
      const set<int>& right_censored_events,
      const set<string>& na_strings,
      const map<string, int>& event_type_to_index,
      const map<string, int>& event_cause_to_index,
      map<string, int>* family_to_index,
      double* left_time, double* right_time,
      int* family_index, int* event_type, int* event_cause,
      vector<DataHolder>* indep_var_values,
      map<string, set<string> >* nominal_variables,
      bool* right_censored_row,
      bool* is_na_row, bool* is_na_event_cause,
      bool* is_new_nominal_col, set<int>* na_indep_vars, string* error_msg);

  // For VariableParams items in in_params that have the col_.name_ set
  // but not the col_.index_, reads title_line and then sets col_.index_.
  static bool FillColumnIndices(
      const string& title_line,
      TimeIndepIntervalCensoredInParams* in_params, string* error_msg);

  // If collapse_time_params' fields indicate one or more of the time variable
  // columns should be bucketed/rounded (to reduce the number of distinct
  // times), then this function performs such rounding.
  static bool UpdateBucketTimeColumns(
      const map<string, CollapseTimesParams>& collapse_time_params,
      vector<vector<pair<double, double>>>* subjects_time_intervals,
      string* error_msg);

  // Buckets the family column, if family_params indicates to do so.
  // DEPRECATED. Not sure why I ever needed this function?...
  /*
  static bool UpdateBucketFamilyColumn(
      const VariableParams& family_params,
      vector<string>* subjects_families, string* error_msg);
  */
};

}  // namespace file_reader_utils

#endif
