// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "read_time_indep_interval_censored_data.h"

#include "FileReaderUtils/read_file_utils.h"
#include "FileReaderUtils/read_table_with_header.h"
#include "FileReaderUtils/read_time_dep_interval_censored_data.h"
#include "MapUtils/map_utils.h"
#include "MathUtils/constants.h"
#include "MathUtils/data_structures.h"
#include "MathUtils/eq_solver.h"
#include "MathUtils/statistics_utils.h"
#include "StringUtils/string_utils.h"

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using file_reader_utils::VariableNormalization;
using namespace map_utils;
using namespace math_utils;
using namespace string_utils;
using namespace std;

namespace file_reader_utils {

namespace {

// Assigns/forces certain special columns to be of NOMINAL type:
//   ID, Family, Event-Type, and Event-Cause.
void AssignKnownNominalColumns(
    const TimeIndepIntervalCensoredInParams& in_params,
    map<string, set<string>>* nominal_variables) {
  if (nominal_variables == nullptr) return;
  if (in_params.family_params_.col_.index_ >= 0) {
    nominal_variables->insert(make_pair(
        in_params.family_params_.col_.name_, set<string>()));
  }
  if (in_params.event_type_params_.col_.index_ >= 0) {
    nominal_variables->insert(make_pair(
        in_params.event_type_params_.col_.name_, set<string>()));
  }
  if (in_params.event_cause_params_.col_.index_ >= 0) {
    nominal_variables->insert(make_pair(
        in_params.event_cause_params_.col_.name_, set<string>()));
  }
}

// Adds column (variable) names to 'data_header' in the same order they
// were added to the internal field 'SubjectValues.indep_var_values_'
// in ReadFile() below.
bool GetHeaderForInternalDataValuesHolder(
    const vector<VariableParams>& all_indep_vars,
    vector<string>* data_header, string* error_msg) {
  if (data_header == nullptr) return false;
  data_header->clear();

  // In ReadFile(), only columns corresponding to the independent variables
  // (covariates) were added to indep_var_values_, and they were added in
  // the order they appear in all_indep_vars.
  for (const VariableParams& indep_vars_params : all_indep_vars) {
    data_header->push_back(indep_vars_params.col_.name_);
  }

  return true;
}

}  // (anonymous) namespace

bool ReadTimeIndepIntervalCensoredData::ReadLine(
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
      bool* is_new_nominal_col, set<int>* na_indep_vars, string* error_msg) {
  if (line.empty()) {
    return true;
  }

  // Split line around delimiter.
  vector<string> col_strs;
  Split(line, delimiter, false /* do not collapse empty strings */, &col_strs);

  // Fetch the Family column, if present.
  if (family_params.col_.index_ >= 0) {
    DataHolder family_value;
    if (!ReadTimeDepIntervalCensoredData::ReadValue(
            false, family_params.col_.name_, family_params.col_.index_, col_strs,
            infinity_char, na_strings, family_params.collapse_params_,
            nominal_variables, is_na_row, is_new_nominal_col,
            &family_value, error_msg)) {
      return false;
    }
    if (*is_na_row) return true;
    const string& family_id = family_value.name_;
    // Check if this Family has already been seen.
    int* temp_family_index = FindOrNull(family_id, *family_to_index);
    if (temp_family_index == nullptr) {
      // Haven't seen this family id yet. Add an entry.
      const int new_family_index = family_to_index->size();
      family_to_index->insert(make_pair(family_id, new_family_index));
      *family_index = new_family_index;
    } else {
      *family_index = *temp_family_index;
    }
  }

  // Fetch the Event-Type column, if present.
  if (event_type_params.col_.index_ >= 0) {
    DataHolder event_value;
    if (!ReadTimeDepIntervalCensoredData::ReadValue(
            false, event_type_params.col_.name_, event_type_params.col_.index_,
            col_strs, infinity_char, na_strings,
            event_type_params.collapse_params_, nominal_variables, is_na_row,
            is_new_nominal_col, &event_value, error_msg)) {
      return false;
    }
    if (*is_na_row) return true;
    const string& event_name = event_value.name_;
    // Find the index for this event type.
    if (event_type_to_index.empty()) {
      *event_type = 0;
    } else {
      const int* temp_event_index = FindOrNull(event_name, event_type_to_index);
      if (temp_event_index == nullptr) {
        cout << "ERROR: Unexpected event type '" << event_name
             << "' found on line " << line_num << " of the input data file."
             << endl;
        return false;
      }
      *event_type = *temp_event_index;
    }
    *right_censored_row =
        right_censored_events.find(*event_type) !=
        right_censored_events.end();
  } else if (!right_censored_events.empty()) {
    // In case everything is a Right-Censored event (so no Event-Type column
    // present or required).
    *right_censored_row =
        right_censored_events.find(0) !=
        right_censored_events.end();
  }

  // Fetch the Event-Cause column, if present and Status was '1'.
  if (event_cause_params.col_.index_ >= 0) {
    DataHolder event_cause_value;
    if (!ReadTimeDepIntervalCensoredData::ReadValue(
            false, event_cause_params.col_.name_, event_cause_params.col_.index_,
            col_strs, infinity_char, na_strings,
            event_cause_params.collapse_params_, nominal_variables, is_na_event_cause,
            is_new_nominal_col, &event_cause_value, error_msg)) {
      return false;
    }
    if (!(*is_na_event_cause)) {
      const string& event_name = event_cause_value.name_;
      // Find the index for this event cause.
      if (event_cause_to_index.empty()) {
        *event_cause = 0;
      } else {
        const int* temp_event_index = FindOrNull(event_name, event_cause_to_index);
        if (temp_event_index == nullptr) {
          cout << "ERROR: Unexpected event cause '" << event_name
               << "' found on line " << line_num << " of the input data file."
               << endl;
          return false;
        }
        *event_cause = *temp_event_index;
      }
    }
  }

  // Fetch the Time Interval column(s).
  // NOTE: Special attention must be paid if this is a Right-Censored event,
  // in which case the (L, R) columns are overloaded:
  //   - Left-Endpoint column holds \tilde{T}: The Survival/Censoring Time
  //   - Right-Endpoint column holds the Status (1 or 0)
  if (!*right_censored_row) {
    // Read (L, R) timepoints.
    DataHolder left_endpoint;
    if (!ReadTimeDepIntervalCensoredData::ReadValue(
                   true, left_time_params.col_.name_,
                   left_time_params.col_.index_,
                   col_strs, infinity_char, na_strings,
                   left_time_params.collapse_params_, nominal_variables,
                   is_na_row, nullptr, &left_endpoint, error_msg)) {
      return false;
    }
    if (*is_na_row) return true;
    if (left_endpoint.value_ < 0.0) {
      // Negative time-values are invalid. Treat these as missing value (NA).
      *is_na_row = true;
      return true;
    }
    *left_time = left_endpoint.value_;
    DataHolder right_endpoint;
    if (!ReadTimeDepIntervalCensoredData::ReadValue(
                   true, right_time_params.col_.name_,
                   right_time_params.col_.index_,
                   col_strs, infinity_char, na_strings,
                   right_time_params.collapse_params_, nominal_variables,
                   is_na_row, nullptr, &right_endpoint, error_msg)) {
      return false;
    }
    if (*is_na_row) return true;
    if (right_endpoint.value_ < 0.0) {
      // Negative time-values are invalid. Treat these as missing value (NA).
      *is_na_row = true;
      return true;
    }
    *right_time = right_endpoint.value_;
  } else {
    // Read Right-Censored (\tilde{T}, Status) values.
    DataHolder tilde_t;
    if (!ReadTimeDepIntervalCensoredData::ReadValue(
                   true, left_time_params.col_.name_,
                   left_time_params.col_.index_,
                   col_strs, infinity_char, na_strings,
                   left_time_params.collapse_params_, nominal_variables,
                   is_na_row, nullptr, &tilde_t, error_msg)) {
      return false;
    }
    if (*is_na_row) return true;
    if (tilde_t.value_ < 0.0) {
      // Negative time-values are invalid. Treat these as missing value (NA).
      *is_na_row = true;
      return true;
    }
    *left_time = tilde_t.value_;
    DataHolder status_value;
    if (!ReadTimeDepIntervalCensoredData::ReadValue(
                   true, right_time_params.col_.name_,
                   right_time_params.col_.index_,
                   col_strs, infinity_char, na_strings,
                   vector<VariableCollapseParams>(), nominal_variables,
                   is_na_row, nullptr, &status_value, error_msg)) {
      return false;
    }
    if (*is_na_row) return true;
    if (status_value.value_ != 1.0 && status_value.value_ != 0.0) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to parse Status column (index " +
                      Itoa(right_time_params.col_.index_) + ") on data row " +
                      Itoa(line_num) + " as a '0' or '1':\t" +
                      Itoa(status_value.value_) + "\n";
      }
      return false;
    }
    // Overload right_time to contain status.
    *right_time = status_value.value_;
  }

  // Iterate through the independent variable columns, copying values to
  // 'indep_var_values'.
  indep_var_values->clear();
  for (int i = 0; i < indep_vars_params.size(); ++i) {
    DataHolder value;
    bool is_na_value = false;
    if (!ReadTimeDepIntervalCensoredData::ReadValue(
            false, indep_vars_params[i].col_.name_,
            indep_vars_params[i].col_.index_, col_strs,
            infinity_char, na_strings, indep_vars_params[i].collapse_params_,
            nominal_variables, &is_na_value, is_new_nominal_col,
            &value, error_msg)) {
      return false;
    }
    if (is_na_value) {
      na_indep_vars->insert(i);
      indep_var_values->push_back(DataHolder());
    } else {
      indep_var_values->push_back(value);
    }
  }

  return true;
}

bool ReadTimeIndepIntervalCensoredData::ReadInputData(
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
    const vector<double> bucketed_left_times,
    const vector<double> bucketed_right_times,
    const map<string, int>& event_type_to_index,
    const map<string, int>& event_cause_to_index,
    map<int, string>* family_index_to_id,
    vector<set<int>>* na_rows,
    map<string, set<string> >* nominal_variables,
    vector<set<double>>* distinct_times,
    vector<vector<vector<DataHolder>>>* subject_to_data_by_model,
    vector<SubjectInfo>* subject_info,
    string* error_msg) {
  // Sanity check input.
  if (subject_to_data_by_model == nullptr || nominal_variables == nullptr ||
      subject_info == nullptr) {
    if (error_msg != nullptr) *error_msg += "ERROR: Null values.\n";
    return false;
  }

  // Open file, and advance past first (header) line, to the first row of data.
  ifstream file(filename);
  if (!file.is_open()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to open file '" + filename + "'.\n";
    }
    return false;
  }
  // Read header line (won't do anything with it;
  // just need to advance to first data line).
  string title_line;
  if (!getline(file, title_line)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Empty Input Data file '" + filename + "'\n";
    }
    return false;
  }

  // Determine if the model depends on event type, event cause, or neither.
  const bool model_depends_on_type =
      (num_models > 1 && num_models == event_type_to_index.size());
  const bool model_depends_on_cause =
      (num_models > 1 && num_models == event_cause_to_index.size());

  // The time-independent data format necessary groups all data
  // corresponding to a single Subject in a contiguous block of lines.
  // The number of lines in this block equals the number of events.
  const int num_event_types = 
      event_type_to_index.empty() ? 1 : event_type_to_index.size();
  const int num_lines_per_subject = num_event_types;

  // Read values from file.
  distinct_times->clear();
  distinct_times->resize(num_event_types);
  subject_to_data_by_model->clear();
  subject_info->clear();
  string line;
  int line_num = 1;
  int bucketed_time_index = -1;
  bool need_another_pass = false;
  vector<bool> is_first_data_row(num_models, true);
  map<string, int> family_id_to_index;
  int subject_index = -1;
  int num_lines_read_for_current_subject = num_lines_per_subject;
  while (getline(file, line)) {
    ++bucketed_time_index;
    // Determine if this is a new Subject.
    if (num_lines_read_for_current_subject == num_lines_per_subject) {
      // New Subject; add entry for it in the data holders, and update
      // parameters.
      num_lines_read_for_current_subject = 0;
      ++subject_index;
      // Add an entry to subject_to_data_by_model and subject_info if this is the
      // first row for this subject.
      subject_to_data_by_model->push_back(
          vector<vector<DataHolder>>(num_models));
      subject_info->push_back(SubjectInfo());
      subject_info->back().times_.resize(num_event_types);
    }
    ++num_lines_read_for_current_subject;
    file_reader_utils::RemoveWindowsTrailingCharacters(&line);
    int family_index = -1;
    int event_type = 0;
    int event_cause = 0;
    double left_time, right_time;
    bool is_na_row = false;
    bool is_na_event_cause = false;
    set<int> na_indep_vars;
    vector<DataHolder> indep_var_values;
    bool right_censored_row = false;
    bool new_nominal_column = false;
    if (!ReadLine(line, delimiter, infinity_char, line_num,
                  left_time_params, right_time_params, family_params,
                  event_type_params, event_cause_params,
                  indep_vars_params, right_censored_events, na_strings,
                  event_type_to_index, event_cause_to_index,
                  &family_id_to_index, &left_time, &right_time,
                  &family_index, &event_type, &event_cause,
                  &indep_var_values, nominal_variables,
                  &right_censored_row,
                  &is_na_row, &is_na_event_cause,
                  &new_nominal_column, &na_indep_vars, error_msg)) {
      return false;
    }

    const bool status = (right_censored_row ? right_time == 1.0 : false);
    if (is_na_row ||
        (is_na_event_cause && model_depends_on_cause) ||
        (is_na_event_cause && status)) {
      // Rows that have an NA in a key column are skipped; but record
      // which rows were skipped.
      for (set<int>& na_rows_k : *na_rows) {
        na_rows_k.insert(line_num);
      }
    } else {
      if (subject_index + 1 != subject_info->size()) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unexpected subject index '" + Itoa(subject_index) +
                        "' doesn't match the number of Subjects already read in (" +
                        Itoa(static_cast<int>(subject_info->size())) + ").\n";
        }
        return false;
      }

      // Update time to be the bucketed time, if appropriate.
      if (!bucketed_left_times.empty()) {
        left_time = bucketed_left_times[bucketed_time_index];
      }
      if (!bucketed_right_times.empty()) {
        right_time = bucketed_right_times[bucketed_time_index];
      }

      // Determine the appropriate model index (either the event_type,
      // event_cause, or '0' if there is just one model).
      const int model_index =
          num_models == 1 ? 0 :
          model_depends_on_type ? event_type :
          event_cause;

      // Store row. Row is one of two flavors:
      //   a) Right-Censored data row (i.e. all non-special rows for Right-Censored data)
      //   b) Interval-Censored data row.
      if (right_censored_row) {
        // The row just read was flavor (a) above, a right-censored row.
        SubjectInfo& current_subject_info = (*subject_info)[subject_index];
        current_subject_info.family_index_ = family_index;
        EventTimeAndCause& current_event_info =
            current_subject_info.times_[event_type];
        if (current_event_info.type_ != CensoringType::CENSOR_TYPE_UNKNOWN) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR: Multiple Right-Cenosred special rows found "
                          "for subject index '" + Itoa(subject_index) +
                          "' (second one found on row " + Itoa(line_num) + ").\n";
          }
          return false;
        }
        current_event_info.type_ = CensoringType::CENSOR_TYPE_RIGHT;
        if (!is_na_event_cause) {
          current_event_info.event_cause_ = event_cause;
        }
        CensoringData& data = current_event_info.censoring_info_;
        data.is_alive_ = !status;
        data.censoring_time_ = data.is_alive_ ? left_time : left_time + 1.0;
        data.survival_time_ = data.is_alive_ ? left_time + 1.0 : left_time;
        if (left_time > 0.0) {
          (*distinct_times)[event_type].insert(left_time);
        }

        // Check that none of the columns that are involved in the model
        // for this row have NA values.
        const set<int>& cols_in_model_k = cols_in_models[model_index];
        set<int> intersection;
        set_intersection(cols_in_model_k.begin(), cols_in_model_k.end(),
                         na_indep_vars.begin(), na_indep_vars.end(),
                         inserter(intersection, intersection.begin()));
        if (!intersection.empty()) {
          (*na_rows)[model_index].insert(line_num);
        } else {
          vector<DataHolder>& current_subject_to_data =
              (*subject_to_data_by_model)[subject_index][model_index];
          for (const int col_index : cols_in_model_k) {
            if (col_index >= indep_var_values.size()) {
              if (error_msg != nullptr) {
                *error_msg += "ERROR: Invalid column index " + Itoa(col_index) +
                              " is larger than the number of independent covariates (" +
                              Itoa(static_cast<int>(indep_var_values.size())) + ").\n";
              }
              return false;
            }
            current_subject_to_data.push_back(indep_var_values[col_index]);
          }
        }
        need_another_pass |=
            (!is_first_data_row[model_index] && new_nominal_column);
        is_first_data_row[model_index] = false;
      } else {
        // The row just read was flavor (b) above, an interval-censored row.
        SubjectInfo& current_subject_info = (*subject_info)[subject_index];
        if (current_subject_info.family_index_ < 0) {
          current_subject_info.family_index_ = family_index;
        }
        if (current_subject_info.family_index_ != family_index) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR: Different family values found for subject '" +
                          Itoa(subject_index) +
                          "' (second one found on row " + Itoa(line_num) + ").\n";
          }
          return false;
        }
        EventTimeAndCause& current_event_info =
            current_subject_info.times_[event_type];
        current_event_info.type_ = CensoringType::CENSOR_TYPE_INTERVAL;
        if (!is_na_event_cause) {
          current_event_info.event_cause_ = event_cause;
        }
        current_event_info.lower_ = left_time;
        current_event_info.upper_ = right_time;

        // Add (left-time, right-time) to distinct times.
        if (left_time > 0.0) {
          (*distinct_times)[event_type].insert(left_time);
        }
        if (right_time != numeric_limits<double>::infinity()) {
          (*distinct_times)[event_type].insert(right_time);
        }

        // Add the independent variable values corresponding to this time,
        // provided that none of the columns that are involved in the model
        // for this row have NA values.
        const set<int>& cols_in_model_k = cols_in_models[model_index];
        set<int> intersection;
        set_intersection(cols_in_model_k.begin(), cols_in_model_k.end(),
                         na_indep_vars.begin(), na_indep_vars.end(),
                         inserter(intersection, intersection.begin()));
        if (!intersection.empty()) {
          (*na_rows)[model_index].insert(line_num);
        } else {
          vector<DataHolder>& current_subject_to_data =
              (*subject_to_data_by_model)[subject_index][model_index];
          for (const int col_index : cols_in_model_k) {
            if (col_index >= indep_var_values.size()) {
              if (error_msg != nullptr) {
                *error_msg += "ERROR: Invalid column index " + Itoa(col_index) +
                              " is larger than the number of independent covariates (" +
                              Itoa(static_cast<int>(indep_var_values.size())) + ").\n";
              }
              return false;
            }
            current_subject_to_data.push_back(indep_var_values[col_index]);
          }
        }
        need_another_pass |=
            (!is_first_data_row[model_index] && new_nominal_column);
        is_first_data_row[model_index] = false;
      }
    }
    ++line_num;
  }

  // If a column wasn't detected by the first row of data (i.e. if the column
  // title didn't indicate it was nominal by a $-suffix, and the value in the
  // first row wasn't obviously nominal), then we did not store the first
  // row(s) of numeric values for the column. But since the column was then
  // later identified as Nominal (by a value in a lower row not being parsable
  // as a numeric value), we now need to go back and treat all the earlier row
  // values for such columns as nominal.
  if (need_another_pass) {
    return ReadInputData(
               num_models, filename, delimiter, infinity_char,
               left_time_params, right_time_params,
               family_params, event_type_params, event_cause_params,
               indep_vars_params, cols_in_models,
               right_censored_events, na_strings,
               bucketed_left_times, bucketed_right_times,
               event_type_to_index, event_cause_to_index,
               family_index_to_id, na_rows,
               nominal_variables, distinct_times,
               subject_to_data_by_model, subject_info, error_msg);
  }

  // Reverse map.
  for (const pair<string, int>& family_id_and_index : family_id_to_index) {
    family_index_to_id->insert(
        make_pair(family_id_and_index.second, family_id_and_index.first));
  }

  return true;
}

// DEPRECATED.
/*
bool ReadTimeIndepIntervalCensoredData::UpdateBucketFamilyColumn(
    const VariableParams& family_params,
    vector<string>* subjects_families, string* error_msg) {
  if (subjects_families == nullptr) return false;

  // If no bucketing is required, return.
  if (family_params.collapse_params_.empty() ||
      family_params.collapse_params_[0].num_buckets_ <= 0) {
    return true;
  }

  const int num_buckets = family_params.collapse_params_[0].num_buckets_;

  // Make sure every family can be parsed as a numeric value, and find
  // the min and max family values.
  double min = DBL_MAX;
  double max = DBL_MIN;
  vector<double> family_values;
  for (const string& family : *subjects_families) {
    double family_value;
    if (!Stod(family, &family_value)) {
      return false;
    }
    if (family_value < min) min = family_value;
    if (family_value > max) max = family_value;
    family_values.push_back(family_value);
  }

  // Collapse all indicated params into buckets.
  for (int i = 0; i < family_values.size(); ++i) {
    double new_family_value;
    if (!ReadTableWithHeader::GetBucketValue(
           family_values[i], num_buckets, min, max,
           &new_family_value, error_msg)) {
      return false;
    }
    (*subjects_families)[i] = Itoa(new_family_value);
  }

  return true;
}
*/

bool ReadTimeIndepIntervalCensoredData::ParseModelLhs(
    const string& model_lhs_str,
    vector<tuple<string, string, string>>* dep_var_name_and_endpoints) {
  if (dep_var_name_and_endpoints == nullptr) return false;
  dep_var_name_and_endpoints->clear();

  // We allow the user to pass in the entire model string (LHS and RHS);
  // check for '=', and in present, just take the LHS.
  vector<string> model_parts;
  Split(model_lhs_str, "=", &model_parts);  

  // Check for and remove enclosing parentheses.
  string cleaned_lhs = StripQuotes(RemoveAllWhitespace(model_parts[0]));
  if (!HasPrefixString(cleaned_lhs, "(") ||
      !HasPrefixString(cleaned_lhs, ")")) {
    cout << "ERROR: Expected model LHS to be enclosed in parentheses." << endl;
    return false;
  }
  cleaned_lhs = StripParentheses(cleaned_lhs);

  // Split around the K dependent covariates. 
  vector<string> dep_cov_parts;
  Split(cleaned_lhs, ";", &dep_cov_parts);

  // Parse each dep covariate on model LHS.
  for (const string& dep_cov : dep_cov_parts) {
    // Extract dep var name, if present.
    vector<string> name_values_parts;
    Split(dep_cov, ":", &name_values_parts);

    if (name_values_parts.size() > 2) {
      cout << "ERROR: Unexpected term on model LHS: '" << dep_cov << endl;
      return false;
    }

    const int values_index = name_values_parts.size() == 2 ? 1 : 0;
    const string dep_var_name =
        name_values_parts.size() == 2 ? name_values_parts[0] : "";

    // The rest of the string should contain the left and right endpoint column
    // names, separated by a comma.
    vector<string> endpoint_parts;
    Split(name_values_parts[values_index], ",", &endpoint_parts);
    if (endpoint_parts.size() != 2) {
      cout << "ERROR: Unexpected term on model LHS: '"
           << name_values_parts[values_index] << endl;
      return false;
    }

    dep_var_name_and_endpoints->push_back(make_tuple(
        dep_var_name, endpoint_parts[0], endpoint_parts[1]));
  }

  return true;
}

bool ReadTimeIndepIntervalCensoredData::GetStatisticsForTimeIndependentLinearTerms(
    const bool do_sample_variance, const int model_index,
    const map<string, set<string>>& nominal_variables,
    const vector<VariableParams>& indep_vars_params_k,
    const vector<Expression>& linear_terms_k,
    const vector<string>& data_header,
    const vector<vector<vector<DataHolder>>>& data_values,
    vector<tuple<bool, double, double>>* linear_terms_mean_and_std_dev,
    string* error_msg) {
  if (linear_terms_mean_and_std_dev == nullptr) return false;

  // Get the column indices of the independent variables.
  map<string, int> indep_var_name_to_col;
  for (int i = 0; i < indep_vars_params_k.size(); ++i) {
    const string& indep_var_name = indep_vars_params_k[i].col_.name_;
    int col_index = -1;
    for (int j = 0; j < data_header.size(); ++j) {
      if (data_header[j] == indep_var_name) {
        col_index = j;
        break;
      }
    }
    if (col_index == -1) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to find covariate column '" +
                      indep_var_name + "' in the input data file.\n";
      }
      return false;
    }
    indep_var_name_to_col.insert(make_pair(indep_var_name, col_index));
  }

  // Compute values for each linear term (for each row), and keep
  // running total (sum) and it's square across all rows. Also keep track
  // if the linear term had any values that weren't 0 or 1 (we treat
  // binary linear terms differently).
  const int p_k = linear_terms_k.size();
  vector<double> sums(p_k, 0.0);
  vector<double> sums_squared(p_k, 0.0);
  set<int> non_binary_linear_terms;
  int denominator = 0;
  for (int i = 0; i < data_values.size(); ++i) {
    const vector<DataHolder>& values_ik = data_values[i][model_index];
    // Check if there are any valid values for this (Subject, Event-Type) pair.
    if (values_ik.empty()) continue;

    ++denominator;
    // Expand non-numeric variables to numeric values.
    map<string, double> var_to_numeric_value;
    if (!GetVariableValuesFromDataRow(
            nominal_variables, indep_var_name_to_col, values_ik,
            &var_to_numeric_value, error_msg)) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: failed to GetVariableValuesFromDataRow for Subject " +
                      Itoa(i) + " and model " + Itoa(model_index) + ".\n";
      }
      return false;
    }

    for (int linear_term_index = 0; linear_term_index < p_k; ++linear_term_index) {
      const Expression& linear_term = linear_terms_k[linear_term_index];
      double term_value;
      if (!EvaluateExpression(
              linear_term, var_to_numeric_value,
              &term_value, error_msg)) {
        return false;
      }
      sums[linear_term_index] += term_value;
      sums_squared[linear_term_index] += (term_value * term_value);
      if (term_value != 0.0 &&  term_value != 1.0) {
        non_binary_linear_terms.insert(linear_term_index);
      }
    }
  }

  // Finally, compute mean and standard deviation for each linear term.
  const int n = denominator;
  linear_terms_mean_and_std_dev->clear();
  for (int linear_term_index = 0; linear_term_index < p_k; ++linear_term_index) {
    const bool is_binary =
        non_binary_linear_terms.find(linear_term_index) ==
        non_binary_linear_terms.end();
    const double& sum = sums[linear_term_index];
    const double mean = sums[linear_term_index] / n;
    const double denominator = do_sample_variance ? n - 1 : n;
    const double std_dev =
        sqrt((sums_squared[linear_term_index] - (sum * sum) / n) / denominator);
    linear_terms_mean_and_std_dev->push_back(make_tuple(!is_binary, mean, std_dev));
  }

  return true;
}

bool ReadTimeIndepIntervalCensoredData::PopulateSubjectInfo(
    const VariableNormalization& standardize_linear_terms,
    const vector<tuple<bool, double, double>>& linear_terms_mean_and_std_dev,
    const vector<VariableParams>& indep_vars_params,
    const map<string, set<string>>& nominal_variables,
    const vector<Expression>& linear_terms,
    const vector<vector<DataHolder>>& subject_to_data,
    vector<SubjectInfo>* subject_info,
    string* error_msg) {

  vector<vector<vector<DataHolder>>> subject_to_data_by_model;
  for (const vector<DataHolder>& subject_data : subject_to_data) {
    subject_to_data_by_model.push_back(vector<vector<DataHolder>>());
    vector<vector<DataHolder>>& subject_data_for_model =
        subject_to_data_by_model.back();
    subject_data_for_model.push_back(subject_data);
  }

  return PopulateSubjectInfo(
      0, standardize_linear_terms, linear_terms_mean_and_std_dev,
      indep_vars_params, nominal_variables, linear_terms,
      subject_to_data_by_model, subject_info, error_msg);
}
 
bool ReadTimeIndepIntervalCensoredData::PopulateSubjectInfo(
    const int model_index,
    const VariableNormalization& standardize_linear_terms,
    const vector<tuple<bool, double, double>>& linear_terms_mean_and_std_dev_k,
    const vector<VariableParams>& indep_vars_params_k,
    const map<string, set<string>>& nominal_variables,
    const vector<Expression>& linear_terms_k,
    const vector<vector<vector<DataHolder>>>& subject_to_data_by_model,
    vector<SubjectInfo>* subject_info,
    string* error_msg) {
  if (subject_info == nullptr ||
      subject_info->size() != subject_to_data_by_model.size() ||
      error_msg == nullptr) {
    return false;
  }

  const int n = subject_info->size();
  const int p_k = linear_terms_k.size();
  const int p_prime_k = indep_vars_params_k.size();

  // Loop through all Subjects, setting linear_term_values_[k] and
  // is_time_indep_[k].
  for (int i = 0; i < n; ++i) {
    const vector<DataHolder>& subject_to_data_ik =
        subject_to_data_by_model[i][model_index];
    SubjectInfo& subject_info_ik = (*subject_info)[i];

    // Add entry to linear_term_values_ for this (Subject, event-type).
    subject_info_ik.linear_term_values_.push_back(make_pair(VectorXd(), MatrixXd()));
    VectorXd& linear_term_values_ik =
        subject_info_ik.linear_term_values_.back().first;

    // Add entry to is_time_indep_ for this (Subject, event-type).
    subject_info_ik.is_time_indep_.push_back(vector<bool>());
    vector<bool>& is_time_indep_k = subject_info_ik.is_time_indep_.back();
    is_time_indep_k.resize(p_k, true);

    if (subject_to_data_ik.empty()) {
      // There are no rows for this (Subject, model) pair that had valid
      // values for all of the relevant covariates. Nothing to do (i.e.,
      // linear_term_values_ will be empty for this model).
      continue;
    }
    linear_term_values_ik.resize(p_k);

    // Get the values to use for each variable, at each distinct time.
    // Notes:
    //   - Here, by "Variable", we mean the names of the variables, as opposed
    //     to the final linear terms. These may be the same thing if the RHS
    //     is a simple linear combination of the covariates, but they may
    //     not exactly correspond, if some of the variables are NOMINAL, and/or
    //     some of the linear terms involve mathematical combinations of the
    //     variables. Notation: There are p'_k variables, and p_k linear-terms
    map<string, DataHolder> var_to_value;
    for (int p_prime = 0; p_prime < p_prime_k; ++p_prime) {
      const VariableParams& var_params = indep_vars_params_k[p_prime];
      var_to_value.insert(make_pair(var_params.col_.name_, subject_to_data_ik[p_prime]));
    }

    // Expand non-numeric variables to numeric values.
    map<string, double> var_to_numeric_value;
    if (!GetVariableValuesFromDataRow(
            Keys(var_to_value), nominal_variables, var_to_value,
            &var_to_numeric_value, error_msg)) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Failed to GetVariableValuesFromDataRow for Subject " +
                      Itoa(i) + ".\n";
      }
      return false;
    }

    // Evaluate each linear term, using the variable values.
    for (int linear_term_index = 0; linear_term_index < p_k; ++linear_term_index) {
      if (!ReadTimeDepIntervalCensoredData::EvaluateLinearTerm(
              linear_terms_k[linear_term_index], standardize_linear_terms,
              (linear_terms_mean_and_std_dev_k.empty() ?
               make_tuple<bool, double, double>(false, 0.0, 1.0) :
               linear_terms_mean_and_std_dev_k[linear_term_index]),
              var_to_numeric_value, set<string>(),  // Not used.
              &linear_term_values_ik(linear_term_index), nullptr, error_msg)) {
        return false;
      }
    }
  }

  return true;
}

bool ReadTimeIndepIntervalCensoredData::ReadFile(
    const TimeIndepIntervalCensoredInParams& in_params,
    const vector<VariableParams>& all_indep_var_params,
    const vector<set<int>>& cols_in_models,
    vector<vector<vector<DataHolder>>>* subject_to_data_by_model,
    TimeDepIntervalCensoredData* out_data) {
  if (out_data == nullptr) return false;

  // Bucket Left, Right Time columns (if appropriate).
  vector<double> bucketed_left_times;
  if (!ReadTimeDepIntervalCensoredData::BucketTimes(
          in_params.file_info_, in_params.left_time_params_,
          in_params.file_info_.na_strings_,
          &bucketed_left_times, &out_data->error_msg_)) {
    return false;
  }
  vector<double> bucketed_right_times;
  if (!ReadTimeDepIntervalCensoredData::BucketTimes(
          in_params.file_info_, in_params.right_time_params_,
          in_params.file_info_.na_strings_,
          &bucketed_right_times, &out_data->error_msg_)) {
    return false;
  }

  // Read all lines (rows) of input data file.
  if (!ReadInputData(
          in_params.indep_vars_params_.size(),
          in_params.file_info_.name_, in_params.file_info_.delimiter_,
          in_params.file_info_.infinity_char_,
          in_params.left_time_params_,
          in_params.right_time_params_,
          in_params.family_params_,
          in_params.event_type_params_, in_params.event_cause_params_,
          all_indep_var_params,
          cols_in_models,
          in_params.right_censored_events_,
          in_params.file_info_.na_strings_,
          bucketed_left_times, bucketed_right_times,
          in_params.event_type_to_index_,
          in_params.event_cause_to_index_,
          &(out_data->family_index_to_id_),
          &(out_data->na_rows_),
          &(out_data->nominal_variables_),
          &(out_data->distinct_times_),
          subject_to_data_by_model,
          &(out_data->subject_info_),
          &(out_data->error_msg_))) {
    return false;
  }

  return true;
}

bool ReadTimeIndepIntervalCensoredData::FillColumnIndices(
    const string& title_line,
    TimeIndepIntervalCensoredInParams* in_params, string* error_msg) {
  if (in_params == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in FillColumnIndices: Null Input.\n";
    }
    return false;
  }

  const bool need_family_col =
      in_params->family_params_.col_.index_ < 0 &&
      !in_params->family_params_.col_.name_.empty();
  const bool need_left_time_col = in_params->left_time_params_.col_.index_ == -1;
  const bool need_right_time_col = in_params->right_time_params_.col_.index_ == -1;
  const bool need_event_type_col =
      in_params->event_type_params_.col_.index_ < 0 &&
      !in_params->event_type_params_.col_.name_.empty();
  const bool need_event_cause_col =
      in_params->event_cause_params_.col_.index_ < 0 &&
      !in_params->event_cause_params_.col_.name_.empty();
  bool need_indep_var_cols = false;
  for (const vector<VariableParams>& indep_vars_params_k :
       in_params->indep_vars_params_) {
    for (const VariableParams& indep_vars_k : indep_vars_params_k) {
      if (indep_vars_k.col_.index_ == -1) {
        need_indep_var_cols = true;
        break;
      }
    }
    // No need to check next model's indep_vars_params if we already
    // know we need some.
    if (need_indep_var_cols) break;
  }

  const bool need_at_least_one_col_index =
      need_family_col || need_left_time_col || need_right_time_col ||
      need_event_type_col || need_event_cause_col ||
      need_indep_var_cols;

  if (!need_at_least_one_col_index) return true;

  // Get a list of columns (names) for which we need to determine the index.
  vector<string> columns_needed;
  if (need_family_col) {
    columns_needed.push_back(in_params->family_params_.col_.name_);
  }
  if (need_left_time_col) {
    columns_needed.push_back(in_params->left_time_params_.col_.name_);
  }
  if (need_right_time_col) {
    columns_needed.push_back(in_params->right_time_params_.col_.name_);
  }
  if (need_event_type_col) {
    columns_needed.push_back(in_params->event_type_params_.col_.name_);
  }
  if (need_event_cause_col) {
    columns_needed.push_back(in_params->event_cause_params_.col_.name_);
  }
  if (need_indep_var_cols) {
    set<string> indep_var_col_names;
    for (const vector<VariableParams>& indep_vars_k : in_params->indep_vars_params_) {
      for (const VariableParams& indep_vars : indep_vars_k) {
        if (indep_vars.col_.index_ == -1) {
          indep_var_col_names.insert(indep_vars.col_.name_);
        }
      }
    }
    for (const string& indep_var_col_name : indep_var_col_names) {
      columns_needed.push_back(indep_var_col_name);
    }
  }

  // Get the column indices.
  vector<int> columns_needed_indices;
  if (!ReadTimeDepIntervalCensoredData::GetColumnIndices(
          title_line, in_params->file_info_.delimiter_,
          columns_needed, &columns_needed_indices, error_msg)) {
    return false;
  }

  // Assign column indices to the corresponding VariableParams.
  int current_index = 0;
  if (need_family_col) {
    in_params->family_params_.col_.index_ = columns_needed_indices[current_index];
    current_index++;
  }
  if (need_left_time_col) {
    in_params->left_time_params_.col_.index_ =
            columns_needed_indices[current_index];
    current_index++;
  }
  if (need_right_time_col) {
    in_params->right_time_params_.col_.index_ =
            columns_needed_indices[current_index];
    current_index++;
  }
  if (need_event_type_col) {
    in_params->event_type_params_.col_.index_ =
        columns_needed_indices[current_index];
    current_index++;
  }
  if (need_event_cause_col) {
    in_params->event_cause_params_.col_.index_ =
        columns_needed_indices[current_index];
    current_index++;
  }
  if (need_indep_var_cols) {
    map<string, int> var_name_to_col;
    for (vector<VariableParams>& indep_vars_k : in_params->indep_vars_params_) {
      for (VariableParams& indep_vars : indep_vars_k) {
        if (indep_vars.col_.index_ == -1) {
          if (var_name_to_col.insert(
                  make_pair(indep_vars.col_.name_, current_index)).second) {
            indep_vars.col_.index_ = columns_needed_indices[current_index];
            current_index++;
          } else {
            indep_vars.col_.index_ = var_name_to_col[indep_vars.col_.name_];
          }
        }
      }
    }
  }

  return true;
}

bool ReadTimeIndepIntervalCensoredData::ReadFile(
    TimeIndepIntervalCensoredInParams& in_params,
    TimeDepIntervalCensoredData* out_data) {
  // Sanity-check input.
  if (out_data == nullptr) return false;
  if (in_params.file_info_.name_.empty()) {
    out_data->error_msg_ += "ERROR: Empty Filename.\n";
    return false;
  }

  // Open input file.
  ifstream input_file(in_params.file_info_.name_.c_str());
  if (!input_file.is_open()) {
    out_data->error_msg_ +=
        "ERROR: Unable to open file '" + in_params.file_info_.name_ + "'.\n";
    return false;
  }

  // Read Title line.
  string title_line;
  if (!getline(input_file, title_line)) {
    out_data->error_msg_ += "ERROR: Empty Input Data file '" +
                            in_params.file_info_.name_ + "'\n";
    return false;
  }
  RemoveWindowsTrailingCharacters(&title_line);

  // Get indices for the important columns, if not already populated.
  if (!FillColumnIndices(title_line, &in_params, &(out_data->error_msg_))) {
    out_data->error_msg_ += "ERROR: Failed to find all variables in the model "
                            "on the header line of data file '" +
                            in_params.file_info_.name_ + "'.\n";
    return false;
  }

  // Assign/Force some special columns to be Nominal:
  //   ID, Family, Event-Type, Event-Cause.
  AssignKnownNominalColumns(in_params, &(out_data->nominal_variables_));

  // Read Variable Names on header line, looking for Nominal variables, as
  // identified by those with a '$'-suffix.
  if (!ReadTimeDepIntervalCensoredData::GetNominalVariablesFromHeader(
          title_line, in_params.file_info_.delimiter_,
          &(out_data->nominal_variables_), &(out_data->error_msg_))) {
    return false;
  }
  input_file.close();

  // Aggregate all the independent variables into one container, get the
  // set of all independent variable names (across all models), and get
  // a header for the data structure that will hold the independent
  // variable values.
  vector<VariableParams> all_indep_var_params;
  set<string> indep_vars;
  vector<string> data_header;
  for (const vector<VariableParams>& indep_var_params_k :
       in_params.indep_vars_params_) {
    for (const VariableParams& indep_vars_k : indep_var_params_k) {
      if (indep_vars.insert(indep_vars_k.col_.name_).second) {
        // First time seeing this variable. Add it.
        all_indep_var_params.push_back(indep_vars_k);
        data_header.push_back(indep_vars_k.col_.name_);
      }
    }
  }

  // For each model, get a list of variables that are part of the model.
  vector<set<int>> cols_in_models;
  vector<vector<string>> data_header_by_model;
  for (int k = 0; k < in_params.indep_vars_params_.size(); ++k) {
    const Expression& model_rhs_k = in_params.model_rhs_[k];
    set<string> vars_in_model_k;
    if (!ExtractVariablesFromExpression(
            model_rhs_k, &vars_in_model_k, &out_data->error_msg_)) {
      return false;
    }
    cols_in_models.push_back(set<int>());
    set<int>& cols_in_model_k = cols_in_models.back();
    for (const string& var_in_model_k : vars_in_model_k) {
      for (int i = 0; i < data_header.size(); ++i) {
        if (var_in_model_k == data_header[i]) {
          cols_in_model_k.insert(i);
          break;
        }
      }
    }
    if (cols_in_model_k.size() != vars_in_model_k.size()) {
      out_data->error_msg_ +=
          "ERROR: Unable to find all variables in model " +
          Itoa(k + 1) + " in the header of the input data file. "
          "Variables in the model: {'" + Join(vars_in_model_k, "', '") +
          "'}; Variables in the input data file: {'" +
          Join(data_header, "', '") + "'}.\n";
      return false;
    }
    // Construct a header for model_k in the same order as the data will
    // be added to the data structure holding the independent values for
    // model_k (i.e. same order as subject_to_data_by_model[][k], which is
    // populated below via ReadFile->ReadInputFile).
    data_header_by_model.push_back(vector<string>());
    vector<string>& data_header_k = data_header_by_model.back();
    for (const int col_index : cols_in_model_k) {
      data_header_k.push_back(data_header[col_index]);
    }
  }

  // Read the rest of the file (i.e. the data).
  // This function will populate subject_to_data_by_model, as well as
  // filling out some fields of out_data:
  //   - distinct_times_
  //   - na_rows_
  //   - subject_info_:
  //       - family_index_
  //       - times_
  //   - family_index_to_id_
  //   - nominal_variables_
  vector<vector<vector<DataHolder>>> subject_to_data_by_model;
  if (!ReadFile(
          in_params, all_indep_var_params, cols_in_models,
          &subject_to_data_by_model, out_data)) {
    return false;
  }

  // Now that we know all NOMINAL variables and all of their values, sanity check
  // that the set of nominal values in the Event-Type/Cause columns exactly
  // match those specified by the user.
  if (!ReadTimeDepIntervalCensoredData::CheckEventTypeAndCauseConsistency(
          out_data->nominal_variables_,
          in_params.event_type_params_,
          in_params.event_cause_params_,
          Keys(in_params.event_type_to_index_),
          Keys(in_params.event_cause_to_index_))) {
    return false;
  }

  // Determine the kind of standardization required.
  const bool do_population_variance =
      in_params.linear_terms_normalization_ == VAR_NORM_STD ||
      in_params.linear_terms_normalization_ == VAR_NORM_STD_NON_BINARY;
  const bool do_sample_variance =
      in_params.linear_terms_normalization_ == VAR_NORM_STD_W_N_MINUS_ONE ||
      in_params.linear_terms_normalization_ ==
          VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;
  const bool standardize_linear_terms =
      do_population_variance || do_sample_variance;

  // Loop over all models, filling in the model's legend and finding each
  // linear-term's mean and standard deviation.
  // NOTE: This has to be done at the model level, and not just over the
  // data columns, since we will normalize each *linear-term*, as opposed
  // to normalizing each column of the input data.
  vector<vector<Expression>> linear_terms;
  for (int k = 0; k < in_params.model_rhs_.size(); ++k) {
    const Expression& model_k_rhs = in_params.model_rhs_[k];

    // Get the legend for the final model (i.e. the string representation
    // of the linear terms, where the set of linear terms has been
    // expanded if necessary to include a indicator function(s) for terms
    // involving non-numeric variables).
    linear_terms.push_back(vector<Expression>());
    vector<Expression>& linear_terms_k = linear_terms.back();
    out_data->legend_.push_back(vector<string>());
    if (!GetLegendAndLinearTerms(
            false, indep_vars, vector<vector<string>>() /* No Subgroups */,
            out_data->nominal_variables_, model_k_rhs,
            &linear_terms_k, &(out_data->legend_.back()), &(out_data->error_msg_))) {
      return false;
    }

    // Get mean and standard deviation for each linear term.
    out_data->linear_terms_mean_and_std_dev_.push_back(
        vector<tuple<bool, double, double>>());
    if (standardize_linear_terms &&
        !GetStatisticsForTimeIndependentLinearTerms(
            do_sample_variance, k,
            out_data->nominal_variables_, in_params.indep_vars_params_[k],
            linear_terms_k, data_header_by_model[k], subject_to_data_by_model,
            &(out_data->linear_terms_mean_and_std_dev_.back()),
            &(out_data->error_msg_))) {
      return false;
    }
  }

  // Finally, loop over all models, filling in remaining items of out_data:
  //   - subject_info_: The parts not yet filled:
  //       - linear_term_values_
  //       - is_time_indep_
  for (int k = 0; k < in_params.model_rhs_.size(); ++k) {
    if (!PopulateSubjectInfo(
            k, in_params.linear_terms_normalization_,
            out_data->linear_terms_mean_and_std_dev_[k],
            in_params.indep_vars_params_[k],
            out_data->nominal_variables_,
            linear_terms[k],
            subject_to_data_by_model,
            &out_data->subject_info_, &out_data->error_msg_)) {
      out_data->error_msg_ += "ERROR filling in covariate values for linear terms "
                              "in model " + Itoa(k + 1) + ".\n";
      return false;
    }
  }

  // Time-independent data format has no Subject Id column; each line is treated
  // as a new Subject (or if there are multiple events, then each block of lines
  // with distinct event types is one Subject). Go ahead and do a dummy-filling
  // of out_data->subject_index_to_id_.
  for (int i = 0; i < out_data->subject_info_.size(); ++i) {
    out_data->subject_index_to_id_.insert(make_pair(i, Itoa(i + 1)));
  }

  return true;
}

bool ReadTimeIndepIntervalCensoredData::ReadFile(
    const set<string>& right_censored_events,
    const string& event_type_col,
    const string& event_cause_col,
    const vector<string>& event_types,
    const vector<string>& event_causes,
    const vector<ModelAndDataParams>& all_models,
    TimeDepIntervalCensoredData* out_data) {
  if (all_models.empty()) return false;

  TimeIndepIntervalCensoredInParams in_params;

  // Sanity-Check event-type/cause specifications are valid.
  if (!ReadTimeDepIntervalCensoredData::SanityCheckEventTypeAndCauseConsistency(
          all_models.size(),  
          event_type_col, event_cause_col, event_types, event_causes,
          right_censored_events, &in_params.right_censored_events_)) {
    return false;
  }

  // Copy fields from all_models to a TimeIndepIntervalCensoredInParams object.
  // First, copy fields that are common to all models.
  const ModelAndDataParams& params = all_models[0];
  in_params.file_info_ = params.file_;
  in_params.linear_terms_normalization_ = params.standardize_vars_;
  // Keep track of covariate columns versus other columns of interest
  // (e.g. left-time, right-time, event_type).
  set<string> non_indep_vars;
  // (L, R) column names.
  if (params.model_lhs_.time_vars_names_.size() != 2) {
    cout << "ERROR: Expected the names of the (Left-Endpoint, Right-Endpoint) "
         << "columns in the specification of the model LHS, found: {'"
         << Join(params.model_lhs_.time_vars_names_, "', '") << "'}" << endl;
    return false;
  }
  in_params.left_time_params_.col_.name_ = params.model_lhs_.time_vars_names_[0];
  non_indep_vars.insert(params.model_lhs_.time_vars_names_[0]);
  in_params.right_time_params_.col_.name_ = params.model_lhs_.time_vars_names_[1];
  non_indep_vars.insert(params.model_lhs_.time_vars_names_[1]);
  // Family (Cluster) column, if present.
  if (!params.family_str_.empty()) {
    in_params.family_params_.col_.name_ = params.family_str_;
    non_indep_vars.insert(params.family_str_);
  }
  // Event-type column, if present.
  if (!event_type_col.empty()) {
    in_params.event_type_params_.col_.name_ = event_type_col;
    non_indep_vars.insert(event_type_col);
    if (event_types.empty()) {
      if (right_censored_events.size() > 1) {
        cout << "ERROR: Multiple events specified via the "
             << "--right_censored_events argument, but a list of "
             << "all event types was not specified (via --event_types "
             << "argument)." << endl;
        return false;
      }
      if (!right_censored_events.empty()) {
        in_params.right_censored_events_.insert(0);
      }
    }
    for (int i = 0; i < event_types.size(); ++i) {
      in_params.event_type_to_index_.insert(make_pair(event_types[i], i));
      out_data->event_type_index_to_name_.insert(make_pair(i, event_types[i]));
      if (right_censored_events.find(event_types[i]) !=
          right_censored_events.end()) {
        in_params.right_censored_events_.insert(i);
      }
    }
    // Sanity-check all the right_censored_events were found among the list
    // of events in event_types.
    if (in_params.right_censored_events_.size() != right_censored_events.size()) {
      cout << "ERROR: Not all of the events specified in the "
           << "--right_censored_events argument are present in the "
           << "--event_types argument." << endl;
      return false;
    }
  } else {
    // Univariate (Single event) case. Go ahead and add a phony event name.
    out_data->event_type_index_to_name_.insert(make_pair(0, "Status"));
  }
  // Event-cause column, if present.
  if (!event_cause_col.empty()) {
    in_params.event_cause_params_.col_.name_ = event_cause_col;
    non_indep_vars.insert(event_cause_col);
    for (int i = 0; i < event_causes.size(); ++i) {
      in_params.event_cause_to_index_.insert(make_pair(event_causes[i], i));
      out_data->event_cause_index_to_name_.insert(make_pair(i, event_causes[i]));
    }
  }

  // Now copy fields specific to each model/event-type.
  out_data->na_rows_.resize(all_models.size());
  for (int k = 0; k < all_models.size(); ++k) {
    const ModelAndDataParams& params_k = all_models[k];

    // Sanity-Check the LHS of the model has an even number of terms
    // (representing the pairs of (L_i, R_i) column names).
    if (params_k.model_lhs_.model_lhs_time_indep_npmle_.empty() ||
        params_k.model_lhs_.model_lhs_time_indep_npmle_.size() % 2 != 0 ||
        (params_k.model_lhs_.model_lhs_time_indep_npmle_.size() !=
         params_k.model_lhs_.time_vars_names_.size()) ||
        (params_k.model_lhs_.dep_vars_names_.size() !=
         params_k.model_lhs_.model_lhs_time_indep_npmle_.size() / 2)) {
      cout << "ERROR: model LHS is improperly formatted." << endl;
      return false;
    }

    // Model for the k^th event type.
    in_params.model_rhs_.push_back(params_k.model_rhs_);

    // Pick out the variable parameters for the independent variables.
    in_params.indep_vars_params_.push_back(vector<VariableParams>());
    vector<VariableParams>& indep_vars_params_k =
        in_params.indep_vars_params_.back();
    for (const VariableParams& var_params : params_k.var_params_) {
      // Event-Type column.
      if (!event_type_col.empty() && var_params.col_.name_ == event_type_col) {
        in_params.event_type_params_ = var_params;
      // Event-Cause column.
      } else if (!event_cause_col.empty() &&
                 var_params.col_.name_ == event_cause_col) {
        in_params.event_cause_params_ = var_params;
      // Independent Variable column.
      } else if (
          non_indep_vars.find(var_params.col_.name_) == non_indep_vars.end() &&
          params_k.input_cols_used_.find(var_params.col_.index_) !=
          params_k.input_cols_used_.end()) {
        indep_vars_params_k.push_back(var_params);
      // Time Left-Endpoint column.
      } else if (var_params.col_.name_ == in_params.left_time_params_.col_.name_) {
        in_params.left_time_params_ = var_params;
      // Time Right-Endpoint column.
      } else if (var_params.col_.name_ == in_params.right_time_params_.col_.name_) {
        in_params.right_time_params_ = var_params;
      // Family column.
      } else if (!params_k.family_str_.empty() &&
                 var_params.col_.name_ == in_params.family_params_.col_.name_) {
        in_params.family_params_ = var_params;
      }
    }
  }

  return ReadFile(in_params, out_data);
}

}  // namespace file_reader_utils
