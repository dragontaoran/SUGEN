// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "read_time_dep_interval_censored_data.h"

#include "FileReaderUtils/read_file_utils.h"
#include "FileReaderUtils/read_table_with_header.h"
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

// In case a variable has ET_CONSTANT for left/right extrapolation,
// and the constant has numeric value, and the variable is NOMINAL,
// then we don't know how to handle this (i.e. what nominal value
// should be used?). Similarly, cannot linearly interpolate a NOMINAL
// variable. Return false for either of these cases.
bool CheckExtrapolationSettings(
    const map<string, set<string>>& nominal_variables,
    vector<vector<VariableParams>>& indep_vars_params,
    string* error_msg) {
  for (vector<VariableParams>& var_params_k : indep_vars_params) {
    for (VariableParams& var_params : var_params_k) {
      if (nominal_variables.find(var_params.col_.name_) !=
          nominal_variables.end()) {
        // This is a nominal variable. Check:
        //  1) Its interpolation type is not IT_LINEAR_INTERP
        const InterpolationType interp_type = var_params.time_params_.interp_type_;
        if (interp_type == InterpolationType::IT_LINEAR_INTERP) {
          if (error_msg != nullptr) {
            *error_msg +=
                "ERROR: Unable to apply interpolation type 'IT_LINEAR_INTERP' "
                "for NOMINAL variable '" + var_params.col_.name_ + "'.\n";
          }
          return false;
        }
        //  2) Its left/right ExtrapoloationType is not ET_CONSTANT with a numeric value.
        OutsideIntervalParams& outside_params = var_params.time_params_.outside_params_;
        if (outside_params.outside_left_type_ == ExtrapolationType::ET_CONSTANT &&
            outside_params.default_left_val_.type_ != DataType::DATA_TYPE_STRING) {
          if (error_msg != nullptr) {
            *error_msg +=
                "ERROR: Unable to apply 'ET_CONSTANT' with default numeric value " +
                Itoa(outside_params.default_left_val_.value_) +
                " as the outside left extrapolation type, for variable '" +
                var_params.col_.name_ + "'.\n";
          }
          return false;
        }
        if (outside_params.outside_right_type_ == ExtrapolationType::ET_CONSTANT &&
            outside_params.default_right_val_.type_ != DataType::DATA_TYPE_STRING) {
          // PHB_OLD
          // I can't do this, as the first example in the unireg documentation
          // doesn't override default values of extrapoloation parameters, and
          // then the GENDER column causes a problem, since it is nominal, but
          // the default right-extrapoloation (which isn't even used) is to a
          // numeric value (0).
          // Instead, just change the constant to be "0" (string); I could
          // actually change it to be anything, since it's not used anyway.
          *error_msg +=
              "ERROR: Unable to apply 'ET_CONSTANT' with default numeric value " +
              Itoa(outside_params.default_right_val_.value_) +
              " as the outside right extrapolation type, for variable '" +
              var_params.col_.name_ + "'.\n";
          return false;

          // PHB NEW/OLD: Updated code so that default extrapolation RIGHT is
          // to use RIGHTMOST value; so the above code works again. However,
          // if upon testing, there is a case where it doesn't work, use the
          // commented-out code below instead of the lines above.
          /*
          // Since this is a NOMINAL variable, change the default constant to be
          // used to "0" (string). NOTE: it doesn't actually matter what default
          // string value that gets used (so long as its type is STRING), since
          // the algorithm will not use right-extrapolation values.
          double name_value;
          if (outside_params.default_right_val_.name_.empty() ||
              !Stod(outside_params.default_right_val_.name_, &name_value) ||
              name_value != outside_params.default_right_val_.value_) {
            outside_params.default_right_val_.name_ =
                Itoa(outside_params.default_right_val_.value_);
          }
          outside_params.default_right_val_.type_ = DataType::DATA_TYPE_STRING;
          */
        }
      }
    }
  }

  return true;
}

// If value_two.type_ == DataType::DATA_TYPE_UNKNOWN, then sets
// value_two equal to value_one, and returns true. Otherwise, returns
// whether the two type_ fields match, and if so, whether the
// respective value_ (if type_ == DATA_TYPE_STRING) or name_
// (if type_ == DATA_TYPE_NUMERIC) fields match.
bool ValuesMatch(const DataHolder& value_one, DataHolder* value_two) {
  if (value_two == nullptr || value_one.type_ == DATA_TYPE_UNKNOWN) return false;

  // Value_two hasn't been set yet. Set it to equal value_one.
  if (value_two->type_ == DATA_TYPE_UNKNOWN) {
    *value_two = value_one;
    return true;
  }

  if (value_two->type_ != value_one.type_) return false;

  if (value_one.type_ == DATA_TYPE_STRING) {
    return value_one.name_ == value_two->name_;
  }

  if (value_one.type_ == DATA_TYPE_NUMERIC) {
    return value_one.value_ == value_two->value_;
  }

  return false;
}

// Assigns/forces certain special columns to be of NOMINAL type:
//   ID, Family, Event-Type, and Event-Cause.
void AssignKnownNominalColumns(
    const TimeDepIntervalCensoredInParams& in_params,
    map<string, set<string>>* nominal_variables) {
  if (nominal_variables == nullptr) return;
  if (in_params.id_params_.col_.index_ >= 0) {
    nominal_variables->insert(make_pair(
        in_params.id_params_.col_.name_, set<string>()));
  }
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

}  // (anonymous) namespace

bool ReadTimeDepIntervalCensoredData::CheckEventTypeAndCauseConsistency(
  const map<string, set<string>>& nominal_values,
  const VariableParams& event_type_params,
  const VariableParams& event_cause_params,
  const set<string>& event_types,
  const set<string>& event_causes) {
  // Check consistency of Event-Types.
  if (!event_types.empty() ||
      event_type_params.col_.index_ >= 0) {
    const string& event_type_col_name = event_type_params.col_.name_;
    const set<string>* found_values = FindOrNull(event_type_col_name, nominal_values);
    if (found_values == nullptr) {
      cout << "ERROR: Could not find Event Type column '" << event_type_col_name
           << "' among the set of columns identified as NOMINAL in the input "
           << "data file." << endl;
      return false;
    }
    if (event_types.empty()) {
      if (found_values->size() > 1) {
        cout << "ERROR: Found " << found_values->size() << " distinct "
             << "event types. Must specify the --event_types argument "
             << "if you use the --event_type argument and there are "
             << "more than one event types." << endl;
        return false;
      }
    } else if (*found_values != event_types) {
      cout << "ERROR: The provided --event_types (" << Join(event_types, ", ")
           << ") does not match the set of values found in the Event Type "
           << "column '" << event_type_col_name << "' ("
           << Join(*found_values, ", ") << ")." << endl;
      return false;
    }
  }

  // Check consistency of Event-Causes.
  if (!event_causes.empty() ||
      event_cause_params.col_.index_ >= 0) {
    const string& event_cause_col_name = event_cause_params.col_.name_;
    const set<string>* found_values = FindOrNull(event_cause_col_name, nominal_values);
    if (found_values == nullptr) {
      cout << "ERROR: Could not find Event Cause column '" << event_cause_col_name
           << "' among the set of columns identified as NOMINAL in the input "
           << "data file." << endl;
      return false;
    }
    if (event_causes.empty()) {
      if (found_values->size() > 1) {
        cout << "ERROR: Found " << found_values->size() << " distinct "
             << "event causes. Must specify the --event_causes argument "
             << "if you use the --event_cause argument and there are "
             << "more than one event causes." << endl;
        return false;
      }
    } else if (*found_values != event_causes) {
      cout << "ERROR: The provided --event_causes (" << Join(event_causes, ", ")
           << ") does not match the set of values found in the Event Cause "
           << "column '" << event_cause_col_name << "' ("
           << Join(*found_values, ", ") << ")." << endl;
      return false;
    }
  }
 
  return true;
}

bool ReadTimeDepIntervalCensoredData::SanityCheckEventTypeAndCauseConsistency(
    const int num_models,
    const string& event_type_col,
    const string& event_cause_col,
    const vector<string>& event_types,
    const vector<string>& event_causes,
    const set<string>& right_censored_events,
    set<int>* right_censored_event_indices) {
  if (event_type_col.empty() != event_types.empty()) {
    // It is okay to specify only --event_type (and not --event_types) if
    // all models are the same; otherwise user must specify neither or both.
    if (num_models > 1 || event_type_col.empty()) {
      cout << "ERROR: You must specify neither or both of --event_type and "
           << "--event_types" << endl;
      return false;
    }
  }
  if (event_cause_col.empty() != event_causes.empty()) {
    cout << "ERROR: You must specify neither or both of --event_cause and "
         << "--event_causes" << endl;
    return false;
  }
  if (event_types.size() > 1 && event_causes.size() > 1) {
    cout << "ERROR: The scenario in which there is both multiple "
         << "Event-Types as well as Multiple Event-Causes is not "
         << "currently supported." << endl;
    return false;
  }
  if (!right_censored_events.empty() && event_type_col.empty()) {
    if (*(right_censored_events.begin()) == "ALL") {
      // Check for special keyword "ALL", indicating all rows are for
      // a (single) Right-Censored event.
      right_censored_event_indices->insert(0);
    } else {
      cout << "ERROR: You must indicate the Event-Type column via "
           << "the --event_type command-line argument if you are "
           << "going to use the --right_censored_events argument." << endl;
      return false;
    }
  }
  // Check that the number of models is either '1' or that it matches
  // either the number of event types or the number of event causes;
  // and in the case it matches the number of event causes, check that
  // the number of event types is 0 or 1 (we currently do not allow
  // multiple event types when the model is determined by event cause).
  if (num_models != 1 &&
      num_models != event_types.size() &&
      num_models != event_causes.size()) {
    cout << "ERROR: the number of models (" << num_models
         << ") should either match the number of event types ("
         << event_types.size() << ") or the number of event causes ("
         << event_causes.size() << ")." << endl;
    return false;
  }
  if (num_models != 1 &&
      num_models != event_types.size() &&
      event_types.size() > 1) {
    cout << "ERROR: It appears that model depends on event cause, since "
         << "the number of models matches the number of event causes ("
         << num_models << "), which is different than the number "
         << "of event types (" << event_types.size() << "). In this case, "
         << "we do not allow multiple event types." << endl;
    return false;
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::GetColumnIndex(
    const string& title_line, const string& delimiter, const string& col_name,
    int* col_index, string* error_msg) {
  if (col_index == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in GetColumnIndex: Null input.\n";
    }
    return false;
  }

  vector<string> titles;
  Split(title_line, delimiter, &titles);

  for (int i = 0; i < titles.size(); ++i) {
    if (col_name == titles[i]) {
      *col_index = i;
      return true;
    }
  }

  if (error_msg != nullptr) {
    *error_msg += "ERROR: Unable to find column '" + col_name +
                  "' among the titles in Header:\n\t" + title_line + "\n";
  }
  return false;
}

bool ReadTimeDepIntervalCensoredData::GetColumnIndices(
    const string& title_line, const string& delimiter,
    const vector<string>& col_names,
    vector<int>* col_indices, string* error_msg) {
  vector<string> titles;
  Split(title_line, delimiter, &titles);

  for (const string& col_name_to_find : col_names) {
    bool found_col = false;
    for (int i = 0; i < titles.size(); ++i) {
      if (col_name_to_find == titles[i]) {
        found_col = true;
        col_indices->push_back(i);
        break;
      }
    }
    if (!found_col) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to find column '" + col_name_to_find +
                      "' among the titles in Header:\n\t" +
                      Join(titles, ", ") + "\n";
      }
      return false;
    }
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::GetNominalVariablesFromHeader(
    const string& title_line, const string& delimiter,
    map<string, set<string>>* nominal_variables, string* error_msg) {
  if (nominal_variables == nullptr) return true;  // Nothing to do.
  vector<string> titles;
  Split(title_line, delimiter, &titles);
  for (const string& current_title : titles) {
    if (current_title.length() > 0 &&
        current_title.substr(current_title.length() - 1) == "$") {
      nominal_variables->insert(make_pair(current_title, set<string>()));
    }
  }
  return true;
}

bool ReadTimeDepIntervalCensoredData::UnstandardizeResults(
    const VariableNormalization std_type,
    const VectorXd& standardized_beta, const VectorXd& standardized_lambda,
    const MatrixXd& standardized_variance,
    const vector<tuple<bool, double, double>>& coordinate_mean_and_std_dev,
    VectorXd* beta, VectorXd* lambda, MatrixXd* variance, string* error_msg) {
  // Return if nothing to do.
  if (std_type == VAR_NORM_NONE || coordinate_mean_and_std_dev.empty()) {
    (*beta) = standardized_beta;
    (*lambda) = standardized_lambda;
    (*variance) = standardized_variance;
    return true;
  }

  // Unstandardize variance.
  if (!UnstandardizeMatrix(
          std_type, standardized_variance, coordinate_mean_and_std_dev,
          variance, error_msg)) {
    return false;
  }

  // Unstandardize beta and lambda.
  return UnstandardizeBetaAndLambda(
      std_type, standardized_beta, standardized_lambda,
      coordinate_mean_and_std_dev, beta, lambda, error_msg);
}

bool ReadTimeDepIntervalCensoredData::UnstandardizeBetaAndLambda(
    const VariableNormalization std_type,
    const VectorXd& standardized_beta, const VectorXd& standardized_lambda,
    const vector<tuple<bool, double, double>>& coordinate_mean_and_std_dev,
    VectorXd* beta, VectorXd* lambda, string* error_msg) {
  // Return if nothing to do.
  if (std_type == VAR_NORM_NONE || coordinate_mean_and_std_dev.empty()) {
    (*beta) = standardized_beta;
    (*lambda) = standardized_lambda;
    return true;
  }

  // Constants to help whether unstandardization is necessary.
  const bool undo_std =
      std_type == VAR_NORM_STD || std_type == VAR_NORM_STD_W_N_MINUS_ONE;
  const bool undo_non_binary_std =
      std_type == VAR_NORM_STD_NON_BINARY ||
      std_type == VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;

  // Unstandardize beta.
  if (!UnstandardizeVector(
          std_type, standardized_beta, coordinate_mean_and_std_dev,
          beta, error_msg)) {
    return false;
  }

  // Unstandardize lambda.
  const int p = coordinate_mean_and_std_dev.size();
  double dot_product = 0.0;
  for (int i = 0; i < p; ++i) {
    const auto& stats = coordinate_mean_and_std_dev[i];
    const bool not_binary = get<0>(stats);
    const double mean = get<1>(stats);
    if (undo_std || (undo_non_binary_std && not_binary)) {
      dot_product += (*beta)(i) * mean;
    }
  }
  const double lambda_unstd_factor = 1.0 / exp(dot_product);
  *lambda = lambda_unstd_factor * standardized_lambda;

  return true;
}

bool ReadTimeDepIntervalCensoredData::UnstandardizeBetaAndLambda(
    const VariableNormalization std_type,
    const vector<DependentCovariateEstimates>& standardized_estimates,
    const vector<vector<tuple<bool, double, double>>>& coordinate_mean_and_std_dev,
    vector<DependentCovariateEstimates>* estimates, string* error_msg) {
  // Return if nothing to do.
  if (std_type == VAR_NORM_NONE || coordinate_mean_and_std_dev.empty()) {
    (*estimates) = standardized_estimates;
    return true;
  }

  const int num_models = standardized_estimates.size();
  if (num_models == 0 || coordinate_mean_and_std_dev.size() != num_models) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Mismatching number of models in provided "
                    "estimates (" + Itoa(num_models) +
                    ") vs. provided mean and standard deviation info (" +
                    Itoa(static_cast<int>(coordinate_mean_and_std_dev.size())) + ").\n";
    }
    return false;
  }

  // Unstandardize beta and lambda for each of the K dependent covariates.
  estimates->clear();
  for (int k = 0; k < standardized_estimates.size(); ++k) {
    const VectorXd& standardized_beta = standardized_estimates[k].beta_;
    const VectorXd& standardized_lambda = standardized_estimates[k].lambda_;
    estimates->push_back(DependentCovariateEstimates());
    DependentCovariateEstimates& current_estimates = estimates->back();
    if (!UnstandardizeBetaAndLambda(
            std_type, standardized_beta, standardized_lambda,
            coordinate_mean_and_std_dev[k],
            &current_estimates.beta_, &current_estimates.lambda_, error_msg)) {
      return false;
    }
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::ReadValue(
    const bool enforce_numeric,
    const string& col_name, const int col_index,
    const vector<string>& str_values,
    const string& infinity_char,
    const set<string>& na_strings,
    const vector<VariableCollapseParams>& params,
    map<string, set<string> >* nominal_variables,
    bool* is_na, bool* is_new_nominal_col,
    DataHolder* parsed_value, string* error_msg) {
  if (is_na == nullptr || parsed_value == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Reading data value: Null Input.\n";
    }
    return false;
  }

  // Sanity-Check given column index is within the range of str_values.
  if (col_index >= str_values.size()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Reading data value: Cannot find id column (index " +
                    Itoa(col_index) + "), data row has only " +
                    Itoa(static_cast<int>(str_values.size())) + " columns:\n" +
                    Join(str_values, ", ") + "\n";
    }
    return false;
  }

  const string& orig_token = str_values[col_index];

  // Check if str_value is NA.
  if (na_strings.find(orig_token) != na_strings.end() ||
      (na_strings.empty() &&
       NA_STRINGS.find(orig_token) != NA_STRINGS.end())) {
    *is_na = true;
    return true;
  }

  // Collapse value, if params dictates.
  DataHolder orig_value;
  orig_value.name_ = orig_token;
  // Try to parse value as a double.
  if (Stod(orig_token, &orig_value.value_)) {
      orig_value.type_ = DataType::DATA_TYPE_NUMERIC;
  } else if (!infinity_char.empty() && orig_token == infinity_char) {
    orig_value.value_ = numeric_limits<double>::infinity();
    orig_value.type_ = DataType::DATA_TYPE_NUMERIC;
  } else if (!infinity_char.empty() && orig_token == ("-" + infinity_char)) {
    orig_value.value_ = -1.0 * numeric_limits<double>::infinity();
    orig_value.type_ = DataType::DATA_TYPE_NUMERIC;
  } else {
    // Temporary set DataType to String (this will be adjusted below, as appropriate).
    orig_value.type_ = DataType::DATA_TYPE_STRING;
  }
  bool is_collapsed = false;
  if (!ReadTableWithHeader::CollapseValue(
          orig_value, params, parsed_value, &is_collapsed)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR Failed to Collapse Value.";
    }
    return false;
  }

  // Re-Check if new (collapsed) value is NA.
  const string& collapsed_str = is_collapsed ? parsed_value->name_ : orig_token;
  if (na_strings.find(collapsed_str) != na_strings.end() ||
      (na_strings.empty() &&
       NA_STRINGS.find(collapsed_str) != NA_STRINGS.end())) {
    *is_na = true;
    return true;
  }

  // Check if this column is known to be Nominal.
  if (nominal_variables != nullptr) {
    map<string, set<string>>::iterator nominal_itr =
        nominal_variables->find(col_name);
    if (nominal_itr != nominal_variables->end()) {
      parsed_value->value_ = 0.0;
      parsed_value->name_ = collapsed_str;
      parsed_value->type_ = DataType::DATA_TYPE_STRING;
      nominal_itr->second.insert(collapsed_str);
      return true;
    }
  }

  // Parse as numeric value, if possible.
  double value = -1.0;
  if (!infinity_char.empty() && collapsed_str == infinity_char) {
    parsed_value->value_ = numeric_limits<double>::infinity();
    parsed_value->type_ = DataType::DATA_TYPE_NUMERIC;
  } else if (!infinity_char.empty() && collapsed_str == ("-" + infinity_char)) {
    parsed_value->value_ = -1.0 * numeric_limits<double>::infinity();
    parsed_value->type_ = DataType::DATA_TYPE_NUMERIC;
  } else if (Stod(collapsed_str, &value)) {
    parsed_value->value_ = value;
    parsed_value->type_ = DataType::DATA_TYPE_NUMERIC;
  // Enforce numeric value, if appropriate.
  } else if (enforce_numeric) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Reading data value: Unable to parse '" +
                    collapsed_str + "' as a numeric value.\n";
    }
    return false;
  } else {
    // Variable is Nominal, and we didn't already know this was a nominal
    // column. Mark the column as nominal.
    parsed_value->value_ = 0.0;
    parsed_value->name_ = collapsed_str;
    parsed_value->type_ = DataType::DATA_TYPE_STRING;
    if (nominal_variables != nullptr) {
      if (enforce_numeric) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in Reading data value: unable to parse '" +
                        collapsed_str + "' as a numeric value.\n";
        }
        return false;
      }
      nominal_variables->insert(make_pair(col_name, set<string>())).
          first->second.insert(collapsed_str);
    }
    *is_new_nominal_col = true;
    return true;
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::ReadLine(
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
    map<string, int>* subject_to_subject_index,
    map<string, int>* family_to_index,
    int* subject_index, bool* status, double* time,
    int* family_index, int* event_type, int* event_cause,
    set<pair<int, int>>* right_censored_special_rows_seen,
    vector<DataHolder>* indep_var_values,
    map<string, set<string> >* nominal_variables,
    bool* right_censored_special_row, bool* right_censored_row,
    bool* is_na_row, bool* is_na_status, bool* is_na_event_cause,
    bool* is_new_nominal_col, set<int>* na_indep_vars, string* error_msg) {
  if (line.empty()) {
    return true;
  }

  // Split line around delimiter.
  vector<string> col_strs;
  Split(line, delimiter, false /* do not collapse empty strings */, &col_strs);

  // Fetch the ID column, if present.
  if (id_params.col_.index_ >= 0) {
    DataHolder id_value;
    if (!ReadValue(false, id_params.col_.name_, id_params.col_.index_, col_strs,
                   infinity_char, na_strings, id_params.collapse_params_,
                   nominal_variables, is_na_row, is_new_nominal_col,
                   &id_value, error_msg)) {
      return false;
    }
    if (*is_na_row) return true;
    const string& subject_id = id_value.name_;
    // Check if this Subject has already been seen.
    int* temp_subject_index = FindOrNull(subject_id, *subject_to_subject_index);
    if (temp_subject_index == nullptr) {
      // Haven't seen this subject id yet. Add an entry.
      const int row_index = subject_to_subject_index->size();
      subject_to_subject_index->insert(make_pair(subject_id, row_index));
      *subject_index = row_index;
    } else {
      *subject_index = *temp_subject_index;
    }
  } else {
    // No Subject Id column present. Treat each row as a unique subject,
    // and use the row index as the Subject ID.
    const int row_index = subject_to_subject_index->size();
    subject_to_subject_index->insert(make_pair(Itoa(row_index), row_index));
    *subject_index = row_index;
  }

  // Fetch the Family column, if present.
  if (family_params.col_.index_ >= 0) {
    DataHolder family_value;
    if (!ReadValue(false, family_params.col_.name_, family_params.col_.index_, col_strs,
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
    if (!ReadValue(
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
    if (*right_censored_row) {
      pair<set<pair<int, int>>::iterator, bool> insert_itr =
          right_censored_special_rows_seen->insert(
              make_pair(*subject_index, *event_type));
      *right_censored_special_row = insert_itr.second;
    }
  } else if (!right_censored_events.empty()) {
    // In case everything is a Right-Censored event (so no Event-Type column
    // present or required).
    *right_censored_row =
        right_censored_events.find(0) !=
        right_censored_events.end();
    if (*right_censored_row) {
      pair<set<pair<int, int>>::iterator, bool> insert_itr =
          right_censored_special_rows_seen->insert(
              make_pair(*subject_index, 0));
      *right_censored_special_row = insert_itr.second;
    }
  }

  // Fetch the Event-Cause column, if present and Status was '1'.
  if (event_cause_params.col_.index_ >= 0) {
    DataHolder event_cause_value;
    if (!ReadValue(
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

  // Fetch the Time column.
  DataHolder time_value;
  if (!ReadValue(true, time_params.col_.name_, time_params.col_.index_, col_strs,
                 infinity_char, na_strings, time_params.collapse_params_,
                 nominal_variables, is_na_row, is_new_nominal_col,
                 &time_value, error_msg)) {
    return false;
  }
  if (*is_na_row) return true;
  if (time_value.value_ < 0.0) {
    // Negative time-values are invalid. Treat these as missing value (NA).
    *is_na_row = true;
    return true;
  }
  *time = time_value.value_;

  // Fetch the Status column value (if this is a row corresponding
  // to a Right-censored event, only read the status if it is the
  // "special row", i.e. if it is this Subject's first row for this
  // event type).
  if (!*right_censored_row || *right_censored_special_row) {
    DataHolder status_value;
    if (!ReadValue(true, status_params.col_.name_,
                   status_params.col_.index_, col_strs,
                   infinity_char, na_strings, status_params.collapse_params_,
                   nominal_variables, is_na_status, is_new_nominal_col,
                   &status_value, error_msg)) {
      return false;
    }
    if (!(*is_na_status)) {
      if (status_value.value_ != 1.0 && status_value.value_ != 0.0) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unable to parse Status column (index " +
                        Itoa(status_params.col_.index_) + ") on data row " +
                        Itoa(line_num) + " as a '0' or '1':\t" +
                        Itoa(status_value.value_) + "\n";
        }
        return false;
      }
      *status = status_value.value_ == 1.0;
    }
  }

  // If this is a Right-Censored "special" row, there are no covariate
  // values to fetch. Return.
  if (*right_censored_special_row) return true;


  // Iterate through the independent variable columns, copying values to
  // 'indep_var_values'.
  indep_var_values->clear();
  for (int i = 0; i < indep_vars_params.size(); ++i) {
    DataHolder value;
    bool is_na_value = false;
    if (!ReadValue(false, indep_vars_params[i].col_.name_,
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

bool ReadTimeDepIntervalCensoredData::ReadInputData(
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

  // Read values from file.
  subject_to_time_and_status_by_event->clear();
  subject_to_data_by_model->clear();
  subject_info->clear();
  string line;
  int line_num = 1;
  int bucketed_time_index = -1;
  bool need_another_pass = false;
  vector<bool> is_first_data_row(num_models, true);
  set<pair<int, int>> right_censored_special_rows_seen;
  map<string, int> subject_id_to_index, family_id_to_index;
  while (getline(file, line)) {
    ++bucketed_time_index;
    file_reader_utils::RemoveWindowsTrailingCharacters(&line);
    int subject_index = -1;
    int family_index = -1;
    int event_type = 0;
    int event_cause = 0;
    bool status = false;
    double time;
    bool is_na_row = false;
    bool is_na_status = false;
    bool is_na_event_cause = false;
    set<int> na_indep_vars;
    vector<DataHolder> indep_var_values;
    bool right_censored_special_row = false;
    bool right_censored_row = false;
    bool new_nominal_column = false;
    if (!ReadLine(line, delimiter, infinity_char, line_num,
                  id_params, family_params, event_type_params,
                  event_cause_params, time_params, status_params,
                  indep_vars_params, right_censored_events, na_strings,
                  event_type_to_index, event_cause_to_index,
                  &subject_id_to_index, &family_id_to_index,
                  &subject_index, &status, &time,
                  &family_index, &event_type, &event_cause,
                  &right_censored_special_rows_seen,
                  &indep_var_values, nominal_variables,
                  &right_censored_special_row, &right_censored_row,
                  &is_na_row, &is_na_status, &is_na_event_cause,
                  &new_nominal_column, &na_indep_vars, error_msg)) {
      return false;
    }
    if (is_na_row ||
        (is_na_event_cause && model_depends_on_cause) ||
        (is_na_event_cause && status)) {
      // Rows that have an NA in a key column are skipped; but record
      // which rows were skipped.
      for (set<int>& na_rows_k : *na_rows) {
        na_rows_k.insert(line_num);
      }
    } else if (is_na_status && right_censored_special_row) {
      // 'Status' is a key column if this is the Right-Censored special row.
      // Skip this row if it is the special row, and record that it was skipped.
      const int model_index =
          num_models == 1 ? 0 :
          model_depends_on_type ? event_type :
          event_cause;
      (*na_rows)[model_index].insert(line_num);
    } else {
      // Add an entry to subject_to_data_by_model and subject_info if this is the
      // first row for this subject.
      if (subject_index < 0) return false;
      if (subject_index >= subject_info->size()) {
        if (subject_index != subject_info->size()) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR: Unexpected subject index '" + Itoa(subject_index) +
                          "' is bigger than the number of Subjects already read in (" +
                          Itoa(static_cast<int>(subject_info->size())) + ").\n";
          }
          return false;
        }
        subject_to_time_and_status_by_event->push_back(
            vector<map<double, bool>>(
                event_type_to_index.empty() ? 1 : event_type_to_index.size()));
        subject_to_data_by_model->push_back(
            vector<map<double, vector<DataHolder>>>(num_models));
        subject_info->push_back(SubjectInfo());
        subject_info->back().times_.resize(
            event_type_to_index.empty() ? 1 : event_type_to_index.size());
      }

      // Update time to be the bucketed time, if appropriate.
      if (!bucketed_times.empty()) {
        time = bucketed_times[bucketed_time_index];
      }

      // Determine the appropriate model index (either the event_type,
      // event_cause, or '0' if there is just one model).
      const int model_index =
          num_models == 1 ? 0 :
          model_depends_on_type ? event_type :
          event_cause;

      // Store row. Row is one of three flavors:
      //   a) Right-Censored "special" row.
      //   b) Right-Censored data row (i.e. all non-special rows for Right-Censored data)
      //   c) Interval-Censored data row.
      if (right_censored_special_row) {
        // The row just read was a "special" right-censored row: it contained
        // the (Time, Status) and family of the Subject, but not any covariate values.
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
        data.censoring_time_ = data.is_alive_ ? time : time + 1.0;
        data.survival_time_ = data.is_alive_ ? time + 1.0 : time;
      } else if (right_censored_row) {
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
          // The row just read was for a right-censored event, but wasn't the
          // first "special" row; i.e. this current row's covariate values
          // should be read (a previous row recorded the (Time, Status) and
          // event cause).
          map<double, vector<DataHolder>>& current_subject_to_data =
              (*subject_to_data_by_model)[subject_index][model_index];
          vector<DataHolder>& values =
              current_subject_to_data.insert(make_pair(time, vector<DataHolder>())).
              first->second;
          for (const int col_index : cols_in_model_k) {
            if (col_index >= indep_var_values.size()) {
              if (error_msg != nullptr) {
                *error_msg += "ERROR: Invalid column index " + Itoa(col_index) +
                              " is larger than the number of independent covariates (" +
                              Itoa(static_cast<int>(indep_var_values.size())) + ").\n";
              }
              return false;
            }
            values.push_back(indep_var_values[col_index]);
          }
        }
        need_another_pass |=
            (!is_first_data_row[model_index] && new_nominal_column);
        is_first_data_row[model_index] = false;
      } else {
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

        // Add (time, status), provided a valid (non-missing) status was read.
        if (is_na_status) {
          (*na_rows)[model_index].insert(line_num);
        } else {
          map<double, bool>& current_time_and_status =
              (*subject_to_time_and_status_by_event)[subject_index][event_type];
          current_time_and_status.insert(make_pair(time, status));
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
          map<double, vector<DataHolder>>& current_subject_to_data =
              (*subject_to_data_by_model)[subject_index][model_index];
          vector<DataHolder>& values =
              current_subject_to_data.insert(make_pair(time, vector<DataHolder>())).
              first->second;
          for (const int col_index : cols_in_model_k) {
            if (col_index >= indep_var_values.size()) {
              if (error_msg != nullptr) {
                *error_msg += "ERROR: Invalid column index " + Itoa(col_index) +
                              " is larger than the number of independent covariates (" +
                              Itoa(static_cast<int>(indep_var_values.size())) + ").\n";
              }
              return false;
            }
            values.push_back(indep_var_values[col_index]);
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
               num_models, filename, delimiter, infinity_char, id_params,
               family_params, event_type_params, event_cause_params,
               time_params, status_params, indep_vars_params, cols_in_models,
               right_censored_events, na_strings, bucketed_times,
               event_type_to_index, event_cause_to_index,
               subject_index_to_id, family_index_to_id, na_rows,
               nominal_variables, subject_to_time_and_status_by_event,
               subject_to_data_by_model, subject_info, error_msg);
  }

  // Reverse map.
  for (const pair<string, int>& subject_id_and_index : subject_id_to_index) {
    subject_index_to_id->insert(
        make_pair(subject_id_and_index.second, subject_id_and_index.first));
  }
  for (const pair<string, int>& family_id_and_index : family_id_to_index) {
    family_index_to_id->insert(
        make_pair(family_id_and_index.second, family_id_and_index.first));
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::ParseModelLhs(
    const string& model_lhs_str,
    string* time_col, vector<string>* dep_var_names) {
  if (time_col == nullptr || dep_var_names == nullptr) return false;
  dep_var_names->clear();

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

  // Split around the ","; the first term is the time column, the remaining
  // K > 1 terms are the dependent covariates. 
  vector<string> dep_cov_parts;
  Split(cleaned_lhs, ",", &dep_cov_parts);
  if (dep_cov_parts.size() < 2) {
     cout << "ERROR: Model LHS is improperly formatted: '" << cleaned_lhs
          << "'" << endl;
     return false;
  }
  *time_col = dep_cov_parts[0];

  // Parse each dep covariate on model LHS.
  for (int i = 1; i < dep_cov_parts.size(); ++i) {
    dep_var_names->push_back(dep_cov_parts[i]);
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::GetStatisticsForLinearTerms(
    const bool do_sample_variance,
    const int model_index, const int id_col,
    const set<string>& na_strings,
    const map<string, set<string>>& nominal_variables,
    const vector<VariableParams>& indep_vars_params_k,
    const vector<Expression>& linear_terms_k,
    const vector<string>& data_header,
    const vector<vector<map<double, vector<DataHolder>>>>& subject_to_data_by_model,
    vector<tuple<bool, double, double>>* linear_terms_mean_and_std_dev_k,
    string* error_msg) {
  if (linear_terms_mean_and_std_dev_k == nullptr) return false;

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

  // There are different ways normalization can be done. To prevent one
  // Subject's values from dominating the normalization (e.g. if one
  // Subject has a ton of observation times), we take a single
  // value (the first one), for each covariate, from each Subject.
  //
  // Go through data, and pick out the first non-NA value for each Subject
  // (for each variable).
  vector<vector<DataHolder>> first_data_values_k;
  for (const vector<map<double, vector<DataHolder>>>& subjects_values :
       subject_to_data_by_model) {
    if (!subjects_values[model_index].empty()) {
      first_data_values_k.push_back(subjects_values[model_index].begin()->second);
    }
  }

  // Now Compute values for each linear term (for each row), and keep
  // running total (sum) and it's square across all rows. Also keep track
  // if the linear term had any values that weren't 0 or 1 (we treat
  // binary linear terms differently).
  const int p_k = linear_terms_k.size();
  vector<double> sums(p_k, 0.0);
  vector<double> sums_squared(p_k, 0.0);
  set<int> non_binary_linear_terms;
  int n = first_data_values_k.size();
  for (int i = 0; i < first_data_values_k.size(); ++i) {
    // Expand non-numeric variables to numeric values.
    map<string, double> var_to_numeric_value;
    bool is_na_row = false;
    if (!GetVariableValuesFromDataRow(
            na_strings, nominal_variables, indep_var_name_to_col,
            first_data_values_k[i],
            &is_na_row, &var_to_numeric_value, error_msg)) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Failed to GetVariableValuesFromDataRow for subject " +
                      Itoa(i) + "'s first row of data.\n";
      }
      return false;
    }
    // This Subject had at least one NA value; skip it.
    if (is_na_row) {
      --n;
      continue;
    }

    for (int linear_term_index = 0; linear_term_index < p_k; ++linear_term_index) {
      const Expression& linear_term_k = linear_terms_k[linear_term_index];
      double term_value;
      if (!EvaluateExpression(
              linear_term_k, var_to_numeric_value,
              &term_value, error_msg)) {
        return false;
      }
      sums[linear_term_index] += term_value;
      sums_squared[linear_term_index] += (term_value * term_value);
      if (term_value != 0.0 && term_value != 1.0) {
        non_binary_linear_terms.insert(linear_term_index);
      }
    }
  }

  if (n <= 1) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to standardize columns: there were "
                    "no Subjects that had non-NA values in all covariate "
                    "columns.\n";
    }
    return false;
  }

  // Finally, compute mean and standard deviation for each linear term.
  linear_terms_mean_and_std_dev_k->clear();
  for (int linear_term_index = 0; linear_term_index < p_k; ++linear_term_index) {
    const bool is_binary =
        non_binary_linear_terms.find(linear_term_index) ==
        non_binary_linear_terms.end();
    const double& sum = sums[linear_term_index];
    const double mean = sums[linear_term_index] / n;
    const double denominator = do_sample_variance ? n - 1 : n;
    const double std_dev =
        sqrt((sums_squared[linear_term_index] - (sum * sum) / n) / denominator);
    linear_terms_mean_and_std_dev_k->push_back(make_tuple(!is_binary, mean, std_dev));
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::PopulateTimes(
    const int num_events,
    const set<int>& right_censored_events,
    const vector<vector<map<double, bool>>>& subject_to_time_and_status_by_event,
    vector<SubjectInfo>* subject_info,
    vector<set<double>>* distinct_times,
    string* error_msg) {
  if (subject_info == nullptr || distinct_times == nullptr ||
      subject_info->size() != subject_to_time_and_status_by_event.size()) {
    return false;
  }

  const int n = subject_info->size();
  distinct_times->resize(num_events);

  // Get (L, R) for each Subject, and from them the set of distinct times.
  // TODO(PHB): Some use-cases may want to include the t_0 = 0.0 timepoint in
  // the analysis (or at least, in the output). For the present implementation,
  // \lambda_0 is defined to always be 0.0, so we don't include it in the
  // analysis (as discussed with Donglin and Danyu, 11/19/15).
  //for (set<double>& cov_distinct_times : distinct_times) cov_distinct_times.insert(0.0);
  for (int subject_index = 0; subject_index < n; ++subject_index) {
    SubjectInfo& subject_info_i = (*subject_info)[subject_index];
    const vector<map<double, bool>>& time_and_status_i =
        subject_to_time_and_status_by_event[subject_index];

    // Loop through each of the dependent covariates, setting appropriate
    // values for the corresponding entry of distinct_times_
    // and all_subject_info[subject_index].times_
    for (int k = 0; k < num_events; ++k) {
      EventTimeAndCause& event_info_ik = subject_info_i.times_[k];

      // No need to find upper/lower times for Right-Censored events; but do add
      // the Censoring/Survival time as one of the distinct times.
      if (right_censored_events.find(k) != right_censored_events.end()) {
        if (event_info_ik.type_ == CensoringType::CENSOR_TYPE_UNKNOWN) {
          // Nothing to do here: the status for this Subject was NA.
        } else {
          const double& time = event_info_ik.censoring_info_.is_alive_ ?
              event_info_ik.censoring_info_.censoring_time_ :
              event_info_ik.censoring_info_.survival_time_;
          if (time > 0.0 && time != numeric_limits<double>::infinity()) {
            (*distinct_times)[k].insert(time);
          }
        }
        continue;
      }

      // Set Event-Type to Interval-Censored.
      event_info_ik.type_ = CensoringType::CENSOR_TYPE_INTERVAL;

      const map<double, bool>& time_and_status_ik = time_and_status_i[k];
      if (time_and_status_ik.empty()) {
        // There were no valid (non-NA) status values for this
        // (Subject, Event-Type) pair, so no (L, R) values exist.
        continue;
      }

      // Get the upper-time R_i for this Subject: the first time status was
      // '1', or infinity if it never was.
      double upper_time_ik = numeric_limits<double>::infinity();
      for (const pair<double, bool>& time_and_status : time_and_status_ik) {
        if (time_and_status.second) {
          if (upper_time_ik == numeric_limits<double>::infinity() ||
              time_and_status.first < upper_time_ik) {
            upper_time_ik = time_and_status.first;
          }
        }
      }
      event_info_ik.upper_ = upper_time_ik;
      if (upper_time_ik != numeric_limits<double>::infinity()) {
        (*distinct_times)[k].insert(upper_time_ik);
      }

      // Set L_i.
      // For this, we loop through all of the rows for this Subject, looking for
      // the time just before R_i (which was already set above).
      int timepoint_index = 0;
      double prev_time = -1.0;
      for (const pair<double, bool>& time_and_status : time_and_status_ik) {
        bool is_subjects_last_timepoint =
            timepoint_index == time_and_status_ik.size() - 1;
        const double& current_time = time_and_status.first;

        // Set Lower-time L_i, if appropriate:
        //   - upper_time_ik is infty (so status was never '1' for this subject) and
        //     event_info_ik.lower_ < current_time (since we are iterating through times
        //     from smallest to largest, this will always be true); OR
        //   - upper_time_ik is current_time, then set lower_time to be the previous
        //     time (or zero, in case this is the first time); since we are
        //     iterating through times from smallest to largest, this is equal
        //     to 'prev_time' (or zero, in case prev_time hasn't yet been set).
        if (event_info_ik.upper_ == numeric_limits<double>::infinity() &&
            is_subjects_last_timepoint) {
          event_info_ik.lower_ = current_time;
          if (event_info_ik.lower_ > 0.0) {
            (*distinct_times)[k].insert(event_info_ik.lower_);
          }
        } else if (event_info_ik.upper_ == current_time) {
          if (prev_time == -1.0) {
            event_info_ik.lower_ = 0.0;
          } else {
            event_info_ik.lower_ = prev_time;
            if (event_info_ik.lower_ > 0.0) {
              (*distinct_times)[k].insert(event_info_ik.lower_);
            }
          }
        }

        // Done with this iteration. Update prev_time to the current_time.
        prev_time = current_time;
        timepoint_index++;
      }
    }
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::PopulateIndependentVariableValues(
    const bool use_time_indep_data_structure, const int model_index,
    const VariableNormalization& standardize_linear_terms,
    const vector<tuple<bool, double, double>>& linear_terms_mean_and_std_dev_k,
    const vector<VariableParams>& indep_vars_params_k,
    const map<string, set<string>>& nominal_variables,
    const vector<Expression>& linear_terms_k,
    const set<double>& distinct_times_k,
    const vector<vector<map<double, vector<DataHolder>>>>& subject_to_data_by_model,
    vector<SubjectInfo>* subject_info,
    string* error_msg) {
  if (subject_info == nullptr ||
      subject_info->size() != subject_to_data_by_model.size() ||
      error_msg == nullptr) {
    return false;
  }

  const int n = subject_info->size();
  const int p_k = linear_terms_k.size();
  const int M_k = distinct_times_k.size();
  if (M_k == 0) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: The set of distinct times for model " +
                    Itoa(model_index + 1) + " is empty; this means "
                    "all Subjects had missing values in the Status "
                    "column for this model.\n";
    }
    return false;
  }

  // Loop through all Subjects, setting linear_term_values_[k] and
  // is_time_indep_[k].
  for (int i = 0; i < n; ++i) {
    const map<double, vector<DataHolder>>& subject_to_data_ik =
        subject_to_data_by_model[i][model_index];
    SubjectInfo& subject_info_ik = (*subject_info)[i];

    // Add entry to linear_term_values_ for this (Subject, event-type).
    subject_info_ik.linear_term_values_.push_back(make_pair(VectorXd(), MatrixXd()));
    pair<VectorXd, MatrixXd>& linear_term_values_ik =
        subject_info_ik.linear_term_values_.back();
    // Add entry to is_time_indep_ for this (Subject, event-type).
    subject_info_ik.is_time_indep_.push_back(vector<bool>());
    vector<bool>& is_time_indep_k = subject_info_ik.is_time_indep_.back();
    is_time_indep_k.resize(p_k, false);

    if (subject_to_data_ik.empty()) {
      // There are no rows for this (Subject, model) pair that had valid
      // values for all of the relevant covariates. Nothing to do (i.e.,
      // linear_term_values_ will be empty for this model).
      continue;
    }

    // Get the values to use for each variable, at each distinct time.
    // Notes:
    //   - Here, by "Variable", we mean the names of the variables, as opposed
    //     to the final linear terms. These may be the same thing if the RHS
    //     is a simple linear combination of the covariates, but they may
    //     not exactly correspond, if some of the variables are NOMINAL, and/or
    //     some of the linear terms involve mathematical combinations of the
    //     variables. Notation: There are p'_k variables, and p_k linear-terms
    //   - We' first get the values for the (p'_k) Variables at each of the M_k
    //     timepoints; below we'll use these values to evaluate each of the
    //     (p_k) linear-terms.
    //   - Some variables are time-indep, some time-dependent. For the former,
    //     we only need to store one value for this Subject, for the latter,
    //     we need to store M_k values. Notation: Let
    //       p'_k = p'_k_dep + p'_k_indep
    //     be the breakdown of time-dep and time-indep variables.
    const int p_prime_k = indep_vars_params_k.size();
    // Container to hold the values for the (p'_k_dep) time-dependent variables.
    // The vector has size M_k, and each map has size p'_k_dep and is Keyed by
    // the (time-dep) Variable name and has Value equal to that variable's value
    // at that time.
    vector<map<string, DataHolder>> time_dep_var_to_values_by_time(
        distinct_times_k.size());
    set<string> time_dep_var_names;
    // Container to hold the value of each (of the p'_k_indep) time-independent
    // variable values.
    map<string, DataHolder> time_indep_var_to_value;
    for (int p_prime = 0; p_prime < p_prime_k; ++p_prime) {
      const VariableParams& var_params = indep_vars_params_k[p_prime];
      const string& var_name = var_params.col_.name_;
      // First grab the values of this variable at all M_k times.
      vector<DataHolder> var_values_at_all_times;
      bool is_time_indep = false;
      bool all_na_values = false;
      if (!GetTimeDependentValues(
              p_prime, var_params.time_params_, distinct_times_k,
              subject_to_data_ik,
              &is_time_indep, &var_values_at_all_times, error_msg)) {
        return false;
      }

      // If current Variable is time-independent, store its value in
      // time_indep_var_to_value and proceed to next Variable.
      // NOTE: This is not a perfect indicator of whether a linear-term
      // will be time-independent. For example, if this Variable is NOMINAL,
      // it may have different values (e.g. RED and BLUE), but one of the
      // indicator terms might be I_VAR=PINK, which will always evaluate
      // to zero (so time-independent), but there is no way to know this
      // when evaluating time independence at the Variable level instead
      // of at the linear-term level (i.e. after Indicator terms have been
      // expanded). Hopefully, the cost of missing out on treating
      // these linear-terms properly as time-indep is minimal.
      if (var_values_at_all_times.size() == 1 || is_time_indep) {
        time_indep_var_to_value.insert(make_pair(
            var_name, var_values_at_all_times[0]));
        continue;
      }

      time_dep_var_names.insert(var_name);

      // While it was convenient to retrieve all of the values (i.e.
      // at all distinct timepoints) for this Variable at once (via
      // GetTimeDependentValues() above), below we'll be looping through
      // timepoints, and thus need to store data so that common timepoints
      // (but for different Variables) are stored together. Thus, store
      // the recently retrieved values for this Variable in a data
      // structure that makes per-time lookup convenient.
      int distinct_times_k_index = -1;
      for (const double& time : distinct_times_k) {
        distinct_times_k_index++;
        time_dep_var_to_values_by_time[distinct_times_k_index].insert(make_pair(
            var_name, var_values_at_all_times[distinct_times_k_index]));
      }
    }

    // For time-independent Variables, get the numeric value (either
    // directly from time_indep_var_to_value, or if this is a NOMINAL
    // variable, then via GetVariableValuesFromDataRow, which
    // expands it into the appropriate number of indicator variables,
    // and evaluates each to be '0' or '1' as appropriate.
    const set<string> time_indep_vars = Keys(time_indep_var_to_value);
    map<string, double> time_indep_var_to_numeric_value;
    if (!GetVariableValuesFromDataRow(
            time_indep_vars, nominal_variables,
            time_indep_var_to_value,
            &time_indep_var_to_numeric_value, error_msg)) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Failed to GetVariableValuesFromDataRow for Subject " +
                      Itoa(i) + ".\n";
      }
      return false;
    }

    // Early termination if all values for this Subject are time-independent.
    if (use_time_indep_data_structure &&
        time_indep_vars.size() == p_prime_k) {
      // Set is_time_indep_ vector to be all true.
      for (int p = 0; p < p_k; ++p) is_time_indep_k[p] = true;

      // Populate all of the linear-term values for this Subject in the
      // time-independent data holder; i.e. linear_term_values_[k].first.
      VectorXd& values = linear_term_values_ik.first;
      values.resize(p_k);
      for (int linear_term_index = 0; linear_term_index < p_k; ++linear_term_index) {
        if (!EvaluateLinearTerm(
                linear_terms_k[linear_term_index], standardize_linear_terms,
                linear_terms_mean_and_std_dev_k[linear_term_index],
                time_indep_var_to_numeric_value, set<string>(),  // Not used.
                &values(linear_term_index), nullptr, error_msg)) {
          return false;
        }
      }
      continue;
    }

    // Loop over all distinct times for this Status variable.
    VectorXd& linear_term_time_indep_values_ik = linear_term_values_ik.first;
    MatrixXd& linear_term_time_dep_values_ik = linear_term_values_ik.second;
    // We don't yet know how many of the linear-terms are time-independent;
    // i.e. we don't know p_k_dep and p_k_indep. So we can't yet set the
    // dimensions of the VectorXd and MatrixXd above. Instead, we store the
    // values for the first time-point in termporary vectors, and once we've
    // evaluated the first time-point, we'll know p_k_dep and p_k_indep, at
    // which point (for the rest of the time points) we can directly populate
    // the MatrixXd of time-dependent linear-term values.
    vector<double> temp_linear_term_time_indep_values;
    vector<double> temp_linear_term_time_dep_values;
    int distinct_times_k_index = -1;
    for (const double& time : distinct_times_k) {
      distinct_times_k_index++;

      // Expand non-numeric (time-dependent) variables to numeric values.
      map<string, double> var_to_numeric_value;
      if (!GetVariableValuesFromDataRow(
              time_dep_var_names, nominal_variables,
              time_dep_var_to_values_by_time[distinct_times_k_index],
              &var_to_numeric_value, error_msg)) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: failed to GetVariableValuesFromDataRow for Subject " +
                        Itoa(distinct_times_k_index) + ".\n";
        }
        return false;
      }
      // Add the time-independent Variable values.
      for (const pair<string, double>& time_indep_value :
           time_indep_var_to_numeric_value) {
        var_to_numeric_value.insert(time_indep_value);
      }

      // Evaluate each linear term, using the variable values.
      int time_dep_index = 0;
      for (int linear_term_index = 0; linear_term_index < p_k; ++linear_term_index) {
        // Nothing to do if this is a time-indep linear-term (it's value
        // will have already been added to linear_term_values_[k].first_).
        if (is_time_indep_k[linear_term_index]) continue;

        double term_value;
        bool is_time_indep;
        if (!EvaluateLinearTerm(
              linear_terms_k[linear_term_index], standardize_linear_terms,
              linear_terms_mean_and_std_dev_k[linear_term_index],
              var_to_numeric_value, Keys(time_indep_var_to_value),
              &term_value, &is_time_indep, error_msg)) {
          return false;
        }
        if (distinct_times_k_index == 0) {
          if (is_time_indep && use_time_indep_data_structure) {
            is_time_indep_k[linear_term_index] = true;
            temp_linear_term_time_indep_values.push_back(term_value);
          } else {
            temp_linear_term_time_dep_values.push_back(term_value);
          }
        } else {
          if (is_time_indep && use_time_indep_data_structure) {
            cout << "ERROR: Only expected to identify time-independent linear "
                 << "terms on the first pass; but found that linear term "
                 << linear_term_index << " was time-independent on (disinct "
                 << "time) iteration " << distinct_times_k_index
                 << " for Subject " << i << endl;
            return false;
          }
          linear_term_time_dep_values_ik(time_dep_index, distinct_times_k_index) =
              term_value;
          ++time_dep_index;
        }
      }

      if (distinct_times_k_index > 0) continue;

      // This is the first time point. Now that we know p_k_dep and p_k_indep,
      // resize linear_term_time_[in]dep_values_ik, and move values from the
      // temporary holders to their proper place in them.
      linear_term_time_indep_values_ik.resize(
          temp_linear_term_time_indep_values.size());
      for (int j = 0; j < temp_linear_term_time_indep_values.size(); ++j) {
        linear_term_time_indep_values_ik(j) =
            temp_linear_term_time_indep_values[j];
      }

      linear_term_time_dep_values_ik.resize(
          temp_linear_term_time_dep_values.size(), M_k);
      for (int j = 0; j < temp_linear_term_time_dep_values.size(); ++j) {
        linear_term_time_dep_values_ik(j, 0) =
            temp_linear_term_time_dep_values[j];
      }
    }
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::GetTimeDependentValues(
    const int indep_var_index,
    const TimeDependentParams& indep_var_params,
    const set<double>& times,
    const map<double, vector<DataHolder>>& time_to_values,
    bool* is_time_indep,
    vector<DataHolder>* row, string* error_msg) {
  if (row == nullptr) return false;

  if (time_to_values.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in SetTimeDepValues: empty time_to_values.";
    }
    return false;
  }
  
  if (indep_var_params.interp_type_ == InterpolationType::IT_UNKNOWN) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in extrapolation argument: Unknown interpolation "
                    "type " + Itoa(static_cast<int>(indep_var_params.interp_type_)) +
                    " for indep variable " + Itoa(indep_var_index);
    }
    return false;
  }

  row->clear();

  // Check if this row should copy baseline values for all time points.
  if (indep_var_params.interp_type_ == InterpolationType::IT_BASELINE_CONSTANT) {
    row->push_back(time_to_values.begin()->second[indep_var_index]);
    *is_time_indep = true;
    return true;
  }

  row->resize(times.size(), DataHolder());

  // Go through all distinct times, evaluating to the proper value.
  const double& first_time = time_to_values.begin()->first;
  const DataHolder& first_value =
      time_to_values.begin()->second[indep_var_index];
  const double& last_time = time_to_values.rbegin()->first;
  const DataHolder& last_value =
      time_to_values.rbegin()->second[indep_var_index];
  map<double, vector<DataHolder>>::const_iterator time_to_values_itr =
      time_to_values.begin();
  DataHolder first_value_added;
  bool one_distinct_value_added = true;
  int index = 0;
  for (const double& time : times) {
    // Handle timepoints that are earlier than all observation times.
    if (time < first_time) {
      if (indep_var_params.outside_params_.outside_left_type_ ==
          ExtrapolationType::ET_LEFTMOST) {
        (*row)[index] = first_value;
        one_distinct_value_added &= ValuesMatch(first_value, &first_value_added);
      } else if (indep_var_params.outside_params_.outside_left_type_ ==
                 ExtrapolationType::ET_CONSTANT) {
        (*row)[index] = indep_var_params.outside_params_.default_left_val_;
        one_distinct_value_added &= ValuesMatch(
            indep_var_params.outside_params_.default_left_val_,
            &first_value_added);
      } else {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unrecognized extrapolation:outside type: " +
                        Itoa(static_cast<int>(
                            indep_var_params.outside_params_.outside_left_type_));
        }
        return false;
      }
      index++;
      continue;
    }

    // Handle timepoints that are later than all observation times.
    if (time > last_time) {
      if (indep_var_params.outside_params_.outside_right_type_ ==
          ExtrapolationType::ET_RIGHTMOST) {
        (*row)[index] = last_value;
        one_distinct_value_added &= ValuesMatch(last_value, &first_value_added);
      } else if (indep_var_params.outside_params_.outside_right_type_ ==
                 ExtrapolationType::ET_CONSTANT) {
        (*row)[index] = indep_var_params.outside_params_.default_right_val_;
        one_distinct_value_added &= ValuesMatch(
            indep_var_params.outside_params_.default_right_val_,
            &first_value_added);
      } else {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unrecognized extrapolation:outside type: " +
                       Itoa(static_cast<int>(
                           indep_var_params.outside_params_.outside_right_type_));
        }
        return false;
      }
      index++;
      continue;
    }

    // Current time lies between two times. Find the relevant interval it lies
    // in, and set value accordingly.
    while (time > time_to_values_itr->first) {
      time_to_values_itr++;
    }
    if (time == time_to_values_itr->first) {
      (*row)[index] = time_to_values_itr->second[indep_var_index];
      one_distinct_value_added &= ValuesMatch(
          time_to_values_itr->second[indep_var_index],
          &first_value_added);
    } else if (indep_var_params.interp_type_ ==
               InterpolationType::IT_RIGHT) {
      (*row)[index] = time_to_values_itr->second[indep_var_index];
      one_distinct_value_added &= ValuesMatch(
          time_to_values_itr->second[indep_var_index],
          &first_value_added);
    } else if (indep_var_params.interp_type_ ==
               InterpolationType::IT_LEFT) {
      // Use the previous observation's value, then increment back to present.
      time_to_values_itr--;
      (*row)[index] = time_to_values_itr->second[indep_var_index];
      one_distinct_value_added &= ValuesMatch(
          time_to_values_itr->second[indep_var_index],
          &first_value_added);
      time_to_values_itr++;
    } else if (indep_var_params.interp_type_ ==
               InterpolationType::IT_NEAREST) {
      const double right_time = time_to_values_itr->first;
      const DataHolder& right_value =
          time_to_values_itr->second[indep_var_index];
      // Get the previous observation's value, then increment back to present.
      time_to_values_itr--;
      const double left_time = time_to_values_itr->first;
      const DataHolder& left_value =
          time_to_values_itr->second[indep_var_index];
      time_to_values_itr++;
      (*row)[index] = (time - left_time) < (right_time - time) ? 
          left_value : right_value;
      one_distinct_value_added &= ValuesMatch(
          (time - left_time) < (right_time - time) ? left_value : right_value,
          &first_value_added);
    } else if (indep_var_params.interp_type_ ==
               InterpolationType::IT_LINEAR_INTERP) {
      const double right_time = time_to_values_itr->first;
      const double right_value =
          time_to_values_itr->second[indep_var_index].value_;
      // Get the previous observation's value, then increment back to present.
      time_to_values_itr--;
      const double left_time = time_to_values_itr->first;
      const double left_value =
          time_to_values_itr->second[indep_var_index].value_;
      time_to_values_itr++;
      const double slope = (right_value - left_value) / (right_time - left_time);
      DataHolder temp;
      temp.value_ = slope * (time - left_time) + left_value;
      temp.type_ = DataType::DATA_TYPE_NUMERIC;
      (*row)[index] = temp;
      one_distinct_value_added &= ValuesMatch(temp, &first_value_added);
    } else {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in GetTimeDependentValues: Invalid interp_type_."
                     "This should never happen.";
      }
      return false;
    }
    index++;
  }

  *is_time_indep = one_distinct_value_added;
  return true;
}

bool ReadTimeDepIntervalCensoredData::EvaluateLinearTerm(
    const Expression& linear_term,
    const VariableNormalization& standardize_linear_terms,
    const tuple<bool, double, double>& mean_and_std_dev,
    const map<string, double>& var_to_value,
    const set<string>& time_indep_vars,
    double* value, bool* is_time_indep, string* error_msg) {
  // Sanity-check input.
  if (value == nullptr) return false;

  // Evaluate this linear term.
  set<string> seen_vars;
  double term_value;
  if (!EvaluateExpression(
          linear_term, var_to_value,
          &term_value, &seen_vars, error_msg)) {
    return false;
  }

  // Test if this linear term used any time-dependent variables.
  if (is_time_indep != nullptr) {
    *is_time_indep = true;
    for (const string& seen_var : seen_vars) {
      if (time_indep_vars.find(seen_var) == time_indep_vars.end()) {
        *is_time_indep = false;
        break;
      }
    }
  }

  // Standardize, if appropriate.
  const bool do_std =
      standardize_linear_terms == VAR_NORM_STD ||
      standardize_linear_terms == VAR_NORM_STD_W_N_MINUS_ONE;
  const bool do_non_binary_std =
      standardize_linear_terms == VAR_NORM_STD_NON_BINARY ||
      standardize_linear_terms ==
          VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;

  if (do_std || do_non_binary_std) {
    const bool not_binary = get<0>(mean_and_std_dev);
    const double& mean = get<1>(mean_and_std_dev);
    const double& std_dev = get<2>(mean_and_std_dev);
    if (do_std || (not_binary && do_non_binary_std)) {
      *value = (term_value - mean) / std_dev;
    } else {
      *value = term_value;
    }
  } else {
      *value = term_value;
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::BucketTimes(
    const FileInfo& file_info,
    const VariableParams& time_params,
    const set<string>& na_strings,
    vector<double>* bucketed_times,
    string* error_msg) {
  // Return if user did not specify bucketing/rounding for time column.
  const vector<VariableCollapseParams>& time_collapse_params =
      time_params.collapse_params_;
  const string& time_col_name = time_params.col_.name_;
  if (time_collapse_params.empty() || time_collapse_params[0].num_buckets_ == 0) {
    return true;
  }

  // Open file, and advance past first (header) line, to the first row of data.
  const string& filename = file_info.name_;
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

  // Constants needed below.
  const int num_buckets = time_collapse_params[0].num_buckets_;
  double min = DBL_MAX;
  double max = DBL_MIN;
  
  // Bucketing requires two passes: First pass to determine the range of
  // time values (so we can determine bucket endpoints), and the second
  // to actually bucket the times. In contrast, Rounding can be done in
  // a single pass. Create a temporary time holder for bucketing, or use
  // the passed-in holder for rounding.
  vector<double> orig_times;

  // Read times from file.
  string line;
  int line_num = 1;
  bool encountered_inf = false;
  while (getline(file, line)) {
    file_reader_utils::RemoveWindowsTrailingCharacters(&line);

    // Split line around delimiter.
    vector<string> col_strs;
    Split(line, file_info.delimiter_, false /* do not collapse empty strings */,
          &col_strs);

    // Fetch the Time column.
    DataHolder time_value;
    bool is_na_row = false;
    bool is_new_nominal_col = false;
    if (!ReadValue(true, time_col_name, time_params.col_.index_, col_strs,
                   file_info.infinity_char_, na_strings,
                   time_params.collapse_params_,
                   nullptr, &is_na_row, &is_new_nominal_col,
                   &time_value, error_msg)) {
      return false;
    }
    const double& time = is_na_row ? -1.0 : time_value.value_;
    orig_times.push_back(time);
    if (is_na_row) {
      continue;
    }
    // Cannot use bucketing if there are infinite values.
    if (std::isinf(time)) {
      // Log a warning to user that infinite time values were not bucketed.
      if (!encountered_inf && error_msg != nullptr) {
        *error_msg += "WARNING: Found infinite time on line " +
                      Itoa(line_num + 1) + " (and possibly others); "
                      "will ignore such values for the time bucketing.\n";
      }
      encountered_inf = true;
      continue;
    }
    if (time < min) min = time;
    if (time > max) max = time;
  }

  // Now that we have min/max, do bucketing.
  for (const double& time : orig_times) {
    if (time == -1.0 || std::isinf(time)) {
      // NA times will be handled later; for now, just mark them by using -1.0.
      bucketed_times->push_back(time);
      continue;
    }
    double bucketed_time;
    if (!ReadTableWithHeader::GetBucketValue(
            time, num_buckets, min, max, &bucketed_time, error_msg)) {
      return false;
    }
    bucketed_times->push_back(bucketed_time);
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::ReadFile(
    const TimeDepIntervalCensoredInParams& in_params,
    const vector<VariableParams>& all_indep_var_params,
    const vector<set<int>>& cols_in_models,
    vector<vector<map<double, bool>>>* subject_to_time_and_status_by_event,
    vector<vector<map<double, vector<DataHolder>>>>* subject_to_data_by_model,
    TimeDepIntervalCensoredData* out_data) {
  if (out_data == nullptr) return false;

  vector<double> bucketed_times;
  if (!BucketTimes(
          in_params.file_info_, in_params.time_params_,
          in_params.file_info_.na_strings_,
          &bucketed_times, &out_data->error_msg_)) {
    return false;
  }
 
  // Read all lines (rows) of input data file.
  if (!ReadInputData(
          in_params.indep_vars_params_.size(),
          in_params.file_info_.name_, in_params.file_info_.delimiter_,
          in_params.file_info_.infinity_char_, in_params.id_params_,
          in_params.family_params_, in_params.event_type_params_,
          in_params.event_cause_params_, in_params.time_params_,
          in_params.status_params_, all_indep_var_params,
          cols_in_models,
          in_params.right_censored_events_, in_params.file_info_.na_strings_,
          bucketed_times,
          in_params.event_type_to_index_,
          in_params.event_cause_to_index_,
          &(out_data->subject_index_to_id_),
          &(out_data->family_index_to_id_),
          &(out_data->na_rows_),
          &(out_data->nominal_variables_),
          subject_to_time_and_status_by_event,
          subject_to_data_by_model,
          &(out_data->subject_info_),
          &(out_data->error_msg_))) {
    return false;
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::FillColumnIndices(
    const string& title_line,
    TimeDepIntervalCensoredInParams* in_params, string* error_msg) {
  if (in_params == nullptr) return false;

  // Check to see if we need any column indices (i.e. we know a column name
  // is important, but haven't determined that columns index w.r.t. input file).
  const bool need_id_col =
      in_params->time_indep_type_ != TimeIndependentType::TIME_INDEP &&
      in_params->id_params_.col_.index_ < 0 &&
      !in_params->id_params_.col_.name_.empty();
  const bool need_status_col =
      in_params->status_params_.col_.index_ < 0 &&
      !in_params->status_params_.col_.name_.empty();
  const bool need_family_col =
      in_params->family_params_.col_.index_ < 0 &&
      !in_params->family_params_.col_.name_.empty();
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

  // If we already know all column indices for the relevant columns, return.
  const bool need_at_least_one_col_index =
      need_id_col || need_family_col || need_indep_var_cols || 
      need_event_type_col || need_event_cause_col || need_status_col ||
      in_params->time_params_.col_.index_ == -1;
  if (!need_at_least_one_col_index) return true;

  // Get a list of columns (names) for which we need to determine the index.
  vector<string> columns_needed;
  if (need_id_col) {
    columns_needed.push_back(in_params->id_params_.col_.name_);
  }
  if (need_status_col) {
    columns_needed.push_back(in_params->status_params_.col_.name_);
  }
  if (need_family_col) {
    columns_needed.push_back(in_params->family_params_.col_.name_);
  }
  if (need_event_type_col) {
    columns_needed.push_back(in_params->event_type_params_.col_.name_);
  }
  if (need_event_cause_col) {
    columns_needed.push_back(in_params->event_cause_params_.col_.name_);
  }
  if (in_params->time_params_.col_.index_ == -1) {
    columns_needed.push_back(in_params->time_params_.col_.name_);
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
  if (!GetColumnIndices(
          title_line, in_params->file_info_.delimiter_,
          columns_needed, &columns_needed_indices, error_msg)) {
    return false;
  }

  // Assign column indices to the corresponding VariableParams.
  int current_index = 0;
  if (need_id_col) {
    in_params->id_params_.col_.index_ = columns_needed_indices[current_index];
    current_index++;
  }
  if (need_status_col) {
    in_params->status_params_.col_.index_ = columns_needed_indices[current_index];
    current_index++;
  }
  if (need_family_col) {
    in_params->family_params_.col_.index_ = columns_needed_indices[current_index];
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
  if (in_params->time_params_.col_.index_ == -1) {
    in_params->time_params_.col_.index_ = columns_needed_indices[current_index];
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

bool ReadTimeDepIntervalCensoredData::ReadFile(
    TimeDepIntervalCensoredInParams& in_params,
    TimeDepIntervalCensoredData* out_data) {
  // Sanity-check input.
  if (out_data == nullptr) return false;
  if (in_params.file_info_.name_.empty()) {
    out_data->error_msg_ += "ERROR: Empty Filename.\n";
    return false;
  }

  // Sanity-check ID column was provided, if there are any
  // Right-Censored Events.
  if (!in_params.right_censored_events_.empty() &&
      in_params.id_params_.col_.index_ < 0 &&
      in_params.id_params_.col_.name_.empty()) {
    out_data->error_msg_ += "ERROR: Input files with Right-censored "
                            "event types must have a Subject ID column "
                            "(this is necessary to link a Subject's "
                            "special Time/Status row to the rest of "
                            "that Subject's rows holding the covariate "
                            "values at each Observation time).\n";
    return false;
  }

  // Open input file.
  ifstream input_file(in_params.file_info_.name_.c_str());
  if (!input_file.is_open()) {
    out_data->error_msg_ +=
        "ERROR: Unable to open file '" + in_params.file_info_.name_ + "'\n";
    return false;
  }

  // Read Title line.
  string title_line;
  if (!getline(input_file, title_line)) {
    out_data->error_msg_ += "ERROR: Empty Input Data file '" +
                            in_params.file_info_.name_ + "'.\n";
    return false;
  }
  RemoveWindowsTrailingCharacters(&title_line);

  // Get indices for the important columns, if not already populated.
  if (!FillColumnIndices(title_line, &in_params, &(out_data->error_msg_))) {
    out_data->error_msg_ += "ERROR: Failed to find column indices for all of "
                            "the variables in the model.\n";
    return false;
  }

  // Assign/Force some special columns to be Nominal:
  //   ID, Family, Event-Type, Event-Cause.
  AssignKnownNominalColumns(in_params, &(out_data->nominal_variables_));

  // Read Variable Names on header line, looking for Nominal variables, as
  // identified by those with a '$'-suffix.
  if (!GetNominalVariablesFromHeader(
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
  //   - na_rows_: For those rows that have NA in a column that is
  //     necessarily part of any model: id, family, time, event type,
  //     event cause, status (if not a right-censored data row).
  //     Note that more may be added later to na_rows_ on a per-model
  //     basis, based on NA values in covariate columns for covariates
  //     that are part of that model's linear terms.
  //   - subject_info_:
  //       - family_index_
  //       - Some of times_:
  //           - type_
  //           - censoring_info_ (for right-censored events)
  //           - event_cause_
  //   - subject_to_subject_index_
  //   - family_to_index_
  //   - nominal_variables_
  // The following two fields have outer-vector size n = Num Subjects,
  // and inner-vector size K' = Num Models.
  vector<vector<map<double, bool>>> subject_to_time_and_status_by_event;
  vector<vector<map<double, vector<DataHolder>>>> subject_to_data_by_model;
  if (!ReadFile(
          in_params, all_indep_var_params, cols_in_models,
          &subject_to_time_and_status_by_event, &subject_to_data_by_model, out_data)) {
    return false;
  }

  // Now that we know all NOMINAL variables, sanity-check the extrapolation
  // settings are consistent (i.e. that NOMINAL variables don't have an
  // extrapolation-type set to a numeric constant, nor an interpolation
  // set to be the average of the left and right closest points).
  if (!CheckExtrapolationSettings(
          out_data->nominal_variables_, in_params.indep_vars_params_,
          &(out_data->error_msg_))) {
    return false;
  }

  // Now that we know all NOMINAL variables and all of their values, sanity check
  // that the set of nominal values in the Event-Type/Cause columns exactly
  // match those specified by the user.
  if (!CheckEventTypeAndCauseConsistency(
          out_data->nominal_variables_,
          in_params.event_type_params_,
          in_params.event_cause_params_,
          Keys(in_params.event_type_to_index_),
          Keys(in_params.event_cause_to_index_))) {
    out_data->error_msg_ += "ERROR: Inconsistent Event type and cause.\n";
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
        !GetStatisticsForLinearTerms(
            do_sample_variance, k,
            in_params.id_params_.col_.index_,
            in_params.file_info_.na_strings_,
            out_data->nominal_variables_, in_params.indep_vars_params_[k],
            linear_terms_k, data_header_by_model[k], subject_to_data_by_model,
            &(out_data->linear_terms_mean_and_std_dev_.back()),
            &out_data->error_msg_)) {
      out_data->error_msg_ += "ERROR getting statistics for linear terms "
                              "in model " + Itoa(k + 1) + ".\n";
      return false;
    }
  }

  // Find (L, R) for each interval-censored event, and the set of all
  // distinct times for each event. In particular, fills out_data:
  //   - distinct_times_
  //   - subject_info_.times_: the parts not yet filled:
  //       - lower_ (for interval-censored events)
  //       - upper_ (for interval-censored events)
  if (!PopulateTimes(
          out_data->event_type_index_to_name_.size(),
          in_params.right_censored_events_, subject_to_time_and_status_by_event,
          &out_data->subject_info_, &out_data->distinct_times_, &out_data->error_msg_)) {
    return false;
  }

  // Finally, loop over all models, filling in remaining items of out_data:
  //   - na_rows_
  //   - subject_info_: The parts not yet filled:
  //       - linear_term_values_
  //       - is_time_indep_
  for (int k = 0; k < in_params.model_rhs_.size(); ++k) {
    if (!PopulateIndependentVariableValues(
            in_params.use_time_indep_data_structure_, k,
            in_params.linear_terms_normalization_,
            out_data->linear_terms_mean_and_std_dev_[k],
            in_params.indep_vars_params_[k],
            out_data->nominal_variables_,
            linear_terms[k],
            in_params.event_type_to_index_.size() == in_params.model_rhs_.size() ?
            out_data->distinct_times_[k] : out_data->distinct_times_[0],
            subject_to_data_by_model,
            &out_data->subject_info_, &out_data->error_msg_)) {
      out_data->error_msg_ += "ERROR filling in covariate values for linear terms "
                              "in model " + Itoa(k + 1) + ".\n";
      return false;
    }
  }

  return true;
}

bool ReadTimeDepIntervalCensoredData::ReadFile(
    const bool use_time_indep_data_structure,
    const set<string>& right_censored_events,
    const string& event_type_col,
    const string& event_cause_col,
    const vector<string>& event_types,
    const vector<string>& event_causes,
    const vector<ModelAndDataParams>& all_models,
    TimeDepIntervalCensoredData* out_data) {
  if (all_models.empty()) return false;

  TimeDepIntervalCensoredInParams in_params;

  // Sanity-Check event-type/cause specifications are valid.
  if (!SanityCheckEventTypeAndCauseConsistency(
          all_models.size(),  
          event_type_col, event_cause_col, event_types, event_causes,
          right_censored_events, &in_params.right_censored_events_)) {
    return false;
  }

  in_params.use_time_indep_data_structure_ =
      use_time_indep_data_structure;
  in_params.time_indep_type_ = TimeIndependentType::TIME_DEP;

  // Copy fields from all_models to a TimeDepIntervalCensoredInParams object.
  // First, copy fields that are common to all models.
  const ModelAndDataParams& params = all_models[0];
  in_params.file_info_ = params.file_;
  in_params.linear_terms_normalization_ = params.standardize_vars_;
  // Keep track of covariate columns versus other columns of interest
  // (e.g. ID, time, status, event_type).
  set<string> non_indep_vars;
  // ID column, if present.
  in_params.id_params_.col_.name_ = params.id_str_;
  non_indep_vars.insert(params.id_str_);
  // Time column.
  if (params.model_lhs_.time_vars_names_.size() != 1) {
    cout << "ERROR: The model specification's LHS should have format: \n\t"
         << "(Time, Status)\nWhere 'Time' and 'Status' refer to the names "
         << "of the columns that hold the Monitoring Time and Status." << endl;
    return false;
  }
  in_params.time_params_.col_.name_ = params.model_lhs_.time_vars_names_[0];
  non_indep_vars.insert(params.model_lhs_.time_vars_names_[0]);
  // Status column.
  if (params.model_lhs_.dep_vars_names_.size() != 1) {
    cout << "ERROR: The model specification's LHS should have format: \n\t"
         << "(Time, Status)\nWhere 'Time' and 'Status' refer to the names "
         << "of the columns that hold the Monitoring Time and Status." << endl;
    return false;
  }
  in_params.status_params_.col_.name_ = params.model_lhs_.dep_vars_names_[0];
  non_indep_vars.insert(params.model_lhs_.dep_vars_names_[0]);
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
      // Univariate (Single event) case. Go ahead and add a phony event name.
      out_data->event_type_index_to_name_.insert(make_pair(0, "Status"));
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

  // Now copy fields specific to each model.
  out_data->na_rows_.resize(all_models.size());
  for (int k = 0; k < all_models.size(); ++k) {
    const ModelAndDataParams& params_k = all_models[k];
    // Model for the k^th event type.
    in_params.model_rhs_.push_back(params_k.model_rhs_);
    // Pick out the variable parameters for the independent variables.
    in_params.indep_vars_params_.push_back(vector<VariableParams>());
    vector<VariableParams>& indep_vars_params_k =
        in_params.indep_vars_params_.back();
    // Set the Variable Parameters for each of the columns.
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
      // Time column.
      } else if (var_params.col_.name_ == in_params.time_params_.col_.name_) {
        in_params.time_params_ = var_params;
      // Family column.
      } else if (!params_k.family_str_.empty() &&
                 var_params.col_.name_ == in_params.family_params_.col_.name_) {
        in_params.family_params_ = var_params;
      }
    }
  }

  return ReadFile(in_params, out_data);
}

/* DEPRECATED. See notes in .h file.
bool ReadTimeDepIntervalCensoredData::DeleteTimeDepIntervalCensoredData(
    TimeDepIntervalCensoredData* data) {
  if (data == nullptr) return true;
  for (map<int, vector<SubjectInfo>>::iterator subjects_values_itr =
       data->subject_to_values_.begin();
       subjects_values_itr != data->subject_to_values_.end();
       ++subjects_values_itr) {
    for (SubjectInfo& info : subjects_values_itr->second) {
      if (info.linear_term_values_ != nullptr) delete info.linear_term_values_;
    }
  }
  return true;
}
*/

bool ReadTimeDepIntervalCensoredData::ParseIntervalCensoredCommandLineArguments(
    int argc, char* argv[],
    string* event_type_col, string* event_cause_col,
    vector<string>* event_types, vector<string>* event_causes,
    set<string>* right_censored_events,
    vector<ModelAndDataParams>* params) {
  for (int i = 1; i < argc; ++i) {
    string arg = string(argv[i]);
    if (arg == "--event_type") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--event_type'.\n";
        return false;
      }
      ++i;
      *event_type_col = StripQuotes(string(argv[i]));
    } else if (arg == "--event_cause") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--event_cause'.\n";
        return false;
      }
      ++i;
      *event_cause_col = StripQuotes(string(argv[i]));
    } else if (arg == "--event_types") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--event_types'.\n";
        return false;
      }
      ++i;
      Split(StripQuotes(string(argv[i])), ",", event_types);
    } else if (arg == "--event_causes") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--event_causes'.\n";
        return false;
      }
      ++i;
      Split(StripQuotes(string(argv[i])), ",", event_causes);
    } else if (arg == "--right_censored_events") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after "
             << "'--right_censored_events'.\n";
        return false;
      }
      ++i;
      vector<string> temp_rc_events;
      Split(StripQuotes(string(argv[i])), ",", &temp_rc_events);
      for (const string& event : temp_rc_events) {
        right_censored_events->insert(event);
      }
    } else if (HasPrefixString(arg, "--model_event_")) {
      // Determine which model is being specified.
      int model_index;
      const string suffix = StripPrefixString(arg, "--model_event_");
      if (!Stoi(suffix, &model_index)) {
        cout << "ERROR Reading Command: Unrecognized argument '"
             << arg << "' could not be parsed. Did you mean to use "
             << "--model_event_k?" << endl;
        return false;
      }

      // Make sure there are enough models already in params; if not, add them.
      while (params->size() < model_index) {
        params->push_back(ModelAndDataParams());
      }

      ModelAndDataParams& params_k = (*params)[model_index - 1];
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--model_event_k'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      params_k.model_str_ = StripQuotes(RemoveAllWhitespace(string(argv[i])));
    } else if (HasPrefixString(arg, "--model_cause_")) {
      // Determine which model is being specified.
      int model_index;
      const string suffix = StripPrefixString(arg, "--model_cause_");
      if (!Stoi(suffix, &model_index)) {
        cout << "ERROR Reading Command: Unrecognized argument '"
             << arg << "' could not be parsed. Did you mean to use "
             << "--model_cause_k?" << endl;
        return false;
      }

      // Make sure there are enough models already in params; if not, add them.
      while (params->size() < model_index) {
        params->push_back(ModelAndDataParams());
      }

      ModelAndDataParams& params_k = (*params)[model_index - 1];
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--model_cause_k'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      params_k.model_str_ = StripQuotes(RemoveAllWhitespace(string(argv[i])));
    }
  }

  // Currently, we only support *either* having multiple event-types *or*
  // having multiple event-causes (or not having either). Check that user
  // has not specified multiple event-types and causes.
  if (event_types->size() > 1 && event_causes->size() > 1) {
    cout << "ERROR: Program for reading Interval-Censored data is currently "
         << "only setup to handle either multiple event-types or event-causes, "
         << "but not both." << endl;
    return false;
  }

  // If the user specified more event_types/causes than models, add more models.
  int num_desired_models = max(event_types->size(), event_causes->size());
  while (params->size() < num_desired_models) {
    params->push_back(ModelAndDataParams());
  }

  if (params->size() > 1 && num_desired_models != params->size()) {
    cout << "ERROR: " << params->size() << " models were specified, which "
         << "must either match the number of event-types specified ("
         << event_types->size() << ") or the number of event-causes "
         << "specified (" << event_causes->size() << ")." << endl;
    return false;
  }

  // Fields that are common to all models were only populated for the first
  // model. Now, we go through and populate all of the other models'
  // corresponding fields by copying over the values from the first model.
  // NOTE: Some fields are allowed to be different for different models, and
  // these must be explicitly populated by the user in the command-line (i.e.
  // these fields will *not* copy over the values of the corresponding fields
  // from the first model):
  //   - subgroup_str_
  //   - starata_str_
  // For model_str_ and model_type_, the first model's values for these will be
  // copied over if they weren't explicitly provided by the user for this model.
  for (int i = 1; i < params->size(); ++i) {
    CopyModelAndDataParams((*params)[0], &((*params)[i]));
  }

  return true;
}

void ReadTimeDepIntervalCensoredData::PrintTimeDepIntervalCensoredData(
    const TimeDepIntervalCensoredData& data) {
  const int n = data.subject_info_.size();
  const int K = data.event_type_index_to_name_.size();
  const int K_prime = data.legend_.size();
  // Early return, if any requisite fields are missing/empty.
  if (n == 0 || K_prime == 0 || K == 0) {
    cout << "ERROR: Empty model, number of events, or no Subjects:\n"
         << "\tNum Models: " << K_prime << endl
         << "\tNum Events: " << K << endl
         << "\tNum Subjects: " << n << endl;
    return;
  }

  // Number of Models, and their RHS.
  cout << endl << "Found " << K_prime << " model"
       << (K_prime == 1 ? "" : "s") << ":" << endl;
  for (int k_prime = 0; k_prime < K_prime; ++k_prime) {
    cout << "\tModel " << k_prime + 1 << " RHS: "
         << Join(data.legend_[k_prime], " + ") << endl;
  }
  cout << endl;

  // Print mapping between Model and Event-Type/Cause.
  if (K_prime > 1) {
    if (K_prime == data.event_cause_index_to_name_.size()) {
      cout << "Model determined by event-cause:" << endl;
      for (const pair<int, string>& model_index_by_event_cause :
           data.event_cause_index_to_name_) {
        cout << "\tModel " << model_index_by_event_cause.first + 1
             << " corresponds to Event-Cause '"
             << model_index_by_event_cause.second << "'" << endl;
      }
    } else if (K_prime == K) {
      cout << "Model determined by event-type:" << endl;
      for (const pair<int, string>& model_index_by_event_type :
           data.event_type_index_to_name_) {
        cout << "\tModel " << model_index_by_event_type.first + 1
             << " corresponds to Event-Type '"
             << model_index_by_event_type.second << "'" << endl;
      }
    } else {
      cout << "ERROR: Number of models (" << K_prime << ") matches neither "
           << "the number of Event-Types (" << K << ") nor the number of "
           << "Event-Causes (" << data.event_cause_index_to_name_.size()
           << ")" << endl;
      return;
    }
  } else {
    cout << "Single-Event detected." << endl;
  }
  cout << endl;

  // Print the set of Distinct Times for each event type.
  cout << "The Set of Distinct Times for "
       << (K > 1 ? "each Event-Type " : "this event ")
       << "is:" << endl;
  for (int k = 0; k < K; ++k) {
    cout << "\tFor Event-Type " << k + 1 << ", distinct times: {"
         << Join(data.distinct_times_[k], ", ") << "}" << endl;
  }
  cout << endl;

  // Print rows that had missing information.
  for (int k_prime = 0; k_prime < K_prime; ++k_prime) {
    if (!data.na_rows_[k_prime].empty()) {
      cout << "For Model " << k_prime + 1 << ", the following rows "
           << "had at least one missing value:\n\t{"
           << Join(data.na_rows_[k_prime], ", ") << "}" << endl;
    }
  }  
  cout << endl;

  // Print out Subject Info.
  cout << "##################################################################"
       << endl;
  cout << endl << "Subject Information (" << n << " Subjects)" << endl << endl;
  int i = 0;
  for (const pair<int, string>& subject_index_to_id :
       data.subject_index_to_id_) {
    if (subject_index_to_id.first != i) {
      cout << "ERROR: Unexpected Subject index " << subject_index_to_id.first
           << " should have been " << i << endl;
      return;
    }
    cout << "Info for Subject " << subject_index_to_id.second << ":" << endl;
    
    // Check that this Subject had at least one valid row of data.
    const SubjectInfo& current_subject_info = data.subject_info_[i];
    if (current_subject_info.times_.empty()) {
      cout << "Subject should not be used (no valid rows in data file)." << endl;
      continue;
    } else if (current_subject_info.times_.size() != K) {
      cout << "ERROR: times_ field of each SubjectInfo should always match "
           << "the number of event-types. But for Subject " << i + 1
           << ", the times_ field has size " << current_subject_info.times_.size()
           << ", while there are " << K << " distinct event-types." << endl;
      cout << "NOTE: This should never happen, and means there is bug in "
           << "the code. Contact Paul with details..." << endl;
      continue;
    }

    // Family Index.
    const string failed_lookup = "ERROR";
    if (current_subject_info.family_index_ >= 0) {
      cout << "\tFamily: "
           << FindWithDefault(current_subject_info.family_index_,
                              data.family_index_to_id_, failed_lookup)
           << endl;
    }

    // Event Times.
    for (int k = 0; k < K; ++k) {
      const EventTimeAndCause& time_and_cause = current_subject_info.times_[k];
      if (K > 1) {
        cout << "\tFor Event-Type '"
             << FindWithDefault(k, data.event_type_index_to_name_, failed_lookup)
             << "', ";
      }
      cout << "\tSubject ";
      if (time_and_cause.type_ == CensoringType::CENSOR_TYPE_RIGHT) {
        if (time_and_cause.censoring_info_.is_alive_) {
          cout << "is Right-Censored with censoring time "
               << time_and_cause.censoring_info_.censoring_time_;
        } else {
          cout << "has Survival-Time "
               << time_and_cause.censoring_info_.survival_time_;
          if (!data.event_cause_index_to_name_.empty()) {
            cout << ", with event caused by '"
                 << FindWithDefault(
                       time_and_cause.event_cause_,
                       data.event_cause_index_to_name_, failed_lookup)
                 << "'";
          }
        }
      } else if (time_and_cause.type_ == CensoringType::CENSOR_TYPE_INTERVAL) {
        if (time_and_cause.upper_ == numeric_limits<double>::infinity()) {
          cout << "is Right-Censored with censoring time "
               << time_and_cause.lower_;
        } else {
          cout << "is Interval-Censored with (L, R) = ("
               << time_and_cause.lower_ << ", "
               << time_and_cause.upper_ << ")";
          if (!data.event_cause_index_to_name_.empty()) {
            cout << ", with event caused by '"
                 << FindWithDefault(
                       time_and_cause.event_cause_,
                       data.event_cause_index_to_name_, failed_lookup)
                 << "'";
          }
        }
      } else {
        cout << "should not be used (no valid rows in data file).";
      }
      cout << endl;
    }
    cout << endl;

    // Independent Variable (a.k.a. Covariate; a.k.a. linear-term) values.
    for (int k_prime = 0; k_prime < K_prime; ++k_prime) {
      const pair<VectorXd, MatrixXd>& linear_term_values =
          current_subject_info.linear_term_values_[k_prime];
      if (linear_term_values.first.size() == 0 &&
          (linear_term_values.second.rows() == 0 ||
           linear_term_values.second.cols() == 0)) {
        cout << "\tSubject has no covariate values for Model "
             << k_prime + 1 << endl;
        continue;
      }
      const vector<bool>& time_indep_covariates =
          current_subject_info.is_time_indep_[k_prime];
      const int p_k = time_indep_covariates.size();
      cout << "\tSubject's covariate values for Model " << k_prime + 1
           << " at each of the " << data.distinct_times_[k_prime].size()
           << " distinct times:" << endl;
      int indep_row_index = 0;
      int dep_row_index = 0;
      for (int p = 0; p < p_k; ++p) {
        if (time_indep_covariates[p]) {
          cout << "\t\tFor Linear-Term '" << data.legend_[k_prime][p]
               << "': " << endl << "\t\t\t"
               << linear_term_values.first(indep_row_index)
               << " (Time-Independent, so same value at all times)" << endl;
          ++indep_row_index;
        } else {
          cout << "\t\tFor Linear-Term '" << data.legend_[k_prime][p]
               << "':" << endl << "\t\t\t"
               << linear_term_values.second.row(dep_row_index)
               << endl;
          ++dep_row_index;
        }
      }
    }
    ++i;
  }
  cout << endl;
}

}  // namespace file_reader_utils
