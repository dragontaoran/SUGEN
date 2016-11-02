#include "read_input.h"

#include "FileReaderUtils/read_table_with_header.h"
#include "MapUtils/map_utils.h"
#include "MathUtils/constants.h"
#include "MathUtils/data_structures.h"
#include "MathUtils/eq_solver.h"
#include "MathUtils/number_comparison.h"
#include "StringUtils/string_utils.h"
#include "TestUtils/test_utils.h"

#include <Eigen/Dense>
#include <errno.h>
#include <map>
#include <math.h>
#include <set>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace map_utils;
using namespace math_utils;
using namespace string_utils;
using namespace test_utils;
using namespace std;

static const char kErrorTermId[] = "Error_Term";

namespace file_reader_utils {

string ReadInput::GetErrorTermId() { return kErrorTermId; }

bool ReadInput::IsTrueIndicator(const string& value) {
  return (value == "1") || (value == "1.0") || (value == "T") ||
         (value == "True") || (value == "true") || (value == "TRUE");
}

bool ReadInput::IsFalseIndicator(const string& value) {
  return (value == "0") || (value == "0.0") || (value == "F") ||
         (value == "False") || (value == "false") || (value == "FALSE");
}

int ReadInput::GetIntegerFromUser(const string& query_text) {
  int n;
  bool valid_input = false;
  char temp;
  cout << query_text;
  while (!valid_input) {
    string user_response = "";
    cin.get(temp);
    while (temp != '\n') {
      user_response += temp;
      cin.get(temp);
    }
    if (!Stoi(user_response, &n)) {
      cout << "\nUnable to process your input '" << user_response
           << "' as an integer.\n" 
           << query_text;
      continue;
    }
    valid_input = true;
  }
  return n;
}

double ReadInput::GetDoubleFromUser(const string& query_text) {
  double d;
  bool valid_input = false;
  char temp;
  cout << query_text;
  while (!valid_input) {
    string user_response = "";
    cin.get(temp);
    while (temp != '\n') {
      user_response += temp;
      cin.get(temp);
    }
    if (!Stod(user_response, &d)) {
      cout << "\nUnable to process your input '" << user_response
           << "' as a numerical value.\n" 
           << query_text;
      continue;
    }
    valid_input = true;
  }
  return d;
}

int ReadInput::GetNumDataRowsPerSimulationFromUser() {
  const string user_input_request =
      "\nPlease enter the number of sample rows (input data points) 'n'\n"
      " $> [Enter value for 'n']: ";
  return GetIntegerFromUser(user_input_request);
}

int ReadInput::GetNumSimulationsFromUser() {
  const string user_input_request =
      "\nPlease enter the number of simulations 'k' you want to run\n"
      " $> [Enter value for 'k']: ";
  return GetIntegerFromUser(user_input_request);
}

bool ReadInput::GetColumnIndicesFromString(
    const string& strata, const vector<string>& header, set<int>* strata_cols) {
  if (strata.empty()) return true;
  vector<string> titles;
  string strata_stripped = StripPrefixString(strata, "(");
  strata_stripped = StripSuffixString(strata_stripped, ")");
  Split(strata_stripped, ",", &titles);
  for (const string& title : titles) {
    bool found_match = false;
    for (int i = 0; i < header.size(); ++i) {
      if (header[i] == title) {
        strata_cols->insert(i);
        found_match = true;
        break;
      }
    }
    if (!found_match) {
      cout << "ERROR in finding strata column: "
           << "Unable to find strata column '"
           << title << "' among the " << header.size()
           << " titles in the Header row of the input file:" << endl
           << "\"" << Join(header, ", ") << "\"" << endl;
      return false;
    }
  }
  return true;
}

bool ReadInput::GetColumnIndicesFromString(
    const string& subgroup, const vector<string>& header,
    vector<int>* subgroup_cols) {
  vector<string> titles;
  string subgroup_stripped = StripPrefixString(subgroup, "(");
  subgroup_stripped = StripSuffixString(subgroup_stripped, ")");
  Split(subgroup_stripped, ",", &titles);
  for (const string& title : titles) {
    bool found_match = false;
    for (int i = 0; i < header.size(); ++i) {
      if (header[i] == title) {
        subgroup_cols->push_back(i);
        found_match = true;
        break;
      }
    }
    if (!found_match) {
      cout << "ERROR in finding subgroup column: "
           << "Unable to find subgroup column '"
           << title << "' among the titles in the Header row of the input file:"
           << endl << "\"" << Join(header, ", ") << "\"" << endl;
      return false;
    }
  }
  return true;
}

bool ReadInput::ParseSubgroup(
    const vector<string>& data_values_header,
    const vector<VariableColumn>& subgroup_cols,
    const vector<vector<string>>& subgroups,
    const vector<DataHolder>& sample_values,
    int* subgroup_index) {
  // It is valid that some calls to this shouldn't do anything; such
  // calls will have NULL subgroup_cols and NULL subgroup.
  if (subgroup_cols.empty()) return true;

  // Go through sample_values, picking out the values in the relevant
  // (subgroup) columns, and storing them in values_for_subgroup_cols.
  vector<string> values_for_subgroup_cols;
  for (int i = 0; i < subgroup_cols.size(); ++i) {
    // Ignore the column index specified by subgroup_cols[i].index_: it is
    // with respect to the original data file, *not* the labelling
    // of sample_values. Instead, lookup the name in data_values_header
    // to find the appropriate column.
    const string& col_name = subgroup_cols[i].name_;
    int subgroup_col = -1;
    for (int i = 0; i < data_values_header.size(); ++i) {
      if (data_values_header[i] == col_name) {
        subgroup_col = i;
        break;
      }
    }
    if (subgroup_col == -1) {
      cout << "ERROR: Unable to find Subgroup column '" << col_name
           << " among the columns of the data file." << endl;
      return false;
    }
    if (subgroup_col >= sample_values.size()) {
      cout << "\nERROR in Parsing Subgroup: The subgroup column index ("
           << subgroup_col << ") is bigger than the number of columns in "
           << "input data (" << sample_values.size() << "). Aborting." << endl;
      return false;
    }
    const string& subgroup_str =
        GetDataHolderString(sample_values[subgroup_col], true, 17);
    if (subgroup_str.empty()) {
      cout << "\nERROR in Parsing Subgroup: Empty column ("
           << subgroup_col << ")." << endl;
      return false;
    }
    values_for_subgroup_cols.push_back(subgroup_str);
  }

  // Try to match the values in values_for_subgroup_cols to one of the
  // indices in subgroups. If a match is found, set subgroup index
  // accordingly.
  for (int i = 0; i < subgroups.size(); ++i) {
    const vector<string>& subgroup_type = subgroups[i];
    if (equal(values_for_subgroup_cols.begin(), values_for_subgroup_cols.end(),
              subgroup_type.begin())) {
      *subgroup_index = i;
      return true;
    }
  }
  return true;
}

bool ReadInput::ParseStrata(
    const int row_index,
    const vector<string>& data_values_header,
    const set<VariableColumn>& strata_cols,
    const vector<DataHolder>& sample_values,
    int* strata_index,
    vector<vector<string>>* strata_names,
    map<int, int>* row_to_strata) {
  // It is valid that some calls to this shouldn't do anything; such
  // calls will have row_to_strata = NULL.
  if (row_to_strata == nullptr) return true;
  
  // Sanity-check we haven't already assigned a strata index to this row.
  if (row_to_strata->find(row_index) != row_to_strata->end()) {
    cout << "ERROR: Already have assigned a strata index ("
         << row_to_strata->find(row_index)->second
         << ") to row " << row_index << endl;
    return false;
  }

  // If strata_cols is empty, then all input samples belong to same
  // strata; mark this row as belonging to strata '0' and return.
  if (strata_cols.empty()) {
    row_to_strata->insert(make_pair(row_index, 0));
    *strata_index = 0;
    return true;
  }
  if (strata_names == nullptr || strata_index == nullptr) {
    return false;
  }

  vector<string> strata_values;
  for (const VariableColumn& strata_col : strata_cols) {
    // Ignore the column index specified by strata_col.index_: it is
    // with respect to the original data file, *not* the labelling
    // of sample_values. Instead, lookup the name in data_values_header
    // to find the appropriate column.
    const string& col_name = strata_col.name_;
    int col_index = -1;
    for (int i = 0; i < data_values_header.size(); ++i) {
      if (data_values_header[i] == col_name) {
        col_index = i;
        break;
      }
    }
    if (col_index == -1) {
      cout << "ERROR: Unable to find Strata column '" << col_name
           << " among the columns of the data file." << endl;
      return false;
    }
    
    // Sanity check it is a valid column index.
    if (col_index >= sample_values.size()) {
      cout << "\nERROR in Parsing Strata: The strata column index ("
           << col_index << ") is bigger than the number of columns in "
           << "input data (" << sample_values.size() << "). Aborting." << endl;
      return false;
    }
    const string& strata_str =
        GetDataHolderString(sample_values[col_index], true, 17);
    if (strata_str.empty()) {
      cout << "\nERROR in Parsing Strata: Empty column ("
           << col_index << ")." << endl;
      return false;
    }
    strata_values.push_back(strata_str);
  }
  

  // Look to see if this strata_name is already present in strata_names.
  bool found_match = false;
  for (int i = 0; i < strata_names->size(); ++i) {
    const vector<string>& strata_name = (*strata_names)[i];
    if (equal(strata_name.begin(), strata_name.end(), strata_values.begin())) {
      // A strata with this name already exists. Add 'row_index' to it.
      *strata_index = i;
      row_to_strata->insert(make_pair(row_index, i));
      found_match = true;
      break;
    }
  }
  if (!found_match) {
    // This name not in strata_names. Create a new strata with this name,
    // and add 'row_index' to it.
    row_to_strata->insert(make_pair(row_index, strata_names->size()));
    *strata_index = strata_names->size();
    strata_names->push_back(strata_values);
  }
  
  return true;
}

// TODO(PHB): Is this function still used anywhere?
bool ReadInput::ReadFileAndGetModel(ModelAndDataParams* params) {
  // Sanity check input.
  if (params == nullptr) {
     params->error_msg_ +=
          "ERROR in ReadFile: null input parameters.\n";
    return false;
  }

  // Open and read input file.
  vector<string> data_values_header;
  vector<vector<DataHolder>> data_values;
  if (!ReadTableWithHeader::ReadDataFile(
          params->file_, params->header_, params->input_cols_used_, params->var_params_,
          &data_values_header, &data_values, &params->nominal_columns_and_values_,
          &params->na_rows_and_columns_, &params->error_msg_)) {
     params->error_msg_ +=
         "ERROR: Unable to Read File (" + params->file_.name_ + ").\n";
    return false;
  }

  // Fetch header from first line of data file.
  if (!GetHeader(params->file_, &params->header_, &params->error_msg_)) {
    return false;
  }

  // Print the detected nominal columns, and get input from user for
  // the actual nominal columns.
  if (!ReadInput::GetNominalColumns(
          params->header_, data_values,
          &params->nominal_columns_and_values_, &params->error_msg_)) {
    params->error_msg_ += "ERROR in getting nominal columns.\n";
    return false;
  }

  // Get Model Specifications from User.
  if (!GetModel(
          params->model_type_, params->header_, params->nominal_columns_and_values_,
          &params->model_lhs_, &params->model_rhs_, &params->input_cols_used_,
          &params->error_msg_)) {
      params->error_msg_ += "ERROR in getting model.\n";
    return false;
  }

  return true; 
}

string ReadInput::PromptUserForModelText(const ModelType& type) {
  if (type == ModelType::MODEL_TYPE_LINEAR ||
      type == ModelType::MODEL_TYPE_LOGISTIC) {
    return
        "\nEnter your linear Model."
         "\n\nYour expression can involve '+', '*', 'Log', 'exp', 'sqrt', and "
         "'pow' terms, and should have one equals sign with the dependent "
         "variable on the left and linear terms (involving the independent "
         "variables) on the right. Regression coefficients, explicit mention "
         "of a constant term (assumed implicitly), and the error term should "
         "not be specified. Nominal variables can optionally be specified by a "
         "'$' suffix (otherwise they will be automatically detected based on t"
         "he data in the input file). Variable names should not contain spaces";
  } else {
    return
        "\nEnter your Cox Model."
         "\n\nYour expression can involve '+', '*', 'Log', 'exp', 'sqrt', and "
         "'pow' terms, and should have one equals sign with the dependent "
         "variables (e.g. time, status) on the left and linear terms "
         "(involving the independent variables) on the right. Regression "
         "coefficients and the error term should not be specified; and no "
         "constant term is assumed. Nominal variables can optionally be "
         "specified by a '$' suffix (otherwise they will be automatically "
         "detected based on the data in the input file). Variable names "
         "should not contain spaces";
  }
}

string ReadInput::ExampleModel(const ModelType& type) {
  if (type == ModelType::MODEL_TYPE_LINEAR ||
      type == ModelType::MODEL_TYPE_LOGISTIC) {
    return
        "Example:\n\tLog(Y) = "
        "RACE$ * Log(AGE) + RACE$ + AGE * pow(HEIGHT, 0.5) + WEIGHT\n";
  } else {
    return
        "Example:\n\t(Log(Survival_Time), Log(Censoring_Time), Status) = "
        "RACE$ * Log(AGE) + RACE$ + AGE * pow(HEIGHT, 0.5) + WEIGHT\n";
  }
}

bool ReadInput::PromptForAndParseModel(
    const bool from_file, const ModelType& model_type,
    DepVarDescription* model_lhs, Expression* model_rhs, string* error_msg) {
  // Sanity check input.
  if (model_lhs == nullptr || model_rhs == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Null input variables in PromptForAndParseModel(). "
                    "Check API in call stack and try again. Aborting.";
    }
    return false;
  }

  // Request Linear Model from user.
  const string divider =
      "\n############################################################";
  const string data_from_file = !from_file ? "" :
      ", and should correspond to (a subset of) the names "
      "used in the input data file";
  string user_input_request = PromptUserForModelText(model_type);
  user_input_request += data_from_file + ".\n" + ExampleModel(model_type);
  cout << divider << user_input_request << " $> [Linear Model]: ";

  // Read in Linear Model.
  bool valid_input = false;
  char temp;
  while (!valid_input) {
    string user_response = "";
    cin.get(temp);
    while (temp != '\n') {
      user_response += temp;
      cin.get(temp);
    }
    string input_error = "";
    if (!ProcessUserEnteredModel(
            user_response, model_type, model_lhs, model_rhs, &input_error)) {
      cout << divider << "\nUnable to process your model: " << input_error
           << "\n" << user_input_request << " $> [Linear Model]: ";
      continue;
    }
    valid_input = true;
  }

  return true;  
}

bool ReadInput::GetModel(
    const ModelType& model_type,
    const vector<string>& variable_names,
    const map<int, set<string> >& nominal_columns,
    DepVarDescription* model_lhs, Expression* model_rhs,
    set<int>* input_cols_used, string* error_msg) {
  // Sanity check input.
  if (model_lhs == nullptr || model_rhs == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Null input variables in GetModel(). Check API in "
                    "call stack and try again. Aborting.";
    }
    return false;
  }

  // Will need the following set of titles to make sure all the user-
  // entered variables used in the model match one of the variables
  // from the header line of the input file.
  map<string, int> titles;
  for (int i = 0; i < variable_names.size(); ++i) {
    titles.insert(make_pair(variable_names[i], i));
  }

  // Read User-Entered model.
  bool valid_model = false;
  while (!valid_model) {
    string err_msg = "";
    if (!PromptForAndParseModel(true, model_type, model_lhs, model_rhs, &err_msg)) {
      cout << "ERROR: " << err_msg << "\n";
      continue;
    }

    // Sanity Check User-Entered Model.
    if (!SanityCheckModel(
            model_type, titles, nominal_columns, *model_lhs, *model_rhs,
            input_cols_used, &err_msg)) {
      cout << "ERROR: " << err_msg << "\n";
      continue;
    }
    valid_model = true;
  }

  return true; 
}

bool ReadInput::GetModel(
    const ModelType& model_type,
    DepVarDescription* model_lhs, Expression* model_rhs, string* error_msg) {
  // Sanity check input.
  if (model_lhs == nullptr || model_rhs == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Null input variables in GetModel(). Check API in "
                    "call stack and try again. Aborting.";
    }
    return false;
  }

  // Read User-Entered model.
  bool valid_model = false;
  while (!valid_model) {
    string err_msg = "";
    if (!PromptForAndParseModel(true, model_type, model_lhs, model_rhs, &err_msg)) {
      cout << "ERROR: " << err_msg << "\n";
      continue;
    }
    valid_model = true;
  }

  return true; 
}


bool ReadInput::ProcessUserEnteredNominalColumns(
    const string& user_input, const vector<string>& variable_names,
    set<int>* nominal_columns, string* error_msg) {
  // This should never happen based on current calls to this function.
  if (user_input.empty()) return false;

  // Check the special tokens that represent the user has indicated
  // that all input variables are numeric. In this case, nothing to do.
  if (user_input == "NONE" || user_input == "None" || user_input == "none") {
    return true;
  }
  
  // Sanity check inputs.
  if (variable_names.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: No variable names found. Check that the first line "
                    "in your input file is appropriately formatted.\n";
    }
    return false;
  }
  if (nominal_columns == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: NULL 'nominal_columns'. Check API in call to "
                    "ProcessUserEnteredNominalColumns.\n";
    }
    return false;
  }

  // Clear nominal_columns.
  nominal_columns->clear();

  // Parse user_input.
  vector<string> new_nominal_columns;
  if (!Split(user_input, ", ", &new_nominal_columns)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unreadable user input.\n";
    }
    return false;
  }

  // Iterate through the user-input column names, and match them
  // to the column indices of the header line of the input file.
  for (int i = 0; i < new_nominal_columns.size(); ++i) {
    bool found_match = false;
    for (int j = 0; j < variable_names.size(); ++j) {
      if (new_nominal_columns[i] == variable_names[j]) {
        nominal_columns->insert(j);
        found_match = true;
        break;
      }
    }
    if (!found_match) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to find '" + new_nominal_columns[i] +
                      "' as one of the variable names in the header "
                      "line of the input file.\n";
      }
      return false;
    }
  }

  return true;
}

bool ReadInput::GetNominalColumns(
    const vector<string>& variable_names,
    vector<vector<DataHolder> >& data_values,
    map<int, set<string> >* nominal_columns, string* error_msg) {
  // Sanity Check input.
  if (variable_names.empty() || nominal_columns == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in GetNominalColumns: Null input.\n";
    }
    return false;
  }

  // Get the columns that have been inferred to be NOMINAL (based on
  // presence of '$' or a non-double value).
  string nominal_columns_detected;
  if (!nominal_columns->empty()) {
    string nominal_column_names;
    for (map<int, set<string> >::const_iterator itr = nominal_columns->begin();
         itr != nominal_columns->end(); ++itr) {
      if (itr->first >= variable_names.size()) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in processing header line: column index is "
                        "out of bounds. Aborting.\n";
        }
        return false;
      }
      if (!nominal_column_names.empty()) {
        nominal_column_names += ", ";
      }
      nominal_column_names += variable_names[itr->first];
    }
    nominal_columns_detected =
        "\n############################################################"
        "\nNominal Columns detected:\n" + nominal_column_names + "\n";
  } else {
    nominal_columns_detected =
        "\n############################################################"
        "\nNo nominal columns detected.\n";
  }

  // Display inferred NOMINAL columns to user.
  const string nominal_col_input_request = 
       "\nTo accept the above listed columns as the nominal ones, press "
       "<ENTER>.\nOtherwise, enter (comma-seperated) nominal columns, "
       "or enter 'NONE' if all variables are numeric\n(column "
       "name must exactly match input file, and is case-sensitive):\n> ";
  cout << nominal_columns_detected << nominal_col_input_request;

  // Prompt user for actual NOMINAL columns.
  set<int> new_nominal_columns;
  bool valid_input = false;
  string user_response;
  char temp;
  while (!valid_input) {
    user_response = "";
    cin.get(temp);
    while (temp != '\n') {
      user_response += temp;
      cin.get(temp);
    }
    string input_error = "";
    if (user_response.empty()) {
      // User has accepted the detected nominal columns.
      valid_input = true;
    } else if (ProcessUserEnteredNominalColumns(
          user_response, variable_names, &new_nominal_columns, &input_error)) {
      // Update set of nominal columns based on user input.
      string input_error_str = "";
      if (!UpdateNominalColumns(
              data_values, new_nominal_columns, nominal_columns, &input_error_str)) {
        cout << "\nUnable to honor your selection for NOMINAL variables: "
             << input_error_str << "\n"
             << nominal_columns_detected << nominal_col_input_request; 
      } else {
        valid_input = true;
      }
    } else {
      cout << "\nUnable to process your response: " << input_error << "\n"
           << nominal_columns_detected << nominal_col_input_request; 
    }
  }

  return true;
}

bool ReadInput::UpdateNominalColumns(
    vector<vector<DataHolder> >& data_values,
    const set<int>& new_columns,
    map<int, set<string> >* columns,
    string* error_msg) {
  // Go through 'columns', checking for indices that are not present in
  // 'new_columns'. If any are found, make sure that all the values in
  // the corresponding set can be treated as a double: if so, remove the
  // (index, set) from 'columns'; otherwise, log an error and return false.
  set<int> columns_to_delete;
  for (map<int, set<string> >::const_iterator old_itr = columns->begin();
       old_itr != columns->end(); ++old_itr) {
    if (new_columns.find(old_itr->first) == new_columns.end()) {
      // Old index does not appear in new_columns. Make sure all values can
      // be represented as a double, and update data_values accordingly.
      for (set<string>::const_iterator value_itr = (old_itr->second).begin();
           value_itr != (old_itr->second).end(); ++value_itr) {
        double temp;
        if (!Stod(*value_itr, &temp)) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR: Column " + Itoa(1 + old_itr->first) +
                          " was not entered as a NOMINAL column, but value '" +
                          *value_itr + "' (which is not parsable as a numerical "
                          "value) appears in one of the data rows for this column.";
          }
          return false;
        }
      }
      // All values in the column are parsable as a double. Update data_values
      // to take the double versions of all values in this column.
      for (vector<vector<DataHolder> >::iterator data_itr = data_values.begin();
           data_itr != data_values.end(); ++data_itr) {
        DataHolder& holder = (*data_itr)[old_itr->first];
        if (holder.name_.empty() ||
            !Stod(holder.name_, &(holder.value_))) {
          if (error_msg != nullptr) {
            *error_msg += "Unknown ERROR: unable to interpret '" + holder.name_ +
                          "' from Column " + Itoa(1 + old_itr->first) +
                          " as a numerical value.";
          }
          return false;
        }
        holder.name_ = "";
        holder.type_ = DataType::DATA_TYPE_NUMERIC;
      }
      columns_to_delete.insert(old_itr->first);
    }
  }

  // Remove extra columns from 'old_columns' (could've done this in above loop,
  // but didn't want to mess up the iteration through 'old_columns' be deleting
  // elements while iterating through it).
  for (set<int>::const_iterator itr = columns_to_delete.begin();
       itr != columns_to_delete.end(); ++itr) {
    columns->erase(*itr);
  }

  // As of now, 'old_columns' is a subset of 'columns'. If they are equal, we're done,
  // and can return.
  if (columns->size() == new_columns.size()) return true;

  // 'old_columns' is a proper subset of 'columns'. For all the new columns, we
  // have to populate 'old_columns' with this index, together with a set that
  // represents all of the distinct values in that column.
  // First, get a list of all the new columns to add.
  set<int> new_columns_to_add;
  for (set<int>::const_iterator new_itr = new_columns.begin();
       new_itr != new_columns.end(); ++new_itr) {
    if (columns->find(*new_itr) == columns->end()) {
      new_columns_to_add.insert(*new_itr);
    }
  }

  // For each column in 'new_columns_to_add', create a set of all the distinct
  // values in this column, and populate 'columns' with it and column index.
  for (vector<vector<DataHolder> >::iterator data_itr = data_values.begin();
       data_itr != data_values.end(); ++ data_itr) {
    for (set<int>::const_iterator itr = new_columns_to_add.begin();
         itr != new_columns_to_add.end(); ++itr) {
      // Sanity check, should never be true.
      if (*itr >= data_itr->size()) {
        if (error_msg != nullptr) {
          *error_msg += "Unknown ERROR: column " + Itoa(*itr) +
                        " is outside of the range of input variables (" +
                        Itoa(static_cast<int>(data_itr->size())) + ").\n";
        }
        return false;
      }

      // Get data value for this (row, column).
      DataHolder& holder = (*data_itr)[*itr];
      if (!holder.name_.empty() || holder.type_ != DataType::DATA_TYPE_NUMERIC) {
        if (error_msg != nullptr) {
          *error_msg += "Unknown ERROR: column " + Itoa(*itr) +
                        " has a stored data value that is a string (" +
                        holder.name_ + "), even though column was "
                        "hitherto considered a non-NOMINAL column.\n";
        }
        return false;
      }

      // Update data value to be a string (instead of a numerical value).
      holder.name_ = Itoa(holder.value_);
      holder.type_ = DataType::DATA_TYPE_STRING;

      // Update 'columns' by adding the new string value to the set.
      map<int, set<string> >::iterator old_itr = columns->find(*itr);
      if (old_itr == columns->end()) {
        // This is the first value being added for this column.
        set<string> value_to_add;
        value_to_add.insert(holder.name_);
        columns->insert(make_pair(*itr, value_to_add));
      } else {
        // This column was already known to be nominal. Add the new value to the
        // set of values for this column.
        set<string>& existing_values = old_itr->second;
        existing_values.insert(holder.name_);
      }
    }
  }

  return true;
}

bool ReadInput::ParseSampleId(
    const int orig_id_col, const set<int>& input_cols_used,
    const vector<DataHolder>& sample_values,
    vector<string>* ids, string* error_msg) {
  int id_col;
  if (!OrigColToDataValuesCol(
          input_cols_used, orig_id_col, &id_col, error_msg)) {
    return false;
  }
  if (id_col < 0 || ids == nullptr) return true;
  if (id_col >= sample_values.size()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing Sample Id: id_col (" + Itoa(id_col) +
                    ") is greater than number of columns in data (" +
                    Itoa(static_cast<int>(sample_values.size())) + ")";
    }
    return false;
  }

  ids->push_back(GetDataHolderString(sample_values[id_col], true, 17));
  return true;
}

bool ReadInput::ParseSampleWeight(
    const int orig_weight_col, const set<int>& input_cols_used,
    const vector<DataHolder>& sample_values,
    vector<double>* weights, string* error_msg) {
  int weight_col;
  if (!OrigColToDataValuesCol(
          input_cols_used, orig_weight_col, &weight_col, error_msg)) {
    return false;
  }
  if (weight_col < 0 || weights == nullptr) return true;
  if (weight_col >= sample_values.size()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing Sample Weight: weight_col (" +
                    Itoa(weight_col) + ") is greater than number of "
                    "columns in data (" +
                    Itoa(static_cast<int>(sample_values.size())) + ")";
    }
    return false;
  }

  if (sample_values[weight_col].type_ != DataType::DATA_TYPE_NUMERIC) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing Sample Weight: DataType is not NUMERIC: " +
                    Itoa(static_cast<int>(sample_values[weight_col].type_)) + "\n";
    }
    return false;
  }

  weights->push_back(sample_values[weight_col].value_);
  return true;
}

bool ReadInput::ParseSampleFamilies(
    const vector<VariableColumn>& orig_family_cols,
    const set<int>& input_cols_used,
    const vector<DataHolder>& sample_values,
    vector<vector<string>>* families, string* error_msg) {
  if (orig_family_cols.size() < 0 || families == nullptr) return true;
  families->push_back(vector<string>());
  vector<string>& sample_families = families->back();
  for (const VariableColumn& orig_family_col : orig_family_cols) {
    int col_index;
    if (!OrigColToDataValuesCol(
            input_cols_used, orig_family_col.index_, &col_index, error_msg)) {
      return false;
    }
    if (col_index < 0 || col_index >= sample_values.size()) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing Sample Families: col_index (" +
                      Itoa(col_index) + ") is greater than number of "
                      "columns in data (" +
                      Itoa(static_cast<int>(sample_values.size())) + ")";
      }
      return false;
    }
    sample_families.push_back(
        GetDataHolderString(sample_values[col_index], true, 17));
  }
  return true;
}

bool ReadInput::ComputeRowDependentVariableValues(
    const ModelType& model_type,
    const vector<DataHolder>& sample_values,
    const map<string, int>& name_to_column,
    const map<int, set<string> >& nominal_columns,
    const DepVarDescription& dependent_var,
    DepVarHolder* dep_var, string* error_msg) {
  // Create the mapping of variable name to value.
  map<string, double> var_values;
  if (!GetVariableValuesFromDataRow(
          nominal_columns, name_to_column, sample_values,
          &var_values, error_msg)) {
    return false;
  }

  // Parse Dependent Variable(s), according to model_type.
  if (model_type == ModelType::MODEL_TYPE_LINEAR) {
    dep_var->dep_vars_linear_.push_back(double());
    if (!EvaluateExpression(
            dependent_var.model_lhs_linear_, var_values,
            &dep_var->dep_vars_linear_.back(), error_msg)) {
      return false;
    }
    return true;
  } else if (model_type == ModelType::MODEL_TYPE_LOGISTIC) {
    double logistic_value;
    if (!EvaluateExpression(
            dependent_var.model_lhs_logistic_, var_values,
            &logistic_value, error_msg)) {
      return false;
    }
    dep_var->dep_vars_logistic_.push_back(logistic_value == 1.0);
  } else if (model_type == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL) {
    // Sanity check input.
    if (dependent_var.model_lhs_cox_.size() != 2 &&
        dependent_var.model_lhs_cox_.size() != 3) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing Dependent Variables: Too many terms (" +
                      Itoa(static_cast<int>(dependent_var.model_lhs_cox_.size())) +
                      ") in dependent_var.\n";
      }
      return false;
    }

    // Parse the Dependent variables based on their values.
    dep_var->dep_vars_cox_.push_back(CensoringData());
    CensoringData& data = dep_var->dep_vars_cox_.back();

    // Parse Time (may be Survival Time or Censoring Time, depending on format
    // of dependent_var.model_lhs_cox_).
    double time;
    if (!EvaluateExpression(
            dependent_var.model_lhs_cox_[0], var_values,
            &time, error_msg)) {
      return false;
    }

    // Parse Status variable, and sanity-check it is True or False, or the
    // dummy value of -1.
    double status;
    if (!EvaluateExpression(
            dependent_var.model_lhs_cox_.back(), var_values,
            &status, error_msg)) {
      return false;
    }
    if (status != 0.0 && status != 1.0 && status != -1.0) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Status should be 0.0 (false) or 1.0 (true); found: " +
                      Itoa(status) + ". NOTE: If you want to specify (survival "
                      "time, censoring time), you need to also include status "
                      "as a third argument (set it to -1, which indicates it "
                      "should be ignored and computed as min(survival time, "
                      "censoring time)).\n";
      }
      return false;
    }

    // Parse Censoring Time, if available. Then combine Time, Censoring Time,
    // and Status into a single CensoringData object.
    if (dependent_var.model_lhs_cox_.size() == 3) {
      double censoring_time;
      if (!EvaluateExpression(
              dependent_var.model_lhs_cox_[1], var_values,
              &censoring_time, error_msg)) {
        return false;
      }
      data.survival_time_ = time;
      data.censoring_time_ = censoring_time;
      if (FloatEq(status, -1.0)) {
        // -1 Indicates that we should determine status based on survival and
        // censoring times.
        data.is_alive_ = (data.survival_time_ > data.censoring_time_);
      } else if (FloatEq(status, 0.0)) {
        if (data.survival_time_ > data.censoring_time_) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR: Status indicates not alive, but survival time (" +
                          Itoa(data.survival_time_) + ") is larger than censoring time (" +
                          Itoa(data.censoring_time_) + ").\n";
          }
          return false;
        }
        data.is_alive_ = true;
      } else if (FloatEq(status, 1.0)) {
        if (data.survival_time_ < data.censoring_time_) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR: Status indicates alive, but survival time (" +
                          Itoa(data.survival_time_) + ") is less than censoring time (" +
                          Itoa(data.censoring_time_) + ").\n";
          }
          return false;
        }
        data.is_alive_ = false;
      }
    } else {
      if (status == -1.0) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Status should be 0.0 (false) or 1.0 (true); found: " +
                        Itoa(status) + ". NOTE: If you want to specify (survival "
                        "time, censoring time), you need to also include status "
                        "as a third argument (set it to -1, which indicates it "
                        "should be ignored and computed as min(survival time, "
                        "censoring time)).\n";
        }
        return false;
      }
      data.is_alive_ = status == 0.0;
      if (data.is_alive_) {
        data.censoring_time_ = time;
        // Artificially set survival time to be bigger than censoring_time_,
        // to ensure it is not used: All functions using CensoringData should
        // look at is_alive_ to determine which (among survival_time_ and
        // censoring_time_) is used; this is just a safeguard, in case there
        // are accidentally any places that use the minimum of the two, in
        // which case we don't want the default value of 0.0 for an
        // uninitialized survival_time_ to be the min.
        data.survival_time_ = time + 1.0;
      } else {
        data.survival_time_ = time;
        // Artificially set censoring_time_ to be bigger than survival_time_,
        // for same reason as above (in reverse).
        data.censoring_time_ = time + 1.0;
      }
    }

    // Parse left-truncation time, if appropriate.
    if (!dependent_var.left_truncation_name_.empty()) {
      map<string, double>::const_iterator itr =
          var_values.find(dependent_var.left_truncation_name_);
      if (itr == var_values.end()) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unable to find value for Left-truncation column.\n";
        }
        return false;
      }
      const double left_truncation_time = itr->second;

      // Sanity-Check the left-truncation time is positive, and is no bigger
      // than the survival/censoring time.
      if (left_truncation_time < 0.0) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Invalid Left-truncation time: " +
                        Itoa(left_truncation_time) + ".\n";
        }
        return false;
      } else if (data.is_alive_ && left_truncation_time > data.censoring_time_) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Invalid Left-truncation time: " +
                        Itoa(left_truncation_time) + " is larger than the "
                        "Censoring Time (" + Itoa(data.censoring_time_) + ").\n";
        }
        return false;
      } else if (!data.is_alive_ && left_truncation_time > data.survival_time_) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Invalid Left-truncation time: " +
                        Itoa(left_truncation_time) + " is larger than the "
                        "Survival Time (" + Itoa(data.survival_time_) + ").\n";
        }
        return false;
      }

      // Store left-truncation time.
      data.left_truncation_time_ = left_truncation_time;
    }

    return true;
  } else if (model_type == ModelType::MODEL_TYPE_INTERVAL_CENSORED) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in parsing dep variable value: Unexpected model type "
                    "MODEL_TYPE_INTERVAL_CENSORED (At this point, this model "
                    "type should have been refined to either the time-dep or "
                    "time-indep type.\n";
    }
    return false;
  } else if (
        model_type == ModelType::MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED ||
        model_type == ModelType::MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in parsing dep variable value: Unexpected model type "
                    "MODEL_TYPE_TIME_[IN]DEPENDENT_INTERVAL_CENSORED: the "
                    "ComputeRowDependentVariableValues() function, called by "
                    "StoreDataValuesInParams(), should not be used for "
                    "Interval-Censored NPMLE (rather, use PopulateSubjectInfo(), "
                    "perhaps via ReadTime[In]DepIntervalCensoredData::ReadFile(), "
                    "to parse Dependent (and independent) variable values.\n";
    }
    return false;
  } else {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in parsing dep variable value: Unexpected model type " +
                    Itoa(static_cast<int>(model_type)) + ".\n";
    }
    return false;
  }
 
  return true;
}

bool ReadInput::StandardizeLinearTerms(
    const VariableNormalization standardize_vars,
    const set<int>& non_binary_linear_terms,
    const vector<double>& linear_terms_sums,
    const vector<double>& linear_terms_sums_squared,
    vector<tuple<bool, double, double>>* linear_terms_mean_and_std_dev,
    MatrixXd* linear_terms, string* error_msg) {
  // Early abort if standardization shouldn't be done.
  if (standardize_vars == VariableNormalization::VAR_NORM_NONE) return true;
  if (linear_terms == nullptr ||
      linear_terms_mean_and_std_dev == nullptr) return false;

  // Compute values for each linear term (for each row), and keep
  // running total (sum) and it's square across all rows. Also keep track
  // if the linear term had any values that weren't 0 or 1 (we treat
  // binary linear terms differently).
  const int p = linear_terms->cols();
  const int num_rows = linear_terms->rows();
  if (num_rows == 0 || p == 0 || linear_terms_sums.size() != p ||
      linear_terms_sums_squared.size() != p) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Mismatching sizes in StandardizeLinearTerms.\n";
    }
    return false;
  }

  const bool do_population_variance =
      standardize_vars == VAR_NORM_STD ||
      standardize_vars == VAR_NORM_STD_NON_BINARY;
  const bool do_sample_variance =
      standardize_vars == VAR_NORM_STD_W_N_MINUS_ONE ||
      standardize_vars == VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;
  const bool standardize_linear_terms =
      do_population_variance || do_sample_variance;
  const bool standardize_non_binary_only =
      standardize_vars == VAR_NORM_STD_NON_BINARY ||
      standardize_vars == VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;

  if (num_rows == 0 ||
      (num_rows < 2 && do_sample_variance)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Cannot find mean and standard deviation "
                    "of data, as there were no rows that were kept ("
                    "after skipping rows with missing values).\n";
    }
    return false;
  }

  // Compute mean and standard deviation for each linear term.
  map<int, pair<double, double>> linear_term_to_mean_and_std_dev;
  for (int linear_term_index = 0; linear_term_index < p; ++linear_term_index) {
    const bool is_binary =
        non_binary_linear_terms.find(linear_term_index) ==
        non_binary_linear_terms.end();
    const double& sum = linear_terms_sums[linear_term_index];
    const double& sum_squared = linear_terms_sums_squared[linear_term_index];
    const double mean = sum / num_rows;
    const double denominator =
        do_sample_variance ? num_rows - 1 : num_rows;
    const double std_dev =
        sqrt((sum_squared - (sum * sum) / num_rows) / denominator);
    linear_terms_mean_and_std_dev->push_back(make_tuple(!is_binary, mean, std_dev));
    if (standardize_linear_terms &&
        (!standardize_non_binary_only || !is_binary)) {
      linear_term_to_mean_and_std_dev.insert(
          make_pair(linear_term_index, make_pair(mean, std_dev)));
    }
  }

  if (linear_term_to_mean_and_std_dev.empty()) return true;

  // Go through data, standardizing each value.
  for (int row = 0; row < num_rows; ++row) {
    for (const pair<int, pair<double, double>>& term_index_to_stats :
         linear_term_to_mean_and_std_dev) {
      const int col = term_index_to_stats.first;
      const double& mean = term_index_to_stats.second.first;
      const double& std_dev = term_index_to_stats.second.second;
      if (col >= p) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unable to standardize linear term with index " +
                        Itoa(col) + ", which is bigger than the number of "
                        "linear terms (" + Itoa(p) + ").\n";
        }
        return false;
      }
      if (std_dev == 0.0) continue;
      const double old_value = (*linear_terms)(row, col);
      (*linear_terms)(row, col) = (old_value - mean) / std_dev;
    }
  }

  return true;
}

bool ReadInput::StoreDataValuesInParams(
    const bool is_simulated_data,
    const vector<string>& data_values_header,
    const vector<vector<DataHolder> >& data_values,
    ModelAndDataParams* input) {
  // Get a mapping from the variable title to the column in the input data
  // that the title corresponds to.
  map<string, int> name_to_column;
  for (int i = 0; i < data_values_header.size(); ++i) {
    name_to_column.insert(make_pair(data_values_header[i], i));
  }

  // Get a mapping from the original column index (from input data file)
  // to the column index (w.r.t. data_values). This will be needed to determine
  // which rows in na_rows_and_columns_ should be skipped (only skip a row
  // if it has a NA value in a used data column).
  map<int, int> data_values_to_orig_data_col_map;
  for (int i = 0; i < data_values_header.size(); ++i) {
    if (is_simulated_data) {
      input->header_.push_back(data_values_header[i]);
    } else {
      for (int j = 0; j < input->header_.size(); ++j) {
        if (data_values_header[i] == input->header_[j]) {
          data_values_to_orig_data_col_map.insert(make_pair(i, j));
          break;
        }
      }
      // Make sure a (unique) match was found.
      if (data_values_to_orig_data_col_map.size() != i + 1) {
        input->error_msg_ +=
            "ERROR: Unable to map column index w.r.t. data_values "
            "to column index w.r.t. original data file, for column " +
            Itoa(i + 1) + ": '" + data_values_header[i] +
            "'. Data values header: {" + Join(data_values_header, ", ") +
            "}; input->header_: {" + Join(input->header_, ", ") + "}\n";
        return false;
      }
    }
  }

  // Linear and Logistic Regression uses a constant term on the RHS of the
  // Regression model, Cox Proportional Hazards Model does not.
  const bool no_constant_term =
      (input->model_type_ == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL ||
       input->model_type_ == ModelType::MODEL_TYPE_INTERVAL_CENSORED ||
       input->model_type_ == ModelType::MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED ||
       input->model_type_ == ModelType::MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED);

  // Expand model RHS to the set of all linear terms. This consists of two steps:
  //   1) Parse Expression for model RHS, separating out the linear terms
  //   2) Expand nominal variables to the appropriate number of linear terms.
  vector<Expression> linear_terms;
  if (!GetLegendAndLinearTerms(
          !no_constant_term, name_to_column,
          // Will need to expand I_Subgroup into (#Subgroups - 1) variables.
          input->use_subgroup_as_covariate_ ? input->subgroups_ : vector<vector<string>>(),
          input->nominal_columns_and_values_, input->model_rhs_,
          &input->orig_linear_terms_to_legend_indices_,
          &linear_terms, &input->legend_, &input->error_msg_)) {
    return false;
  }

  // Get a mapping between the original column names/variables (from the input
  // data file) to all of the linear terms that contain that Variable.
  if (!GetLinearTermsInvolvingEachColumn(
          linear_terms, data_values_header, input->header_,
          Keys(input->nominal_columns_and_values_),
          &input->header_index_to_legend_indices_)) {
    cout << "WARNING: Failed to construct header_index_to_legend_indices_ "
         << "field. We should abort here, but since this is a new function "
         << "that hasn't been thoroughly tested yet, we just print a "
         << "warning and proceed, so as not to stop the rest of the "
         << "this method from failing." << endl;
  }

  // Get the set of variables found among the dependent term(s).
  set<string> vars_in_dep_terms;
  if (!ExtractVariablesFromExpression(
          input->model_type_, input->model_lhs_,
          &vars_in_dep_terms, &input->error_msg_)) {
    return false;
  }
  // Pick out the sub-map of name_to_column that just contains info for
  // dependent vars (i.e. those in LHS of model).
  map<string, int> dep_var_name_to_column;
  if (!is_simulated_data) {
    for (const string& var_name : vars_in_dep_terms) {
      int* col_index = FindOrNull(var_name, name_to_column);
      if (col_index == nullptr) {
        input->error_msg_ += "ERROR: Unable to find variable '" + var_name +
                             "' (from the LHS of the model) among the titles "
                             "on the header line of the input data file.\n";
        return false;
      }
      dep_var_name_to_column.insert(make_pair(var_name, *col_index));
    }
  }

  // Get the set of variables found among the linear_terms.
  set<string> vars_in_linear_terms;
  if (!ExtractVariablesFromExpression(
          input->model_rhs_, &vars_in_linear_terms, &input->error_msg_)) {
    return false;
  }
  // Pick out the sub-map of name_to_column that just contains info for
  // indep vars (i.e. those in RHS of model).
  map<string, int> indep_var_name_to_column;
  for (const string& var_name : vars_in_linear_terms) {
    int* col_index = FindOrNull(var_name, name_to_column);
    if (col_index == nullptr) {
      if (var_name == kSubgroupString) {
        // "I_Subgroup" is (likely) not one of the columns in the input data,
        // but rather, it will be computed based on values in other columns.
        // The 'indep_var_name_to_column' is used once below to compute each
        // linear term's values; but this is done separately by ParseSubgroups()
        // and then within ComputeRowCovariateValues() below; so it is okay not
        // to add any column index corresponding to "I_Subgroup" to
        // indep_var_name_to_column.
        continue;
      } else {
        input->error_msg_ += "ERROR: Unable to find variable '" + var_name +
                             "' (from the RHS of the model) among the titles "
                             "on the header line of the input data file.\n";
        return false;
      }
    }
    indep_var_name_to_column.insert(make_pair(var_name, *col_index));
  }

  // Go through all the data values (by sample), and compute each linear term.
  input->ids_.clear();
  input->weights_.clear();
  input->families_.clear();
  input->linear_term_values_.resize(data_values.size(), linear_terms.size());
  vector<vector<string>> strata_names;
  int current_row = 0;
  vector<double> linear_terms_sums(linear_terms.size(), 0.0);
  vector<double> linear_terms_sums_squared(linear_terms.size(), 0.0);
  set<int> non_binary_linear_terms;
  for (int i = 0; i < data_values.size(); ++i) {
    const vector<DataHolder>& sample_values = data_values[i];

    // Skip data rows that have 'NA_STRING' in at least one column that is
    // needed for the model.
    bool keep_row = true;
    map<int, set<int> >::const_iterator na_columns_itr =
        input->na_rows_and_columns_.find(i);
    if (na_columns_itr != input->na_rows_and_columns_.end()) {
      for (const int na_column : na_columns_itr->second) {
        const int orig_column_index =
            data_values_to_orig_data_col_map[na_column];
        if (input->input_cols_used_.find(orig_column_index) !=
            input->input_cols_used_.end()) {
          keep_row = false;
          break;
        }
      }
    }
    if (!keep_row) {
      input->na_rows_skipped_.insert(i + 1);
      continue;
    }

    // Fetch the subgroup index for this row, if necessary. Also determines
    // the subgroup_index (which of the user-entered subgroups the data in
    // this row matches, if any), which will be used (if
    // input->use_subgroup_as_covariate_ is true) to determine the value
    // of the Subgroup Indicator covariate.
    int subgroup_index = -1;
    if (!ParseSubgroup(
            data_values_header, input->subgroup_cols_, input->subgroups_,
            sample_values, &subgroup_index)) {
      return false;
    }
    // If subgroup_index was not set, then this row does not belong to
    // the subgroup.
    if (!input->subgroup_cols_.empty() && subgroup_index == -1) {
      continue;
    } else if (!input->subgroup_cols_.empty()) {
      map<int, set<int>>::iterator subgroup_itr =
          input->subgroup_rows_per_index_.find(subgroup_index);
      if (subgroup_itr == input->subgroup_rows_per_index_.end()) {
        set<int> first_row_index_for_subgroup;
        first_row_index_for_subgroup.insert(i + 1);
        input->subgroup_rows_per_index_.insert(make_pair(
              subgroup_index, first_row_index_for_subgroup));
      } else {
        (subgroup_itr->second).insert(i + 1);
      }
    }

    // Fetch the strata for this row, if necessary.
    if (!input->strata_cols_.empty()) {
      int strata_index = -1;
      if (!ParseStrata(i, data_values_header, input->strata_cols_, sample_values,
                       &strata_index, &strata_names, &input->row_to_strata_)) {
        return false;
      }
    }

    // Grab Id of the Sample corresponding to this row (if appropriate).
    if (!ParseSampleId(input->id_col_.index_, input->input_cols_used_,
                       sample_values, &input->ids_, &input->error_msg_)) {
      input->error_msg_ += "ERROR in Parsing Sample Id for data row: " +
                           Itoa(i + 1) + ".\n";
      return false;
    }

    // Get Weight for this Sample (if appropriate).
    if (!ParseSampleWeight(input->weight_col_.index_, input->input_cols_used_,
                           sample_values, &input->weights_, &input->error_msg_)) {
      input->error_msg_ += "ERROR in Parsing Sample Weight for data row: " +
                           Itoa(i + 1) + ".\n";
      return false;
    }

    // Get Families for this Sample (if appropriate).
    if (!ParseSampleFamilies(input->family_cols_, input->input_cols_used_,
                             sample_values, &input->families_, &input->error_msg_)) {
      input->error_msg_ += "ERROR in Parsing Sample Families for data row: " +
                           Itoa(i + 1) + ".\n";
      return false;
    }

    // Generate the values for the dependent variable(s).
    if (!is_simulated_data &&
        !ComputeRowDependentVariableValues(
            input->model_type_, sample_values, dep_var_name_to_column,
            input->nominal_columns_and_values_, input->model_lhs_,
            &(input->dep_vars_), &input->error_msg_)) {
      input->error_msg_ += "ERROR in Parsing Dependent Variable on data row: " +
                           Itoa(i + 1) + ".\n";
      return false;
    }

    // Generate the vector of values for the linear terms.
    VectorXd temp_row_values;
    if (!ComputeRowCovariateValues(
            subgroup_index, input->subgroups_,
            // Shouldn't reach this point for rows that have a missing
            // value in a relevant column (skipped such rows above);
            // so ComputeRowCovariateValues() shouldn't encounter
            // any missing values.
            set<string>(), input->nominal_columns_and_values_,
            linear_terms, indep_var_name_to_column, sample_values,
            &temp_row_values, &input->error_msg_)) {
      input->error_msg_ +=
           "ERROR reading data values for row: " + Itoa(i + 1) + "\n";
      return false;
    }
    // Update cumulative sums for each linear term (to be used below for
    // standardization).
    for (int p = 0; p < temp_row_values.size(); ++p) {
      const double& term_value = temp_row_values(p);
      linear_terms_sums[p] += term_value;
      linear_terms_sums_squared[p] +=  term_value * term_value;
      if (term_value != 0.0 && term_value != 1.0) {
        non_binary_linear_terms.insert(p);
      }
    }
    input->linear_term_values_.row(current_row) = temp_row_values;
    current_row++;
  }

  // If not all data rows were taken (e.g. skipped rows with missing value, or
  // rows not in the subgroup), then need to resize the matrix of linear term values.
  input->linear_term_values_.conservativeResize(current_row, Eigen::NoChange_t::NoChange);

  // Update input->linear_term_values_ by finding mean, std_dev from
  // linear_term_sums[_squared], and standardize values.
  if (!StandardizeLinearTerms(
          input->standardize_vars_, non_binary_linear_terms,
          linear_terms_sums, linear_terms_sums_squared,
          &input->linear_terms_mean_and_std_dev_,
          &input->linear_term_values_, &input->error_msg_)) {
    input->error_msg_ += "ERROR in standardizing linear terms.\n";
    return false;
  }

  // linear_terms created data on the heap. Remove it now.
  for (Expression& linear_term : linear_terms) {
    DeleteExpression(linear_term);
  }

  // Print out final model.
  PrintFinalModel(
      input->model_type_, input->model_lhs_,
      input->legend_, &input->final_model_);

  return true;
}

bool ReadInput::FillModelAndDataParams(ModelAndDataParams* params) {
  if (params == nullptr) return false;

  // Fill Category (2) fields.
  if (!ParseModelAndDataParams(params)) return false;

  // Fill Category (3) fields.
  vector<string> data_values_header;
  vector<vector<DataHolder>> data_values;
  if (!ReadTableWithHeader::ReadDataFile(
          params->file_, params->header_, params->input_cols_used_, params->var_params_,
          &data_values_header, &data_values, &params->nominal_columns_and_values_,
          &params->na_rows_and_columns_, &params->error_msg_)) {
    return false;
  }

  // Fill Category (4) fields.
  if (!StoreDataValuesInParams(data_values_header, data_values, params)) return false;

  return true;
}

}  // namespace file_reader_utils
