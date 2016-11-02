// Date: March 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "read_file_utils.h"

#include "FileReaderUtils/command_line_utils.h"
#include "FileReaderUtils/read_file_structures.h"
#include "MapUtils/map_utils.h"
#include "MathUtils/eq_solver.h"
#include "TestUtils/test_utils.h"

#include <fstream>   // For sprintf and fstream.
#include <string>

// Start "ls" block: This block is necessary to do a system-independent "ls".
// Required for GetWorkingDirectory() and FilesInDirectory().
#ifdef WINDOWS
  #include <Windows.h>
#else
  #include <dirent.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif
// End "ls" block.

using Eigen::MatrixXd;
using Eigen::NoChange_t;
using Eigen::VectorXd;
using namespace map_utils;
using namespace test_utils;

namespace file_reader_utils {

/* ================================ Constants =============================== */

const set<string> NA_STRINGS {
  "NA", "N/A", "na", "n/a", ""
};

static const char kSubgroupIndicator[] = "I_Subgroup";

/* ============================== END Constants ============================= */

/* =============================== Functions ================================ */
string GetFileName(const string& filepath) {
  size_t last_dir = filepath.find_last_of("/");
  if (last_dir == string::npos) return filepath;
  if (last_dir == filepath.length() - 1) return "";
  return filepath.substr(last_dir + 1);
}

string GetDirectory(const string& filepath) {
  size_t last_dir = filepath.find_last_of("/");
  if (last_dir == string::npos) return ".";
  return filepath.substr(0, last_dir);
}

string GetWorkingDirectory() {
  // Get working dir, either via GetModuleFileName (if working on WINDOWS)
  // or via getcwd otherwise.
  char dir_name[FILENAME_MAX];
  #ifdef WINDOWS
    GetModuleFileName(nullptr, dir_name, FILENAME_MAX);
  #else
    getcwd(dir_name, FILENAME_MAX);
  #endif
  return string(dir_name);
}

bool GetFilesInDirectory(
    const bool full_path, const string& directory, vector<string>* files) {
  if (files == nullptr) return false;

  // We don't know if user passed in trailing "/" or not; also, handle
  // case directory is empty.
  string dir = directory;
  if (dir.empty()) {
    dir = ".";
  } else if (directory.substr(directory.length() - 1) == "/") {
    dir = directory.substr(0, directory.length() - 1);
  }

  // Get files in dir, either via Windows' Find[First|Next]File() or
  // linux's [open | read]dir.
  #ifdef WINDOWS
    // First check directory exists.
    DWORD file_attribute = GetFileAttributesA(dir.c_str());
    if (file_attribute == INVALID_FILE_ATTRIBUTES /* Invalid path structure */ ||
        !(file_attribute & FILE_ATTRIBUTE_DIRECTORY) /* No such directory */) {
      return false;
    }

    // Check if there are any files in this directory.
    const string dir_regexp = dir + "*";
    WIN32_FIND_DATA file_data;
    HANDLE dir_handle = FindFirstFile(dir_regexp.c_str(), &file_data);
    if (dir_handle == INVALID_HANDLE_VALUE) return true;  // No files in dir.

    // Directory contains at least one file. Get them all.
    do {
        const string file_name = file_data.cFileName;

        // Exclude files beginning with "." (e.g. self-reference ".", parent
        // directory "..", and "hidden" files).
        if (file_name[0] == '.') continue;

        // Exclude child directories.
        const bool is_directory =
            (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        if (is_directory) continue;

        // Add file.
        files->push_back(full_path ? dir + "/" + file_name : file_name);
    } while (FindNextFile(dir_handle, &file_data));

    // Close directory.
    FindClose(dir_handle);
  #else  // Linux.
    DIR* dir_ptr = opendir(dir.c_str());
    if (dir_ptr == nullptr) return false;  // No such directory.

    struct dirent* ent;
    while ((ent = readdir(dir_ptr)) != nullptr) {
      const string file_name = ent->d_name;
      // Exclude files beginning with "." (e.g. self-reference ".", parent
      // directory "..", and "hidden" files).
      if (file_name[0] == '.') continue;

      const string full_file_name = directory + "/" + file_name;
      
      // Exclude child directories.
      class stat st;
      if (stat(full_file_name.c_str(), &st) == -1) continue;  // Unable to get filetype.
      if ((st.st_mode & S_IFDIR) != 0) continue;  // Is Directory.

      // Add file.
      files->push_back(full_path ? dir + "/" + file_name : file_name);
    }

    // Close directory.
    closedir (dir_ptr);
  #endif

  return true;
}

bool PrintDataToFile(
    const string& filename, const string& sep, const vector<string>& header,
    const vector<vector<DataHolder> >& data_values) {
  ofstream data_file;
  data_file.open(filename);
  if (!data_file.is_open()) {
    return false;
  }
  data_file << Join(header, sep) << endl;
  for (const vector<DataHolder>& row : data_values) {
    bool is_first = true;
    for (const DataHolder& value : row) {
      if (is_first) {
        is_first = false;
      } else {
        data_file << sep;
      }
      if (value.type_ == DataType::DATA_TYPE_STRING) {
        data_file << value.name_;
      } else if (value.type_ == DataType::DATA_TYPE_NUMERIC) {
        data_file << value.value_;
      } else {
        cout << "ERROR: Unsupported Data type: " << value.type_ << endl;
        return false;
      }
    }
    data_file << endl;
  }
  data_file.close();
  return true;
}

void RemoveWindowsTrailingCharacters(string* input) {
  if (input == nullptr) return;
  if (input->length() > 0) {
    if ((*input)[input->length() - 1] == 10) {
      *input = input->substr(0, input->length() - 1);
    }
  }
  if (input->length() > 0) {
    if ((*input)[input->length() - 1] == 13) {
      *input = input->substr(0, input->length() - 1);
    }
  }
}

string PrintModelAndDataParams(const ModelAndDataParams& params) {
  string to_return = "";
  
  // Model
  to_return += "\tModel Type: " + Itoa(static_cast<int>(params.model_type_)) + "\n";
  if (!params.model_str_.empty()) {
    to_return += "\tModel String: " + params.model_str_ + "\n";
  }
  to_return += "\tModel LHS: ";
  if (params.model_type_ == ModelType::MODEL_TYPE_LINEAR) {
    to_return += GetExpressionString(params.model_lhs_.model_lhs_linear_) + "\n";
  } else if (params.model_type_ == ModelType::MODEL_TYPE_LOGISTIC) {
    to_return += GetExpressionString(params.model_lhs_.model_lhs_logistic_) + "\n";
  } else if (params.model_type_ == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL) {
    to_return += "(";
    for (int i = 0; i < params.model_lhs_.model_lhs_cox_.size(); ++i) {
      if (i != 0) {
        to_return += ", ";
      }
      to_return += GetExpressionString(params.model_lhs_.model_lhs_cox_[i]);
    }
    to_return += ")\n";
  } else if (params.model_type_ == ModelType::MODEL_TYPE_INTERVAL_CENSORED) {
    to_return += "(";
    const vector<Expression>& lhs =
        params.model_lhs_.model_lhs_time_dep_npmle_.empty() ?
        params.model_lhs_.model_lhs_time_indep_npmle_ :
        params.model_lhs_.model_lhs_time_dep_npmle_;
    for (int i = 0; i < lhs.size(); ++i) {
      if (i != 0) {
        to_return += ", ";
      }
      to_return += GetExpressionString(lhs[i]);
    }
    to_return += ")\n";
  } else if (params.model_type_ == ModelType::MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED) {
    to_return += "(";
    for (int i = 0; i < params.model_lhs_.model_lhs_time_dep_npmle_.size(); ++i) {
      if (i != 0) {
        to_return += ", ";
      }
      to_return += GetExpressionString(params.model_lhs_.model_lhs_time_dep_npmle_[i]);
    }
    to_return += ")\n";
  } else if (params.model_type_ == ModelType::MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED) {
    to_return += "(";
    for (int i = 0; i < params.model_lhs_.model_lhs_time_indep_npmle_.size(); ++i) {
      if (i != 0) {
        to_return += ", ";
      }
      to_return += GetExpressionString(params.model_lhs_.model_lhs_time_indep_npmle_[i]);
    }
    to_return += ")\n";
  } else {
    to_return += "??? (Unknown ModelType).\n";
  }
  if (!params.model_lhs_.time_vars_names_.empty()) {
    to_return +=
        "\tModel LHS Time Vars: " +
        Join(params.model_lhs_.time_vars_names_, ",") + "\n";
  }
  if (!params.model_lhs_.dep_vars_names_.empty()) {
    to_return +=
        "\tModel LHS Dependent Vars Names: " +
        Join(params.model_lhs_.dep_vars_names_, ",") + "\n";
  }
  if (!params.model_lhs_.left_truncation_name_.empty()) {
    to_return +=
        "\tModel LHS Left-Truncation time: " + 
        params.model_lhs_.left_truncation_name_ + "\n";
  }
  to_return += "\tModel RHS: " + GetExpressionString(params.model_rhs_) + "\n";
  to_return += "\tFinal Model: " + params.final_model_ + "\n";
  to_return += "\tLegend: " + Join(params.legend_, ", ") + "\n";

  // File Info.
  to_return += "\tData File:\n\t\t'" + params.file_.name_ +
               "'\n\t\tdelimiter: '" + params.file_.delimiter_ +
               "'\n\t\tcomment char: '" + params.file_.comment_char_ +
               "'\n\t\tinfinity_char_: '" + params.file_.infinity_char_ +
               "'\n\t\tna strings: {\"" +
               Join(params.file_.na_strings_, "\", \"") + "\"}\n";
  if (!params.header_.empty()) {
    to_return += "\tHeader: " + Join(params.header_, ", ") + "\n";
  }
  to_return += "\tData Columns used: " + Join(params.input_cols_used_, ", ") + "\n";
  to_return += "\tNominal Columns: " + Join(params.nominal_columns_, ", ") + "\n";
  to_return += "\tNominal Columns And Values:\n";
  for (const pair<int, set<string>>& col_and_values :
       params.nominal_columns_and_values_) {
    if (col_and_values.second.size() < 100) {
      to_return += "\t\tColumn " + Itoa(col_and_values.first) + ": {" +
                   Join(col_and_values.second, ", ") + "}\n";
    } else {
      to_return += "\t\tColumn " + Itoa(col_and_values.first) +
                   " has " + Itoa(static_cast<int>(col_and_values.second.size())) +
                   " distinct values, (Printing top 100): {";
      int i = 0;
      for (const string& val : col_and_values.second) {
        to_return += val + ", ";
        ++i;
        if (i >=100) break;
      }
      to_return += "}\n";
    }
  }
  to_return += "\t" + Itoa(static_cast<int>(params.na_rows_skipped_.size())) +
               " Rows Skipped due to Missing Values";
  if (params.na_rows_skipped_.size() <= 100) {
    to_return +=
        ":\n\t\t" + Join(params.na_rows_skipped_, ", ") + "\n";
  } else {
    to_return += " (Printing first 100 of them):\n\t\t";
    int i = 0;
    for (const int row_skipped : params.na_rows_skipped_) {
      to_return += Itoa(row_skipped) + ", ";
      ++i;
      if (i >=100) break;
    }
  }
  to_return += "\t" + Itoa(static_cast<int>(params.na_rows_and_columns_.size())) +
               " Rows With Missing Values";
  if (params.na_rows_and_columns_.size() < 100) {
    to_return += ":\n";
    for (const pair<int, set<int>>& row_and_columns : params.na_rows_and_columns_) {
      to_return += "\t\tRow " + Itoa(row_and_columns.first) +
                   " has missing values in columns: " +
                   Join(row_and_columns.second, ", ") + "\n";
    }
  } else {
    to_return += " (Printing top 100):\n";
    int i = 0;
    for (const pair<int, set<int>>& row_and_col : params.na_rows_and_columns_) {
       to_return += "\t\tRow " + Itoa(row_and_col.first) +
                    " has missing values in columns: " +
                    Join(row_and_col.second, ", ") + "\n";
       ++i;
       if (i >=100) break;
    }
  }
  to_return += "\tLinear Term Statistics:";
  if (params.linear_terms_mean_and_std_dev_.empty()) {
    to_return += " None (no standardization performed).\n";
  } else {
    to_return += "\n";
  }
  for (int i = 0; i < params.linear_terms_mean_and_std_dev_.size(); ++i) {
    const tuple<bool, double, double>& linear_term_info =
        params.linear_terms_mean_and_std_dev_[i];
    to_return += "\t\tLinear Term " + Itoa(i + 1) + ": Standardize: ";
    to_return += get<0>(linear_term_info) ? "Yes" : "No";
    to_return += ", Mean: " + Itoa(get<1>(linear_term_info));
    to_return += ", Std Dev: " + Itoa(get<2>(linear_term_info)) + "\n";
  }

  // Analysis to run.
  to_return += "\tAnalysis to run:\n";
  to_return += "\t\tStandard: ";
  to_return += (params.analysis_params_.standard_analysis_ ? "Yes" : "No");
  to_return += "\n\t\tRobust: ";
  to_return += (params.analysis_params_.robust_analysis_ ? "Yes" : "No");
  to_return += "\n\t\tPeto: ";
  to_return += (params.analysis_params_.peto_analysis_ ? "Yes" : "No");
  to_return += "\n\t\tLog-Rank: ";
  to_return += (params.analysis_params_.log_rank_analysis_ ? "Yes" : "No");
  to_return += "\n\t\tScoreMethod: ";
  to_return += (params.analysis_params_.score_method_analysis_ ? "Yes" : "No");
  to_return += "; ScoreMethodWidth: ";
  to_return += (params.analysis_params_.score_method_width_analysis_ ? "Yes" : "No");
  to_return += "\n\t\tKME: ";
  to_return += (params.kme_type_ ? "Yes" : "No");
  to_return += "; KME for Log-Rank: ";
  to_return += (params.kme_type_for_log_rank_ ? "Yes" : "No");
  to_return += "\n\t\tUse Ties Constant: ";
  to_return += (params.use_ties_constant_ ? "Yes" : "No");
  to_return += "\n\t\tSubgroup as Covariate: ";
  to_return += (params.use_subgroup_as_covariate_ ? "Yes" : "No");
  to_return += "\n";

  // Id, Family, Weight.
  if (!params.id_str_.empty()) {
    to_return += "\tid_str_: " + params.id_str_ + "\n";
    to_return +=
        "\tId Col Name: " + params.id_col_.name_ +
        ", Col Index: " + Itoa(params.id_col_.index_) + "\n";
  }
  if (!params.weight_str_.empty()) {
    to_return += "\tweight_str_: " + params.weight_str_ + "\n";
    to_return +=
        "\tWeight Col Name: " + params.weight_col_.name_ +
        ", Col Index: " + Itoa(params.weight_col_.index_) + "\n";
  }
  if (!params.family_str_.empty()) {
    to_return += "\tfamily_str_: " + params.family_str_ + "\n";
    for (const VariableColumn& var_col : params.family_cols_) {
      to_return +=
          "\tFamily Col Name: " + var_col.name_ +
          ", Col Index: " + Itoa(var_col.index_) + "\n";
    }
  }

  // Subgroup.
  if (!params.subgroup_str_.empty()) {
    to_return += "\tsubgroup_str_: " + params.subgroup_str_ + "\n";
    for (const VariableColumn& var_col : params.subgroup_cols_) {
      to_return +=
          "\tSubgroup Col Name: " + var_col.name_ +
          ", Col Index: " + Itoa(var_col.index_) + "\n";
    }
    for (int i = 0; i < params.subgroups_.size(); ++i) {
      to_return += "\tSubgroup " + Itoa(i + 1) + " Description: " +
                   Join(params.subgroups_[i], ", ") + "\n";
    }
  }

  // Strata.
  if (!params.strata_str_.empty()) {
    to_return += "\tstrata_str_: " + params.strata_str_ + "\n";
    for (const VariableColumn& var_col : params.strata_cols_) {
      to_return +=
          "\tStrata Col Name: " + var_col.name_ +
          ", Col Index: " + Itoa(var_col.index_) + "\n";
    }
  }
  
  // Subgroup and Strata stats.
  if (!params.subgroup_rows_per_index_.empty() || !params.row_to_strata_.empty()) {
    to_return += "\tSubgroup and Strata row membership information:\n\t";
    string membership = "";
    if (GetModelAndDataParamsSubgroupsAndStrata(params, &membership)) {
      to_return += membership;
    }
  }

  // Variable Parameters.
  to_return += "\tVariableNormalization: " +
               Itoa(static_cast<int>(params.standardize_vars_)) + "\n";
  if (!params.collapse_params_str_.empty()) {
    to_return += "\tCollapse Params String: " + params.collapse_params_str_ + "\n";
  }
  if (!params.time_params_str_.empty()) {
    to_return += "\tTime Params String: " + params.time_params_str_ + "\n";
  }
  if (!params.var_norm_params_str_.empty()) {
    to_return +=
        "\tVariable Normalization Params String: " +
        params.var_norm_params_str_ + "\n";
  }
  if (!params.var_params_.empty()) {
    to_return += "\tvar_params_:\n";
    for (const VariableParams& var_param : params.var_params_) {
      to_return += "\t\t====================================================\n";
      to_return += "\t\tVariable Name: " + var_param.col_.name_ +
                   ", Variable Column Index: " +
                   Itoa(var_param.col_.index_) + "\n";
      to_return += "\t\tVariable Normalization: " +
                   Itoa(static_cast<int>(var_param.norm_)) + "\n";
      const OutsideIntervalParams& outside_params =
          var_param.time_params_.outside_params_;
      to_return += "\t\tVariable TimeDependentParams:\n\t\t\tinterp_type_: " +
                   Itoa(static_cast<int>(var_param.time_params_.interp_type_)) +
                   "\n\t\t\toutside_params_: outside_left_type_: " +
                   Itoa(static_cast<int>(outside_params.outside_left_type_)) +
                   "\n\t\t\tdefault_left_val_: " +
                   GetDataHolderString(outside_params.default_left_val_) +
                   "\n\t\t\toutside_right_type_: " +
                   Itoa(static_cast<int>(outside_params.outside_right_type_)) +
                   "\n\t\t\tdefault_right_val_: " +
                   GetDataHolderString(outside_params.default_right_val_) + "\n";
      to_return += "\t\tVariable Collapse Params:\n";
      for (const VariableCollapseParams& collapse_params :
           var_param.collapse_params_) {
        to_return +=
            "\t\t\t" + GetVariableCollapseParamsString(collapse_params) + "\n";
      }
    }
  }

  return to_return;
}

void CopyModelAndDataParams(
    const ModelAndDataParams& params_one, ModelAndDataParams* params_two) {
  params_two->file_ = params_one.file_;
  params_two->outfile_ = params_one.outfile_;
  // Model Type is only copied over if not explicitly set by the user.
  if (params_two->model_type_ == ModelType::MODEL_TYPE_UNKNOWN) {
    params_two->model_type_ = params_one.model_type_;
  }
  // Model is only copied over if not explicitly set by the user.
  if (params_two->model_str_.empty()) {
    params_two->model_str_ = params_one.model_str_;
  }
  params_two->max_itr_ = params_one.max_itr_;
  params_two->analysis_params_ = params_one.analysis_params_;
  params_two->print_options_ = params_one.print_options_;
  params_two->kme_type_ = params_one.kme_type_;
  params_two->kme_type_for_log_rank_ = params_one.kme_type_for_log_rank_;
  params_two->use_ties_constant_ = params_one.use_ties_constant_;
  // Subgroup and Strata are not copied over; they must be explicitly
  // set for each model by the user in the command-line args.
  params_two->id_str_ = params_one.id_str_;
  params_two->weight_str_ = params_one.weight_str_;
  params_two->family_str_ = params_one.family_str_;
  params_two->left_truncation_str_ = params_one.left_truncation_str_;
  params_two->collapse_params_str_ = params_one.collapse_params_str_;
  params_two->time_params_str_ = params_one.time_params_str_;
  params_two->var_norm_params_str_ = params_one.var_norm_params_str_;
  params_two->standardize_vars_ = params_one.standardize_vars_;
}

string GetVariableCollapseParamsString(const VariableCollapseParams& params) {
  string to_return = "From: ";
  if (params.from_type_ == DataType::DATA_TYPE_STRING) {
    to_return += params.from_str_;
  } else if (params.from_type_ == DataType::DATA_TYPE_NUMERIC) {
    to_return += params.from_val_;
  } else if (params.from_type_ == DataType::DATA_TYPE_NUMERIC_RANGE) {
    to_return += "[" + Itoa(params.from_range_.first) + ".." +
                 Itoa(params.from_range_.second) + "]";
  } else {
    return "ERROR Printing VariableCollapseParams: Unsupported from_type_ " +
           Itoa(static_cast<int>(params.from_type_));
  }

  to_return += ", To: ";
  if (params.to_type_ == DataType::DATA_TYPE_STRING) {
    to_return += params.to_str_;
  } else if (params.to_type_ == DataType::DATA_TYPE_NUMERIC) {
    to_return += params.to_val_;
  } else {
    return "ERROR Printing VariableCollapseParams: Unsupported to_type_ " +
           Itoa(static_cast<int>(params.to_type_));
  }
  return to_return;
}

bool GetModelAndDataParamsSubgroupsAndStrata(
    const ModelAndDataParams& params, string* subgroup_and_strata_str) {
  if (params.row_to_strata_.empty() && params.subgroup_rows_per_index_.empty()) {
    return true;
  }
  if (subgroup_and_strata_str == nullptr) return false;

  // Get the rows in each strata.
  map<int, set<int>> rows_in_each_strata;
  for (const pair<int, int>& row_to_strata_itr : params.row_to_strata_) {
    map<int, set<int>>::iterator itr =
        rows_in_each_strata.find(row_to_strata_itr.second);
    if (itr == rows_in_each_strata.end()) {
      // This is the first row of original data that belongs to this strata.
      set<int> set_w_one_row;
      // row_to_strata indexed input data rows starting at index 0, but
      // params.subgroup_rows_per_index_ indexed input data rows starting at index 1.
      // Since we'll be intersecting these below (and because we want to
      // print row indices starting at index 1), increment index by 1 here.
      set_w_one_row.insert(1 + row_to_strata_itr.first);
      rows_in_each_strata.insert(make_pair(
          row_to_strata_itr.second, set_w_one_row));
    } else {
      // params.row_to_strata_ indexed input data rows starting at index 0, but
      // params.subgroup_rows_per_index_ indexed input data rows starting at index 1.
      // Since we'll be intersecting these below (and because we want to
      // print row indices starting at index 1), increment index by 1 here.
      itr->second.insert(1 + row_to_strata_itr.first);
    }
  }

  return GetSubgroupAndStrataDescription(
      rows_in_each_strata, params.subgroup_rows_per_index_,
      subgroup_and_strata_str);
}

bool GetSubgroupAndStrataDescription(
    const map<int, set<int>>& rows_in_each_strata,
    const map<int, set<int>>& subgroup_rows_per_index,
    string* subgroup_and_strata_str) {
  if (rows_in_each_strata.empty() && subgroup_rows_per_index.empty()) return true;
  if (subgroup_and_strata_str == nullptr) return false;

  // Count the number of rows in all Strata and Subgroups.
  int num_rows_in_all_strata = 0;
  for (const pair<int, set<int>>& strata_rows : rows_in_each_strata) {
    num_rows_in_all_strata += strata_rows.second.size();
  }
  int num_rows_in_all_subgroups = 0;
  for (const pair<int, set<int>>& subgroup_rows : subgroup_rows_per_index) {
    num_rows_in_all_subgroups += subgroup_rows.second.size();
  }

  // Print strata and subgroup sizes.
  const int kMaxPrintableRows = 1000;
  const bool kPrintRows = false;
  const string strata_str =
      rows_in_each_strata.size() <= 1 ? "" :
      " among " + Itoa(static_cast<int>(rows_in_each_strata.size())) + " Strata";
  const string subgroup_str =
      subgroup_rows_per_index.empty() ? "" :
      Itoa(static_cast<int>(subgroup_rows_per_index.size())) + " Subgroup(s)";
  const string separator =
      subgroup_str.empty() ? "" : strata_str.empty() ? " among " : " and ";
  // Case 1: More than one strata.
  if (rows_in_each_strata.size() > 1) {
    // Case 1a: More than one strata, no subgroups.
    if (subgroup_rows_per_index.empty()) {
      *subgroup_and_strata_str +=
          "Kept " + Itoa(num_rows_in_all_strata) + " input rows" + strata_str +
          separator + subgroup_str + ":\n";
      for (const pair<int, set<int>>& strata_rows_itr : rows_in_each_strata) {
        *subgroup_and_strata_str +=
            "\tStrata " + Itoa(strata_rows_itr.first) + " (" +
            Itoa(static_cast<int>(strata_rows_itr.second.size())) + " rows)";
        if (!kPrintRows) {
          *subgroup_and_strata_str += "\n";
          continue;
        } else if (strata_rows_itr.second.size() > kMaxPrintableRows) {
          *subgroup_and_strata_str += ": Too many to print...\n";
          continue;
        }
        *subgroup_and_strata_str += ":";
        int i = 0;
        for (const int strata_row_val : strata_rows_itr.second) {
          if ((i < 1000 && (i % 20) == 0) ||
              (i >= 1000 && i < 10000 && (i % 16) == 0) ||
              (i >= 10000 && i < 100000 && (i % 12) == 0) ||
              (i >= 100000 && (i % 10) == 0)) {
            *subgroup_and_strata_str += "\n\t\t";
          }
          *subgroup_and_strata_str += strata_row_val;
          if (i != strata_rows_itr.second.size() - 1) *subgroup_and_strata_str += ", ";
          ++i;
        }
        *subgroup_and_strata_str += "\n";
      }
    // Case 1b: More than one strata, more than one subgroup.
    } else {
      int total_rows_kept = 0;
      string output_holder = "";
      for (const pair<int, set<int>>& strata_rows_itr : rows_in_each_strata) {
        string output_outer_holder =
            "\tStrata " + Itoa(strata_rows_itr.first) + " (";
        int total_rows_in_strata = 0;
        string output_inner_holder = "";
        for (const pair<int, set<int>>& subgroup_rows_itr : subgroup_rows_per_index) {
          set<int> intersection;
          set_intersection(
              strata_rows_itr.second.begin(), strata_rows_itr.second.end(),
              subgroup_rows_itr.second.begin(), subgroup_rows_itr.second.end(),
              inserter(intersection, intersection.begin()));
          if (!intersection.empty()) {
            total_rows_in_strata += intersection.size();
            output_inner_holder +=
                "\t\tSubgroup_" + Itoa(subgroup_rows_itr.first) +
                " (" + Itoa(static_cast<int>(intersection.size())) + " rows)";
            if (!kPrintRows) {
              output_inner_holder += "\n";
              continue;
            } else if (intersection.size() > kMaxPrintableRows) {
              output_inner_holder += ": Too many to print...\n";
              continue;
            }
            output_inner_holder += ":";
            int i = 0;
            for (const int row_val : intersection) {
              if ((i < 1000 && (i % 20) == 0) ||
                  (i >= 1000 && i < 10000 && (i % 16) == 0) ||
                  (i >= 10000 && i < 100000 && (i % 12) == 0) ||
                  (i >= 100000 && (i % 10) == 0)) {
                output_inner_holder += "\n\t\t";
              }
              output_inner_holder += Itoa(row_val);
              if (i != intersection.size() - 1) output_inner_holder += ", ";
              ++i;
            }
            output_inner_holder += "\n";
          }
        }
        total_rows_kept += total_rows_in_strata;
        if (total_rows_in_strata == 0) {
          output_outer_holder += "0 rows among the Subgroups)\n";
        } else {
          output_outer_holder +=
              Itoa(total_rows_in_strata) +
              " rows among the Subgroups):\n" + output_inner_holder;
        }
        output_holder += output_outer_holder + "\n";
      }
      *subgroup_and_strata_str +=
          "Kept " + Itoa(total_rows_kept) + " input rows" + strata_str +
          separator + subgroup_str + ":\n" + output_holder;
    }
  // Case 2: Exactly one Strata, more than one Subgroup.
  } else if (!subgroup_rows_per_index.empty()) {
    string output_holder = "";
    int num_subgroup_rows = 0;
    for (const pair<int, set<int>>& subgroup_rows_itr : subgroup_rows_per_index) {
      num_subgroup_rows += subgroup_rows_itr.second.size();
      output_holder +=
          "\tSubgroup_" + Itoa(subgroup_rows_itr.first) + " (" +
          Itoa(static_cast<int>(subgroup_rows_itr.second.size())) + " rows)";
      if (!kPrintRows) {
        output_holder += "\n";
        continue;
      } else if (subgroup_rows_itr.second.size() > kMaxPrintableRows) {
        output_holder += ": Too many to print...\n";
        continue;
      }
      output_holder += ":";
      int i = 0;
      for (const int subgroup_row_val : subgroup_rows_itr.second) {
        if ((i < 1000 && (i % 20) == 0) ||
            (i >= 1000 && i < 10000 && (i % 16) == 0) ||
            (i >= 10000 && i < 100000 && (i % 12) == 0) ||
            (i >= 100000 && (i % 10) == 0)) {
          output_holder += "\n\t\t";
        }
        output_holder += Itoa(subgroup_row_val);
        if (i != subgroup_rows_itr.second.size() - 1) output_holder += ", ";
        ++i;
      }
      output_holder += "\n";
    }
    *subgroup_and_strata_str +=
        "Kept " + Itoa(num_subgroup_rows) + " input rows" + strata_str +
        separator + subgroup_str + ":\n" + output_holder;
  // Case 3: Exactly one strata, no Subgroups.
  } else {
    *subgroup_and_strata_str +=
        "Kept " + Itoa(num_rows_in_all_strata) + " input rows" + strata_str +
        separator + subgroup_str;
    if (!kPrintRows) {
      *subgroup_and_strata_str += "\n";
    } else if (rows_in_each_strata.begin()->second.size() > kMaxPrintableRows) {
      *subgroup_and_strata_str += ": Too many to print...\n";
    } else {
      *subgroup_and_strata_str += ":";
      int i = 0;
      for (const int row_val : rows_in_each_strata.begin()->second) {
        if ((i < 1000 && (i % 20) == 0) ||
            (i >= 1000 && i < 10000 && (i % 16) == 0) ||
            (i >= 10000 && i < 100000 && (i % 12) == 0) ||
            (i >= 100000 && (i % 10) == 0)) {
          *subgroup_and_strata_str += "\n\t";
        }
        *subgroup_and_strata_str += Itoa(row_val);
        if (i != rows_in_each_strata.begin()->second.size() - 1) {
          *subgroup_and_strata_str += ", ";
        }
        ++i;
      }
      *subgroup_and_strata_str += "\n";
    }
  }
  *subgroup_and_strata_str += "\n";

  return true;
}

bool PrintSubgroupSynopsis(
    const vector<VariableColumn>& subgroup_cols,
    const vector<vector<string>>& subgroups,
    const map<int, set<int>>& subgroup_index_to_its_rows,
    string* output) {
  if (output == nullptr) return false;

  if (subgroup_cols.empty() || subgroups.empty() ||
      subgroup_index_to_its_rows.empty()) {
    return true;
  }
  vector<string> subgroup_col_names;
  for (const VariableColumn& col : subgroup_cols) {
    subgroup_col_names.push_back(col.name_);
  }
  *output += "Subgroups {" + Join(subgroup_col_names, ", ") + "}:\n";

  if (subgroups.size() != subgroup_index_to_its_rows.size()) {
    cout << "ERROR in PrintSubgroupSynopsis: Mismatching number of subgroups: "
         << subgroup_index_to_its_rows.size() << " found earlier, now "
         << subgroups.size() << " found\n";
    return false;
  }
  for (int i = 0; i < subgroups.size(); ++i) {
    if (subgroups[i].size() != subgroup_col_names.size()) {
      cout << "ERROR in PrintSubgroupSynopsis: Mismatching number of "
           << "values for Subgroup " << i << ": found "
           << subgroups[i].size() << ", expected: "
           << subgroup_col_names.size() << endl;
      return false;
    }
    map<int, set<int>>::const_iterator size_itr =
        subgroup_index_to_its_rows.find(i);
    if (size_itr == subgroup_index_to_its_rows.end()) {
      cout << "ERROR in PrintSubgroupSynopsis: Unable to find size of "
           << "Subgroup " << i << endl;
      return false;
    }
    *output += "\tSubgroup_" + Itoa(i) + " = (";
    for (int j = 0; j < subgroup_col_names.size(); ++j) {
      if (j != 0) {
        *output += ", ";
      }
      *output += subgroup_col_names[j] + " = " + subgroups[i][j];
    }
    *output += ")\t(" + Itoa(static_cast<int>(size_itr->second.size())) + " rows)\n";
  }
  *output += "\n";

  return true;
}

bool PrintSubgroupSynopsis(
    const vector<VariableColumn>& subgroup_cols,
    const vector<vector<string>>& subgroups,
    const map<int, set<int>>& subgroup_index_to_its_rows,
    ofstream& out_file) {
  string output = "";
  if (!PrintSubgroupSynopsis(
          subgroup_cols, subgroups, subgroup_index_to_its_rows, &output)) {
    return false;
  }
  out_file << output;
  return true;
}

bool PrintSubgroupSynopsis(
    const vector<VariableColumn>& subgroup_cols,
    const vector<vector<string>>& subgroups,
    const string& subgroup, ofstream& out_file) {
  if (subgroup_cols.empty() || subgroups.empty()) return true;
  vector<string> subgroup_col_names;
  for (const VariableColumn& col : subgroup_cols) {
    subgroup_col_names.push_back(col.name_);
  }
  out_file << "Subgroups {" << Join(subgroup_col_names, ", ") << "}:" << endl;
  for (int i = 0; i < subgroups.size(); ++i) {
    if (subgroups[i].size() != subgroup_col_names.size()) {
      cout << "ERROR in PrintSubgroupSynopsis: Mismatching number of "
           << "values for Subgroup " << i << ": found "
           << subgroups[i].size() << ", expected: "
           << subgroup_col_names.size() << endl;
      return false;
    }
    out_file << "\tSubgroup_" << i << " = (";
    for (int j = 0; j < subgroup_col_names.size(); ++j) {
      if (j != 0) {
        out_file << ", ";
      }
      out_file << subgroup_col_names[j] << " = " << subgroups[i][j];
    }
    out_file << ")" << endl;
  }
  out_file << endl;
  return true;
}

bool CombineVariableParams(
    const vector<string>& header,
    const map<string, vector<VariableCollapseParams>>& var_name_to_collapse_params,
    const map<string, TimeDependentParams>& var_name_to_time_params,
    const map<string, VariableNormalization>& var_name_to_normalization,
    vector<VariableParams>* var_params, string* error_msg) {
  if (var_params == nullptr) {
    return false;
  }

  // First, collect all the variable names.
  set<string> variable_names;
  for (const string& header_col : header) {
    variable_names.insert(header_col);
  }
  // Make sure that none of the VariableParameters passed in are for
  // an unknown variable name.
  for (const string& name : Keys(var_name_to_collapse_params)) {
    if (variable_names.find(name) == variable_names.end()) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to find variable '" + name +
                      "', appearing in the --collapse argument, in the "
                      "header line of the input data file.\n";
      }
      return false;
    }
  }
  for (const string& name : Keys(var_name_to_time_params)) {
    if (variable_names.find(name) == variable_names.end()) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to find variable '" + name +
                      "', appearing in the --extrapolation argument, in the "
                      "header line of the input data file.\n";
      }
      return false;
    }
  }
  for (const string& name : Keys(var_name_to_normalization)) {
    if (variable_names.find(name) == variable_names.end()) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to find variable '" + name +
                      "', appearing in the --standardization argument, in the "
                      "header line of the input data file.\n";
      }
      return false;
    }
  }

  // Create a VariableParams object for each variable.
  for (int i = 0; i < header.size(); ++i) {
    const string& var_name = header[i];
    var_params->push_back(VariableParams());
    VariableParams& params = var_params->back();
    params.col_.name_ = var_name;
    params.col_.index_ = i;
    // Add VariableCollapseParams, if present.
    if (var_name_to_collapse_params.find(var_name) !=
        var_name_to_collapse_params.end()) {
      params.collapse_params_ =
          var_name_to_collapse_params.find(var_name)->second;
    }
    // Add TimeDependentParams, if present.
    if (var_name_to_time_params.find(var_name) !=
        var_name_to_time_params.end()) {
      params.time_params_ = var_name_to_time_params.find(var_name)->second;
    }
    // Add VariableNormalization Params, if present.
    if (var_name_to_normalization.find(var_name) !=
        var_name_to_normalization.end()) {
      params.norm_ = var_name_to_normalization.find(var_name)->second;
    }
  }
  return true;
}

bool SanityCheckModel(
    const ModelType& model_type,
    const map<string, int>& titles,
    const map<int, set<string> >& nominal_columns,
    const DepVarDescription& model_lhs,
    const Expression& model_rhs,
    set<int>* input_cols_used, string* error_msg) {
  if (!SanityCheckDependentVariable(
          model_type, titles, nominal_columns, model_lhs, input_cols_used, error_msg)) {
    return false;
  }    
  if (!IsEmptyExpression(model_rhs) &&
      !SanityCheckIndependentVariables(
          titles, nominal_columns, model_rhs, input_cols_used, error_msg)) {
    return false;
  }
  return true;
}

bool SanityCheckDependentVariable(
    const ModelType& model_type,
    const map<string, int>& titles,
    const map<int, set<string> >& nominal_columns,
    const DepVarDescription& model_lhs,
    set<int>* input_cols_used, string* error_msg) {
  if (model_type == ModelType::MODEL_TYPE_LINEAR) {
    return SanityCheckVariables(
        titles, nominal_columns, model_lhs.model_lhs_linear_, input_cols_used, error_msg);
  } else if (model_type == ModelType::MODEL_TYPE_LOGISTIC) {
    return SanityCheckVariables(
        titles, nominal_columns, model_lhs.model_lhs_logistic_, input_cols_used, error_msg);
  } else if (model_type == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL) {
    for (const Expression& dep_var_exp : model_lhs.model_lhs_cox_) {
      if (!SanityCheckVariables(
              titles, nominal_columns, dep_var_exp, input_cols_used, error_msg)) {
        return false;
      }
    }
    return true;
  } else if (model_type == ModelType::MODEL_TYPE_INTERVAL_CENSORED) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR parsing model's LHS: Unexpected ModelType: " +
                    Itoa(static_cast<int>(model_type)) + "; at this point, "
                    "should have determined if this was time-indep or time-dep.\n";
    }
    return false;
  } else if (model_type == ModelType::MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED) {
    for (const Expression& dep_var_exp : model_lhs.model_lhs_time_dep_npmle_) {
      if (!SanityCheckVariables(
              titles, nominal_columns, dep_var_exp, input_cols_used, error_msg)) {
        return false;
      }
    }
    return true;
  } else if (model_type == ModelType::MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED) {
    for (const Expression& dep_var_exp : model_lhs.model_lhs_time_indep_npmle_) {
      if (!SanityCheckVariables(
              titles, nominal_columns, dep_var_exp, input_cols_used, error_msg)) {
        return false;
      }
    }
    return true;
  } else {
    if (error_msg != nullptr) {
      *error_msg += "ERROR checking model's LHS: unrecognized ModelType: " +
                    Itoa(static_cast<int>(model_type)) + "\n";
    }
    return false;
  }
}

bool SanityCheckIndependentVariables(
    const map<string, int>& variable_names_and_cols,
    const map<int, set<string> >& nominal_columns,
    const Expression& model_rhs,
    set<int>* input_cols_used, string* error_msg) {
  return SanityCheckVariables(
      variable_names_and_cols, nominal_columns, model_rhs,
      input_cols_used, error_msg);
}


bool SanityCheckVariables(
    const map<string, int>& variable_names_and_cols,
    const map<int, set<string> >& nominal_columns,
    const Expression& variable_expression,
    set<int>* input_cols_used, string* error_msg) {
  // Check that variable_expression variable's names are in variable_names_and_cols.
  set<string> used_variables;
  if (!ExtractVariablesFromExpression(
          variable_expression, Keys(variable_names_and_cols), &used_variables, error_msg)) {
    return false;
  }

  // Add the column indices of the variables found in the variable_expression to
  // 'input_cols_used'.
  if (input_cols_used != nullptr) {
    for (const string& used_var : used_variables) {
      map<string, int>::const_iterator names_to_col_itr =
          variable_names_and_cols.find(used_var);
      if (names_to_col_itr == variable_names_and_cols.end() &&
          used_var != kSubgroupIndicator) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unable to find variable '" + used_var +
                        "' from the model RHS in the set of column "
                        "names on the header row of the input data file:\n\t{" +
                        Join(Keys(variable_names_and_cols), ", ") + "}\n";
        }
        return false;
      }
      if (used_var != kSubgroupIndicator) {
        input_cols_used->insert(names_to_col_itr->second);
      }
    }
  }

  // Check Log, Exp are not applied directly to a nominal variable.
  set<string> nominal_vars;
  for (const pair<string, int>& var_name_and_col : variable_names_and_cols) {
    if (nominal_columns.find(var_name_and_col.second) != nominal_columns.end()) {
      nominal_vars.insert(var_name_and_col.first);
    }
  }
  if (!SanityCheckFunctionsAppliedToNominalVariables(
          variable_expression, nominal_vars, error_msg)) {
    return false;
  }
  return true;
}

bool ParseModel(
    const string& model, const string& left_truncation_col_name,
    const ModelType type, const vector<string>& header,
    DepVarDescription* model_lhs, Expression* model_rhs,
    bool* use_subgroup_as_covariate, set<int>* input_cols_used,
    string* error_msg) {
  if (model_lhs == nullptr || model_rhs == nullptr) return false;
  set<int> nominal_columns;
  if (!GetNominalColumnsFromTitles(header, &nominal_columns)) return false;

  // Convert nominal_columns to the structure that ReadTableWithHeader::ParseModel
  // requires.
  map<int, set<string>> nominal_columns_and_values;
  for (const int nominal_column : nominal_columns) {
    nominal_columns_and_values.insert(make_pair(nominal_column, set<string>()));
  }

  // Parse passed-in model.
  string temp_error_msg = "";
  if (!ProcessUserEnteredModel(
          model, type, model_lhs, model_rhs, &temp_error_msg)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to parse model:\n" + model +
                    "Error message:\n" + temp_error_msg + "\n";
    }
    return false;
  }

  // Parse left-truncation column, for Cox.
  if (type == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL &&
      !left_truncation_col_name.empty()) {
    string left_truncation_str_cleaned =
        StripQuotes(RemoveAllWhitespace(left_truncation_col_name));

    model_lhs->left_truncation_name_ = left_truncation_str_cleaned;
    bool found_column = false;
    for (int i = 0; i < header.size(); ++i) {
      if (header[i] == left_truncation_str_cleaned) {
        input_cols_used->insert(i);
        found_column = true;
        break;
      }
    }
    if (!left_truncation_str_cleaned.empty() && !found_column) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to find left-truncation column '" +
                      left_truncation_str_cleaned +
                      "' among the column names in the header: {'" +
                      Join(header, "', '") + "'}.\n";
      }
      return false;
    }
  }

  // See if "I_Subgroup" appears in RHS of model, and set
  // 'use_subgroup_as_covariate' accordingly.
  if (use_subgroup_as_covariate != nullptr) {
    set<string> model_rhs_vars;
    if (!ExtractVariablesFromExpression(
            *model_rhs, &model_rhs_vars, error_msg)) {
      return false;
    }
    *use_subgroup_as_covariate =
        model_rhs_vars.find(kSubgroupIndicator) != model_rhs_vars.end();
  }

  // Sanity Check Model.
  if (!header.empty()) {
    map<string, int> titles;
    for (int i = 0; i < header.size(); ++i) {
      titles.insert(make_pair(header[i], i));
    }
    if (!SanityCheckModel(
            type, titles, nominal_columns_and_values,
            *model_lhs, *model_rhs, input_cols_used, error_msg)) {
      return false;
    }
  }

  return true;
}

bool ProcessUserEnteredModel(
    const string& model, const ModelType& model_type,
    DepVarDescription* model_lhs, Expression* model_rhs, string* error_msg) {
  // Sanity check input.
  if (model_rhs == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: NULL model_rhs. Check API in call "
                    "to ProcessUserEnteredModel.\n";
    }
    return false;
  }

  // Simplify model by removing whitespace.
  string parsed_model;
  RemoveAllWhitespace(model, &parsed_model);
  if (parsed_model.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Empty model received from user input.\n";
    }
    return false;
  }

  // Split model around the equality (separating (in)dependent variables).
  vector<string> dep_indep_split;
  Split(parsed_model, "=", &dep_indep_split);
  bool has_rhs = true;
  if (dep_indep_split.size() > 2) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Input contains multiple '=' signs.\n";
    }
    return false;
  }
  if (dep_indep_split.size() == 1) {
    // NOTE: We first aborted in this case, then we wanted to support
    // Linear Models that have empty RHS (since Constant and Error terms
    // are not explicitly written); and then finally there are some use
    // cases (e.g. cox_compute_k_m_estimator_main.exe) that don't have a
    // RHS at all. Thus, we don't abort here anymore.
    /*
    // There is one valid use-case for the size to not be 2: If we have a
    // Linear Model, the RHS can be empty (just do constant term and error).
    // Check if this is the case, otherwise abort.
    if (model_type != ModelType::MODEL_TYPE_LINEAR ||
        dep_indep_split.size() != 1 ||
        !HasSuffixString(parsed_model, "=")) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Input contains multiple '=' signs.\n";
      }
      return false;
    } else {
      has_rhs = false;
    }
    */
    has_rhs = false;
  }

  // Parse Model LHS.
  if (!ParseDependentTerm(dep_indep_split[0], model_type, model_lhs, error_msg)) { 
    if (error_msg != nullptr) {
      *error_msg += "ERROR in parsing LHS of model as an expression:\n\t" +
                    dep_indep_split[0] + "\n";
    }
    return false;
  }

  // Parse Model RHS.
  if (has_rhs) {
    if (!ParseExpression(dep_indep_split[1], false, model_rhs)) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in parsing RHS of model as an expression:\n\t" +
                      dep_indep_split[1] + "\n";
      }
      return false;
    }
  }
  return true;
}


bool ParseDependentTerm(
    const string& input, const ModelType& model_type,
    DepVarDescription* dep_term, string* error_msg) {
  if (model_type == ModelType::MODEL_TYPE_LINEAR) {
    return ParseDependentTermLinearModel(
        input, &dep_term->model_lhs_linear_, &dep_term->dep_vars_names_, error_msg);
  } else if (model_type == ModelType::MODEL_TYPE_LOGISTIC) {
    return ParseDependentTermLogisticModel(
        input, &dep_term->model_lhs_logistic_, &dep_term->dep_vars_names_, error_msg);
  } else if (model_type == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL) {
    return ParseDependentTermCoxModel(
        input, &dep_term->model_lhs_cox_, &dep_term->time_vars_names_,
        &dep_term->dep_vars_names_, error_msg);
  } else if (model_type == ModelType::MODEL_TYPE_INTERVAL_CENSORED) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR parsing model's LHS: Unexpected ModelType: " +
                    Itoa(static_cast<int>(model_type)) + "; at this point, "
                    "should have determined if this was time-indep or time-dep.\n";
    }
    return false;
  } else if (model_type == ModelType::MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED) {
    return ParseDependentTermTimeDependentNpmleModel(
        input, &dep_term->model_lhs_time_dep_npmle_, &dep_term->time_vars_names_,
        &dep_term->dep_vars_names_, error_msg);
  } else if (model_type == ModelType::MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED) {
    return ParseDependentTermTimeIndependentNpmleModel(
        input, &dep_term->model_lhs_time_indep_npmle_, &dep_term->time_vars_names_,
        &dep_term->dep_vars_names_, error_msg);
  } else {
    if (error_msg != nullptr) {
      *error_msg += "ERROR parsing model's LHS: unrecognized ModelType: " +
                    Itoa(static_cast<int>(model_type)) + "\n";
    }
    return false;
  }
}

bool ParseDependentTermLinearModel(
    const string& input,
    Expression* dep_term, vector<string>* dep_var_name, string* error_msg) {
  // Sanity check input.
  if (dep_term == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: NULL term in call to "
                    "ParseDependentTermLinearModel. Check API.";
    }
    return false;
  }

  // Return false if input is empty.
  if (input.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Dependent variable (expression on left side of '=' "
                    "sign) is empty (or contains only whitespace).\n";
    }
    return false;
  }

  // Parse the Linear Dependent Variable expression.
  if (!ParseExpression(input, false, dep_term)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Dependent variable (expression on left side of '=' "
                    "sign) cannot be parsed: '" + input + "'.\n";
    }
    return false;
  }

  dep_var_name->push_back(GetExpressionString(*dep_term));

  return true;
}

bool ParseDependentTermLogisticModel(
    const string& input,
    Expression* dep_term, vector<string>* dep_var_name, string* error_msg) {
  // Format of LHS is same for Logistic Model as Linear Model, so just use
  // existing function for that.
  return ParseDependentTermLinearModel(input, dep_term, dep_var_name, error_msg);
}

bool ParseDependentTermCoxModel(
    const string& input,
    vector<Expression>* dep_term, vector<string>* time_vars_names,
    vector<string>* dep_vars_names, string* error_msg) {
  // Sanity check input.
  if (dep_term == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: NULL term in call to "
                    "ParseDependentTermLogisticModel. Check API.";
    }
    return false;
  }

  // Return false if input is empty.
  if (input.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Dependent variable (expression on left side of '=' "
                    "sign) is empty (or contains only whitespace).\n";
    }
    return false;
  }

  // Split LHS around the "," which delineates between [Survival | Censoring] Time
  // and Status.
  vector<string> parts;
  Split(input, ",", &parts);
  if (parts.size() != 2 && parts.size() != 3) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to parse LHS of Cox Regression equation: " +
                    input + "\n";
    }
    return false;
  }

  // Process first dep variable.
  string dep_var, stripped_dep_var;
  StripPrefixString(parts[0], "(", &dep_var);
  RemoveExtraWhitespace(dep_var, &stripped_dep_var);
  dep_term->push_back(Expression());
  if (!ParseExpression(stripped_dep_var, false, &dep_term->back())) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: 1st Dependent variable (expression on left side of '=' "
                    "sign) cannot be parsed: '" + input + "'.\n";
    }
    return false;
  }
  time_vars_names->push_back(GetExpressionString(dep_term->back()));

  // Process second dep variable (if present).
  int status_var_index = 1;
  if (parts.size() == 3) {
    string dep_var_two;
    RemoveExtraWhitespace(parts[1], &dep_var_two);
    dep_term->push_back(Expression());
    if (!ParseExpression(dep_var_two, false, &dep_term->back())) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: 2nd Dependent variable (expression on left side of '=' "
                      "sign) cannot be parsed: '" + input + "'.\n";
      }
      return false;
    }
    time_vars_names->push_back(GetExpressionString(dep_term->back()));
    status_var_index++;
  }

  // Process last dep variable (Status).
  string status_var, stripped_status_var;
  StripSuffixString(parts[status_var_index], ")", &status_var);
  RemoveExtraWhitespace(status_var, &stripped_status_var);
  dep_term->push_back(Expression());
  if (!ParseExpression(stripped_status_var, false, &dep_term->back())) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Last Dependent variable (expression on left side of '=' "
                    "sign) cannot be parsed: '" + input + "'.\n";
    }
    return false;
  }
  dep_vars_names->push_back(GetExpressionString(dep_term->back()));

  return true;
}

bool ParseDependentTermTimeDependentNpmleModel(
    const string& input,
    vector<Expression>* dep_term, vector<string>* time_vars_names,
    vector<string>* dep_vars_names, string* error_msg) {
  // Sanity check input.
  if (dep_term == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: NULL term in call to "
                    "ParseDependentTermTimeDependentNpmleModel. Check API.";
    }
    return false;
  }

  // Return false if input is empty.
  if (input.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Dependent variable (expression on left side of '=' "
                    "sign) is empty (or contains only whitespace).\n";
    }
    return false;
  }
  
  // Remove enclosing parentheses.
  const string input_stripped = StripParentheses(input);

  // Split LHS around the "," which delineates between Examination Time
  // and Status (of each of the K dependent variables).
  vector<string> parts;
  Split(input_stripped, ",", &parts);
  if (parts.size() < 2) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to parse LHS of Time-Dependent NPMLE "
                    "Regression equation: " + input + "\n";
    }
    return false;
  }

  // Process dep variables.
  for (int i = 0; i < parts.size(); ++i) {
    string dep_var;
    RemoveExtraWhitespace(parts[i], &dep_var);
    dep_term->push_back(Expression());
    if (!ParseExpression(dep_var, false, &dep_term->back())) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: " + Itoa(i) + " Dependent variable (expression on "
                      "left side of '=' sign) cannot be parsed: '" + input + "'.\n";
      }
      return false;
    }
    if (i == 0) {
      time_vars_names->push_back(GetExpressionString(dep_term->back()));
    } else {
      dep_vars_names->push_back(GetExpressionString(dep_term->back()));
    }
  }

  return true;
}

bool ParseDependentTermTimeIndependentNpmleModel(
    const string& input,
    vector<Expression>* dep_term, vector<string>* time_vars_names,
    vector<string>* dep_vars_names, string* error_msg) {
  // Sanity check input.
  if (dep_term == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: NULL term in call to "
                    "ParseDependentTermTimeIndependentNpmleModel. Check API.";
    }
    return false;
  }

  // Return false if input is empty.
  if (input.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Dependent variable (expression on left side of '=' "
                    "sign) is empty (or contains only whitespace).\n";
    }
    return false;
  }
  
  // Remove enclosing parentheses.
  const string input_stripped = StripParentheses(input);

  // Split around ";", which delineates between the dependent variables
  // (in the multivariate NPMLE case).
  vector<string> dep_var_parts;
  Split(input_stripped, ";", &dep_var_parts);
  for (int i = 0; i < dep_var_parts.size(); ++i) {
    const string& dep_var_part = dep_var_parts[i];

    // Split LHS around the "," which delineates between the Time Left-Endpoint(s)
    // and Right-Endpoint(s).
    vector<string> parts;
    Split(dep_var_part, ",", &parts);
    if (parts.size() % 2 != 0) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to parse LHS of Time-Dependent NPMLE "
                      "Regression equation: " + input + "\n";
      }
      return false;
    }

    // Extract dependent variable name, if present.
    vector<string> name_left_parts;
    Split(parts[0], ":", &name_left_parts);
    if (name_left_parts.size() > 2) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to parse term " + Itoa(i) + " of LHS of "
                      "Time-Dependent NPMLE Regression equation: " + input + "\n";
      }
      return false;
    }
    if (name_left_parts.size() == 2) {
      dep_vars_names->push_back(name_left_parts[0]);
    } else {
      dep_vars_names->push_back("Dep_Var_" + Itoa(1 + i / 2));
    }

    // Extract Left-End timepoint.
    dep_term->push_back(Expression());
    if (!ParseExpression(
            name_left_parts.back(), false, &dep_term->back())) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: 1st Dependent variable (expression on left side of '=' "
                      "sign) cannot be parsed: '" + input + "'.\n";
      }
      return false;
    }
    time_vars_names->push_back(GetExpressionString(dep_term->back()));

    // Extract Right-End timepoint.
    dep_term->push_back(Expression());
    if (!ParseExpression(parts[1], false, &dep_term->back())) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: 1st Dependent variable (expression on left side of '=' "
                      "sign) cannot be parsed: '" + input + "'.\n";
      }
      return false;
    }
    time_vars_names->push_back(GetExpressionString(dep_term->back()));
  }

  return true;
}

bool GetTitles(
    const string& title_line, const string& delimiter,
    vector<string>* titles, set<int>* nominal_columns) {
  if (titles == nullptr) return false;

  titles->clear();
  Split(
      title_line, delimiter, false /* Do not collapse consecutive delimiters */,
      titles);

  // Go through all titles, looking for nomial columns (based on '$' suffix).
  for (int column_num = 0; column_num < titles->size(); ++column_num) {
    const string& current_title = (*titles)[column_num];
    if (nominal_columns != nullptr && current_title.length() > 0 &&
        current_title.substr(current_title.length() - 1) == "$") {
      nominal_columns->insert(column_num);
    }
  }

  // Sanity check that titles are distinct.
  map<string, int> names;
  for (int i = 0; i < titles->size(); ++i) {
    if (names.find((*titles)[i]) != names.end()) return false;
    names.insert(make_pair((*titles)[i], i));
  }

  return true;
}

bool GetHeader(
    const FileInfo& file_info, vector<string>* header, string* error_msg) {
  if (header == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in GetHeader: Null input.\n";
    }
    return false;
  }

  const string& filename = file_info.name_;
  const string& delimiter = file_info.delimiter_;
  const string& comment_char = file_info.comment_char_;
  if (delimiter.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Reading data file Header: No column delimiter specified.\n";
    }
    return false;
  }

  // Open input file.
  ifstream input_file(filename.c_str());
  if (!input_file.is_open()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to find data file '" + filename +
                    "'. Make sure that it is present in your current directory.\n";
    }
    return false;
  }
  
  // Read Title line of input file.
  string title_line;
  bool found_title_line = false;
  while (!found_title_line && getline(input_file, title_line)) {
    if (comment_char.empty() ||
        !HasPrefixString(title_line, comment_char)) {
      found_title_line = true;
    }
  }
  if (!found_title_line) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in reading data file '" + filename +
                    "': file is empty.\n";
    }
    return false;
  }
  RemoveWindowsTrailingCharacters(&title_line);
  set<int> nominal_columns;
  if (!GetTitles(title_line, delimiter, header, &nominal_columns)) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR parsing header line in data file '" + filename +
                    "': Improper file format.\n";
    }
    return false;
  }
  return true;
}

string GetDependentVarString(
    const ModelType model_type, const DepVarDescription& dependent_var) {
  if (model_type == ModelType::MODEL_TYPE_LINEAR) {
    return GetExpressionString(dependent_var.model_lhs_linear_);
  } else if (model_type == ModelType::MODEL_TYPE_LOGISTIC) {
    return GetExpressionString(dependent_var.model_lhs_logistic_);
  } else if (model_type == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL) {
    string dep_var_str = "(";
    for (const Expression& dep_var_term : dependent_var.model_lhs_cox_) {
      if (dep_var_str != "(") {
        dep_var_str += ", ";
      }
      dep_var_str += GetExpressionString(dep_var_term);
    }
    dep_var_str += ")";
    return dep_var_str;
  } else if (model_type == ModelType::MODEL_TYPE_INTERVAL_CENSORED) {
      return "ERROR getting dep var string: Unexpected ModelType: " +
             Itoa(static_cast<int>(model_type)) + "; at this point, "
             "should have determined if this was time-indep or time-dep.\n";
  } else if (model_type == ModelType::MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED) {
    string dep_var_str = "(";
    for (const Expression& dep_var_term : dependent_var.model_lhs_time_dep_npmle_) {
      if (dep_var_str != "(") {
        dep_var_str += ", ";
      }
      dep_var_str += GetExpressionString(dep_var_term);
    }
    dep_var_str += ")";
    return dep_var_str;
  } else if (model_type == ModelType::MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED) {
    string dep_var_str = "(";
    for (const Expression& dep_var_term : dependent_var.model_lhs_time_indep_npmle_) {
      if (dep_var_str != "(") {
        dep_var_str += ", ";
      }
      dep_var_str += GetExpressionString(dep_var_term);
    }
    dep_var_str += ")";
    return dep_var_str;
  } else {
    return "ERROR getting dep var string: unrecognized ModelType: " +
           Itoa(static_cast<int>(model_type)) + "\n";
  }
}

void PrintFinalModel(
    const ModelType type, const DepVarDescription& model_lhs,
    const vector<string>& legend, string* final_model) {
  if (final_model == nullptr) return;
  string constant_term_in_title = "";
  string constant_term_in_model = "";
  const bool include_constant_term =
      type == ModelType::MODEL_TYPE_LINEAR ||
      type == ModelType::MODEL_TYPE_LOGISTIC;
  if (include_constant_term) {
    constant_term_in_title = ", constant term (B_0)";
    constant_term_in_model = "B_0";
  }
  string indep_vars_str = " = ";
  for (int i = 0; i < legend.size(); ++i) {
    const string& legend_term = legend[i];
    int j = include_constant_term ? i : i + 1;
    if (j == 0) {
      // Constant term.
      indep_vars_str += constant_term_in_model;
      continue;
    }
    // If we already have at least one term (the constant term), add
    // the proper '+' sepearator (here, length 3 is because indep_vars_str
    // is initialized with having 3 characters for ' = ').
    if (indep_vars_str.length() > 3) {
      indep_vars_str += " + ";
    }
    indep_vars_str += "B_" + Itoa(j) + " * " + legend_term;
  }

  const bool include_error_term = type == ModelType::MODEL_TYPE_LINEAR;
  const string dep_var_str = GetDependentVarString(type, model_lhs);
  const string error_term_in_title = include_error_term ?
      ", and error term (e)):\n\t" : "):\n\t";
  const string error_term_in_model =
      include_error_term ? " + e\n" : "\n";
  const string model_msg =
      "Model (includes indicator variables for NOMINAL variables" +
      constant_term_in_title + error_term_in_title + dep_var_str +
      indep_vars_str + error_term_in_model;
  *final_model = model_msg;
}

bool ParseModelAndDataParams(ModelAndDataParams* params) {
  if (params == nullptr) {
    return false;
  }
  
  // Header.
  if (params->header_.empty() && !params->file_.name_.empty() &&
       !GetHeader(params->file_, &params->header_, &params->error_msg_)) {
    params->error_msg_ += "ERROR in Parsing arguments: Unable to get header.\n";
    return false;
  }

  // Nominal Columns.
  if (!GetNominalColumnsFromTitles(params->header_, &params->nominal_columns_)) {
    params->error_msg_ +=
        "ERROR in Parsing arguments: Unable to get nominal columns.\n";
    return false;
  }

  // Update ModelType, if appropriate. NOTE: Must do this before parsing model,
  // as in some cases (ModelType == MODEL_TYPE_INTERVAL_CENSORED), the final
  // model type will be determined by the presence/absence of id_col (i.e. that
  // determines whether data is time-dependent or time-independent).
  if (params->model_type_ == ModelType::MODEL_TYPE_INTERVAL_CENSORED) {
    if (params->id_str_.empty()) {
      params->model_type_ =
          ModelType::MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED;
    } else {
      params->model_type_ =
          ModelType::MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED;
    }
  }

  // Model.
  if (!params->model_str_.empty() &&
      !ParseModel(params->model_str_, params->left_truncation_str_,
                  params->model_type_, params->header_,
                  &params->model_lhs_, &params->model_rhs_,
                  &params->use_subgroup_as_covariate_,
                  &params->input_cols_used_, &params->error_msg_)) {
    params->error_msg_ += "ERROR in Parsing arguments: Unable to parse model.\n";
    return false;
  }

  // Subgroup(s).
  if (!params->subgroup_str_.empty() &&
      !ParseSubgroups(
          params->subgroup_str_, params->header_,
          &params->subgroup_cols_, &params->subgroups_,
          &params->input_cols_used_, &params->error_msg_)) {
    params->error_msg_ +=
        "ERROR in Parsing arguments: Unable to parse subgroups.\n";
    return false;
  }

  // Strata.
  if (!params->strata_str_.empty() &&
      !ParseStrata(
          params->strata_str_, params->header_, &params->strata_cols_,
          &params->input_cols_used_, &params->error_msg_)) {
    params->error_msg_ += "ERROR in Parsing arguments: Unable to parse strata.\n";
    return false;
  }

  // Family, Id, and Weight.
  if (!ParseIdWeightAndFamily(
          params->id_str_, params->weight_str_, params->family_str_,
          params->header_,
          &params->id_col_, &params->weight_col_, &params->family_cols_,
          &params->input_cols_used_, &params->error_msg_)) {
    params->error_msg_ += "ERROR in Parsing arguments: Unable to parse id, "
                  "Family, or Weight.\n";
    return false;
  }

  // Miscellaneous Columns (None as of now).
  /*
  if (!ParseMiscellaneousColumns()) {
    params->error_msg_ += "ERROR in Parsing arguments: Unable to parse "
                  "some columns.\n";
    return false;
  }
  */

  // VariableParams.
  //   - VariableCollapseParams.
  map<string, vector<VariableCollapseParams>> var_name_to_collapse_params;
  if (!params->collapse_params_str_.empty() &&
      !ParseCollapseParams(
          params->collapse_params_str_, params->file_.infinity_char_,
          &var_name_to_collapse_params, &params->error_msg_)) {
    return false;
  }
  //   - TimeDependentParams.
  map<string, TimeDependentParams> var_name_to_time_params;
  if (!params->time_params_str_.empty() &&
      !ParseTimeDependentParams(
          params->time_params_str_, &var_name_to_time_params,
          &params->error_msg_)) {
    return false;
  }
  //   - VariableNormalization Params.
  map<string, VariableNormalization> var_name_to_normalization;
  if (!params->var_norm_params_str_.empty() &&
      !ParseVariableNormalizationParams(
          params->var_norm_params_str_,
          &var_name_to_normalization, &params->error_msg_)) {
    return false;
  }
  // Now combine them all, and update column name and index.
  if (!CombineVariableParams(
          params->header_, var_name_to_collapse_params,
          var_name_to_time_params, var_name_to_normalization,
          &(params->var_params_), &params->error_msg_)) {
    params->error_msg_ += "ERROR in Parsing arguments: Inconsistent variable "
                          "names provided.\n";
    return false;
  }

  return true;
}

bool ComputeRowCovariateValues(
    const int subgroup_index,
    const vector<vector<string>>& subgroups,
    const set<string> na_strings,
    const map<int, set<string> >& nominal_columns,
    const vector<Expression>& linear_terms,
    const map<string, int>& name_to_column,
    const vector<DataHolder>& sample_values,
    VectorXd* linear_terms_values,
    string* error_msg) {
  // Create the mapping of variable name to value.
  map<string, double> var_values;
  bool is_na_row = false;
  if (!GetVariableValuesFromDataRow(
          subgroup_index, subgroups, na_strings, nominal_columns,
          name_to_column, sample_values,
          &is_na_row, &var_values, error_msg)) {
    return false;
  }
  
  // Don't keep rows with missing values.
  if (is_na_row) return true;

  VectorXd& new_sample_row = *linear_terms_values;
  new_sample_row.resize(linear_terms.size());
  for (int i = 0; i < linear_terms.size(); ++i) {
    const Expression& linear_term = linear_terms[i];
    if (!EvaluateExpression(linear_term, var_values, &new_sample_row[i], error_msg)) {
      return false;
    }
  }
  return true;
}

bool GetVariableValuesFromDataRow(
    const int subgroup_index,
    const vector<vector<string>>& subgroups,
    const set<string> na_strings,
    const map<int, set<string> >& nominal_columns,
    const map<string, int>& name_to_column,
    const vector<DataHolder>& sample_values,
    bool* is_na_row, map<string, double>* var_values, string* error_msg) {
  // First, go through all subgroups, determining which one this row
  // (as represented by 'sample_values') belongs to. Put a '1' in
  // the corresponding subgroup indicator, and '0's n the others.
  if (subgroup_index >= 0) {
    // IMPORTANT: This assumes that nominal variables are expanded into
    // k - 1 covariates by skipping the first distinct value (in this case,
    // Subgroup_0), and then setting the k - 1 covariates as I_Subgroup_k.
    for (int i = 1; i < subgroups.size(); ++i) {
      const vector<string>& subgroup_i = subgroups[i];
      // IMPORTANT: the (formation of the) string below should exactly match
      // how it will appear (as a variable name) in an Expression; in particular,
      // this string should match the corresponding string in ExpandExpression.
      var_values->insert(make_pair(
          "I_(Subgroup=" + Join(subgroup_i, ",") + ")",
          (subgroup_index == i) ? 1.0 : 0.0));
    }
  }

  // Go though all variables, adding their value. In particular, for
  // numeric variables, just read their value. For non-numeric (nominal)
  // variables, for all expanded indicator variables, mark them as '1'
  // or '0', as appropriate.
  for (const pair<string, int>& var_name_and_column : name_to_column) {
    const string& var_name = var_name_and_column.first;
    const int var_column = var_name_and_column.second;

    const DataHolder& var_value = sample_values[var_column];

    // Sanity-Check the DataType has been set.
    if (var_value.type_ != DataType::DATA_TYPE_STRING &&
        var_value.type_ != DataType::DATA_TYPE_NUMERIC) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unsupported DataType: " +
                      Itoa(static_cast<int>(var_value.type_)) + ".\n";
      }
      return false;
    }

    // Check if this variable is NA.
    if (is_na_row != nullptr && !na_strings.empty()) {
      const string& var_value_str =
          var_value.type_ == DataType::DATA_TYPE_STRING ?
          var_value.name_ : Itoa(var_value.value_);
      if (na_strings.find(var_value_str) != na_strings.end()) {
        *is_na_row = true;
        return true;
      }
    }

    map<int, set<string>>::const_iterator col_to_values_itr =
        nominal_columns.find(var_column);
    if (col_to_values_itr == nominal_columns.end()) {
      if (var_value.type_ != DataType::DATA_TYPE_NUMERIC) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Expected NUMERIC DataType for non-nominal "
                        "column '" + var_name + "', but found STRING: '" +
                        var_value.name_ + "'.\n";
        }
        return false;
      }
      // This is not a nominal variable. Just add its value to var_values.
      var_values->insert(make_pair(var_name, var_value.value_));
      continue;
    }
    // If reached here, present variable is nominal. Handle it.
    if (var_value.type_ != DataType::DATA_TYPE_STRING) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Expected STRING DataType for nominal "
                      "column '" + var_name + "', but found NUMERIC value: " +
                      Itoa(var_value.value_) + ".\n";
      }
      return false;
    }
    const string& var_value_str = var_value.name_;
    // IMPORTANT: This assumes that nominal variables are expanded into
    // k - 1 covariates by skipping the first distinct value, and then
    // setting the k - 1 covariates as I_distinct_value_k.
    bool is_first_distinct_value = true;
    for (const string& distinct_value : col_to_values_itr->second) {
      if (is_first_distinct_value) {
        is_first_distinct_value = false;
        continue;
      }
      // IMPORTANT: the (formation of the) string below should exactly match
      // how it will appear (as a variable name) in an Expression; in particular,
      // this string should match the corresponding string in ExpandExpression.
      var_values->insert(make_pair(
            "I_(" + var_name + "=" + distinct_value + ")",
            (var_value_str == distinct_value) ? 1.0 : 0.0));
    }
  }

  return true;
}

bool GetVariableValuesFromDataRow(
    const set<string>& vars_in_linear_terms,
    const int subgroup_index,
    const vector<vector<string>>& subgroups,
    const set<string> na_strings,
    const map<string, set<string> >& nominal_variables,
    const map<string, DataHolder>& sample_values,
    bool* is_na_row, map<string, double>* var_values, string* error_msg) {
  // First, go through all subgroups, determining which one this row
  // (as represented by 'sample_values') belongs to. Put a '1' in
  // the corresponding subgroup indicator, and '0's n the others.
  if (subgroup_index >= 0) {
    // IMPORTANT: This assumes that nominal variables are expanded into
    // k - 1 covariates by skipping the first distinct value (in this case,
    // Subgroup_0), and then setting the k - 1 covariates as I_Subgroup_k.
    for (int i = 1; i < subgroups.size(); ++i) {
      const vector<string>& subgroup_i = subgroups[i];
      // IMPORTANT: the (formation of the) string below should exactly match
      // how it will appear (as a variable name) in an Expression; in particular,
      // this string should match the corresponding string in ExpandExpression.
      var_values->insert(make_pair(
          "I_(Subgroup=" + Join(subgroup_i, ",") + ")",
          (subgroup_index == i) ? 1.0 : 0.0));
    }
  }

  // Go though all variables, adding their value. In particular, for
  // numeric variables, just read their value. For non-numeric (nominal)
  // variables, for all expanded indicator variables, mark them as '1'
  // or '0', as appropriate.
  for (const string& col_to_read : vars_in_linear_terms) {
    map<string, DataHolder>::const_iterator var_name_and_value_itr =
        sample_values.find(col_to_read);
    if (var_name_and_value_itr == sample_values.end()) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to find column '" + col_to_read +
                      "' among the columns of the data file.\n";
      }
      return false;
    }
    const string& var_name = var_name_and_value_itr->first;
    const DataHolder& var_value = var_name_and_value_itr->second;

    // Sanity-Check the DataType has been set.
    if (var_value.type_ != DataType::DATA_TYPE_STRING &&
        var_value.type_ != DataType::DATA_TYPE_NUMERIC) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unsupported Datatype: " +
                      Itoa(static_cast<int>(var_value.type_)) + ".\n";
      }
      return false;
    }

    // Check if this variable is NA.
    if (is_na_row != nullptr && !na_strings.empty()) {
      const string& var_value_str =
          var_value.type_ == DataType::DATA_TYPE_STRING ?
          var_value.name_ : Itoa(var_value.value_);
      if (na_strings.find(var_value_str) != na_strings.end()) {
        *is_na_row = true;
        return true;
      }
    }

    map<string, set<string>>::const_iterator nominal_values_itr =
        nominal_variables.find(var_name);
    if (nominal_values_itr == nominal_variables.end()) {
      // This is NOT a nominal variable. Just add its value to var_values.
      if (var_value.type_ != DataType::DATA_TYPE_NUMERIC) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Expected NUMERIC DataType for non-nominal "
                        "column '" + var_name + "', but found STRING: '";
                        var_value.name_ + "'.\n";
        }
        return false;
      }
      var_values->insert(make_pair(var_name, var_value.value_));
      continue;
    }
    // If reached here, present variable IS nominal. Handle it.
    if (var_value.type_ != DataType::DATA_TYPE_STRING) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Expected STRING DataType for nominal "
                      "column '" + var_name + "', but found NUMERIC value: " +
                      Itoa(var_value.value_) + ".\n";
      }
      return false;
    }
    const string& var_value_str = var_value.name_;
    // IMPORTANT: This assumes that nominal variables are expanded into
    // k - 1 covariates by skipping the first distinct value, and then
    // setting the k - 1 covariates as I_distinct_value_k.
    bool is_first_distinct_value = true;
    for (const string& distinct_value : nominal_values_itr->second) {
      if (is_first_distinct_value) {
        is_first_distinct_value = false;
        continue;
      }
      // IMPORTANT: the (formation of the) string below should exactly match
      // how it will appear (as a variable name) in an Expression; in particular,
      // this string should match the corresponding string in ExpandExpression.
      var_values->insert(make_pair(
            "I_(" + var_name + "=" + distinct_value + ")",
            (var_value_str == distinct_value) ? 1.0 : 0.0));
    }
  }

  return true;
}

bool GetLegendAndLinearTerms(
    const bool has_constant_term,
    const map<string, int>& variable_name_to_column,
    const vector<vector<string>>& subgroups,
    const map<int, set<string> >& nominal_columns,
    const Expression& expression,
    vector<set<int>>* orig_linear_terms_to_legend_indices,
    vector<Expression>* linear_terms,
    vector<string>* legend,
    string* error_msg) {
  // First, parse Expression by separating the linear terms into sub-expressions.
  vector<Expression> non_expanded_terms;
  if (!GetLinearTermsInExpression(expression, &non_expanded_terms, error_msg)) {
    return false;
  }

  // Now iterate through the linear terms, expanding non-numeric variables
  // into the appropriate number of covariates.
  linear_terms->clear();
  legend->clear();
  int current_num_linear_terms = 0;
  // For Linear/Logistic Regression Models, there is a constant term
  // assumed for RHS of the model.
  if (has_constant_term) {
    legend->push_back("Constant");
    linear_terms->push_back(Expression());
    Expression& constant_term = linear_terms->back();
    ParseExpression("1.0", &constant_term);
    current_num_linear_terms = linear_terms->size();
  }
  string current_expanded_term = "";
  for (int i = 0; i < non_expanded_terms.size(); ++i) {
    current_num_linear_terms = linear_terms->size();
    const Expression& linear_term = non_expanded_terms[i];
    if (!ExpandExpression(
            linear_term, variable_name_to_column, subgroups, nominal_columns,
            current_expanded_term, linear_terms, legend, error_msg)) {
      return false;
    }
    if (orig_linear_terms_to_legend_indices != nullptr) {
      orig_linear_terms_to_legend_indices->push_back(set<int>());
      set<int>& indices_term_i_is_part_of = orig_linear_terms_to_legend_indices->back();
      for (int j = current_num_linear_terms; j < linear_terms->size(); ++j) {
        indices_term_i_is_part_of.insert(j);
      }
    }
  }
  return true;
}

bool GetLegendAndLinearTerms(
    const bool has_constant_term,
    const set<string>& variable_names,
    const vector<vector<string>>& subgroups,
    const map<string, set<string> >& nominal_variables,
    const Expression& expression,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg) {
  // First, parse Expression by separating the linear terms into sub-expressions.
  vector<Expression> non_expanded_terms;
  if (!GetLinearTermsInExpression(expression, &non_expanded_terms, error_msg)) {
    return false;
  }

  // Now iterate through the linear terms, expanding non-numeric variables
  // into the appropriate number of covariates.
  linear_terms->clear();
  legend->clear();
  // For Linear/Logistic Regression Models, there is a constant term
  // assumed for RHS of the model.
  if (has_constant_term) {
    legend->push_back("Constant");
    linear_terms->push_back(Expression());
    Expression& constant_term = linear_terms->back();
    ParseExpression("1.0", &constant_term);
  }
  string current_expanded_term = "";
  for (const Expression& linear_term : non_expanded_terms) {
    if (!ExpandExpression(
            linear_term, variable_names, subgroups, nominal_variables,
            current_expanded_term, linear_terms, legend, error_msg)) {
      return false;
    }
  }
  return true;
}

bool GetLegendAndLinearTermsForSimulation(
    const bool has_constant_term,
    const Expression& expression,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg) {
  linear_terms->clear();
  if (legend != nullptr) legend->clear();

  // For Linear/Logistic Regression Models, there is a constant term
  // assumed for RHS of the model.
  if (has_constant_term) {
    linear_terms->push_back(Expression());
    Expression& constant_term = linear_terms->back();
    ParseExpression("1.0", &constant_term);
  }

  // Parse Expression by separating the linear terms into sub-expressions.
  if (!GetLinearTermsInExpression(expression, linear_terms, error_msg)) {
    return false;
  }

  // Construct Legend.
  bool is_first_linear_term = true;
  for (const Expression& linear_term : *linear_terms) {
    if (is_first_linear_term && has_constant_term) {
      is_first_linear_term = false;
      if (legend != nullptr) legend->push_back("Constant");
      continue;
    }
    is_first_linear_term = false;
    if (legend != nullptr) {
      legend->push_back(StripParentheses(GetExpressionString(linear_term)));
    }
  }
  return true;
}

bool GetLinearTermsInvolvingEachColumn(
    const vector<Expression>& linear_terms,
    const vector<string>& temp_data_header,
    const vector<string>& orig_header,
    const set<int>& nominal_columns,
    vector<set<int>>* col_index_to_linear_terms) {
  if (col_index_to_linear_terms == nullptr) return true;
  col_index_to_linear_terms->clear();
  col_index_to_linear_terms->resize(orig_header.size(), set<int>());

  // Will need to map variable column, with respect to original data file,
  // to the column index with respect to the values passed in, since the
  // set of nominal columns is with respect to the latter.
  map<int, int> orig_column_to_data_column;
  for (int i = 0; i < orig_header.size(); ++i) {
    const string& orig_col = orig_header[i];
    bool found_match = false;
    for (int j = 0; j < temp_data_header.size(); ++j) {
      if (temp_data_header[j] == orig_col) {
        orig_column_to_data_column.insert(make_pair(i, j));
        found_match = true;
        break;
      }
    }
  }

  // Loop through all linear terms.
  for (int i = 0; i < linear_terms.size(); ++i) {
    const Expression& linear_term = linear_terms[i];
    set<string> vars_in_linear_term;
    string error_msg = "";
    if (!ExtractVariablesFromExpression(
            linear_term, &vars_in_linear_term, &error_msg)) {
      cout << "ERROR parsing linear term '"
           << GetExpressionString(linear_term)
           << "'" << endl;
      return false;
    }

    // Loop over all the variables appearing in this linear term.
    for (const string& var_name : vars_in_linear_term) {
      bool found_match = false;
      // Check each variable in header, to see if it matches the current
      // var_name.
      for (const pair<int, int>& orig_to_data : orig_column_to_data_column) {
        const int orig_index = orig_to_data.first;
        const int data_index = orig_to_data.second;
        const string& col_name = orig_header[orig_index];
             
        // Check the column name itself, as well as what this column would
        // have been expanded to as an indicator variable, in case the column
        // is NOMINAL.
        if (col_name == var_name ||
            (nominal_columns.find(data_index) != nominal_columns.end() &&
             HasPrefixString(var_name, "I_(" + col_name + "=") &&
             HasSuffixString(var_name, ")"))) {
          (*col_index_to_linear_terms)[orig_index].insert(i);
          found_match = true;
          break;
        }
      }
      if (!found_match) {
        cout << "ERROR: Unable to find which original data column the "
             << "variable '" << var_name << "' came from." << endl;
        return false;
      }
    }
  }

  return true;
}

bool GetLinearTermsInvolvingEachColumn(
    const vector<Expression>& linear_terms,
    const vector<string>& header,
    const set<string>& nominal_columns,
    vector<set<int>>* col_index_to_linear_terms) {
  if (col_index_to_linear_terms == nullptr) return true;
  col_index_to_linear_terms->clear();
  col_index_to_linear_terms->resize(header.size(), set<int>());

  // Loop through all linear terms.
  for (int i = 0; i < linear_terms.size(); ++i) {
    const Expression& linear_term = linear_terms[i];
    set<string> vars_in_linear_term;
    string error_msg = "";
    if (!ExtractVariablesFromExpression(
            linear_term, &vars_in_linear_term, &error_msg)) {
      cout << "ERROR parsing linear term '"
           << GetExpressionString(linear_term)
           << "'" << endl;
      return false;
    }

    // Loop over all the variables appearing in this linear term.
    for (const string& var_name : vars_in_linear_term) {
      bool found_match = false;
      // Check each variable in header, to see if it matches the current
      // var_name.
      for (int j = 0; j < header.size(); ++j) {
        const string& col_name = header[j];
             
        // Check the column name itself, as well as what this column would
        // have been expanded to as an indicator variable, in case the column
        // is NOMINAL.
        if (col_name == var_name ||
            (nominal_columns.find(col_name) != nominal_columns.end() &&
             HasPrefixString(var_name, "I_(" + col_name + "=") &&
             HasSuffixString(var_name, ")"))) {
          (*col_index_to_linear_terms)[j].insert(i);
          found_match = true;
          break;
        }
      }
      if (!found_match) {
        cout << "ERROR: Unable to find which original data column the "
             << "variable '" << var_name << "' came from." << endl;
        return false;
      }
    }
  }

  return true;
}

bool GetLinearTermsInExpression(
    const Expression& expression,
    vector<Expression>* linear_terms, string* error_msg) {
  if (IsEmptyExpression(expression)) return true;

  // If the operation is not Addition, then the expression represents
  // a single linear term. Add the whole thing to linear_terms and return.
  if (expression.op_ != Operation::ADD) {
    linear_terms->push_back(CopyExpression(expression));
    return true;
  }

  // The expression consists of two or more linear terms. Process them.
  if (expression.subterm_one_ == nullptr ||
      expression.subterm_two_ == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in parsing Model RHS: one of "
                    "the subterms is null.\n\t" +
                    GetExpressionString(expression) + "\n";
    }
    return false;
  }
  if (!GetLinearTermsInExpression(
          *expression.subterm_one_, linear_terms, error_msg)) {
    return false;
  }
  if (!GetLinearTermsInExpression(
          *expression.subterm_two_, linear_terms, error_msg)) {
    return false;
  }
  return true;
}

bool ExpandExpression(
    const Expression& expression,
    const map<string, int>& variable_name_to_column,
    const vector<vector<string>>& subgroups,
    const map<int, set<string> >& nominal_columns,
    const string& current_title,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg) {
  // Go through all the variable (names) of expression, storing those encountered.
  set<string> variables_seen;
  if (!ExtractVariablesFromExpression(
          expression, Keys(variable_name_to_column), &variables_seen, error_msg)) {
    return false;
  }

  // For non-numeric variables: determine which are present, and prepare the
  // strings of their expanded counterparts.
  map<string, pair<int, vector<string>>> var_name_to_expanded_name;
  vector<int> factors;
  factors.push_back(1);
  int non_numeric_index = 0;
  for (const string& seen_var : variables_seen) {
    if (seen_var == kSubgroupIndicator) {
      if (subgroups.size() < 2) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unable to use I_Subgroup as a covariate, as "
                        "there are only " + Itoa(static_cast<int>(subgroups.size())) +
                        "subgroups.\n";
        }
        return false;
      }
      factors.push_back(subgroups.size() - 1);
      vector<string>& expanded_names = var_name_to_expanded_name.insert(
          make_pair(seen_var, make_pair(non_numeric_index, vector<string>()))).
          first->second.second;
      non_numeric_index++;
      // IMPORTANT: This assumes that nominal variables are expanded into
      // k - 1 covariates by skipping the first distinct value (in this case,
      // Subgroup_0), and then setting the k - 1 covariates as I_Subgroup_k.
      for (int i = 1; i < subgroups.size(); ++i) {
        const vector<string>& subgroup_i = subgroups[i];
        expanded_names.push_back("I_(Subgroup=" + Join(subgroup_i, ",") + ")");
      }
    } else {
      map<string, int>::const_iterator var_to_col_itr =
          variable_name_to_column.find(seen_var);
      if (var_to_col_itr == variable_name_to_column.end()) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: No column in header line of data file has "
                        "name '" + seen_var + "' from term on model RHS:\n\t" +
                        GetExpressionString(expression) + "\n";
        }
        return false;
      }
      const int seen_var_col = var_to_col_itr->second;
      map<int, set<string>>::const_iterator col_to_values_itr =
          nominal_columns.find(seen_var_col);
      if (col_to_values_itr == nominal_columns.end()) {
        // This is not a nominal variable. Nothing to do.
        continue;
      }
      factors.push_back(col_to_values_itr->second.size() - 1);
      vector<string>& expanded_names = var_name_to_expanded_name.insert(
          make_pair(seen_var, make_pair(non_numeric_index, vector<string>()))).
          first->second.second;
      non_numeric_index++;
      // IMPORTANT: This assumes that nominal variables are expanded into
      // k - 1 covariates by skipping the first distinct value, and then
      // setting the k - 1 covariates as I_distinct_value_k.
      bool is_first_distinct_value = true;
      for (const string& distinct_value : col_to_values_itr->second) {
        if (is_first_distinct_value) {
          is_first_distinct_value = false;
          continue;
        }
        expanded_names.push_back("I_(" + seen_var + "=" + distinct_value + ")");
      }
    }
  }

  // Now walk through Expression, replacing each non-numeric variable
  // with one of the expanded list.
  const int num_expansions = factors.back();
  for (int i = 0; i < num_expansions; ++i) {
    linear_terms->push_back(Expression());
    legend->push_back("");
    map<string, string> non_numeric_vars_to_expansion;
    for (const auto& var_name_and_expansions : var_name_to_expanded_name) {
      const string& orig_var_name = var_name_and_expansions.first;
      const vector<string>& var_expansions =
          var_name_and_expansions.second.second;
      const int non_numeric_index = var_name_and_expansions.second.first;
      if (non_numeric_index + 1 >= factors.size()) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in expanding non-numeric covariate to indicator "
                        "function: too many non_numeric variables found (" +
                        Itoa(non_numeric_index) +
                        " exceeds number found " +
                        Itoa(static_cast<int>(factors.size())) + ")\n";
        }
        return false;
      }
      const int expansion_index_to_use =
          (i % factors[non_numeric_index + 1]) / factors[non_numeric_index];
      non_numeric_vars_to_expansion.insert(make_pair(
            orig_var_name, var_expansions[expansion_index_to_use]));
    }
    if (!ReplaceVariableInExpressionWithIndicatorExpansion(
            expression, non_numeric_vars_to_expansion,
            &(linear_terms->back()), &(legend->back()), error_msg)) {
      return false;
    }
  }

  return true;
}

bool ExpandExpression(
    const Expression& expression,
    const set<string>& variable_names,
    const vector<vector<string>>& subgroups,
    const map<string, set<string> >& nominal_variables,
    const string& current_title,
    vector<Expression>* linear_terms, vector<string>* legend, string* error_msg) {
  // Go through all the variable (names) of expression, storing those encountered.
  set<string> variables_seen;
  if (!ExtractVariablesFromExpression(
          expression, variable_names, &variables_seen, error_msg)) {
    return false;
  }

  // For non-numeric variables: determine which are present, and prepare the
  // strings of their expanded counterparts.
  map<string, pair<int, vector<string>>> var_name_to_expanded_name;
  vector<int> factors;
  factors.push_back(1);
  int non_numeric_index = 0;
  for (const string& seen_var : variables_seen) {
    if (seen_var == kSubgroupIndicator) {
      if (subgroups.size() < 2) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unable to use I_Subgroup as a covariate, as "
                        "there are only " + Itoa(static_cast<int>(subgroups.size())) +
                        "subgroups.\n";
        }
        return false;
      }
      factors.push_back(subgroups.size() - 1);
      vector<string>& expanded_names = var_name_to_expanded_name.insert(
          make_pair(seen_var, make_pair(non_numeric_index, vector<string>()))).
          first->second.second;
      non_numeric_index++;
      // IMPORTANT: This assumes that nominal variables are expanded into
      // k - 1 covariates by skipping the first distinct value (in this case,
      // Subgroup_0), and then setting the k - 1 covariates as I_Subgroup_k.
      for (int i = 1; i < subgroups.size(); ++i) {
        const vector<string>& subgroup_i = subgroups[i];
        expanded_names.push_back("I_(Subgroup=" + Join(subgroup_i, ",") + ")");
      }
    } else {
      map<string, set<string>>::const_iterator nominal_var_to_values_itr =
          nominal_variables.find(seen_var);
      if (nominal_var_to_values_itr == nominal_variables.end()) {
        // This is not a nominal variable. Nothing to do.
        continue;
      }
      factors.push_back(nominal_var_to_values_itr->second.size() - 1);
      vector<string>& expanded_names = var_name_to_expanded_name.insert(
          make_pair(seen_var, make_pair(non_numeric_index, vector<string>()))).
          first->second.second;
      non_numeric_index++;
      // IMPORTANT: This assumes that nominal variables are expanded into
      // k - 1 covariates by skipping the first distinct value, and then
      // setting the k - 1 covariates as I_distinct_value_k.
      bool is_first_distinct_value = true;
      for (const string& distinct_value : nominal_var_to_values_itr->second) {
        if (is_first_distinct_value) {
          is_first_distinct_value = false;
          continue;
        }
        expanded_names.push_back("I_(" + seen_var + "=" + distinct_value + ")");
      }
    }
  }

  // Now walk through Expression, replacing each non-numeric variable
  // with one of the expanded list.
  const int num_expansions = factors.back();
  for (int i = 0; i < num_expansions; ++i) {
    linear_terms->push_back(Expression());
    legend->push_back("");
    map<string, string> non_numeric_vars_to_expansion;
    for (const auto& var_name_and_expansions : var_name_to_expanded_name) {
      const string& orig_var_name = var_name_and_expansions.first;
      const vector<string>& var_expansions =
          var_name_and_expansions.second.second;
      const int non_numeric_index = var_name_and_expansions.second.first;
      if (non_numeric_index + 1 >= factors.size()) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in expanding non-numeric covariate to indicator "
                        "function: too many non_numeric variables found (" +
                        Itoa(non_numeric_index) +
                        " exceeds number found " +
                        Itoa(static_cast<int>(factors.size())) + ")\n";
        }
        return false;
      }
      const int expansion_index_to_use =
          (i % factors[non_numeric_index + 1]) / factors[non_numeric_index];
      non_numeric_vars_to_expansion.insert(make_pair(
            orig_var_name, var_expansions[expansion_index_to_use]));
    }
    if (!ReplaceVariableInExpressionWithIndicatorExpansion(
            expression, non_numeric_vars_to_expansion,
            &(linear_terms->back()), &(legend->back()), error_msg)) {
      return false;
    }
  }

  return true;
}

bool ExtractVariablesFromExpression(
    const Expression& expression,
    const set<string>& var_names,
    set<string>* vars_in_expression, string* error_msg) {
  // Operations break down into 1 of 3 categories: Self-Operation (Identity),
  // 1-Term Operations, and 2-Term Operations. Handle each case.
  if (expression.op_ == Operation::IDENTITY) {
    // Self-Operation.
    const string& var_name = expression.var_name_;
    if (!var_name.empty()) {
      if (!var_names.empty() &&
          var_name != kSubgroupIndicator &&
          var_names.find(var_name) == var_names.end()) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Variable '" + var_name + "' in the model "
                        "does not appear on header line of input data file, "
                        "which has column names: {'" +
                        Join(var_names, "', '") + "'}\n";
        }
        return false;
      }
      vars_in_expression->insert(expression.var_name_);
    }
  } else if (expression.op_ == TAN || expression.op_ == SIN ||
             expression.op_ == COS || expression.op_ == SQRT ||
             expression.op_ == LOG || expression.op_ == EXP ||
             expression.op_ == FACTORIAL || expression.op_ == ABS) {
    // 1-Term Operation.
    if (expression.subterm_one_ == nullptr ||
        expression.subterm_two_ != nullptr) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in parsing model RHS covariates: "
                      "Unable to perform operation: one of "
                      "the subterms is null.\n\t" +
                      GetExpressionString(expression) + "\n";
      }
      return false;
    }
    // Iteratively parse the subterms.
    if (!ExtractVariablesFromExpression(
            *expression.subterm_one_, var_names,
            vars_in_expression, error_msg)) {
      return false;
    }
  } else if (expression.op_ == ADD || expression.op_ == SUB ||
             expression.op_ == MULT || expression.op_ == DIV ||
             expression.op_ == POW) {
    // 2-Term Operation.
    if (expression.subterm_one_ == nullptr ||
        expression.subterm_two_ == nullptr) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in parsing model RHS covariates: "
                      "Unable to perform add/mult/pow: one of "
                      "the subterms is null.\n\t" +
                      GetExpressionString(expression) + "\n";
      }
      return false;
    }

    // Iteratively parse the two subterms.
    if (!ExtractVariablesFromExpression(
            *expression.subterm_one_, var_names,
            vars_in_expression, error_msg)) {
      return false;
    }
    if (!ExtractVariablesFromExpression(
            *expression.subterm_two_, var_names,
            vars_in_expression, error_msg)) {
      return false;
    }
  }

  return true;
}

bool ExtractVariablesFromExpression(
    const ModelType& model_type,
    const DepVarDescription& model_lhs,
    const set<string>& var_names,
    set<string>* vars_in_expression, string* error_msg) {
  if (model_type == ModelType::MODEL_TYPE_LINEAR) {
    return ExtractVariablesFromExpression(
        model_lhs.model_lhs_linear_, var_names, vars_in_expression, error_msg);
  } else if (model_type == ModelType::MODEL_TYPE_LOGISTIC) {
    return ExtractVariablesFromExpression(
        model_lhs.model_lhs_logistic_, var_names, vars_in_expression, error_msg);
  } else if (model_type == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL) {
    for (const Expression& lhs_exp : model_lhs.model_lhs_cox_) {
      if (!ExtractVariablesFromExpression(
              lhs_exp, var_names, vars_in_expression, error_msg)) {
        return false;
      }
    }
    return true;
  } else if (model_type == ModelType::MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED) {
    for (const Expression& lhs_exp : model_lhs.model_lhs_time_dep_npmle_) {
      if (!ExtractVariablesFromExpression(
              lhs_exp, var_names, vars_in_expression, error_msg)) {
        return false;
      }
    }
    return true;
  } else if (model_type == ModelType::MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED) {
    for (const Expression& lhs_exp : model_lhs.model_lhs_time_indep_npmle_) {
      if (!ExtractVariablesFromExpression(
              lhs_exp, var_names, vars_in_expression, error_msg)) {
        return false;
      }
    }
    return true;
  } else {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to ExtractVariablesFromExpression for ModelType " +
                    Itoa(static_cast<int>(model_type)) + ".\n";
    }
    return false;
  }

  return true;
}

bool ReplaceVariableInExpressionWithIndicatorExpansion(
    const Expression& orig_expression,
    const map<string, string>& non_numeric_vars_to_expansion_var,
    Expression* new_expression, string* legend, string* error_msg) {
  new_expression->op_ = orig_expression.op_;
  // Operations break down into 1 of 3 categories: Self-Operation (Identity),
  // 1-Term Operations, and 2-Term Operations. Handle each case.
  if (orig_expression.op_ == Operation::IDENTITY) {
    // Self-Operation.
    const string term_name = orig_expression.is_constant_ ?
        Itoa(orig_expression.value_) : orig_expression.var_name_;
    const string term_name_to_use =
        FindWithDefault(term_name, non_numeric_vars_to_expansion_var, term_name);
    (*legend) += term_name_to_use;
    new_expression->var_name_ =
        orig_expression.var_name_.empty() ? "" : term_name_to_use;
    new_expression->value_ = orig_expression.value_;
    new_expression->is_constant_ = orig_expression.is_constant_;
  } else if (orig_expression.op_ == TAN || orig_expression.op_ == SIN ||
             orig_expression.op_ == COS || orig_expression.op_ == SQRT ||
             orig_expression.op_ == LOG || orig_expression.op_ == EXP ||
             orig_expression.op_ == FACTORIAL || orig_expression.op_ == ABS) {
    // 1-Term Operation.
    if (orig_expression.subterm_one_ == nullptr ||
        orig_expression.subterm_two_ != nullptr) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in parsing model RHS: "
                      "Unable to perform operation: one of "
                      "the subterms is null.\n\t" +
                      GetExpressionString(orig_expression) + "\n";
      }
      return false;
    }
    // Iteratively parse the subterms.
    new_expression->subterm_one_ = new Expression();
    string first_term_legend = "";
    if (!ReplaceVariableInExpressionWithIndicatorExpansion(
            *orig_expression.subterm_one_, non_numeric_vars_to_expansion_var,
            new_expression->subterm_one_, &first_term_legend, error_msg)) {
      return false;
    }

    // Store the subterm results.
    (*legend) += GetOpString(orig_expression.op_, first_term_legend);
  } else if (orig_expression.op_ == ADD || orig_expression.op_ == SUB ||
             orig_expression.op_ == MULT || orig_expression.op_ == DIV ||
             orig_expression.op_ == POW) {
    // 2-Term Operation.
    if (orig_expression.subterm_one_ == nullptr ||
        orig_expression.subterm_two_ == nullptr) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in parsing model RHS: "
                      "Unable to perform add/mult/pow: one of "
                      "the subterms is null.\n\t" +
                      GetExpressionString(orig_expression) + "\n";
      }
      return false;
    }

    // Iteratively parse the two subterms.
    new_expression->subterm_one_ = new Expression();
    string first_term_legend = "";
    if (!ReplaceVariableInExpressionWithIndicatorExpansion(
            *orig_expression.subterm_one_, non_numeric_vars_to_expansion_var,
            new_expression->subterm_one_, &first_term_legend, error_msg)) {
      return false;
    }
    new_expression->subterm_two_ = new Expression();
    string second_term_legend = "";
    if (!ReplaceVariableInExpressionWithIndicatorExpansion(
            *orig_expression.subterm_two_, non_numeric_vars_to_expansion_var,
            new_expression->subterm_two_, &second_term_legend, error_msg)) {
      return false;
    }

    // Store the two subterm results.
    (*legend) +=
        "(" + first_term_legend + ")" + GetOpString(orig_expression.op_) +
        "(" + second_term_legend + ")";
  }

  return true;
}

bool SanityCheckFunctionsAppliedToNominalVariables(
    const Expression& expression, const set<string>& nominal_vars,
    string* error_msg) {
  // Sanity check operation specified for the independent variable:
  // should not apply any operation to a nominal variable.
  if (expression.op_ != Operation::IDENTITY &&
      nominal_vars.find(expression.var_name_) != nominal_vars.end()) {
    if (error_msg != nullptr) {
      const string op_name =
          expression.op_ == Operation::MULT ? "Multiplication" :
          expression.op_ == Operation::ADD ? "Addition" :
          expression.op_ == Operation::EXP ? "Exponential" :
          expression.op_ == Operation::SQRT ? "Square Root" :
          expression.op_ == Operation::POW ? "Exponent" :
          expression.op_ == Operation::LOG ? "Log" : "Unknown Operation";
      *error_msg += "Independent variable '" + expression.var_name_ +
                    "' is of NOMINAL type, so you cannot apply " +
                    op_name + " to it.\n";
    }
    return false;
  }

  if (expression.subterm_one_ != nullptr &&
      !SanityCheckFunctionsAppliedToNominalVariables(
          *expression.subterm_one_, nominal_vars, error_msg)) {
    return false;
  }

  if (expression.subterm_two_ != nullptr &&
      !SanityCheckFunctionsAppliedToNominalVariables(
          *expression.subterm_two_, nominal_vars, error_msg)) {
    return false;
  }

  return true;
}

bool ConstructLegend(
    const vector<string>& input_header,
    const map<int, set<string>>& nominal_columns_and_values,
    vector<string>* output_legend, string* error_msg) {
  if (output_legend == nullptr) return false;
  for (int col = 0; col < input_header.size(); ++col) {
    const string& col_title = input_header[col];

    // Check if this column is nominal.
    map<int, set<string>>::const_iterator nominal_col_itr =
        nominal_columns_and_values.find(col);
    if (nominal_col_itr != nominal_columns_and_values.end()) {
      // Nominal column. Iterate over all possible values in
      // the set, creating indicators for each (except the first).
      const int indicators_needed = (nominal_col_itr->second.size() - 1);
      if (indicators_needed == 0) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in data input: Found only one distinct "
                        "nominal value in column " + Itoa(col) +
                        " (this will lead to non-invertible matrix).\n";
        }
        return false;
      }
      bool first_nominal_var = true;
      for (const string& nominal_value : nominal_col_itr->second) {
        if (first_nominal_var) {
          // Should loop through one fewer time than number of values in
          // nominal_column_itr->second.
          first_nominal_var = false;
          continue;
        }
        output_legend->push_back("I_" + col_title + "_" + nominal_value);
      }
    } else {
      output_legend->push_back(col_title);
    }
  }
  return true;
}

bool OrigColToDataValuesCol(
    const set<int>& input_cols_used, const int orig_col_index,
    int* col_index, string* error_msg) {
  if (col_index == nullptr) return false;
  if (orig_col_index == -1 || input_cols_used.empty()) {
    *col_index = orig_col_index;
    return true;
  }

  int i = 0;
  for (const int input_col_used : input_cols_used) {
    if (input_col_used == orig_col_index) {
      *col_index = i;
      return true;
    }
    ++i;
  }

  if (error_msg != nullptr) {
    *error_msg += "ERROR in coverting original column index to the index w.r.t. "
                  "the reduced data structure (where unused columns were omitted): "
                  "The original column index " + Itoa(orig_col_index) +
                  " is not among the list of columns that were kept: {" +
                  Join(input_cols_used, ", ") + "}.\n";
  }
  return false;
}

/* ============================= END Functions ============================== */


}  // namespace file_reader_utils
