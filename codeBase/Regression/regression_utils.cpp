// Date: Dec 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "regression_utils.h"

#include "FileReaderUtils/read_file_utils.h"
#include "FileReaderUtils/read_input.h"
#include "MathUtils/data_structures.h"
#include "MathUtils/eq_solver.h"
#include "MathUtils/statistics_utils.h"

#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

using Eigen::VectorXd;
using namespace math_utils;
using namespace file_reader_utils;

namespace regression {

bool ParseRegressionCommandLineArgs(
    int argc, char* argv[],
    const bool check_model_and_input_file_are_present,
    vector<ModelAndDataParams>* params,
    set<string>* unparsed_args) {
  if (params == nullptr || params->empty()) return false;
  ModelAndDataParams* params_one = &(*params)[0];

  FileInfo* file_info = &params_one->file_;
  FileInfo* outfile_info = &params_one->outfile_;
  AnalysisParams* analysis_params = &params_one->analysis_params_;
  bool print_var_set = false;
  bool print_robust_set = false;

  for (int i = 1; i < argc; ++i) {
    string arg = string(argv[i]);
    if (arg == "--in") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--out'.\n";
        return false;
      }
      ++i;
      file_info->name_ = StripQuotes(string(argv[i]));
    } else if (arg == "--out") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--out'.\n";
        return false;
      }
      ++i;
      outfile_info->name_ = StripQuotes(string(argv[i]));
    } else if (arg == "--sep") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--sep'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      file_info->delimiter_ = StripQuotes(argv[i]);
      outfile_info->delimiter_ = StripQuotes(argv[i]);
    } else if (arg == "--comment_char") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--comment_char'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      file_info->comment_char_ = StripQuotes(argv[i]);
      outfile_info->comment_char_ = StripQuotes(argv[i]);
    } else if (arg == "--inf_char") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--inf_char'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      file_info->infinity_char_ = StripQuotes(argv[i]);
      outfile_info->infinity_char_ = StripQuotes(argv[i]);
    } else if (arg == "--na_string" || arg == "--na_strings" ||
               arg == "--missing_value" || arg == "--missing_values") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--na_strings'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      vector<string> na_strs;
      Split(StripQuotes(argv[i]), ",", &na_strs);
      for (const string& na_str : na_strs) {
        file_info->na_strings_.insert(na_str);
        outfile_info->na_strings_.insert(na_str);
      }
    } else if (arg == "--max_itr" || arg == "max_iterations") {
      if (i == argc - 1) {
        cout << "ERROR Reading Command: Expected argument after '--max_itr'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      if (!Stoi(argv[i], &params_one->max_itr_)) {
        cout << "ERROR Reading Command: Unable to parse max_itr parameter '"
             << argv[i] << "' as a numeric value.\nAborting.\n";
        return false;
      }
    } else if (HasPrefixString(arg, "--model_type")) {
      // Determine which model is being specified.
      int model_index;
      if (arg == "--model_type") {
        model_index = 1;
      } else if (arg == "--model_type_two") {
        // For posterity, we support the old way of specifying the second model,
        // namely with "_two" instead of "_2".
        model_index = 2;
      } else if (!HasPrefixString(arg, "--model_type_")) {
        cout << "ERROR Reading Command: Unrecognized argument '"
             << arg << "' could not be parsed. Did you mean to use "
             << "--model_type[_k]?" << endl;
        return false;
      } else {
        const string suffix = StripPrefixString(arg, "--model_type_");
        if (!Stoi(suffix, &model_index)) {
          cout << "ERROR Reading Command: Unrecognized argument '"
               << arg << "' could not be parsed. Did you mean to use "
               << "--model_type[_k]?" << endl;
          return false;
        }
      }

      // Make sure there are enough models already in params; if not, add them.
      while (params->size() < model_index) {
        params->push_back(ModelAndDataParams());
      }
      // Push_back above may have invalidated the reference of params_one. Reset.
      params_one = &(*params)[0];
      file_info = &params_one->file_;
      outfile_info = &params_one->outfile_;
      analysis_params = &params_one->analysis_params_;

      ModelAndDataParams& params_k = (*params)[model_index - 1];
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--model_type'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      const string model_type = StripQuotes(RemoveAllWhitespace(string(argv[i])));
      if (model_type == "Linear" || model_type == "linear" ||
          model_type == "LINEAR") {
        params_k.model_type_ = ModelType::MODEL_TYPE_LINEAR;
      } else if (model_type == "Logistic" || model_type == "logistic" ||
                 model_type == "LOGISTIC") {
        params_k.model_type_ = ModelType::MODEL_TYPE_LOGISTIC;
      } else if (model_type == "Cox" || model_type == "cox" || model_type == "COX" ||
                 model_type == "Right-Censored" ||
                 model_type == "right-censored" ||
                 model_type == "Right-censored") {
        params_k.model_type_ = ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL;
      } else if (model_type == "Interval-Censored" ||
                 model_type == "Interval-censored" ||
                 model_type == "interval-censored") {
        params_k.model_type_ = ModelType::MODEL_TYPE_INTERVAL_CENSORED;
      } else {
        cout << "\nERROR Reading Command: Unrecognized model type: '"
             << model_type << "'" << endl;
        return false;
      }
    } else if (HasPrefixString(arg, "--model")) {
      // Determine which model is being specified.
      int model_index;
      if (arg == "--model") {
        model_index = 1;
      } else if (arg == "--model_two") {
        // For posterity, we support the old way of specifying the second model,
        // namely with "_two" instead of "_2".
        model_index = 2;
      } else if (!HasPrefixString(arg, "--model_")) {
        cout << "ERROR Reading Command: Unrecognized argument '"
             << arg << "' could not be parsed. Did you mean to use "
             << "--model[_k]?" << endl;
        return false;
      } else {
        const string suffix = StripPrefixString(arg, "--model_");
        if (!Stoi(suffix, &model_index)) {
          // This argument could not be parsed as a model_k, but that may
          // be fine, as there could be a *_main.cpp program that allows
          // users to specify command-line arguments that begin with
          // 'model_'; in which case those programs will be responsible
          // for parsing this argument.
          continue;
        }
      }

      // Make sure there are enough models already in params; if not, add them.
      while (params->size() < model_index) {
        params->push_back(ModelAndDataParams());
      }
      // Push_back above may have invalidated the reference of params_one. Reset.
      params_one = &(*params)[0];
      file_info = &params_one->file_;
      outfile_info = &params_one->outfile_;
      analysis_params = &params_one->analysis_params_;

      ModelAndDataParams& params_k = (*params)[model_index - 1];
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--model'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      params_k.model_str_ = StripQuotes(RemoveAllWhitespace(string(argv[i])));
    } else if (HasPrefixString(arg, "--strata")) {
      // Determine which model is being specified.
      int model_index;
      if (arg == "--strata") {
        model_index = 1;
      } else if (arg == "--strata_two") {
        // For posterity, we support the old way of specifying the second model,
        // namely with "_two" instead of "_2".
        model_index = 2;
      } else if (!HasPrefixString(arg, "--strata_")) {
        cout << "ERROR Reading Command: Unrecognized argument '"
             << arg << "' could not be parsed. Did you mean to use "
             << "--strata[_k]?" << endl;
        return false;
      } else {
        const string suffix = StripPrefixString(arg, "--strata_");
        if (!Stoi(suffix, &model_index)) {
          cout << "ERROR Reading Command: Unrecognized argument '"
               << arg << "' could not be parsed. Did you mean to use "
               << "--strata[_k]?" << endl;
          return false;
        }
      }

      // Make sure there are enough models already in params; if not, add them.
      while (params->size() < model_index) {
        params->push_back(ModelAndDataParams());
      }
      // Push_back above may have invalidated the reference of params_one. Reset.
      params_one = &(*params)[0];
      file_info = &params_one->file_;
      outfile_info = &params_one->outfile_;
      analysis_params = &params_one->analysis_params_;

      ModelAndDataParams& params_k = (*params)[model_index - 1];
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--strata'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      params_k.strata_str_ = StripQuotes(RemoveAllWhitespace(string(argv[i])));
    } else if (HasPrefixString(arg, "--subgroup")) {
      // Determine which model is being specified.
      int model_index;
      if (arg == "--subgroup") {
        model_index = 1;
      } else if (arg == "--subgroup_two") {
        // For posterity, we support the old way of specifying the second model,
        // namely with "_two" instead of "_2".
        model_index = 2;
      } else if (!HasPrefixString(arg, "--subgroup_")) {
        cout << "ERROR Reading Command: Unrecognized argument '"
             << arg << "' could not be parsed. Did you mean to use "
             << "--subgroup[_k]?" << endl;
        return false;
      } else {
        const string suffix = StripPrefixString(arg, "--subgroup_");
        if (!Stoi(suffix, &model_index)) {
          cout << "ERROR Reading Command: Unrecognized argument '"
               << arg << "' could not be parsed. Did you mean to use "
               << "--subgroup[_k]?" << endl;
          return false;
        }
      }

      // Make sure there are enough models already in params; if not, add them.
      while (params->size() < model_index) {
        params->push_back(ModelAndDataParams());
      }
      // Push_back above may have invalidated the reference of params_one. Reset.
      params_one = &(*params)[0];
      file_info = &params_one->file_;
      outfile_info = &params_one->outfile_;
      analysis_params = &params_one->analysis_params_;

      ModelAndDataParams& params_k = (*params)[model_index - 1];
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--subgroup'."
             << "\nAborting.\n";
        return false;
      }
      ++i;
      params_k.subgroup_str_ = StripQuotes(RemoveAllWhitespace(string(argv[i])));
    } else if (arg == "--id_col" || arg == "--subject_id") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--"
             << "id_col'.\nAborting.\n";
        return false;
      }
      ++i;
      params_one->id_str_ = StripQuotes(RemoveAllWhitespace(string(argv[i])));
    } else if (arg == "--family_cols" || arg == "--family_col" || arg == "--family_id") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--"
             << "family_col'.\nAborting.\n";
        return false;
      }
      ++i;
      params_one->family_str_ = StripQuotes(RemoveAllWhitespace(string(argv[i])));
    } else if (arg == "--weight_col") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--"
             << "weight_col'.\nAborting.\n";
        return false;
      }
      ++i;
      params_one->weight_str_ = StripQuotes(RemoveAllWhitespace(string(argv[i])));
    } else if (arg == "--left_truncation_col") {
      if (i == argc - 1) {
        cout << "\nERROR Reading Command: Expected argument after '--"
             << "left_truncation_col'.\nAborting.\n";
        return false;
      }
      ++i;
      params_one->left_truncation_str_ = StripQuotes(RemoveAllWhitespace(string(argv[i])));
    } else if (arg == "--collapse") {
      if (i == argc - 1) {
        cout << "ERROR Reading Command: Expected argument after '--collapse'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      params_one->collapse_params_str_ = StripQuotes(RemoveAllWhitespace(argv[i]));
    } else if (arg == "--standardization") {
      if (i == argc - 1) {
        cout << "ERROR Reading Command: Expected argument after '--standardization'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      params_one->var_norm_params_str_ = StripQuotes(RemoveAllWhitespace(argv[i]));
    } else if (arg == "--extrapolation") {
      if (i == argc - 1) {
        cout << "ERROR Reading Command: Expected argument after '--extrapolation'.\n"
             << "Aborting.\n";
        return false;
      }
      ++i;
      params_one->time_params_str_ = StripQuotes(RemoveAllWhitespace(argv[i]));
    } else if (arg == "--nostd" || arg == "--no_std") {
      params_one->standardize_vars_ = VariableNormalization::VAR_NORM_NONE;
    } else if (arg == "--std") {
      params_one->standardize_vars_ =
          VariableNormalization::VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;
    } else if (arg == "--std_all") {
      params_one->standardize_vars_ =
          VariableNormalization::VAR_NORM_STD_W_N_MINUS_ONE;
    } else if (arg == "--nostandard") {
      analysis_params->standard_analysis_ = false;
    } else if (arg == "--standard") {
      analysis_params->standard_analysis_ = true;
    } else if (arg == "--norobust") {
      analysis_params->robust_analysis_ = false;
    } else if (arg == "--robust") {
      analysis_params->robust_analysis_ = true;
    } else if (arg == "--nolog_rank") {
      analysis_params->log_rank_analysis_ = false;
    } else if (arg == "--log_rank") {
      analysis_params->log_rank_analysis_ = true;
    } else if (arg == "--nopeto") {
      analysis_params->peto_analysis_ = false;
    } else if (arg == "--peto") {
      analysis_params->peto_analysis_ = true;
    } else if (arg == "--noscore_method") {
      analysis_params->score_method_analysis_ = false;
    } else if (arg == "--score_method") {
      analysis_params->score_method_analysis_ = true;
    } else if (arg == "--noscore_method_width") {
      analysis_params->score_method_width_analysis_ = false;
    } else if (arg == "--score_method_width") {
      analysis_params->score_method_width_analysis_ = true;
    } else if (arg == "satterthwaite") {
      analysis_params->satterthwaite_analysis_ = true;
    } else if (arg == "nosatterthwaite") {
      analysis_params->satterthwaite_analysis_ = false;
    } else if (arg == "--ties_constant") {
      params_one->use_ties_constant_ = true;
    } else if (arg == "--noties_constant") {
      params_one->use_ties_constant_ = false;
    } else if (arg == "--kme") {
      params_one->kme_type_ = KaplanMeierEstimatorType::LEFT_CONTINUOUS;
    } else if (arg == "--nokme") {
      params_one->kme_type_ = KaplanMeierEstimatorType::NONE;
    } else if (arg == "--kme_for_log_rank") {
      params_one->kme_type_for_log_rank_ =
          KaplanMeierEstimatorType::LEFT_CONTINUOUS;
    } else if (arg == "--nokme_for_log_rank") {
      params_one->kme_type_for_log_rank_ = KaplanMeierEstimatorType::NONE;
    } else if (arg == "--print_est") {
      params_one->print_options_.print_estimates_ = true;
    } else if (arg == "--noprint_est") {
      params_one->print_options_.print_estimates_ = false;
    } else if (arg == "--print_var") {
      print_var_set = true;
      params_one->print_options_.print_variance_ = true;
    } else if (arg == "--noprint_var") {
      print_var_set = true;
      params_one->print_options_.print_variance_ = false;
    } else if (arg == "--print_robust_var") {
      print_robust_set = true;
      params_one->print_options_.print_robust_variance_ = true;
    } else if (arg == "--noprint_robust_var") {
      print_robust_set = true;
      params_one->print_options_.print_robust_variance_ = false;
    } else if (arg == "--print_se") {
      params_one->print_options_.print_se_ = true;
    } else if (arg == "--noprint_se") {
      params_one->print_options_.print_se_ = false;
    } else if (arg == "--print_t_stat") {
      params_one->print_options_.print_t_stat_ = true;
    } else if (arg == "--noprint_t_stat") {
      params_one->print_options_.print_t_stat_ = false;
    } else if (arg == "--print_p_value") {
      params_one->print_options_.print_p_value_ = true;
    } else if (arg == "--noprint_p_value") {
      params_one->print_options_.print_p_value_ = false;
    } else if (arg == "--print_ci_width") {
      params_one->print_options_.print_ci_width_ = true;
    } else if (arg == "--noprint_ci_width") {
      params_one->print_options_.print_ci_width_ = false;
    } else if (arg == "--print_cov_matrix") {
      params_one->print_options_.print_covariance_matrix_ = true;
    } else if (arg == "--noprint_cov_matrix") {
      params_one->print_options_.print_covariance_matrix_ = false;
    } else if (unparsed_args != nullptr) {
      // Don't do anything for unrecognized arguments; each *_main.cpp
      // program may have its own specific arguments, which it is
      // responsible for parsing itself.
      unparsed_args->insert(arg);
    }
  }

  // Required inputs are: --in and --model. Make sure all these were set.
  if (check_model_and_input_file_are_present) {
    if (params_one->file_.name_.empty() ||
        params_one->model_type_ == ModelType::MODEL_TYPE_UNKNOWN ||
        params_one->model_str_.empty()) {
      cout << "ERROR: Empty input file, model_type, and/or model." << endl;
      return false;
    }
  }

  // Update the default values for params.print_options_ based on the
  // analyses being run.
  if (!print_var_set && params_one->analysis_params_.robust_analysis_) {
    params_one->print_options_.print_variance_ = false;
  }
  if (!print_robust_set && params_one->analysis_params_.robust_analysis_) {
    params_one->print_options_.print_robust_variance_ = true;
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
    CopyModelAndDataParams(*params_one, &((*params)[i]));
  }

  return true;
}

string GetLinearTermTitle(const LinearTerm& term) {
  string to_return = "";
  for (const VariableTerm& var_term : term.terms_) {
    if (!to_return.empty()) {
      to_return += GetOpString(term.op_);
    }
    to_return += GetTermString(var_term);
  }
  return to_return;
}

bool IsNominalExpansionString(
    const string& expanded_term, const string& original_term) {
  const string exp_term_no_space = RemoveAllWhitespace(expanded_term);
  const string orig_term_no_space = RemoveAllWhitespace(original_term);
  if (exp_term_no_space.empty() || orig_term_no_space.empty()) return false;
  vector<string> subterms;
  const string op_str = GetOpString(Operation::MULT);
  Split(orig_term_no_space, op_str, &subterms);
  for (int i = 0; i < subterms.size(); ++i) {
    const string& subterm = subterms[i];
    size_t pos = exp_term_no_space.find(subterm);
    if (pos == string::npos) return false;
    // First subterm in orig_term_no_space should also be the first subterm
    // of exp_term_no_space.
    if (i == 0 && pos != 0) return false;
    // If this is NOT the first subterm of exp_term_no_space, then
    // the character immediately before this subterm should be
    // the LinearTerm's Opeartion (assumed to be multiplication).
    if (i > 0 && (pos == 0 || exp_term_no_space.substr(pos - 1, 1) != op_str)) {
      return false;
    }
    size_t index_of_next_char = pos + subterm.length();
    // If this is NOT the last subterm of orig_term_no_space, then the
    // character immediately after this subterm should either be "_"
    // (in case this subterm is a nominal variable) or "*" (in case it's
    // not a nominal variable).
    if (i < subterms.size() - 1 &&
        (index_of_next_char >= exp_term_no_space.length() ||
         (exp_term_no_space.substr(index_of_next_char, 1) != "_" &&
          exp_term_no_space.substr(index_of_next_char, 1) != op_str))) {
      return false;
    }
    // If this is the last subterm of orig_term_no_space, then the character
    // immediately after it should be nothing (in case it's not a nominal
    // variable), or "_" (in case it is a nominal variable).
    if (i == subterms.size() - 1 &&
        index_of_next_char < exp_term_no_space.length() &&
        exp_term_no_space.substr(index_of_next_char, 1) != "_") {
      return false;
    }
  }
  return true;
}

bool GetBetaFromLegend(
    const map<string, double>& var_name_to_beta_value,
    const vector<string>& legend,
    VectorXd* beta_values, string* error_msg) {
  if (beta_values == nullptr) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in GetBetaFromLegend: Null input.\n";
    }
    return false;
  }

  beta_values->resize(legend.size());

  // Check if var_name_to_beta_value has a beta value for 'I_Subgroup'.
  double subgroup_beta = DBL_MIN;
  map<string, double>::const_iterator subgroup_beta_finder =
      var_name_to_beta_value.find("I_Subgroup");
  if (subgroup_beta_finder != var_name_to_beta_value.end()) {
    subgroup_beta = subgroup_beta_finder->second;
  } else {
    subgroup_beta_finder = var_name_to_beta_value.find("I_subgroup");
    if (subgroup_beta_finder != var_name_to_beta_value.end()) {
      subgroup_beta = subgroup_beta_finder->second;
    }
  }

  for (int i = 0; i < legend.size(); ++i) {
    const string& legend_term = legend[i];
    map<string, double>::const_iterator beta_finder =
        var_name_to_beta_value.find(legend_term);
    if (beta_finder != var_name_to_beta_value.end()) {
      // Found a beta-value for this term. Use it, and proceed to next term.
      (*beta_values)(i) = beta_finder->second;
    } else {
      // We don't have a beta-value for this term. Check to see if the term
      // came from:
      //   1) The expansion of a nominal value
      //   2) An indicator for subgroup membership.
      // First check for (1), by seeing if legend_term is a possible
      // (nominal value) expansion of any of the terms in var_name_to_beta_value.
      bool found_nominal_beta = false;
      for (const pair<string, double>& var_name_and_beta : var_name_to_beta_value) {
        if (IsNominalExpansionString(legend_term, var_name_and_beta.first)) {
          (*beta_values)(i) = var_name_and_beta.second;
          found_nominal_beta = true;
          break;
        }
      }
      if (found_nominal_beta) {
        continue;
      // Now check (2) above: Subgroups. Subgroup terms will have prefix
      // 'I_Subgroup' (and then the subgroup index appended, which we'll
      // ignore, since all subgroups will get the same \beta multiplier).
      } else if (subgroup_beta != DBL_MIN &&
                 (HasPrefixString(legend_term, "I_Subgroup") ||
                  HasPrefixString(legend_term, "I_subgroup"))) {
        (*beta_values)(i) = subgroup_beta;
      } else {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in GetBetaFromLegend: Unable to find term '" +
                        legend_term + "' among the terms for which we have a "
                        "beta value.\n";
        }
        // Resize beta_values to be empty, to emphasize we didn't succeed
        // (Simulations will only call this function on the first iteration,
        // and they will use the size of beta_values to determine if it has
        // already been set; so we must reset size to zero here to emphasize
        // to future iterations that this function hasn't succeeeded yet.
        beta_values->resize(0);
        return false;
      }
    }
  }
  return true;
}

bool GetDependentVariableNames(
    const ModelType& model_type, const DepVarDescription& model_lhs,
    vector<string>* dep_var_names) {
  if (dep_var_names == nullptr) return false;

  dep_var_names->clear();
  if (model_type == ModelType::MODEL_TYPE_LINEAR ||
      model_type == ModelType::MODEL_TYPE_LOGISTIC) {
    if (model_lhs.dep_vars_names_.size() != 1) {
      cout << "ERROR: Expected a single dependent variable (name) for "
           << "model type " << static_cast<int>(model_type)
           << ", but found " << model_lhs.dep_vars_names_.size() << endl;
      return false;
    }
    dep_var_names->push_back(model_lhs.dep_vars_names_[0]);
    return true;
  } else if (model_type == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL) {
    if (model_lhs.time_vars_names_.size() != 1 &&
        model_lhs.time_vars_names_.size() != 2) {
      cout << "ERROR: Expected either one or two time variable names ("
           << "(representing survival and/or censoring time) for "
           << "Cox PH model, but found "
           << model_lhs.time_vars_names_.size() << endl;
      return false;
    }
    for (const string& time_var_name : model_lhs.time_vars_names_) {
      dep_var_names->push_back(time_var_name);
    }
    if (model_lhs.dep_vars_names_.size() != 1) {
      cout << "ERROR: Expected a single dependent variable (name) for "
           << "Cox PH model, but found "
           << model_lhs.dep_vars_names_.size() << endl;
      return false;
    }
    dep_var_names->push_back(model_lhs.dep_vars_names_[0]);
    return true;
  } else {
    cout << "ERROR: GetDependentVariableNames is not supported for type "
         << static_cast<int>(model_type) << endl;
    return false;
  }

  return true;
}

bool PrintRegressionHeader(
    const ModelAndDataParams& params,
    const SummaryStatistics& summary_stats,
    string* output) {
  if (output == nullptr) return false;

  // Print Model.
  *output += params.final_model_ + "\n\n";

  // Print Rows skipped due to missing/NA values.
  if (!params.na_rows_skipped_.empty()) {
    *output += "Skipped the following data rows due to missing/NA values:\n" +
               Join(params.na_rows_skipped_, ", ") + "\n\n";
  }
  
  // Print out Subgroup Synopsis.
  //   - Subgroup Definition.
  if (!PrintSubgroupSynopsis(
        params.subgroup_cols_, params.subgroups_,
        params.subgroup_rows_per_index_, output)) {
    return false;
  }
  //   - Row membership in each Subgroup and Strata.
  string subgp_and_strata_str = "";
  if (!GetModelAndDataParamsSubgroupsAndStrata(params, &subgp_and_strata_str)) {
    return false;
  }
  *output += subgp_and_strata_str;

  /* TODO(PHB): Determine if this is useful for read data, or just simulations.
  // Fraction ties for each variable (for Cox PH Models).
  for (const pair<string, int>& var_fraction_info : fraction_ties_per_var) {
    *output += "Average fraction of ties for " + var_fraction_info.first +
               ": " + Itoa(var_fraction_info.second) + "\n";
  }
  */

  // For Survival models, print out the average percentage of
  // Status 0 (censored) Subjects.
  if (!summary_stats.fraction_alive_.empty()) {
    *output += "Average fraction of censored samples (Delta = 0): ";
    const vector<tuple<int, int, double>>& fraction_alive =
        summary_stats.fraction_alive_;
    if (fraction_alive.size() == 1) {
      const double& fraction = get<2>(fraction_alive[0]);
      const string fraction_str = fraction >= 0.0 ? Itoa(fraction) : "N/A";
      *output += fraction_str + "\n\n";
    } else {
      *output += "\n";
      for (int i = 0; i < fraction_alive.size(); ++i) {
        const double& fraction = get<2>(fraction_alive[i]);
        const pair<int, int> partition = make_pair(
            get<0>(fraction_alive[i]),
            get<1>(fraction_alive[i]));
        const string fraction_str = fraction >= 0.0 ? Itoa(fraction) : "N/A";
        *output += "\tOn Samples [" + Itoa(partition.first) + ", " +
                   Itoa(partition.second) + "]: " + fraction_str + "\n";
      }
      *output += "\n";
    }
  }

  return true;
}

bool PrintRegressionStatistics(
    const ModelAndDataParams& params,
    const SummaryStatistics& summary_stats,
    string* output) {
  if (output == nullptr) return false;

  // Print Header information.
  if (!PrintRegressionHeader(params, summary_stats, output)) return false;

  // Print Summary Statistics.
  return PrintSummaryStatistics(params, summary_stats, output);
}

bool PrintRegressionStatistics(
    const string& command,
    const ModelAndDataParams& params, const SummaryStatistics& summary_stats) {
  if (params.file_.name_.empty()) return false;
  ofstream outfile;
  outfile.open(params.outfile_.name_);
  if (!outfile.is_open()) {
    cout << "ERROR: Unable to open file '" << params.file_.name_
         << "' for writing." << endl;
    return false;
  }

  string output = "";
  if (!PrintSummaryStatistics(params, summary_stats, &output)) {
    return false;
  }

  if (!command.empty()) {
    outfile << "Command:\n\n" << command << endl << endl;
  }
  outfile << output;
  outfile.close();
  return true;
}

/* PHB
bool ComputeDependentVariableValues(
    const ModelType& model_type,
    const VectorXd& actual_beta_values,
    const Expression& model_rhs,
    const DepVarDescription& model_lhs,
    const vector<string>& header,
    const vector<vector<DataHolder>>& sampled_var_values,
    DepVarHolder* dep_vars, string* error_msg) {
  if (dep_vars == nullptr) return false;
  if (!ClearDependentVariables(model_type, dep_vars, error_msg)) return false;

  // Break Model's RHS into the linear terms (we'll need to compute the
  // value of the dependent variable(s) from the (simulated) values for
  // the RHS, and use the actual beta coefficients. We can't just evaluate
  // the entire model RHS at once because the beta coefficients are not
  // stored as part of model_rhs_, and we need to evaluate the expression
  // after multiplying each linear term by its beta coefficient.
  const bool has_constant_term =
      model_type == ModelType::MODEL_TYPE_LINEAR ||
      model_type == ModelType::MODEL_TYPE_LOGISTIC;
  vector<Expression> linear_terms;
  if (!GetLegendAndLinearTermsForSimulation(
          has_constant_term, model_rhs,
          &linear_terms, nullptr, error_msg)) {
    return false;
  }

  if (linear_terms.size() != actual_beta_values.size()) {
    if (error_msg != nullptr) {
      *error_msg += "ERRROR in ComputeDependentVariableValues: Number of linear "
                    "terms found on the RHS of the model (" +
                    Itoa(linear_terms.size()) + ") does not match the number of "
                    "beta coefficients provided (" +
                    Itoa(actual_beta_values.size()) + ").\n";
    }
    return false;
  }
  
  // Go through sampled values, computing the dependent variable values for
  // each "row" of simulated data.
  for (int i = 0; i < sampled_var_values.size(); ++i) {
    const vector<DataHolder>& sampled_row = sampled_var_values[i];

    // Create mapping from variable name to value.
    map<string, double> var_name_to_value;
    if (sampled_row.size() != header.size()) {
      if (error_msg != nullptr) {
        *error_msg +=
            "ERROR: mismatching header size in ComputeDependentVariableValues\n";
      }
      return false;
    }
    for (int j = 0; j < header.size(); ++j) {
      var_name_to_value.insert(make_pair(header[j], sampled_row[j].value_));
    }

    // Evaluate model's RHS.
    double rhs_value = 0.0;
    for (int j = 0; j < linear_terms.size(); ++j) {
      // Evaluate the linear term value, using the simulated values for each
      // of the variables that appear in the linear term's Expression.
      double term_value;
      if (!EvaluateExpression(
              linear_terms[j], var_name_to_value,
              &term_value, error_msg)) {
        if (error_msg != nullptr) {
          *error_msg +=
              "ERROR in SimulateValuesAndEvaluateVariableValues: Unable to "
              "evaluate model's RHS of Sample " + Itoa(i + 1) + ".\n";
        }
        return false;
      }
      rhs_value += actual_beta_values(j) * term_value;
    }
    // model_rhs does not include error term (for linear models).
    // Add it now.
    if (model_type == ModelType::MODEL_TYPE_LINEAR) {
      double* error_value =
          FindOrNull(ReadInput::GetErrorTermId(), var_name_to_value);
      if (error_value == nullptr) {
        cout << "ERROR: Error term value wasn't simulated." << endl;
        return false;
      }
      rhs_value += *error_value;
    }

    if (model_type == ModelType::MODEL_TYPE_LINEAR) {
      // For linear models, we don't allow any mathematical formulas on the LHS
      // of the model; just simply a variable name (e.g. "Y"). So there is nothing
      // to compute here; just return rhs_value.
      dep_vars->dep_vars_linear_.push_back(rhs_value);
      continue;
    } else if (model_type == ModelType::MODEL_TYPE_LOGISTIC) {
      // For Logistic Models,  evaluate the logistic expression:
      //   I(Y > (exp(RHS_STRING) / (1.0 + exp(RHS_STRING)))),
      // where I is indicator function, and Y is the value sampled
      // (typically from the uniform distribution U(0,1)) for the LHS variable.

      // Get RHS value.
      const double exp_rhs_value = exp(rhs_value);
      const double rhs_logistic_value = exp_rhs_value / (1.0 + exp_rhs_value);

      // Get LHS value.
      double lhs_value;
      if (!EvaluateExpression(
              model_lhs.model_lhs_logistic_, var_name_to_value,
              &lhs_value, error_msg)) {
        return false;
      }

      dep_vars->dep_vars_logistic_.push_back(
          lhs_value > rhs_logistic_value ? 1.0 : 0.0);
      continue;
    } else if (model_type == ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL) {
      // As an example, model_lhs_cox_ might look like:
      //   - 1st Element: -1.0 * log (U1) / exp(RHS_STRING), where U1 <- Unif(0, 1)
      //   - 2nd Element: a * U2, where U2 <- Unif(0, 1), and 'a' is some constant
      //   - 3rd Element: -1, which indicates to determine Status via Min(1st, 2nd)
      if (model_lhs.model_lhs_cox_.size() != 2 &&
          model_lhs.model_lhs_cox_.size() != 3) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unable to generate the dependent variable value for "
                        "Cox Model: Unexpected model_lhs has " +
                        Itoa(static_cast<int>(model_lhs.model_lhs_cox_.size())) +
                        " terms.\n";
        }
        return false;
      }

      dep_vars->dep_vars_cox_.push_back(CensoringData());
      CensoringData& data = dep_vars->dep_vars_cox_.back();

      // Compute first Time value.
      if (!EvaluateExpression(
              model_lhs.model_lhs_cox_[0], var_name_to_value,
              &data.survival_time_, error_msg)) {
        return false;
      }

      // Compute Status value.
      double status;
      if (!EvaluateExpression(
              model_lhs.model_lhs_cox_.back(), var_name_to_value,
              &status, error_msg)) {
        return false;
      }
      // Sanity-Check status is -1, 0, or 1.
      if (status != -1.0 && status != 0.0 && status != 1.0) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR: Unable to generate the dependent variable value for "
                        "Cox model: unexpected status: " + Itoa(status) + ".\n";
        }
        return false;
      }

      // Compute second (Censoring) Time value, if appropriate.
      if (model_lhs.model_lhs_cox_.size() == 3) {
        // Censoring Time provided. Compute it, and set status accordingly.
        if (!EvaluateExpression(
                model_lhs.model_lhs_cox_[1], var_name_to_value,
                &data.censoring_time_, error_msg)) {
          return false;
        }

        // Make sure Status was either -1 (in which case set status based on
        // min(time, censoring_time)), or that it agrees with
        // min(time, censoring_time).
        const double actual_status = data.survival_time_ <= data.censoring_time_ ? 1.0 : 0.0;
        if (status == -1.0) data.is_alive_ = !actual_status;
        else if (status != actual_status) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR: Unable to generate the dependent variable value for "
                          "Cox Model: Status (" + Itoa(status) +
                          ") is not consistent with Survival Time (" +
                          Itoa(data.survival_time_) + ") and Censoring Time (" +
                          Itoa(data.censoring_time_) + ").\n";
          }
          return false;
        }
      } else {
        // No censoring time provided. Make sure status is 1 or 0 (above, we also
        // allowed -1 values for status, contingent on Censoring Time being
        // available), and then make the provided time either Survival Time or
        // Censoring Time (depending on status), and set the other to arbitarily
        // be one greater than this (anything larger is fine).
        if (status != 0.0 && status != 1.0) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR: Unable to generate the dependent variable value for "
                          "Cox model: unexpected status: " + Itoa(status) + ".\n";
          }
          return false;
        }
        data.is_alive_ = !status;
        data.censoring_time_ = data.survival_time_;
        if (status == 0.0) {
          data.survival_time_ += 1.0;
        } else {
          data.censoring_time_ += 1.0;
        }
      }
    } else {
      if (error_msg != nullptr) {
        *error_msg += "ERROR: Unable to generate the dependent variable value for "
                      "model type " + Itoa(static_cast<int>(model_type)) + ".\n";
      }
      return false;
    }
  }

  return true;
}
*/

}  // namespace regression
