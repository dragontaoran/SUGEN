// Date: April 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "cox_dual_regression.h"

#include "FileReaderUtils/read_input.h"
#include "FileReaderUtils/read_table_with_header.h"
#include "Regression/regression_utils.h"
#include "MapUtils/map_utils.h"
#include "MathUtils/constants.h"
#include "MathUtils/data_structures.h"
#include "MathUtils/gamma_fns.h"
#include "MathUtils/number_comparison.h"
#include "MathUtils/statistics_utils.h"

#include <cmath>
#include <cstdlib>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdio.h>
#include <string>
#include <vector>

using Eigen::Dynamic;
using Eigen::FullPivLU;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using file_reader_utils::ReadInput;
using file_reader_utils::ReadTableWithHeader;
using namespace map_utils;
using namespace math_utils;
using namespace string_utils;
using namespace std;

namespace regression {

static const int MAX_SCORE_METHOD_WIDTH_ITERATIONS = 50;
static const double kScoreMethodConfidenceIntervalBound = 3.8414588207;
// NOTE. Determine what to set for kUseLeftContinuousKme
// (originally, Danyu wanted to use right-continuous K-M; but R
// uses left-continuous).
static const bool kUseLeftContinuousKme = true;
static const bool kPrintRMatrix = false;
static const bool kPrintRows = false;
static const bool kPrintVariance = false;

bool CoxDualRegression::RunCoxRegression(
    const VectorXd& actual_beta,
    const ModelAndDataParams& params,
    map<int, int>* orig_row_to_unsorted_model_row,
    SummaryStatistics* summary_stats) {
  if (!ConstructMappingFromOrigRowToUnsortedModelRow(
          params, orig_row_to_unsorted_model_row)) {
    return false;
  }

  map<int, pair<int, int>> unsorted_model_row_to_strata_index_and_row;
  if (!RunCoxRegression(
          params, &unsorted_model_row_to_strata_index_and_row, summary_stats)) {
    return false;
  }

  /* PHB OLD
  // Compute the fraction of 'alive' samples, on each partition of the Samples.
  if (!GetFractionAlive(sample_partitions, subgroup_index_to_its_rows, dep_var,
                        &(summary_stats->fraction_alive_))) {
    return false;
  }
  */

  char error_msg_char[512] = "";
  if (!ComputeTestStatistic(summary_stats, error_msg_char)) {
    cout << "ERROR in Computing Test Statistic:\n" << error_msg_char << endl;
    return false;
  }

  if (params.analysis_params_.score_method_analysis_ &&
      params.analysis_params_.standard_analysis_) {
    // TODO(PHB): Currently, Score Method computations are only supported
    // for p = 1. For p > 1, need to define what is meant by U^2 / I
    // (most likely, U^2 := U^T * U), and then in case this is a p x p matrix,
    // how we find the "width" (i.e., probably looking for values of \beta
    // such that all coordinates (or perhaps just diagonal coordinates) of U^2 / I
    // are less than 3.8416?
    if (params.linear_term_values_.cols() != 1) {
      cout << "\nERROR: Computing Score Method width for p = "
           << params.linear_term_values_.cols() << " is not currently supported." << endl;
      return false;
    }

    // Partition params.dep_vars_.dep_vars_cox_ and params.linear_term_values_
    // by their strata.
    map<int, pair<vector<CensoringData>, MatrixXd>> strata_vars;
    if (!GetStrataVars(unsorted_model_row_to_strata_index_and_row,
                       params.dep_vars_.dep_vars_cox_,
                       params.linear_term_values_, &strata_vars)) {
       cout << "\nERROR: Failed to GetStrataVars." << endl;
      return false;
    }

    // Sort Samples based on time.
    map<int, vector<int>> strata_reverse_sorted_indices;
    if (!GetStrataSortedIndices(
            strata_vars, &strata_reverse_sorted_indices)) {
      cout << "\nERROR: Failed to GetStrataSortedIndices." << endl;
      return false;
    }

    // Compute K-M Estimators (if necessary).
    map<int, VectorXd> strata_km_estimators;
    if (!ComputeKaplanMeierEstimator(
            params.kme_type_, strata_vars, &strata_km_estimators)) {
      cout << "\nERROR: Failed to ComputeKaplanMeierEstimator." << endl;
      return false;
    }

    const int n = params.linear_term_values_.rows();
    const int p = params.linear_term_values_.cols();

    // Compute Score Method value: U^2(beta)/I(beta).
    // NOTE: Do this only for simulations (if actual_beta is provided), as:
    //   1) We only know the actual \beta to use in the formula above for
    //      simulations; AND
    //   1) We only need score method value to compute Score Method CP
    //      (Coverage Probability), which is only relevant for simulations
    if (actual_beta.size() > 0) {
      if (!EvaluateScoreMethodFunction(
              n, p, params.use_ties_constant_, actual_beta(0),
              strata_km_estimators, strata_km_estimators,
              strata_vars, strata_reverse_sorted_indices,
              &summary_stats->score_method_value_)) {
        cout << "\nERROR: Failed to EvaluateScoreMethodFunction in "
             << "ReadSimulatedDataAndRunCoxRegression for beta(0): "
             << actual_beta(0) << endl;
        return false;
      }
    }

    // Compute Width (find two roots of: U^2(x)/I(x) = 3.8415, and then take
    // the difference between them).
    if (params.analysis_params_.score_method_width_analysis_ &&
        !CoxDualRegression::ComputeScoreMethodCi(
          n, p, params.use_ties_constant_, summary_stats->estimates_[0],
          summary_stats->standard_error_[0],
          strata_km_estimators, strata_km_estimators, strata_vars,
          strata_reverse_sorted_indices,
          &summary_stats->score_method_ci_left_,
          &summary_stats->score_method_ci_right_)) {
      cout << "\nERROR in ComputeScoreMethodCi." << endl;
      return false;
    }
  }

  return true;
}

/* PHB_OLD
bool CoxDualRegression::RunCoxRegression(
    const ConvergenceCriteria& criteria,
    const ModelAndDataParams& params,
    map<int, int>* orig_row_to_unsorted_model_row,
    SummaryStatistics* summary_stats) {
  if (!ConstructMappingFromOrigRowToUnsortedModelRow(
          params, orig_row_to_unsorted_model_row)) {
    return false;
  }

  map<int, pair<int, int>> unsorted_model_row_to_strata_index_and_row;
  if (!RunCoxRegression(
          criteria, params,
          &unsorted_model_row_to_strata_index_and_row, summary_stats)) {
    return false;
  }

  char error_msg_char[512] = "";
  if (!ComputeTestStatistic(summary_stats, error_msg_char)) {
    cout << "ERROR in Computing Test Statistic:\n" << error_msg_char << endl;
    return false;
  }

  if (params.analysis_params_.score_method_analysis_ &&
      params.analysis_params_.standard_analysis_) {
    // TODO(PHB): Currently, Score Method computations are only supported
    // for p = 1. For p > 1, need to define what is meant by U^2 / I
    // (most likely, U^2 := U^T * U), and then in case this is a p x p matrix,
    // how we find the "width" (i.e., probably looking for values of \beta
    // such that all coordinates (or perhaps just diagonal coordinates) of U^2 / I
    // are less than 3.8416?
    if (params.linear_term_values_.cols() != 1) {
      cout << "\nERROR: Computing Score Method width for p = "
           << params.linear_term_values_.cols() << " is not currently supported." << endl;
      return false;
    }

    // Partition params.dep_vars_.dep_vars_cox_ and params.linear_term_values_
    // by their strata.
    map<int, pair<vector<CensoringData>, MatrixXd>> strata_vars;
    if (!GetStrataVars(unsorted_model_row_to_strata_index_and_row,
                       params.dep_vars_.dep_vars_cox_,
                       params.linear_term_values_, &strata_vars)) {
       cout << "\nERROR: Failed to GetStrataVars." << endl;
      return false;
    }

    // Sort Samples based on time.
    map<int, vector<int>> strata_reverse_sorted_indices;
    if (!GetStrataSortedIndices(
            strata_vars, &strata_reverse_sorted_indices)) {
      cout << "\nERROR: Failed to GetStrataSortedIndices." << endl;
      return false;
    }

    // Compute K-M Estimators (if necessary).
    map<int, VectorXd> strata_km_estimators;
    if (!ComputeKaplanMeierEstimator(
            params.kme_type_, strata_vars, &strata_km_estimators)) {
      cout << "\nERROR: Failed to ComputeKaplanMeierEstimator." << endl;
      return false;
    }

    // NOTE: Unlike ReadSimulatedDataAndRunCoxRegression above, we do not call
    // EvaluateScoreMethodFunction to compute U^2(beta)/I(beta) here, because:
    //   1) We don't need this, as this is needed for CP (Coverage Probability),
    //      which is only relevant for simulations; and
    //   2) We don't know the actual \beta to use

    // Compute Width (find two roots of: U^2(x)/I(x) = 3.8415, and then take
    // the difference between them).
    if (params.analysis_params_.score_method_width_analysis_ &&
        !CoxDualRegression::ComputeScoreMethodCi(
          params.linear_term_values_.rows(), params.linear_term_values_.cols(),
          params.use_ties_constant_,
          summary_stats->estimates_[0],
          summary_stats->standard_error_[0],
          strata_km_estimators, strata_km_estimators, strata_vars,
          strata_reverse_sorted_indices,
          &summary_stats->score_method_ci_left_,
          &summary_stats->score_method_ci_right_)) {
      cout << "\nERROR in ComputeScoreMethodCi." << endl;
      return false;
    }
  }

  return true;
}
*/

bool CoxDualRegression::RunCoxRegression(
    const ModelAndDataParams& params,
    map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row,
    SummaryStatistics* summary_stats) {
  // Set strata to be all data if no strata was specified.
  map<int, int> orig_rows_to_one_strata;
  if (params.row_to_strata_.empty()) {
    for (int i = 0; i < params.dep_vars_.dep_vars_cox_.size(); ++i) {
      orig_rows_to_one_strata.insert(make_pair(i, 0));
    }
  }

  // Run Cox Regression for this model.
  summary_stats->num_iterations_ = 0;
  if (!Compute(
          params,
          params.row_to_strata_.empty() ?
              orig_rows_to_one_strata : params.row_to_strata_,
          unsorted_model_row_to_strata_index_and_row, summary_stats)) {
    cout << "Failed to compute Cox Regression.\n";
    return false;
  }
  return true;
}

void CoxDualRegression::GetCommonRows(
    const map<int, int>& orig_row_to_unsorted_model_row_one,
    const map<int, int>& orig_row_to_unsorted_model_row_two,
    set<int>* common_rows) {
  // Find common rows.
  const set<int> rows_model_one = Keys(orig_row_to_unsorted_model_row_one);
  const set<int> rows_model_two = Keys(orig_row_to_unsorted_model_row_two);
  set_intersection(rows_model_one.begin(), rows_model_one.end(),
                   rows_model_two.begin(), rows_model_two.end(),
                   inserter(*common_rows, common_rows->begin()));
}

CensoringData CoxDualRegression::GetDepVar(
    const vector<CensoringData>& dep_vars, const int index) {
  return dep_vars[index];
}

bool CoxDualRegression::SortInputByTime(
      const vector<CensoringData>& dep_vars,
      const MatrixXd& linear_term_values,
      const map<int, int>& row_to_strata,
      map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row,
      vector<StratificationData>* stratification_data) {
  // Sort by time, and keep track of the original order via
  // 'sorted_row_to_input_row', which maps the sorted rows to the original
  // row index. E.g. if row j has the smallest time, sorted_row_to_input_row
  // will include the pair (1, j).
  map<int, int> sorted_row_to_input_row;
  if (!GetSortLegend(dep_vars, &sorted_row_to_input_row)) return false;

  if (sorted_row_to_input_row.size() != row_to_strata.size()) {
    cout << "ERROR in SortInputByTime: mismatching vectors: "
         << "sorted_row_to_input_row.size(): " << sorted_row_to_input_row.size()
         << ", row_to_strata.size(): " << row_to_strata.size() << endl;
    return false;
  }

  // Partition (sorted) dep_vars and linear_term_values into their strata.
  for (const pair<int, int>& sorted_rows_itr : sorted_row_to_input_row) {
    const int sorted_row_index = sorted_rows_itr.first;
    // The data input in dep_vars may not match the original data that was
    // read in (e.g. it may be a subset of the original data, if --subgroup
    // was specified). This complicates things, because there are now three
    // indices corresponding to a data row:
    //   1) The original index (based on file read in)
    //   2) The index with respect to 'dep_vars'
    //   3) The sorted index
    // The 'row_to_strata' is keyed by indices as in (1);
    // 'sorted_row_to_input_row' has Keys indexed as in (3) and Values
    // indexed as in (2). To get the original row index (as in (1)) of
    // a sorted index (as in (3)), we can use sorted_row_to_input_row
    // to get the index 'j' with respect to dep_vars (i.e. as in (2)),
    // and then look up the j^th element of row_to_strata.
    const int dep_var_row_index = sorted_rows_itr.second;
    if (dep_var_row_index >= row_to_strata.size()) {
      cout << "ERROR in SortInputByTime: Row " << dep_var_row_index
           << " of dep_vars (size: " << dep_vars.size() << ") has index "
           << "that exceeds the number of rows in row_to_strata ("
           << row_to_strata.size() << "). Aborting." << endl;
      return false;
    }
    map<int, int>::const_iterator orig_row_finder = row_to_strata.begin();
    advance(orig_row_finder, dep_var_row_index);
    const int orig_row_index = orig_row_finder->first;
    const int strata_index = orig_row_finder->second;

    StratificationData& data = (*stratification_data)[strata_index];
    // Update unsorted_model_row_to_strata_index_and_row with this row's
    // (sorted) position within the strata.
    const int strata_row = data.dep_vars_.size();
    unsorted_model_row_to_strata_index_and_row->insert(
        make_pair(dep_var_row_index, make_pair(strata_index, strata_row)));
    data.indep_vars_.row(strata_row) =
        linear_term_values.row(dep_var_row_index);
    data.dep_vars_.push_back(GetDepVar(dep_vars, dep_var_row_index));
  }

  for (StratificationData& data : *stratification_data) {
    // Sanity check number of rows in dep_vars matches indep_vars.
    if (data.dep_vars_.size() != data.indep_vars_.rows()) {
      cout << "ERROR in SortInputByTime: number of dep vars ("
           << data.dep_vars_.size() << ") doesn't equal number of rows "
           << "in indep vars (" << data.indep_vars_.rows()
           << "). Aborting." << endl;
      return false;
    }
    // Get transition_indices, i.e. indices i such that:
    //   data.dep_var[i] > data.dep_var[i - 1]
    if (!GetTransitionIndices(data.dep_vars_, &data.transition_indices_)) {
      return false;
    }
  }

  return true;
}

bool CoxDualRegression::InitializeStratificationData(
      const int p,
      const vector<CensoringData>& dep_vars,
      const MatrixXd& linear_term_values,
      const map<int, int>& row_to_strata,
      map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row,
      vector<StratificationData>* stratification_data) {
  PopulateStratificationIndices(p, row_to_strata, stratification_data);

  // Sort linear_term_values based on Survival/Censoring time (computations
  // are easier based on this sorting).
  if (!SortInputByTime(
          dep_vars, linear_term_values, row_to_strata,
          unsorted_model_row_to_strata_index_and_row, stratification_data)) {
    return false;
  }
  return true;
}

bool CoxDualRegression::Compute(
      const ModelAndDataParams& params,
      const string* output_filename,
      const map<int, int>& row_to_strata,
      map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row,
      SummaryStatistics* stats) {
  // Sanity check input for strata.
  const int n = params.linear_term_values_.rows();
  const int p = params.linear_term_values_.cols();
  if (n == 0 || p == 0) {
    cout << "ERROR: No data (n = " << n << ", p = " << p << "). Aborting.\n";
    return false;
  }
  if (params.dep_vars_.dep_vars_cox_.size() != n) {
    cout << "ERROR: Invalid variables. Aborting.\n";
    return false;
  }

  // The number of independent variables must be less than the number
  // of sample rows (data values), or the system will be under-determined.
  if (p >= n) {
    cout << "ERROR: Invalid input file: Number of Cases (" << n
         << ") must exceed the number of independent variables ("
         << p << ").\n";
    return false;
  }

  // Create temporary data structure to hold all summary statistics, if necessary.
  SummaryStatistics temp_stats;
  if (stats == nullptr) {
    stats = &temp_stats;
  }

  // Compute stats of what fraction of samples are alive (status = 1).
  stats->fraction_alive_.push_back(make_tuple(1, n, -1.0));
  double& fraction_alive = get<2>(stats->fraction_alive_.back());
  GetFractionAlive(
      params.dep_vars_.dep_vars_cox_, &fraction_alive);

  // Initialize the data structure that will hold all information (input
  // data values, as well as all summary statistics).
  vector<StratificationData> stratification_data;
  map<int, pair<int, int>> unsorted_model_row_to_strata_index_and_row_temp;
  map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row_ptr =
      unsorted_model_row_to_strata_index_and_row == nullptr ?
      &unsorted_model_row_to_strata_index_and_row_temp :
      unsorted_model_row_to_strata_index_and_row;
  if (!InitializeStratificationData(
          p, params.dep_vars_.dep_vars_cox_, params.linear_term_values_,
          row_to_strata, unsorted_model_row_to_strata_index_and_row_ptr,
          &stratification_data)) {
    return false;
  }

  // Run Cox Proportional Hazards to compute Regression coefficients.
  if (params.analysis_params_.standard_analysis_) {
    VectorXd sorted_regression_coefficients;
    if (!ComputeRegressionCoefficients(
            n, p, params.kme_type_, params.convergence_criteria_,
            &stats->num_iterations_, &sorted_regression_coefficients,
            &stats->final_info_matrix_inverse_, &stratification_data)) {
      cout << "ERROR in Computing Regression Coefficients for full model."
           << endl;
      return false;
    }

    // Compute W-Vectors and Robust Variance.
    MatrixXd* robust_var = &(stats->robust_var_);
    if (params.analysis_params_.robust_analysis_ &&
        !ComputeWVectorsAndRobustVariance(
            p, sorted_regression_coefficients, stats->final_info_matrix_inverse_,
            *unsorted_model_row_to_strata_index_and_row_ptr, &stratification_data,
            robust_var, stats)) {
      return false;
    }

    // Calculate final Summary Statistics.
    char error_msg[512] = "";
    if (!ComputeFinalValues(
            stats->final_info_matrix_inverse_, sorted_regression_coefficients,
            stats, error_msg)) {
      string error_msg_str(error_msg);
      cout << error_msg_str << endl;
      return false;
    }
  }

  // Compute Log Rank Statistics.
  if (params.analysis_params_.log_rank_analysis_ ||
      params.analysis_params_.peto_analysis_) {
    // Can only compute log-rank statistics if p is 1.
    if (p != 1) {
      cout << "Warning: Unable to compute log-rank when there is more than "
           << "one independent variable (found p = " << p << "). Skipping."
           << endl;
      return false;
    }

    // Set the constant multiplier used to handle ties.
    if (params.use_ties_constant_) {
      SetTiesConstant(&stratification_data);
    }

    // Compute log-rank statistics.
    double* log_rank_estimate = &(stats->log_rank_estimate_);
    double* log_rank_variance = &(stats->log_rank_variance_);
    if (!ComputeLogRank(
            params.kme_type_for_log_rank_,
            &stratification_data, log_rank_estimate, log_rank_variance)) {
      return false;
    }
    stats->log_rank_estimate_squared_ =
        (*log_rank_estimate) * (*log_rank_estimate);
    stats->log_rank_standard_estimate_of_error_ = sqrt(*log_rank_variance);
    stats->peto_estimate_ = *log_rank_estimate / *log_rank_variance;
    stats->peto_estimate_squared_ =
        stats->peto_estimate_ * stats->peto_estimate_;
    stats->peto_variance_ = 1.0 / *log_rank_variance;
    stats->peto_standard_estimate_of_error_ = 1.0 / sqrt(*log_rank_variance);
    if (!ComputeLogRankWVectors(
            *unsorted_model_row_to_strata_index_and_row_ptr,
            &stratification_data, stats)) {
      return false;
    }
  }

  // Print Output.
  if (output_filename != nullptr) {
    ofstream out_file;
    out_file.open((*output_filename).c_str());
    PrintSummaryStatistics(out_file, params, *stats);
    out_file.close();
  }

  return true;
}

bool CoxDualRegression::MatrixHasNanTerm(
    const MatrixXd& mat) {
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      if (std::isnan(mat(i, j))) return true;
    }
  }
  return false;
}

double CoxDualRegression::GetKmeOfPreviousTimepoint(
    const CensoringData& data,
    const vector<CensoringData>& dep_var,
    const VectorXd& km_estimate) {
  if (dep_var.size() != km_estimate.size()) {
    cout << "ERROR: Mismatching number of dependent variables ("
         << dep_var.size() << ") and KM estimates ("
         << km_estimate.size() << ")." << endl;
    return -1.0;
  }
  
  const double time =
      data.is_alive_ ? data.censoring_time_ : data.survival_time_;
  int closest_index = -1;
  double closest_time = -1.0;
  for (int i = 0; i < dep_var.size(); ++i) {
    const double sample_time = dep_var[i].is_alive_ ?
        dep_var[i].censoring_time_ : dep_var[i].survival_time_;
    if (sample_time < time && sample_time > closest_time) {
      closest_index = i;
      closest_time = sample_time;
    }
  }
  if (closest_index == -1) {
    return 1.0;
  } else {
    return km_estimate(closest_index);
  }
}

void CoxDualRegression::GetFractionAlive(
    const vector<CensoringData>& dep_var, double* fraction_alive) {
  if (fraction_alive == nullptr) return;
  int n = dep_var.size();
  int num_alive = 0;
  for (const CensoringData& sample : dep_var) {
    if (sample.is_alive_) num_alive++;
  }
  *fraction_alive = static_cast<double>(num_alive) / n;
}

bool CoxDualRegression::GetFractionAlive(
    const vector<pair<int, int>>& sample_partitions,
    const map<int, set<int>>* subgroup_index_to_its_rows,
    const vector<CensoringData>& dep_var,
    vector<double>* fraction_alive) {
  if (fraction_alive == nullptr) return true;
  fraction_alive->clear();
  fraction_alive->resize(sample_partitions.size(), 0.0);

  // The original partitions are with respect to the entire data set.
  // If a subgroup was taken, we should only print fraction alive info
  // for the rows in the subgroups. Here, we find which rows these are.
  map<int, int> orig_row_to_subgroup_row;
  const bool has_subgroups =
      (subgroup_index_to_its_rows != nullptr) &&
      (!subgroup_index_to_its_rows->empty());
  if (has_subgroups) {
    // Iterate over all subgroups, picking out their row index (with respect
    // to the original data).
    set<int> rows_in_subgroup;
    for (const pair<int, set<int>>& subgroups_itr : *subgroup_index_to_its_rows) {
      for (const int row : subgroups_itr.second) {
        rows_in_subgroup.insert(row);
      }
    }
    if (rows_in_subgroup.size() != dep_var.size()) {
      cout << "ERROR: Expected the total number of rows across all Subgroups ("
           << rows_in_subgroup.size() << ") to equal the number of rows in "
           << "dep_var (" << dep_var.size() << ")." << endl;
      return false;
    }

    // Create orig_row_to_subgroup_row mapping. Note that for any two rows
    // with respect to dep_var (i.e. w.r.t. Subgroup indexing), their relative
    // index with respect to the original data will respect the same order.
    int subgroup_index = 0;
    for (const int orig_index : rows_in_subgroup) {
      orig_row_to_subgroup_row.insert(make_pair(orig_index, subgroup_index));
      subgroup_index++;
    }
  }

  for (int i = 0; i < sample_partitions.size(); ++i) {
    const pair<int, int>& current_partition = sample_partitions[i];
    const int left = current_partition.first;
    const int right = current_partition.second;

    // Determine which of these partition indices are in the subgroup
    // (should be all or none, print error otherwise).
    set<int> rows_in_partition_and_subgroup;
    if (has_subgroups) {
      for (int j = left; j <= right; ++j) {
        if (orig_row_to_subgroup_row.find(j) !=
            orig_row_to_subgroup_row.end()) {
          rows_in_partition_and_subgroup.insert(
              orig_row_to_subgroup_row.find(j)->second);
        }
      }
      if (!rows_in_partition_and_subgroup.empty() &&
          rows_in_partition_and_subgroup.size() != (right - left + 1)) {
        cout << "ERROR: Expected each subgroup to either be disjoint "
             << "from each partition, or to be the entire partition. "
             << "instead, the intersection of partition [" << left
             << ", " << right << "] with the rows in the Subgroups "
             << "has size " << rows_in_partition_and_subgroup.size() << endl;
        return false;
      }
    }

    // Pick out the relevant rows of dep_var (those that are in this
    // partition).
    if (has_subgroups) {
      // Nothing to do if there are no rows for this subgroup in the partition.
      if (rows_in_partition_and_subgroup.empty()) {
        (*fraction_alive)[i] = -1.0;
      } else {
        vector<CensoringData> samples_in_partition;
        for (const int subgroup_index_itr : rows_in_partition_and_subgroup) {
          samples_in_partition.push_back(dep_var[subgroup_index_itr]);
        }
        GetFractionAlive(samples_in_partition, &((*fraction_alive)[i]));
      }
    } else {
      if (left < 1 || left > dep_var.size() ||
          right < left || right > dep_var.size()) {
        // Should never happen: one of the sampling partitions lies outside
        // the number of samples.
        cout << "ERROR: Unexpected partition [" << left << ", " << right
             << "] is not a subset of the Sample indices [1, "
             << dep_var.size() << "]. This should never happen. "
             << "Fraction Alive printed numbers may be off." << endl;
        return false;
      }
      vector<CensoringData> samples_in_partition(right - left + 1);
      copy(dep_var.begin() + (left - 1), dep_var.begin() + right,
           samples_in_partition.begin());
      GetFractionAlive(samples_in_partition, &((*fraction_alive)[i]));
    }
  }
  return true;
}

void CoxDualRegression::PushBackStratificationIndices(
    const int n, const int p,
    vector<StratificationData>* stratification_data) {
  // Initialize stratification_data.
  stratification_data->push_back(StratificationData());
  StratificationData& data = stratification_data->back();
  data.ties_constant_.resize(n);
  // By default, we set tie-breaking constant to 1.0. If user specified
  // log-rank, peto, or Score Method, then this will get updated
  // appropriately in SetTiesConstant().
  for (int i = 0; i < n; ++i) {
    data.ties_constant_(i) = 1.0;
  }
  data.indep_vars_.resize(n, p);
}
    
void CoxDualRegression::PopulateStratificationIndices(
    const int p, const map<int, int>& row_to_strata,
    vector<StratificationData>* stratification_data) {
  // First compute the size of each strata.
  map<int, int> strata_index_to_size;
  for (const pair<int, int>& row_to_strata_itr : row_to_strata) {
    const int strata_index = row_to_strata_itr.second;
    map<int, int>::iterator itr = strata_index_to_size.find(strata_index);
    if (itr == strata_index_to_size.end()) {
      strata_index_to_size.insert(make_pair(strata_index, 1));
    } else {
      (itr->second)++;
    }
  }

  // Now go through and initialize elements of each strata.
  for (const pair<int, int>& itr : strata_index_to_size) {
    PushBackStratificationIndices(itr.second, p, stratification_data);
  }
}

void CoxDualRegression::SetTiesConstant(
    const vector<CensoringData>& dep_vars,
    VectorXd* ties_constant) {
  ties_constant->resize(dep_vars.size());
  for (int j = 0; j < dep_vars.size(); ++j) {
    double r_value = 0.0;
    double d_value = 0.0;
    const CensoringData& current_data = dep_vars[j];
    const double current_time =
        min(current_data.survival_time_, current_data.censoring_time_);
    for (int i = 0; i < dep_vars.size(); ++i) {
      const CensoringData& data = dep_vars[i];
      const double time = min(data.survival_time_, data.censoring_time_);
      const bool is_alive = data.survival_time_ > data.censoring_time_;
      if (time >= current_time) {
        r_value += 1.0;
      }
      if (!is_alive && time == current_time) {
        d_value += 1.0;
      }
    }
    (*ties_constant)(j) =
        r_value == 1.0 ? 1.0 : (r_value - d_value) / (r_value - 1.0);
  }
}

void CoxDualRegression::SetTiesConstant(
    vector<StratificationData>* stratification_data) {
  for (StratificationData& data : *stratification_data) {
    const vector<CensoringData>& dep_vars = data.dep_vars_;
    VectorXd ties_constant;
    SetTiesConstant(dep_vars, &ties_constant);
    data.ties_constant_ = ties_constant;
  }
}

bool CoxDualRegression::GetSortLegend(
    const vector<CensoringData>& dep_vars,
    map<int, int>* sorted_row_to_orig_row) {
  // Get minimum of Survival Time and Censoring Time.
  vector<double> times;
  for (const CensoringData& data : dep_vars) {
    times.push_back(min(data.survival_time_, data.censoring_time_));
  }

  // Sort times.
  sort(times.begin(), times.end());

  // Now go through sorted times, finding transition indices, and mapping
  // back to the original index that had that time.
  set<int> used_indices;
  for (int i = 0; i < times.size(); ++i) {
    const double& time = times[i];
    bool found_match = false;
    for (int j = 0; j < dep_vars.size(); ++j) {
      if (used_indices.find(j) != used_indices.end()) continue;
      const CensoringData& data = dep_vars[j];
      if (FloatEq(time, min(data.survival_time_, data.censoring_time_))) {
        used_indices.insert(j);
        sorted_row_to_orig_row->insert(make_pair(i, j));
        found_match = true;
        break;
      }
    }
    if (!found_match) {
      cout << "ERROR in GetSortLegend: Unable to find time "
           << time << ". Aborting.\n";
      return false;
    }
  }
  return true;
}

bool CoxDualRegression::GetTransitionIndices(
    const vector<CensoringData>& dep_vars, vector<int>* transition_indices) {
  double previous_time = -1.0;
  for (int i = 0; i < dep_vars.size(); ++i) {
    const  CensoringData& data = dep_vars[i];
    const double& time = data.is_alive_ ?
        data.censoring_time_ : data.survival_time_;
    if (time != previous_time) {
      transition_indices->push_back(i);
      previous_time = time;
    }
  }
  return true;
}

bool CoxDualRegression::GetStrataVars(
    const map<int, pair<int, int>>& unsorted_model_row_to_strata_index_and_row,
    const vector<CensoringData>& dep_var,
    const MatrixXd& indep_vars,
    map<int, pair<vector<CensoringData>, MatrixXd>>* strata_vars) {
  if (strata_vars == nullptr) return false;
  // Revert to ordinary (unstratified) version if no strata.
  if (unsorted_model_row_to_strata_index_and_row.empty()) {
    strata_vars->insert(make_pair(0, make_pair(dep_var, indep_vars)));
    return true;
  }

  // Get a map from strata index to strata size.
  map<int, int> strata_index_to_size;
  for (const pair<int, pair<int, int>>& row_to_strata_and_row_itr :
       unsorted_model_row_to_strata_index_and_row) {
    pair<map<int, int>::iterator, bool> insertion_itr =
        strata_index_to_size.insert(make_pair(
            row_to_strata_and_row_itr.second.first, 1));
    if (!insertion_itr.second) {
      // Strata index already seen; update count by one.
      insertion_itr.first->second += 1;
    }
  }

  // Go through each strata, computing Score Function and Information Matrix
  // for it, and then add those to a running sum.
  for (map<int, int>::const_iterator itr = strata_index_to_size.begin();
       itr != strata_index_to_size.end(); ++itr) {
    const int strata_index = itr->first;
    pair<vector<CensoringData>, MatrixXd>& current_strata_vars =
        (strata_vars->insert(make_pair(strata_index, make_pair(
            vector<CensoringData>(), MatrixXd()))).first)->second;
    // Pick out the subset of (In)dependent vars among the (In)dependent
    // vars that belong to this strata.
    MatrixXd& strata_indep_vars = current_strata_vars.second;
    strata_indep_vars.resize(itr->second, indep_vars.cols());
    vector<CensoringData>& strata_dep_var = current_strata_vars.first;
    for (const pair<int, pair<int, int>>& row_to_strata_and_row_itr :
         unsorted_model_row_to_strata_index_and_row) {
      const int curr_strata_index = row_to_strata_and_row_itr.second.first;
      if (curr_strata_index != strata_index) continue;
      strata_indep_vars.row(strata_dep_var.size()) =
          indep_vars.row(row_to_strata_and_row_itr.first);
      strata_dep_var.push_back(dep_var[row_to_strata_and_row_itr.first]);
    }
  }
  return true;
}

bool CoxDualRegression::GetStrataSortedIndices(
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    map<int, vector<int>>* strata_reverse_sorted_indices) {
  if (strata_reverse_sorted_indices == nullptr) return false;
  for (const pair<int, pair<vector<CensoringData>, MatrixXd>>& strata_itr :
       strata_vars) {
    const int strata_index = strata_itr.first;
    const vector<CensoringData>& strata_dep_vars = strata_itr.second.first;
    vector<int>& current_sorted_indices =
        (strata_reverse_sorted_indices->insert(
             make_pair(strata_index, vector<int>()))).first->second;
    map<int, int> sorted_row_to_orig_row;
    if (!GetSortLegend(strata_dep_vars, &sorted_row_to_orig_row)) {
      return false;
    }
    for (map<int, int>::reverse_iterator reverse_itr = sorted_row_to_orig_row.rbegin();
         reverse_itr != sorted_row_to_orig_row.rend(); ++reverse_itr) {
      current_sorted_indices.push_back(reverse_itr->second);
    }
  }
  return true;
}

bool CoxDualRegression::ConstructMappingFromOrigRowToUnsortedModelRow(
    const ModelAndDataParams& params,
    map<int, int>* orig_row_to_unsorted_model_row) {
  if (!params.row_to_strata_.empty()) {
    // For stratified models, we can use params.row_to_strata_ to give us the
    // row index (w.r.t. data in format after Step (1)) of all rows that
    // were kept. The corresponding index of each row w.r.t. data in format
    // after Step (3) is simply that row's rank (sorted order) among the rows
    // that were kept.
    int i = 0;
    for (const pair<int, int>& row_to_strata_itr : params.row_to_strata_) {
      orig_row_to_unsorted_model_row->insert(make_pair(
          row_to_strata_itr.first, i));
      ++i;
    }
  } else if (!params.subgroup_rows_per_index_.empty()) {
    // For models that use Subgroups, we can use params.subgroup_rows_per_index_
    // to give us the row index (w.r.t. data in format after Step (1)) of all
    // rows that were kept. The corresponding index of each row w.r.t. data in
    // format after Step (3) is simply that row's rank (sorted order) among the
    // rows that were kept.
    set<int> kept_rows;
    for (const pair<int, set<int>>& subgroup_index_to_its_rows :
         params.subgroup_rows_per_index_) {
      for (const int row : subgroup_index_to_its_rows.second) {
        if (!kept_rows.insert(row).second) {
          cout << "ERROR: Row " << row << " appears in multiple Subgroups."
               << endl;
          return false;
        }
      }
    }
    int i = 0;
    for (const int& row_index : kept_rows) {
      // We subtract 1 from row_index, because rows in subgroup_rows_per_index_
      // are 1-based, and we want rows in orig_row_to_unsorted_model_row to
      // be 0-based.
      orig_row_to_unsorted_model_row->insert(make_pair(row_index - 1, i));
      ++i;
    }
  } else {
    // Step (3) above didn't remove any rows (since there aren't subgroups);
    // thus, the order of rows in params.linear_term_values_ matches the order
    // in data_values, except for the NA rows. We use params.na_rows_skipped_
    // to map.
    int num_na_rows_before_current_row = 0;
    for (int i = 0; i < params.linear_term_values_.rows(); ++i) {
      bool current_row_not_na = true;
      int orig_row_index = i + num_na_rows_before_current_row;
      // Add 1 to orig_row_index, since na_rows_skipped_ is 1-based.
      while (params.na_rows_skipped_.find(orig_row_index + 1) !=
             params.na_rows_skipped_.end()) {
        // The original row was skipped, got to next row index.
        current_row_not_na = false;
        ++num_na_rows_before_current_row;
        orig_row_index = i + num_na_rows_before_current_row;
      }
      // It's possible that all of the last rows were skipped due to
      // missing values; make sure orig_row_index is within bounds.
      if (num_na_rows_before_current_row < params.na_rows_skipped_.size() ||
          current_row_not_na) {
        orig_row_to_unsorted_model_row->insert(make_pair(orig_row_index, i));
      }
    }
  }

  return true;
}

bool CoxDualRegression::ComputePartialSums(
    const vector<int>& transition_indices,
    const VectorXd& exp_logistic_eq,
    VectorXd* partial_sums) {
  double running_total = 0.0;
  partial_sums->resize(transition_indices.size());
  int backwards_itr = exp_logistic_eq.size() - 1;  // n_k - 1; n_k = strata size
  for (int i = transition_indices.size() - 1; i >= 0; --i) {
    const int transition_index = transition_indices[i];
    while (backwards_itr >= transition_index) {
      running_total += exp_logistic_eq(backwards_itr);
      --backwards_itr;
    }
    (*partial_sums)(i) = running_total;
  }
  return true;
}

bool CoxDualRegression::ComputeExponentialOfLogisticEquation(
    const VectorXd& beta_hat,
    const vector<VectorXd>& indep_vars,
    VectorXd* logistic_eq,
    VectorXd* exp_logistic_eq) {
  if (indep_vars.empty() || indep_vars[0].size() != beta_hat.size()) {
    return false;
  }
  logistic_eq->resize(indep_vars.size());
  exp_logistic_eq->resize(indep_vars.size());
  for (int i = 0; i < indep_vars.size(); ++i) {
    (*logistic_eq)(i) = beta_hat.transpose() * indep_vars[i];
    (*exp_logistic_eq)(i) = exp((*logistic_eq)(i));
  }
  return true;
}

bool CoxDualRegression::ComputeExponentialOfLogisticEquation(
    const VectorXd& beta_hat,
    const MatrixXd& indep_vars,
    VectorXd* logistic_eq,
    VectorXd* exp_logistic_eq) {
  if (indep_vars.cols() != beta_hat.size()) return false;
  *logistic_eq = beta_hat.transpose() * indep_vars.transpose();
  exp_logistic_eq->resize(indep_vars.rows());
  for (int i = 0; i < indep_vars.rows(); ++i) {
    (*exp_logistic_eq)(i) = exp((*logistic_eq)(i));
  }
  return true;
}

bool CoxDualRegression::ComputeLogLikelihood(
    const VectorXd& logistic_eq,
    const VectorXd& exp_logistic_eq,
    const VectorXd& partial_sums,
    const vector<int>& transition_indices,
    const vector<CensoringData>& dep_var,
    double* log_likelihood) {
  *log_likelihood = 0;
  int current_transition_index = dep_var.size();
  int transition_itr = transition_indices.size();
  for (int i = dep_var.size() - 1; i >= 0; --i) {
    // Check if i is in the previous block (i.e. if i + 1 is a transition index)
    if (i < current_transition_index) {
      transition_itr--;
      current_transition_index = transition_indices[transition_itr];
    }
    // Only terms with Delta_i = 1 (i.e. I(T <= C) = 1; i.e. patient has died)
    // contribute to Score Function.
    if (dep_var[i].is_alive_) continue;
    *log_likelihood += logistic_eq(i) - log(partial_sums(transition_itr));
  }
  return true;
}

bool CoxDualRegression::ComputeGlobalScoreFunctionAndInfoMatrix(
    const vector<StratificationData>& stratification_data,
    VectorXd* score_function, MatrixXd* info_matrix) {
  const int p = stratification_data[0].score_function_.size();
  score_function->resize(p);
  score_function->setZero();
  info_matrix->resize(p, p);
  info_matrix->setZero();

  for (const StratificationData& data : stratification_data) {
    *score_function += data.score_function_;
    *info_matrix += data.information_matrix_;
  }

  // Sanity check matrix is invertible.
  FullPivLU<MatrixXd> lu = info_matrix->fullPivLu();
  if (!lu.isInvertible()) {
    cout << "ERROR in ComputeGlobalScoreFunctionAndInfoMatrix: "
         << "info_matrix is not invertible:\n" << *info_matrix << "\n";
    return false;
  }
  return true;
}

bool CoxDualRegression::ComputeKaplanMeierEstimator(
    const KaplanMeierEstimatorType& kme_type,
    const VectorXd& partial_sums, const vector<int>& transition_indices,
    const vector<CensoringData>& dep_var,
    VectorXd* estimator) {
  if (estimator == nullptr) return true;

  const int n = dep_var.size();
  estimator->resize(n);
  
  if (kme_type == KaplanMeierEstimatorType::NONE) {
    for (int i = 0; i < n; ++i) {
      (*estimator)[i] = 1.0;
    }
    return true;
  }

  // Use K-M Estimator formula to compute KME:
  //   - If kme_type = RIGHT_CONTINUOUS
  //       KME(T_i) = \Pi_{T_j <= T_i } (1 - (\Delta_j / S^0(0, T_j)))
  //   - If kme_type = LEFT_CONTINUOUS
  //       KME(T_i) = \Pi_{T_j < T_i } (1 - (\Delta_j / S^0(0, T_j)))
  //   - If there are ties, then for each {T_j} that are equal, only
  //     take one term in the product, and instead of using \Delta_j,
  //     use \sum_{T_k \in T_j} \Delta_k
  for (int i = 0; i < n; ++i) {
    (*estimator)[i] = 1.0;
    map<pair<double, int>, int> time_and_S0_to_delta;
    const double time =
        min(dep_var[i].survival_time_, dep_var[i].censoring_time_);
    int current_transition_index = n;
    int transition_itr = transition_indices.size();
    // Go through all other T_j, multiplying by the appropriate factor
    // for each T_j >= T_i.
    for (int j = dep_var.size() - 1; j >= 0; --j) {
      // Check if i is in the previous block (i.e. if j + 1 is a transition index)
      if (j < current_transition_index) {
        transition_itr--;
        current_transition_index = transition_indices[transition_itr];
      }

      const double time_j =
          min(dep_var[j].survival_time_, dep_var[j].censoring_time_);
      // Only times less than current time contribute to KME.
      if (kme_type == KaplanMeierEstimatorType::LEFT_CONTINUOUS &&
          time_j >= time) {
        continue;
      }
      if (kme_type == KaplanMeierEstimatorType::RIGHT_CONTINUOUS &&
          time_j > time) {
        continue;
      }

      // If \Delta_j = 0, the term contributes '1' to the product; i.e.
      // these indices do not contribute to KME.
      if (dep_var[j].is_alive_) continue;

      const pair<double, int> key =
          make_pair(time_j, static_cast<int>(partial_sums(transition_itr)));
      map<pair<double, int>, int>::iterator term_itr =
          time_and_S0_to_delta.find(key);
      if (term_itr == time_and_S0_to_delta.end()) {
        time_and_S0_to_delta.insert(make_pair(key, 1));
      } else {
        (term_itr->second)++;
      }
    }
    for (const auto& terms_itr : time_and_S0_to_delta) {
      (*estimator)[i] *=
          (1.0 - (static_cast<double>(terms_itr.second) /
                  terms_itr.first.second));
    }
  }
  return true;
}

bool CoxDualRegression::ComputeKaplanMeierEstimator(
    const KaplanMeierEstimatorType& kme_type,
    const VectorXd& S0, const vector<CensoringData>& dep_var,
    VectorXd* estimator) {
  if (estimator == nullptr) return true;

  const int n = dep_var.size();
  estimator->resize(n);
 
  if (kme_type == KaplanMeierEstimatorType::NONE) {
    for (int i = 0; i < n; ++i) {
      (*estimator)[i] = 1.0;
    }
    return true;
  }

  // Sanity-check input.
  if (S0.size() != n) {
    cout << "ERROR: S^0 vector has different size (" << S0.size()
         << ") than the number of samples (" << n << ")." << endl;
    return false;
  }

  // Use K-M Estimator formula to compute KME:
  //   - If kme_type = RIGHT_CONTINUOUS
  //       KME(T_i) = \Pi_{T_j <= T_i } (1 - (\Delta_j / S^0(0, T_j)))
  //   - If kme_type = LEFT_CONTINUOUS
  //       KME(T_i) = \Pi_{T_j < T_i } (1 - (\Delta_j / S^0(0, T_j)))
  //   - If there are ties, then for each {T_j} that are equal, only
  //     take one term in the product, and instead of using \Delta_j,
  //     use \sum_{T_k \in T_j} \Delta_k
  for (int i = 0; i < n; ++i) {
    (*estimator)[i] = 1.0;
    map<pair<double, int>, int> time_and_S0_to_delta;
    const double time =
        min(dep_var[i].survival_time_, dep_var[i].censoring_time_);
    // Go through all other T_j, multiplying by the appropriate factor
    // for each T_j >= T_i.
    for (int j = 0; j < n; ++j) {
      const double time_j =
          min(dep_var[j].survival_time_, dep_var[j].censoring_time_);
      // Only times less than current time contribute to KME.
      if (kme_type == KaplanMeierEstimatorType::LEFT_CONTINUOUS &&
          time_j >= time) {
        continue;
      }
      if (kme_type == KaplanMeierEstimatorType::RIGHT_CONTINUOUS &&
          time_j > time) {
        continue;
      }

      // If \Delta_j = 0, the term contributes '1' to the product; i.e.
      // these indices do not contribute to KME.
      if (dep_var[j].is_alive_) continue;
      const pair<double, int> key = make_pair(time_j, static_cast<int>(S0(j)));
      map<pair<double, int>, int>::iterator term_itr =
          time_and_S0_to_delta.find(key);
      if (term_itr == time_and_S0_to_delta.end()) {
        time_and_S0_to_delta.insert(make_pair(key, 1));
      } else {
        (term_itr->second)++;
      }
    }
    for (const auto& terms_itr : time_and_S0_to_delta) {
      (*estimator)[i] *=
          (1.0 - (static_cast<double>(terms_itr.second) /
                  terms_itr.first.second));
    }
  }
  return true;
}

bool CoxDualRegression::ComputeKaplanMeierEstimator(
    const KaplanMeierEstimatorType& kme_type,
    const vector<CensoringData>& dep_var, VectorXd* estimator) {
  if (estimator == nullptr) return true;

  // Compute S^0(0, T_j), which is the term in the denominator of K-M.
  const int n = dep_var.size();
  // Initialize S0 to be zero.
  VectorXd S0;
  S0.resize(n);
  S0.setZero();
  for (int i = 0; i < n; ++i) {
    // Get min(survival, sensoring) time for T_i.
    const double time =
        min(dep_var[i].survival_time_, dep_var[i].censoring_time_);
    // Go through all other T_j, adding one to S0[i] for each T_j >= T_i.
    for (int j = 0; j < n; ++j) {
      const double time_j =
          min(dep_var[j].survival_time_, dep_var[j].censoring_time_);
      if (time_j < time) continue;
      S0[i] += 1;
    }
  }

  return ComputeKaplanMeierEstimator(kme_type, S0, dep_var, estimator);
}

bool CoxDualRegression::ComputeKaplanMeierEstimator(
    const KaplanMeierEstimatorType& kme_type,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    map<int, VectorXd>* strata_km_estimators) {
  for (const pair<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars_itr :
       strata_vars) {
    const int strata_index = strata_vars_itr.first;
    const vector<CensoringData>& strata_dep_vars = strata_vars_itr.second.first;
    VectorXd& current_strata_km_estimators =
        (strata_km_estimators->insert(make_pair(
             strata_index, VectorXd())).first)->second;
    if (!ComputeKaplanMeierEstimator(
            kme_type, strata_dep_vars, &current_strata_km_estimators)) {
      cout << "ERROR: Failed to ComputeKaplanMeierEstimator for strata "
           << strata_index << endl;
      return false;
    }
  }
  return true;
}

bool CoxDualRegression::ComputePartialSums(
    const int n, const int p,
    const VectorXd& beta,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    vector<double>* S0, vector<VectorXd>* S1, vector<MatrixXd>* S2) {
  // Sanity-check input.
  if (S0 == nullptr || S1 == nullptr || S2 == nullptr) return false;
  if (n == 0 || p == 0 ||
      indep_vars.rows() != n || indep_vars.cols() != p || dep_var.size() != n) {
    return false;
  }
  
  // Compute <\beta|X> and exp(<\beta|X>) once.
  VectorXd logistic_eq, exp_logistic_eq;
  if (!ComputeExponentialOfLogisticEquation(
          beta, indep_vars, &logistic_eq, &exp_logistic_eq)) {
    return false;
  }

  for (int i = 0; i < n; ++i) {
    // Initialize S0 to be zero.
    S0->push_back(double());
    double& S0_value = S0->back();
    S0_value = 0.0;

    // Initialize S1 to be a vector of zeros.
    S1->push_back(VectorXd());
    VectorXd& S1_value = S1->back();
    S1_value.resize(p);
    S1_value.setZero();

    // Initialize S2 to be a matrix of zeros.
    S2->push_back(MatrixXd());
    MatrixXd& S2_value = S2->back();
    S2_value.resize(p, p);
    S2_value.setZero();

    // Get min(survival, sensoring) time for T_i.
    const double time =
        min(dep_var[i].survival_time_, dep_var[i].censoring_time_);
    for (int j = 0; j < n; ++j) {
      const double time_j =
          min(dep_var[j].survival_time_, dep_var[j].censoring_time_);
      if (time_j < time) continue;
      S0_value += exp_logistic_eq(j);
      S1_value += exp_logistic_eq(j) * indep_vars.row(j);
      S2_value +=
          exp_logistic_eq(j) *
          (indep_vars.row(j).transpose() * indep_vars.row(j));
    }
  }
  return true;
}

bool CoxDualRegression::ComputePartialSums(
    const int n, const int p,
    const VectorXd& beta,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    const vector<int>& reverse_sorted_indices,
    vector<double>* S0, vector<VectorXd>* S1, vector<MatrixXd>* S2) {
  // Sanity-check input.
  if (S0 == nullptr || S1 == nullptr || S2 == nullptr) return false;
  if (n == 0 || p == 0 ||
      indep_vars.rows() != n || indep_vars.cols() != p || dep_var.size() != n ||
      reverse_sorted_indices.size() != n) {
    return false;
  }
  
  // Compute <\beta|X> and exp(<\beta|X>) once.
  VectorXd logistic_eq, exp_logistic_eq;
  if (!ComputeExponentialOfLogisticEquation(
          beta, indep_vars, &logistic_eq, &exp_logistic_eq)) {
    return false;
  }

  S0->resize(n);
  S1->resize(n);
  S2->resize(n);

  double running_S0 = 0.0;
  VectorXd running_S1;
  running_S1.resize(p);
  running_S1.setZero();
  MatrixXd running_S2;
  running_S2.resize(p, p);
  running_S2.setZero();
  set<int> to_update;
  // Artificially set prev_time to be smallest double value, so that the
  // check involving it below will fail on the first pass through the loop.
  double prev_time = DBL_MIN;
  for (int j = 0; j < n; ++j) {
    const int i = reverse_sorted_indices[j];
    // Get min(survival, sensoring) time for T_i.
    const double time =
        min(dep_var[i].survival_time_, dep_var[i].censoring_time_);

    // Because we're iterating through sorted times (starting from largest),
    // we have time <= prev_time. If we have exact equality, we don't yet
    // update the previous time points' S-values.
    if (time < prev_time) {
      // This time point is strictly less than the previous one, which
      // means all previous S values (for indices in 'to_update') should
      // be updated.
      for (const int sample_index : to_update) {
        (*S0)[sample_index] = running_S0;
        (*S1)[sample_index] = running_S1;
        (*S2)[sample_index] = running_S2;
      }
      to_update.clear();
    }

    prev_time = time;

    // We don't set S values for this Sample yet, because we don't know
    // if Samples with earlier timepoints (based on sort-order) are
    // equal to this one or strictly before it. Mark it as needing to be
    // updated (which will happen as soon as we hit a timepoint that is
    // strictly less than this one).
    to_update.insert(i);

    // Update running S values.
    running_S0 += exp_logistic_eq(i);
    running_S1 += exp_logistic_eq(i) * indep_vars.row(i);
    running_S2 +=
        exp_logistic_eq(i) *
        (indep_vars.row(i).transpose() * indep_vars.row(i));
  }

  // There will have been some samples (those with earliest timepoint)
  // that haven't had their S values set. Do so now.
  for (const int sample_index : to_update) {
    (*S0)[sample_index] = running_S0;
    (*S1)[sample_index] = running_S1;
    (*S2)[sample_index] = running_S2;
  }
  return true;
}

bool CoxDualRegression::ComputeScoreFunction(
    const int n, const int p,
    const VectorXd& beta,
    const VectorXd& kaplan_meier_estimators,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    const vector<int>& reverse_sorted_indices,
    VectorXd* score_function) {
  // Sanity-check input.
  if (n == 0 || p == 0 || beta.size() != p ||
      indep_vars.rows() != n || indep_vars.cols() != p || dep_var.size() != n) {
    return false;
  }

  vector<double> S0;
  vector<VectorXd> S1;
  vector<MatrixXd> S2;
  if (reverse_sorted_indices.empty()) {
    ComputePartialSums(n, p, beta, indep_vars, dep_var, &S0, &S1, &S2);
  } else {
    ComputePartialSums(n, p, beta, indep_vars, dep_var, reverse_sorted_indices,
                       &S0, &S1, &S2);
  }

  score_function->resize(indep_vars.cols());
  score_function->setZero();
  for (int i = 0; i < n; ++i) {
    // Only terms with Delta_i = 1 (i.e. I(T <= C) = 1; i.e. patient has died)
    // contribute to Score Function.
    if (dep_var[i].is_alive_) continue;
    if (S0[i] == 0.0) return false;
    *score_function += kaplan_meier_estimators(i) *
        (static_cast<VectorXd>(indep_vars.row(i)) - (S1[i] / S0[i]));
  }
  return true;
}

bool CoxDualRegression::ComputeScoreFunction(
    const int n, const int p,
    const VectorXd& beta,
    const VectorXd& kaplan_meier_estimators,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    VectorXd* score_function) {
  return ComputeScoreFunction(
      n, p, beta, kaplan_meier_estimators, indep_vars, dep_var, vector<int>(),
      score_function);
}

bool CoxDualRegression::ComputeScoreFunction(
    const int n, const int p,
    const KaplanMeierEstimatorType& kme_type, 
    const VectorXd& beta,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    const vector<int>& reverse_sorted_indices,
    VectorXd* score_function) {
  VectorXd kme;
  if (!ComputeKaplanMeierEstimator(kme_type, dep_var, &kme)) {
    cout << "\nERROR: Failed to ComputeKaplanMeierEstimator." << endl;
    return false;
  }
  return ComputeScoreFunction(
      n, p, beta, kme, indep_vars, dep_var, reverse_sorted_indices,
      score_function);
}

bool CoxDualRegression::ComputeScoreFunction(
    const int n, const int p,
    const KaplanMeierEstimatorType& kme_type, 
    const VectorXd& beta,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    VectorXd* score_function) {
  return ComputeScoreFunction(
      n, p, kme_type, beta, indep_vars, dep_var, vector<int>(),
      score_function);
}

bool CoxDualRegression::ComputeScoreFunction(
    const int n, const int p,
    const VectorXd& beta,
    const map<int, VectorXd>& strata_kaplan_meier_estimators,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    const map<int, vector<int>>& strata_reverse_sorted_indices,
    map<int, VectorXd>* strata_to_score_function) {
  // Sanity-check number of strata is consistent between strata_vars and
  // strata_kaplan_meier_estimators.
  if (strata_kaplan_meier_estimators.size() != strata_vars.size()) {
    cout << "ERROR: mismatching strata size for KME ("
         << strata_kaplan_meier_estimators.size()
         << ") and strata vars (" << strata_vars.size() << ")" << endl;
    return false;
  }

  // Go through each strata, computing score function for it.
  for (const pair<int, pair<vector<CensoringData>, MatrixXd>>& strata_itr :
       strata_vars) {
    const int strata_index = strata_itr.first;
    map<int, VectorXd>::const_iterator kme_itr =
        strata_kaplan_meier_estimators.find(strata_index);
    if (kme_itr == strata_kaplan_meier_estimators.end()) {
      cout << "ERROR: No KME info for strata " << strata_index << endl;
      return false;
    }
    map<int, vector<int>>::const_iterator sort_itr =
        strata_reverse_sorted_indices.find(strata_index);
    if (sort_itr == strata_reverse_sorted_indices.end()) {
      cout << "ERROR: No Sort info for strata " << strata_index << endl;
      return false;
    }

    // Pick out the subset of (In)dependent vars among the (In)dependent
    // vars that belong to this strata.
    const vector<CensoringData>& strata_dep_var = strata_itr.second.first;
    const MatrixXd& strata_indep_vars = strata_itr.second.second;
    const VectorXd& strata_kme = kme_itr->second;
    const vector<int>& strata_sort_order = sort_itr->second;

    // Compute Score Function for this strata.
    VectorXd& curr_score_function =
        (strata_to_score_function->insert(
             make_pair(strata_index, VectorXd())).first)->second;
    if (!ComputeScoreFunction(
            n, p, beta,
            strata_kme, strata_indep_vars, strata_dep_var, strata_sort_order,
            &curr_score_function)) {
      return false;
    }
  }
  return true;
}

bool CoxDualRegression::ComputeScoreFunction(
    const int n, const int p,
    const KaplanMeierEstimatorType& kme_type, 
    const VectorXd& beta,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    const map<int, vector<int>>& strata_reverse_sorted_indices,
    map<int, VectorXd>* strata_to_score_function) {
  // Compute KME for each strata.
  map<int, VectorXd> strata_kaplan_meier_estimators;
  if (!ComputeKaplanMeierEstimator(
          kme_type, strata_vars, &strata_kaplan_meier_estimators)) {
    return false;
  }

  return ComputeScoreFunction(
      n, p, beta, strata_kaplan_meier_estimators, strata_vars,
      strata_reverse_sorted_indices, strata_to_score_function);
}

bool CoxDualRegression::ComputeScoreFunction(
    const VectorXd& exp_logistic_eq,
    const VectorXd& partial_sums,
    const vector<int>& transition_indices,
    const VectorXd& kaplan_meier_estimators,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    VectorXd* score_function) {
  // Sanity-Check input.
  if (score_function == nullptr ||
      indep_vars.rows() != dep_var.size() ||
      indep_vars.rows() != exp_logistic_eq.size() ||
      indep_vars.rows() != kaplan_meier_estimators.size()) {
    cout << "\nComputeScoreFunction Failure. indep_vars.size(): "
         << indep_vars.rows() << ", dep_var.size(): "
         << dep_var.size() << ", exp_logistic_eq.size(): "
         << exp_logistic_eq.size() << ", kaplan_meier_estimators.size(): "
         << kaplan_meier_estimators.size() << endl;
    return false;
  }
  if (partial_sums.size() != transition_indices.size()) {
    cout << "\nComputeScoreFunction Failure. partial_sums size: "
         << partial_sums.size() << ", transition indices size: "
         << transition_indices.size() << endl;
    return false;
  }

  score_function->resize(indep_vars.cols());
  score_function->setZero();
  VectorXd numerator;
  numerator.resize(indep_vars.cols());
  numerator.setZero();
  int current_transition_index = indep_vars.rows();
  int transition_itr = transition_indices.size();
  for (int i = indep_vars.rows() - 1; i >= 0; --i) {
    // Check if i is in the previous block (i.e. if i + 1 is a transition index)
    if (i < current_transition_index) {
      transition_itr--;
      current_transition_index = transition_indices[transition_itr];
      int numerator_index = i;
      while (numerator_index >= current_transition_index) {
        numerator +=
            exp_logistic_eq(numerator_index) * indep_vars.row(numerator_index);
        --numerator_index;
      }
    }
    // Only terms with Delta_i = 1 (i.e. I(T <= C) = 1; i.e. patient has died)
    // contribute to Score Function.
    if (dep_var[i].is_alive_) continue;
    *score_function += kaplan_meier_estimators(i) *
        (static_cast<VectorXd>(indep_vars.row(i)) -
         (numerator / partial_sums(transition_itr)));
  }
  return true;
}

bool CoxDualRegression::ComputeInformationMatrix(
    const int n, const int p,
    const bool use_ties_constant,
    const VectorXd& beta,
    const VectorXd& kaplan_meier_estimators,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    const vector<int>& reverse_sorted_indices,
    MatrixXd* info_matrix) {
  // Sanity-check input.
  if (n == 0 || p == 0 || beta.size() != p ||
      indep_vars.rows() != n || indep_vars.cols() != p || dep_var.size() != n) {
    return false;
  }

  // Compute S^0, S^1, and S^2.
  vector<double> S0;
  vector<VectorXd> S1;
  vector<MatrixXd> S2;
  if (reverse_sorted_indices.empty()) {
    ComputePartialSums(n, p, beta, indep_vars, dep_var, &S0, &S1, &S2);
  } else {
    ComputePartialSums(n, p, beta, indep_vars, dep_var, reverse_sorted_indices,
                       &S0, &S1, &S2);
  }

  info_matrix->resize(p, p);
  info_matrix->setZero();
  VectorXd ties_constant;
  if (use_ties_constant) {
    SetTiesConstant(dep_var, &ties_constant);
  }
  for (int i = 0; i < n; ++i) {
    // Only terms with Delta_i = 1 (i.e. I(T <= C) = 1; i.e. patient has died)
    // contribute to Score Function.
    if (dep_var[i].is_alive_) continue;
    if (S0[i] == 0.0) return false;
    const double ties_constant_multiplier = use_ties_constant ?
        ties_constant(i) : 1.0;
    // If S values are large, then we may overflow if we perform the
    // multiplication before the division. For p > 1, this is unavoidable,
    // as we need to compute S1 * S1^T. But for p = 1, we can do division first.
    // We do so here, to minimize chance of overflow.
    if (p == 1) {
      const double first_term = (S2[i] / S0[i])(0, 0);
      const double second_term = (S1[i] / S0[i])(0);
      (*info_matrix)(0, 0) +=
          ties_constant_multiplier * kaplan_meier_estimators(i) *
          (first_term - (second_term * second_term));
    } else {
      *info_matrix +=
          ((S2[i] / S0[i]) - ((S1[i] * S1[i].transpose()) / (S0[i] * S0[i]))) *
          ties_constant_multiplier * kaplan_meier_estimators(i);
    }
  }
  if (MatrixHasNanTerm(*info_matrix)) return false;
  return true;
}

bool CoxDualRegression::ComputeInformationMatrix(
    const int n, const int p,
    const bool use_ties_constant,
    const VectorXd& beta,
    const VectorXd& kaplan_meier_estimators,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    MatrixXd* info_matrix) {
  return ComputeInformationMatrix(
      n, p, use_ties_constant, beta, kaplan_meier_estimators,
      indep_vars, dep_var, vector<int>(), info_matrix);
}

bool CoxDualRegression::ComputeInformationMatrix(
    const int n, const int p,
    const bool use_ties_constant,
    const KaplanMeierEstimatorType& kme_type,
    const VectorXd& beta,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    const vector<int>& reverse_sorted_indices,
    MatrixXd* info_matrix) {
  VectorXd kme;
  if (!ComputeKaplanMeierEstimator(kme_type, dep_var, &kme)) {
    cout << "\nERROR: Failed to ComputeKaplanMeierEstimator." << endl;
    return false;
  }
  return ComputeInformationMatrix(
      n, p, use_ties_constant, beta, kme, indep_vars,
      dep_var, reverse_sorted_indices, info_matrix);
}

bool CoxDualRegression::ComputeInformationMatrix(
    const int n, const int p,
    const bool use_ties_constant,
    const KaplanMeierEstimatorType& kme_type,
    const VectorXd& beta,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    MatrixXd* info_matrix) {
  return ComputeInformationMatrix(
      n, p, use_ties_constant, kme_type, beta,
      indep_vars, dep_var, vector<int>(), info_matrix);

}

bool CoxDualRegression::ComputeInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const VectorXd& beta,
      const map<int, VectorXd>& strata_kaplan_meier_estimators,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      map<int, MatrixXd>* strata_to_info_matrix) {
  // Sanity-check number of strata is consistent between strata_vars and
  // strata_kaplan_meier_estimators.
  if (strata_kaplan_meier_estimators.size() != strata_vars.size()) {
    cout << "ERROR: mismatching strata size for KME ("
         << strata_kaplan_meier_estimators.size()
         << ") and strata vars (" << strata_vars.size() << ")" << endl;
    return false;
  }

  // Go through each strata, computing Information Matrix for it.
  for (const pair<int, pair<vector<CensoringData>, MatrixXd>>& strata_itr :
       strata_vars) {
    const int strata_index = strata_itr.first;
    map<int, VectorXd>::const_iterator kme_itr =
        strata_kaplan_meier_estimators.find(strata_index);
    if (kme_itr == strata_kaplan_meier_estimators.end()) {
      cout << "ERROR: No KME info for strata " << strata_index << endl;
      return false;
    }
    map<int, vector<int>>::const_iterator sort_itr =
        strata_reverse_sorted_indices.find(strata_index);
    if (sort_itr == strata_reverse_sorted_indices.end()) {
      cout << "ERROR: No Sort info for strata " << strata_index << endl;
      return false;
    }

    // Pick out the subset of (In)dependent vars among the (In)dependent
    // vars that belong to this strata.
    const vector<CensoringData>& strata_dep_var = strata_itr.second.first;
    const MatrixXd& strata_indep_vars = strata_itr.second.second;
    const VectorXd& strata_kme = kme_itr->second;
    const vector<int>& strata_sort_order = sort_itr->second;

    // Compute Information Matrix for this strata.
    MatrixXd& curr_info_matrix =
        (strata_to_info_matrix->insert(
             make_pair(strata_index, MatrixXd())).first)->second;
    if (!ComputeInformationMatrix(
            n, p, use_ties_constant, beta, strata_kme,
            strata_indep_vars, strata_dep_var, strata_sort_order,
            &curr_info_matrix)) {
      return false;
    }
  }
  return true;
}

bool CoxDualRegression::ComputeInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const KaplanMeierEstimatorType& kme_type,
      const VectorXd& beta,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      map<int, MatrixXd>* strata_to_info_matrix) {
  // Compute KME for each strata.
  map<int, VectorXd> strata_kaplan_meier_estimators;
  if (!ComputeKaplanMeierEstimator(
          kme_type, strata_vars, &strata_kaplan_meier_estimators)) {
    return false;
  }

  return ComputeInformationMatrix(
      n, p, use_ties_constant, beta, strata_kaplan_meier_estimators,
      strata_vars, strata_reverse_sorted_indices, strata_to_info_matrix);
}

bool CoxDualRegression::ComputeInformationMatrix(
    const VectorXd& exp_logistic_eq,
    const VectorXd& partial_sums,
    const VectorXd& ties_constant,
    const vector<int>& transition_indices,
    const VectorXd& kaplan_meier_estimators,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    MatrixXd* info_matrix) {
  // Sanity-Check input.
  if (info_matrix == nullptr || indep_vars.cols() == 0 ||
      indep_vars.rows() != exp_logistic_eq.size() ||
      indep_vars.rows() != dep_var.size() ||
      indep_vars.rows() != ties_constant.size() ||
      indep_vars.rows() != kaplan_meier_estimators.size()) {
    cout << "\nComputeInformationMatrix Failure. indep_vars.size(): "
         << indep_vars.rows() << ", dep_var.size(): "
         << dep_var.size() << ", ties_constant.size(): "
         << ties_constant.size() << ", exp_logistic_eq.size(): "
         << exp_logistic_eq.size() << ", kaplan_meier_estimators.size(): "
         << kaplan_meier_estimators.size() << endl;
    return false;
  }
  if (partial_sums.size() != transition_indices.size()) {
    cout << "\nComputeInformationMatrix Failure. partial_sums size: "
         << partial_sums.size() << ", transition indices size: "
         << transition_indices.size() << endl;
    return false;
  }

  const int p = indep_vars.cols();
  const int n =  indep_vars.rows();
  info_matrix->resize(p, p);
  info_matrix->setZero();
  MatrixXd first_term_numerator;
  first_term_numerator.resize(p, p);
  first_term_numerator.setZero();
  VectorXd second_term_numerator;
  second_term_numerator.resize(p);
  second_term_numerator.setZero();
  int current_transition_index = indep_vars.rows();
  int transition_itr = transition_indices.size();
  for (int i = indep_vars.rows() - 1; i >= 0; --i) {
    // Check if i is in the previous block (i.e. if i + 1 is a transition index)
    if (i < current_transition_index) {
      transition_itr--;
      current_transition_index = transition_indices[transition_itr];
      int numerator_index = i;
      while (numerator_index >= current_transition_index) {
        const VectorXd& indep_var_value = indep_vars.row(numerator_index);
        first_term_numerator +=
            exp_logistic_eq(numerator_index) *
            (indep_var_value * indep_var_value.transpose());
        second_term_numerator +=
            exp_logistic_eq(numerator_index) * indep_var_value;
        --numerator_index;
      }
    }
    // Only terms with Delta_i = 1 (i.e. I(T <= C) = 1; i.e. patient has died)
    // contribute to Score Function.
    if (dep_var[i].is_alive_) continue;

    *info_matrix +=
        (first_term_numerator / partial_sums(transition_itr) -
         (second_term_numerator * second_term_numerator.transpose() /
          (partial_sums(transition_itr) * partial_sums(transition_itr)))) *
        ties_constant[i] * kaplan_meier_estimators(i);
  }
  if (MatrixHasNanTerm(*info_matrix)) return false;

  // Sanity check matrix is invertible.
  FullPivLU<MatrixXd> lu = info_matrix->fullPivLu();
  if (!lu.isInvertible()) {
    cout << "ERROR: info_matrix is not invertible:\n" << *info_matrix << "\n";
    return false;
  }
 
  return true;
}

bool CoxDualRegression::ComputeScoreFunctionAndInformationMatrix(
    const int n, const int p,
    const bool use_ties_constant,
    const VectorXd& beta,
    const VectorXd& kaplan_meier_estimators_for_score_fn,
    const VectorXd& kaplan_meier_estimators_for_info_matrix,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    const vector<int>& reverse_sorted_indices,
    VectorXd* score_function, MatrixXd* info_matrix) {
  // Sanity-check input.
  if (score_function == nullptr || info_matrix == nullptr ||
      n == 0 || p == 0 || beta.size() != p ||
      indep_vars.rows() != n || indep_vars.cols() != p || dep_var.size() != n) {
    return false;
  }

  // Compute S^0, S^1, and S^2.
  vector<double> S0;
  vector<VectorXd> S1;
  vector<MatrixXd> S2;
  if (reverse_sorted_indices.empty()) {
    ComputePartialSums(n, p, beta, indep_vars, dep_var, &S0, &S1, &S2);
  } else {
    ComputePartialSums(n, p, beta, indep_vars, dep_var, reverse_sorted_indices,
                       &S0, &S1, &S2);
  }

  score_function->resize(indep_vars.cols());
  score_function->setZero();
  info_matrix->resize(p, p);
  info_matrix->setZero();
  VectorXd ties_constant;
  if (use_ties_constant) {
    SetTiesConstant(dep_var, &ties_constant);
  }
  for (int i = 0; i < n; ++i) {
    // Only terms with Delta_i = 1 (i.e. I(T <= C) = 1; i.e. patient has died)
    // contribute to Score Function.
    if (dep_var[i].is_alive_) continue;
    if (S0[i] == 0.0) return false;
    *score_function += kaplan_meier_estimators_for_score_fn(i) *
        (static_cast<VectorXd>(indep_vars.row(i)) - (S1[i] / S0[i]));
    const double ties_constant_multiplier = use_ties_constant ?
        ties_constant(i) : 1.0;
    *info_matrix +=
        ((S2[i] / S0[i]) - ((S1[i] * S1[i].transpose()) / (S0[i] * S0[i]))) *
        ties_constant_multiplier * kaplan_meier_estimators_for_info_matrix(i);
  }
  return true;
}

bool CoxDualRegression::ComputeScoreFunctionAndInformationMatrix(
    const int n, const int p,
    const bool use_ties_constant,
    const VectorXd& beta,
    const VectorXd& kaplan_meier_estimators_for_score_fn,
    const VectorXd& kaplan_meier_estimators_for_info_matrix,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    VectorXd* score_function, MatrixXd* info_matrix) {
  return ComputeScoreFunctionAndInformationMatrix(
      n, p, use_ties_constant, beta, kaplan_meier_estimators_for_score_fn,
      kaplan_meier_estimators_for_info_matrix, indep_vars, dep_var,
      vector<int>(), score_function, info_matrix);
}

bool CoxDualRegression::ComputeScoreFunctionAndInformationMatrix(
    const int n, const int p,
    const bool use_ties_constant,
    const VectorXd& beta,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    const map<int, vector<int>>& strata_reverse_sorted_indices,
    VectorXd* score_function, MatrixXd* info_matrix) {
  // Sanity-check number of strata is consistent between strata_vars and
  // strata_kaplan_meier_estimators.
  if (strata_kaplan_meier_estimators_for_score_fn.size() != strata_vars.size() ||
      strata_kaplan_meier_estimators_for_info_matrix.size() != strata_vars.size()) {
    cout << "ERROR: mismatching strata size for KME for score function ("
         << strata_kaplan_meier_estimators_for_score_fn.size()
         << "), KME for information matrix ("
         << strata_kaplan_meier_estimators_for_info_matrix.size()
         << "), and strata vars (" << strata_vars.size() << ")" << endl;
    return false;
  }

  // Initialize Score Function and Info Matrix to zero.
  score_function->resize(p);
  score_function->setZero();
  info_matrix->resize(p, p);
  info_matrix->setZero();

  // Go through each strata, computing Information Matrix for it.
  for (const pair<int, pair<vector<CensoringData>, MatrixXd>>& strata_itr :
       strata_vars) {
    const int strata_index = strata_itr.first;
    map<int, VectorXd>::const_iterator score_fn_kme_itr =
        strata_kaplan_meier_estimators_for_score_fn.find(strata_index);
    if (score_fn_kme_itr == strata_kaplan_meier_estimators_for_score_fn.end()) {
      cout << "ERROR: No Score Function KME info for strata "
           << strata_index << endl;
      return false;
    }
    map<int, VectorXd>::const_iterator info_matrix_kme_itr =
        strata_kaplan_meier_estimators_for_info_matrix.find(strata_index);
    if (info_matrix_kme_itr == strata_kaplan_meier_estimators_for_info_matrix.end()) {
      cout << "ERROR: No Info Matrix KME info for strata "
           << strata_index << endl;
      return false;
    }
    map<int, vector<int>>::const_iterator sort_itr =
        strata_reverse_sorted_indices.find(strata_index);
    if (sort_itr == strata_reverse_sorted_indices.end()) {
      cout << "ERROR: No Sort info for strata " << strata_index << endl;
      return false;
    }

    // Pick out the subset of (In)dependent vars among the (In)dependent
    // vars that belong to this strata.
    const vector<CensoringData>& strata_dep_var = strata_itr.second.first;
    const MatrixXd& strata_indep_vars = strata_itr.second.second;
    const VectorXd& strata_kaplan_meier_estimators_for_score_fn =
        score_fn_kme_itr->second;
    const VectorXd& strata_kaplan_meier_estimators_for_info_matrix =
        info_matrix_kme_itr->second;
    const vector<int>& strata_sort_order = sort_itr->second;

    // Compute Score Function and Information Matrix for this strata.
    VectorXd strata_score_vector;
    MatrixXd strata_info_matrix;
    if (!ComputeScoreFunctionAndInformationMatrix(
            strata_dep_var.size(), p, use_ties_constant, beta,
            strata_kaplan_meier_estimators_for_score_fn,
            strata_kaplan_meier_estimators_for_info_matrix,
            strata_indep_vars, strata_dep_var, strata_sort_order,
            &strata_score_vector, &strata_info_matrix)) {
      return false;
    }
    *score_function += strata_score_vector;
    *info_matrix += strata_info_matrix;
  }
  return true;
}

bool CoxDualRegression::EvaluateScoreMethodFunction(
    const int n, const int p, const double& x,
    const VectorXd& score_function, const MatrixXd& info_matrix,
    double* value) {
  // Sanity check dimensions.
  if (score_function.size() != info_matrix.rows() ||
      info_matrix.rows() != info_matrix.cols()) {
    return false;
  }

  // NOTE: The below code works for p > 1, but the formula isn't quite right,
  // as Danyu said in email "The formula is correct, but we need to use the 
  // percentile of the chi-squared distribution with p degrees of freedom,
  // and there is no easy way to get the confidence limits in the p > 1 case."
  // So for now, we abort if p > 1, to emphasize it isn't correctly implemented.
  if (p > 1) return false;

  // Sanity check matrix is invertible.
  FullPivLU<MatrixXd> lu = info_matrix.fullPivLu();
  if (!lu.isInvertible()) return false;

  const MatrixXd score_method_evaluation =
      (score_function.transpose() * info_matrix.inverse() * score_function);
  *value = score_method_evaluation(0, 0);
  return true;
}

bool CoxDualRegression::EvaluateScoreMethodFunction(
    const int n, const int p,
    const bool use_ties_constant,
    const double& x,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    const map<int, vector<int>>& strata_reverse_sorted_indices,
    double* value) {
  if (value == nullptr) return false;

  // Get U(x) and I(x)
  VectorXd score_function;
  MatrixXd info_matrix;
  VectorXd x_vec;
  x_vec.resize(1);
  x_vec(0) = x;
  if (!ComputeScoreFunctionAndInformationMatrix(
          n, p, use_ties_constant, x_vec,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
         &score_function, &info_matrix)) {
    cout << "ERROR: Unable to compute Score Function and Info Matrix for "
         << "getting CI width of Score Method for x: "
         << x << endl;
    return false;
  }

  return EvaluateScoreMethodFunction(
      n, p, x, score_function, info_matrix, value);
}

bool CoxDualRegression::FindScoreMethodCiRoot(
    const int num_iterations,
    const int n, const int p,
    const bool use_ties_constant,
    const double& prev_neg, const double& prev_pos,
    const double& beta, const double& guess,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    const map<int, vector<int>>& strata_reverse_sorted_indices,
    double* root) {
  if (root == nullptr) return false;

  // Early-bailout, if prev_pos and prev_neg are sufficiently close, no
  // need to get more accuracy.
  if (prev_pos != DBL_MAX && prev_pos != DBL_MIN &&
      prev_neg != DBL_MAX && prev_neg != DBL_MIN &&
      (num_iterations > (MAX_SCORE_METHOD_WIDTH_ITERATIONS / 2)) &&
      AbsoluteConvergenceSafeTwo(
          prev_neg, prev_pos, DELTA, 0.001)) {
    *root = (prev_neg + prev_pos) / 2.0;
    return true;
  }

  if (num_iterations > MAX_SCORE_METHOD_WIDTH_ITERATIONS) {
    cout << "ERROR: Search for roots of Score Method's "
         << "Confidence Interval width failed to converge "
         << "after " << MAX_SCORE_METHOD_WIDTH_ITERATIONS
         << " iterations." << endl;
    return false;
  }

  // Evaluate U^2(guess)/I(guess).
  double curr_value;
  if (!EvaluateScoreMethodFunction(
          n, p, use_ties_constant, guess,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
          &curr_value)) {
    // Information Matrix evaluated to zero. Bring guess closer to beta, and
    // try again.
    cout << "WARNING: In attempting to FindScoreMethodCiRoot (iteration = "
         << num_iterations << "), got a zero-valued Information matrix at "
         << "guess: " << guess << " (Note: beta = " << beta << ")" << endl;
    if (FloatEq(guess, beta)) return false;
    cout << "Attempting to FindScoreMethodCiRoot at new guess: "
         << ((guess + beta) / 2.0) << endl;
    return FindScoreMethodCiRoot(
        num_iterations + 1, n, p, use_ties_constant, prev_neg, prev_pos, beta,
        ((guess + beta) / 2.0), strata_kaplan_meier_estimators_for_score_fn,
        strata_kaplan_meier_estimators_for_info_matrix, strata_vars,
        strata_reverse_sorted_indices, root);
  }

  // Check if we are sufficiently close to a root.
  if (FloatEq(curr_value, kScoreMethodConfidenceIntervalBound)) {
    *root = guess;
    return true;
  }

  if (curr_value < kScoreMethodConfidenceIntervalBound) {
    // Need to move further away from \hat{\beta}.
    if (prev_pos == DBL_MAX || prev_pos == DBL_MIN) {
      // Haven't found a value above 3.8415 yet; move twice as far from \hat{\beta}.
      const double next_guess = FloatEq(guess, beta) ?
          2.0 * (prev_pos == DBL_MAX ? (beta + EPSILON) : (beta - EPSILON)) :
          2.0 * guess - beta;
      return FindScoreMethodCiRoot(
          num_iterations + 1, n, p, use_ties_constant, guess, prev_pos, beta,
          next_guess, strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices, root);
    // NOTE: Originally I had the following condition in place, to ensure that
    // next guess (guess + prev_pos) / 2 would be different from current guess;
    // but there are times that these values are within EPSILON of each other,
    // but that f(guess) < kScoreMethodConfidenceIntervalBound and
    // f(prev_pos) > kScoreMethodConfidenceIntervalBound. Instead of aborting,
    // allow the iterations to continue, realizing that it is possible that
    // next_guess = guess, and just rely on MAX_SCORE_METHOD_WIDTH_ITERATIONS
    // to control infinite iterations in this case.
    /*
    } else if (FloatEq(guess, prev_pos)) {
      if (FloatEq(
              curr_value, kScoreMethodConfidenceIntervalBound, DELTA, 0.001)) {
        cout << "WARNING: Taking Score Method confidence interval endpoint as: "
             << guess << " which evaluates to: " << curr_value
             << " even though it is only within 10^-3 of "
             << kScoreMethodConfidenceIntervalBound
             << " (doing this because the Midpoint Method will fail with more "
             << "iterations, because the interval is already too small)"
             << endl;
        *root = guess;
        return true;
      } else {
        cout << "ERROR: Current guess " << guess << " evaluated to "
             << curr_value << " which is less than "
             << kScoreMethodConfidenceIntervalBound
             << ", and yet it equals prev_pos (" << prev_pos
             << "), which should have evaluated to something more than "
             << kScoreMethodConfidenceIntervalBound << endl;
        return false;
      }
      */
    } else {
      return FindScoreMethodCiRoot(
          num_iterations + 1, n, p, use_ties_constant, guess, prev_pos, beta,
          (guess + prev_pos) / 2.0,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices, root);
    }
  } else if (FloatEq(beta, guess)) {
    // Abort: U^2(\hat{beta})/I(\hat{beta}) is above 3.8415, even though
    // U^2(\hat{beta}/I(\hat{beta} should equal 0 (by definition of \hat{beta}).
    cout << "ERROR: U^2(hat{beta})/I(hat{beta}) > "
         << kScoreMethodConfidenceIntervalBound
         << ", which should be impossible (should evaluate to zero)."
         << endl;
    return false;
  } else {
    // Need to move closer to \hat{\beta}.
    const double next_guess = (prev_neg == DBL_MAX || prev_neg == DBL_MIN) ?
        (guess + beta) / 2.0 : (guess + prev_neg) / 2.0;
    return FindScoreMethodCiRoot(
        num_iterations + 1, n, p, use_ties_constant, prev_neg, guess, beta,
        next_guess, strata_kaplan_meier_estimators_for_score_fn,
        strata_kaplan_meier_estimators_for_info_matrix,
        strata_vars, strata_reverse_sorted_indices, root);
  }

  // Shouldn't reach here.
  return false;
}

bool CoxDualRegression::FindScoreMethodCiNegValue(
    const int n, const int p,
    const bool use_ties_constant,
    const double& beta,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    const map<int, vector<int>>& strata_reverse_sorted_indices,
    double* close_to_min) {
  if (close_to_min == nullptr) return false;

  // Compute U^2(beta)/I(beta).
  double f_beta;
  if (!EvaluateScoreMethodFunction(
          n, p, use_ties_constant, beta,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
          &f_beta)) {
    cout << "\nERROR: In attempting to FindScoreMethodCiNegValue, failed to "
         << "EvaluateScoreMethodFunction at beta: " << beta << endl;
    return false;
  }

  // For most simulation iterations, the following should hold.
  if (f_beta < kScoreMethodConfidenceIntervalBound) {
    *close_to_min = beta;
    return true;
  }

  // Do Step (2) (see notes in header for this method).
  double L1 = DBL_MIN;
  double R1 = DBL_MAX;
  if (!DoScoreMethodStepTwo(
          0, n, p, use_ties_constant, beta, f_beta, beta - 0.5, beta + 0.5,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
          &L1, &R1)) {
    return false;
  }
  // Check Step 2's early abort (in case it found close_to_min).
  if (L1 == R1) {
    *close_to_min = L1;
    return true;
  }

  // Do Step (3) (see notes in header for this method).
  return DoScoreMethodStepThree(
      0, n, p, use_ties_constant, L1, R1,
      (L1 + R1) / 2.0, (3.0 * L1 + R1) / 4.0, (3.0 * R1 + L1) / 4.0,
      strata_kaplan_meier_estimators_for_score_fn,
      strata_kaplan_meier_estimators_for_info_matrix,
      strata_vars, strata_reverse_sorted_indices,
      close_to_min);
}

bool CoxDualRegression::DoScoreMethodStepTwo(
    const int num_iterations,
    const int n, const int p,
    const bool use_ties_constant,
    const double& beta, const double& f_beta,
    const double& guess_left, const double& guess_right,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    const map<int, vector<int>>& strata_reverse_sorted_indices,
    double* L1, double* R1) {
  if (L1 == nullptr || R1 == nullptr) return false;

  if (num_iterations > MAX_SCORE_METHOD_WIDTH_ITERATIONS) {
    cout << "ERROR: Search for roots of Score Method's "
         << "Confidence Interval width via Step 2 failed to converge "
         << "after " << MAX_SCORE_METHOD_WIDTH_ITERATIONS
         << " iterations." << endl;
    return false;
  }

  // Compute U^2(guess_left)/I(guess_left).
  if (*L1 == DBL_MIN) {
    // L1 not yet found. Try guess_left.
    double f_left;
    if (!EvaluateScoreMethodFunction(
            n, p, use_ties_constant, guess_left,
            strata_kaplan_meier_estimators_for_score_fn,
            strata_kaplan_meier_estimators_for_info_matrix,
            strata_vars, strata_reverse_sorted_indices,
            &f_left)) {
      cout << "\nWARNING: Failed to EvaluateScoreMethodFunction in "
           << "DoScoreMethodStepTwo for guess_left: " << guess_left << endl;
      return DoScoreMethodStepTwo(
          num_iterations + 1, n, p, use_ties_constant, beta, f_beta,
          (beta + guess_left) / 2.0, guess_right,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
          L1, R1);
    }
    if (f_left < kScoreMethodConfidenceIntervalBound) {
      // Early abort. The whole goal is to find such a value.
      // Artificially set L1 and R1 to guess_left, which
      // will indicate to calling function that close_to_min was found.
      *L1 = guess_left;
      *R1 = guess_left;
      return true;
    }
    if (f_left > f_beta) {
      *L1 = f_left;
    }
  }

  // Compute U^2(guess_right)/I(guess_right).
  if (*R1 == DBL_MAX) {
    // R1 not yet found. Try guess_right.
    double f_right;
    if (!EvaluateScoreMethodFunction(
            n, p, use_ties_constant, guess_right,
            strata_kaplan_meier_estimators_for_score_fn,
            strata_kaplan_meier_estimators_for_info_matrix,
            strata_vars, strata_reverse_sorted_indices,
            &f_right)) {
      cout << "\nWARNING: Failed to EvaluateScoreMethodFunction in "
           << "DoScoreMethodStepTwo for guess_right: " << guess_right << endl;
      return DoScoreMethodStepTwo(
          num_iterations + 1, n, p, use_ties_constant, beta, f_beta,
          guess_left, (beta + guess_right) / 2.0,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
          L1, R1);
    }
    if (f_right < kScoreMethodConfidenceIntervalBound) {
      // Early abort. The whole goal is to find such a value.
      // Artificially set L1 and R1 to guess_left, which
      // will indicate to calling function that close_to_min was found.
      *L1 = guess_right;
      *R1 = guess_right;
      return true;
    }
    if (f_right > f_beta) {
      *R1 = f_right;
    }
  }

  if (*L1 != DBL_MIN && *R1 != DBL_MAX) return true;
  return DoScoreMethodStepTwo(
      num_iterations + 1, n, p, use_ties_constant, beta, f_beta,
      2.0 * guess_left - beta, 2.0 * guess_right + beta,
      strata_kaplan_meier_estimators_for_score_fn,
      strata_kaplan_meier_estimators_for_info_matrix,
      strata_vars, strata_reverse_sorted_indices,
      L1, R1);
}

bool CoxDualRegression::DoScoreMethodStepThree(
    const int num_iterations,
    const int n, const int p,
    const bool use_ties_constant,
    const double& left, const double& right, const double& midpoint,
    const double& guess_left, const double& guess_right,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    const map<int, vector<int>>& strata_reverse_sorted_indices,
    double* close_to_min) {
  if (close_to_min == nullptr) return false;

  if (num_iterations > MAX_SCORE_METHOD_WIDTH_ITERATIONS) {
    cout << "ERROR: Search for roots of Score Method's "
         << "Confidence Interval width via Step 3 failed to converge "
         << "after " << MAX_SCORE_METHOD_WIDTH_ITERATIONS
         << " iterations." << endl;
    return false;
  }

  // Compute U^2(midpoint)/I(midpoint).
  double f_midpoint;
  if (!EvaluateScoreMethodFunction(
          n, p, use_ties_constant, midpoint,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
          &f_midpoint)) {
    cout << "\nERROR: Failed to EvaluateScoreMethodFunction at midpoint: "
         << midpoint << " in DoScoreMethodStepThree." << endl;
    return false;
  }
  if (f_midpoint < kScoreMethodConfidenceIntervalBound) {
    // Early abort if midpoint is sufficient.
    *close_to_min = midpoint;
    return true;
  }

  // Compute U^2(guess_left)/I(guess_left).
  double f_left;
  if (!EvaluateScoreMethodFunction(
          n, p, use_ties_constant, guess_left,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
          &f_left)) {
    cout << "\nERROR: Failed to EvaluateScoreMethodFunction at guess_left: "
         << guess_left << " in DoScoreMethodStepThree." << endl;
    return false;
  }
  if (f_left < kScoreMethodConfidenceIntervalBound) {
    // Early abort if midpoint is sufficient.
    *close_to_min = guess_left;
    return true;
  }

  // Compute U^2(guess_right)/I(guess_right).
  double f_right;
  if (!EvaluateScoreMethodFunction(
          n, p, use_ties_constant, guess_right,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
          &f_right)) {
    cout << "\nERROR: Failed to EvaluateScoreMethodFunction at guess_right: "
         << guess_right << " in DoScoreMethodStepThree." << endl;
    return false;
  }
  if (f_right < kScoreMethodConfidenceIntervalBound) {
    // Early abort if midpoint is sufficient.
    *close_to_min = guess_right;
    return true;
  }

  if (f_midpoint < f_right && f_midpoint < f_left) {
    // This is the hard case.
    return DoScoreMethodStepThree(
        num_iterations + 1, n, p, use_ties_constant, guess_left, guess_right,
        midpoint, (midpoint + guess_left) / 2.0, (guess_right + midpoint) / 2.0,
        strata_kaplan_meier_estimators_for_score_fn,
        strata_kaplan_meier_estimators_for_info_matrix,
        strata_vars, strata_reverse_sorted_indices,
        close_to_min);
  } else if (f_midpoint > f_right) {
    return DoScoreMethodStepThree(
        num_iterations + 1, n, p, use_ties_constant, midpoint, guess_right,
        (midpoint + guess_right) / 2.0, (3.0 * midpoint + guess_right) / 4.0,
        (3.0 * guess_right + midpoint) / 4.0,
        strata_kaplan_meier_estimators_for_score_fn,
        strata_kaplan_meier_estimators_for_info_matrix,
        strata_vars, strata_reverse_sorted_indices,
        close_to_min);
  } else {  // f_midpoint > f_left
    return DoScoreMethodStepThree(
        num_iterations + 1, n, p, use_ties_constant, guess_left, midpoint,
        (midpoint + guess_left) / 2.0, (3.0 * guess_left + midpoint) / 4.0,
        (3.0 * midpoint + guess_left) / 4.0,
        strata_kaplan_meier_estimators_for_score_fn,
        strata_kaplan_meier_estimators_for_info_matrix,
        strata_vars, strata_reverse_sorted_indices,
        close_to_min);
  }

  // Shouldn't reach here.
  return false;
}

bool CoxDualRegression::ComputeScoreMethodCi(
    const int n, const int p,
    const bool use_ties_constant,
    const double& hat_beta, const double& hat_std_err,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
    const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
    const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
    const map<int, vector<int>>& strata_reverse_sorted_indices,
    double* score_method_ci_left, double* score_method_ci_right) {
  if (score_method_ci_left == nullptr || score_method_ci_right == nullptr) {
    return false;
  }

  // Let f(x) := U^2(x)/I(x). Then find a value x for which f(x) < 3.8415
  // (hat_beta should always work; see comments in header file).
  double close_to_min;
  if (!FindScoreMethodCiNegValue(
          n, p, use_ties_constant, hat_beta,
          strata_kaplan_meier_estimators_for_score_fn,
          strata_kaplan_meier_estimators_for_info_matrix,
          strata_vars, strata_reverse_sorted_indices,
          &close_to_min)) {
    return false;
  }

  // Look for the "upper" root (the right-end of the confidence interval).
  // This value should be close to (\hat{\beta} + 1.96 * \hat{SE}), so we
  // start with that guess (plus some fudge).
  *score_method_ci_right = DBL_MAX;
  if (!FindScoreMethodCiRoot(
      0, n, p, use_ties_constant, close_to_min, DBL_MAX, close_to_min,
      close_to_min + Z_SCORE_FOR_TWO_TAILED_FIVE_PERCENT * hat_std_err + EPSILON,
      strata_kaplan_meier_estimators_for_score_fn,
      strata_kaplan_meier_estimators_for_info_matrix,
      strata_vars, strata_reverse_sorted_indices,
      score_method_ci_right)) {
    cout << "ERROR: Unable to find upper root for Score Method's "
         << "Confidence Interval." << endl;
    return false;
  }

  // Look for the "lower" root (the left-end of the confidence interval).
  // NOTE: Like above, we could start our search at:
  //   (\hat{\beta} - 1.96 * \hat{SE})
  // However, since we already have the "upper_root", and U^2/I should be
  // symmetric about beta_hat, a better guess is:
  //   \hat{\beta} - (upper_root - \hat{\beta}) = 2 * \hat{\beta} - upper_root
  *score_method_ci_left = DBL_MIN;
  if (!FindScoreMethodCiRoot(
      0, n, p, use_ties_constant, close_to_min, DBL_MIN, close_to_min,
      2 * close_to_min - *score_method_ci_right - EPSILON,
      strata_kaplan_meier_estimators_for_score_fn,
      strata_kaplan_meier_estimators_for_info_matrix,
      strata_vars, strata_reverse_sorted_indices,
      score_method_ci_left)) {
    cout << "ERROR: Unable to find lower root for Score Method's "
         << "Confidence Interval." << endl;
    return false;
  }

  return true;
}

bool CoxDualRegression::ComputeSTerms(
    const vector<int>& transition_indices,
    const VectorXd& regression_coefficients,
    const MatrixXd& indep_vars, const vector<CensoringData>& dep_vars,
    VectorXd* s_zero, MatrixXd* s_one) {
  const int strata_size = dep_vars.size();
  s_zero->resize(strata_size);
  s_zero->setZero();
  s_one->resize(strata_size, indep_vars.cols());
  s_one->setZero();

  double running_s_zero = 0.0;
  VectorXd running_s_one;
  running_s_one.resize(indep_vars.cols());
  running_s_one.setZero();
  int current_transition_index = strata_size;
  int transition_itr = transition_indices.size();
  for (int i = strata_size - 1; i >= 0; --i) {
    // Check if i is in the previous block (i.e. if i + 1 is a transition index)
    if (i < current_transition_index) {
      transition_itr--;
      current_transition_index = transition_indices[transition_itr];
      int numerator_index = i;
      while (numerator_index >= current_transition_index) {
        const double exp_logistic_eq =
            exp(regression_coefficients.transpose() *
                indep_vars.row(numerator_index).transpose());
        running_s_zero += exp_logistic_eq;
        running_s_one += exp_logistic_eq * indep_vars.row(numerator_index);
        --numerator_index;
      }
    }
    (*s_zero)(i) = running_s_zero;
    (*s_one).row(i) = running_s_one;
  }
  return true;
}

VectorXd CoxDualRegression::WVectorFirstTerm(
    const double& kaplan_meier_multiplier,
    const VectorXd& row, const CensoringData& dep_var,
    const double& s_zero, const VectorXd& s_one) {
  // Only terms with Delta_i = 1 (i.e. I(T <= C) = 1; i.e. patient has died)
  // have non-zero FirstTerm.
  if (dep_var.is_alive_) {
    VectorXd first_term;
    first_term.resize(row.size());
    first_term.setZero();
    return first_term;
  }
  return kaplan_meier_multiplier * (row - (s_one / s_zero));
}

VectorXd CoxDualRegression::WVectorSecondTerm(
    const double& exp_logistic_eq, const int stop_index,
    const vector<CensoringData>& dep_vars,
    const VectorXd& kaplan_meier_estimators,
    const VectorXd& row, const VectorXd& s_zero, const MatrixXd& s_one) {
  VectorXd second_term;
  second_term.resize(row.size());
  second_term.setZero();
  for (int i = 0; i < stop_index; ++i) {
    // Only terms with Delta_i = 1 (i.e. I(T <= C) = 1; i.e. patient has died)
    // have non-zero contribution to SecondTerm.
    if (dep_vars[i].is_alive_) continue;
    second_term += kaplan_meier_estimators(i) *
        (row / s_zero(i) - (s_one.row(i).transpose() / (s_zero(i) * s_zero(i))));
  }
  return exp_logistic_eq * second_term;
}

bool CoxDualRegression::ComputeWVectors(
    const vector<int>& transition_indices,
    const VectorXd& kaplan_meier_estimators,
    const VectorXd& regression_coefficients,
    const MatrixXd& indep_vars, const vector<CensoringData>& dep_vars,
    MatrixXd* w_vectors) {
  const int n = dep_vars.size();

  // Sanity-check input.
  if (n != kaplan_meier_estimators.size() ||
      n != indep_vars.rows()) {
    cout << "ERROR: Mismatching sizes: dep_vars.size(): "
         << dep_vars.size() << ", indep_vars.rows(): "
         << indep_vars.rows() << ", kaplan_meier_estimators.size(): "
         << kaplan_meier_estimators.size() << endl;
    return false;
  }

  VectorXd s_zero;
  MatrixXd s_one;
  if (!ComputeSTerms(
          transition_indices, regression_coefficients, indep_vars, dep_vars,
          &s_zero, &s_one)) {
    return false;
  }

  for (int i = 0; i < n; ++i) {
    const CensoringData& data = dep_vars[i];
    const double& time = data.is_alive_ ?
        data.censoring_time_ : data.survival_time_;
    const double exp_logistic_eq =
        exp(regression_coefficients.transpose() * indep_vars.row(i).transpose());
    // Only indices l with T_l <= T_i will be needed in the SecondTerm.
    int stop_index = -1;
    for (int j = 0; j < transition_indices.size(); ++j) {
      const int row_index = transition_indices[j];
      const CensoringData& temp_data = dep_vars[row_index];
      const double& temp_time = temp_data.is_alive_ ?
          temp_data.censoring_time_ : temp_data.survival_time_;
      if (temp_time == time) {
        stop_index = (j == transition_indices.size() - 1) ?
            n : transition_indices[j + 1];
        break;
      }
    }
    if (stop_index == -1) {
      cout << "ERROR in ComputeWVectors: Unable to find stop index for i: "
           << i << ". Aborting." << endl;
      return false;
    }
    w_vectors->row(i) =
        WVectorFirstTerm(
            kaplan_meier_estimators(i), indep_vars.row(i), data,
            s_zero(i), s_one.row(i).transpose()) -
        WVectorSecondTerm(
            exp_logistic_eq, stop_index,
            dep_vars, kaplan_meier_estimators, indep_vars.row(i),
            s_zero, s_one);
  }

  return true;
}

bool CoxDualRegression::ComputeWVectors(
    const int p,
    const VectorXd& regression_coefficients,
    vector<StratificationData>* stratification_data) {
  // Compute the strata W-vectors.
  for (int strata_index = 0; strata_index < stratification_data->size();
       ++strata_index) {
    StratificationData& data = (*stratification_data)[strata_index];
    data.w_vectors_.resize(data.dep_vars_.size(), p);
    data.w_vectors_.setZero();
    if (!ComputeWVectors(
          data.transition_indices_, data.kaplan_meier_estimators_,
          regression_coefficients, data.indep_vars_, data.dep_vars_,
          &data.w_vectors_)) {
      cout << "ERROR in ComputeWVectors, strata index: " << strata_index
           << ". Aborting." << endl;
      return false;
    }
  }
  return true;
}

bool CoxDualRegression::ComputeWVectorsAndRobustVariance(
    const int p,
    const VectorXd& regression_coefficients,
    const MatrixXd& info_matrix_inverse,
    const map<int, pair<int, int>>& unsorted_model_row_to_strata_index_and_row,
    vector<StratificationData>* stratification_data,
    MatrixXd* robust_var, SummaryStatistics* stats) {
  // Compute W-vectors.
  if (!ComputeWVectors(p, regression_coefficients, stratification_data)) {
    return false;
  }

  // Copy W-vectors to stats.
  if (stats != nullptr) {
    CopyWVectors(false, p, unsorted_model_row_to_strata_index_and_row,
                 *stratification_data, &(stats->w_vectors_));
  }

  // Sanity check W-vectors by comparing against Robust Variance.
  ComputeRobustVariance(info_matrix_inverse, *stratification_data, robust_var);
  return true;
}

bool CoxDualRegression::ComputeLogRankWVectors(
    const map<int, pair<int, int>>& unsorted_model_row_to_strata_index_and_row,
    vector<StratificationData>* stratification_data,
    SummaryStatistics* stats) {
  // Log Rank W-Vectors are computed the same way as ordinary W-Vectors are,
  // but for regression coefficient = 0.0.
  VectorXd regression_coefficients;
  regression_coefficients.resize(1);
  regression_coefficients.setZero();

  // Compute the strata Log Rank W-vectors.
  for (int strata_index = 0; strata_index < stratification_data->size();
       ++strata_index) {
    StratificationData& data = (*stratification_data)[strata_index];
    data.log_rank_w_vectors_.resize(data.dep_vars_.size(), 1);
    data.log_rank_w_vectors_.setZero();
    if (!ComputeWVectors(
          data.transition_indices_, data.kaplan_meier_estimators_,
          regression_coefficients,
          data.indep_vars_, data.dep_vars_, &data.log_rank_w_vectors_)) {
      cout << "ERROR in ComputeWVectors, strata index: " << strata_index
           << ". Aborting." << endl;
      return false;
    }
  }

  // Copy W-vectors (in strata-independent manner) to Summary Statistics.
  CopyWVectors(true, 1, unsorted_model_row_to_strata_index_and_row,
               *stratification_data, &(stats->log_rank_w_vectors_));
  return true;
}

void CoxDualRegression::CopyWVectors(
      const bool is_log_rank, const int p,
      const map<int, pair<int, int>>& unsorted_model_row_to_strata_index_and_row,
      const vector<StratificationData>& stratification_data,
      MatrixXd* w_vectors) {
  w_vectors->resize(unsorted_model_row_to_strata_index_and_row.size(), p);
  w_vectors->setZero();
  for (map<int, pair<int, int>>::const_iterator itr =
           unsorted_model_row_to_strata_index_and_row.begin();
       itr != unsorted_model_row_to_strata_index_and_row.end(); ++ itr) {
    const int row = itr->first;
    const pair<int, int>& strata_index_and_row = itr->second;
    const int strata_index = strata_index_and_row.first;
    const int strata_row = strata_index_and_row.second;
    w_vectors->row(row) = is_log_rank ?
        stratification_data[strata_index].log_rank_w_vectors_.row(strata_row) :
        stratification_data[strata_index].w_vectors_.row(strata_row);
  }
}

void CoxDualRegression::ComputeRobustVariance(
    const MatrixXd& info_matrix_inverse,
    const vector<StratificationData>& stratification_data,
    MatrixXd* robust_var) {
  const int p = stratification_data[0].w_vectors_.cols();
  MatrixXd w_vectors_sum;
  w_vectors_sum.resize(p, p);
  w_vectors_sum.setZero();
  for (int i = 0; i < stratification_data.size(); ++i) {
    const MatrixXd& w_vectors = stratification_data[i].w_vectors_;
    w_vectors_sum += w_vectors.transpose() * w_vectors;
  }
  *robust_var = info_matrix_inverse * w_vectors_sum * info_matrix_inverse;
}

bool CoxDualRegression::ComputeDualCovariance(
    const map<int, int>& orig_row_to_model_one_row,
    const map<int, int>& orig_row_to_model_two_row,
    const AnalysisParams& analysis_params,
    const SummaryStatistics& stats_one, const SummaryStatistics& stats_two,
    set<int>* common_rows, DualStatistics* dual_stats) {
  if (common_rows == nullptr || dual_stats == nullptr) return false;

  // Get rows (Subjects) commone to both models.
  GetCommonRows(
      orig_row_to_model_one_row, orig_row_to_model_two_row, common_rows);

  // Compute Dual Covariance Matrix (if Robust Analysis was requested).
  if (analysis_params.standard_analysis_ &&
      analysis_params.robust_analysis_ &&
      !ComputeDualCovarianceMatrix(
          *common_rows,
          orig_row_to_model_one_row, orig_row_to_model_two_row,
          stats_one.w_vectors_, stats_two.w_vectors_,
          stats_one.final_info_matrix_inverse_,
          stats_two.final_info_matrix_inverse_,
          dual_stats)) {
    cout << "\nERROR in ComputeDualCovarianceMatrix. Aborting." << endl;
    return false;
  }

  // Compute Dual Log-Rank Covariance (if Log-Rank or Peto analysis was requested).
  if (analysis_params.log_rank_analysis_ || analysis_params.peto_analysis_) {
    if (!ComputeDualLogRank(
            *common_rows,
            stats_one.log_rank_variance_, stats_two.log_rank_variance_,
            orig_row_to_model_one_row, orig_row_to_model_two_row,
            stats_one.log_rank_w_vectors_, stats_two.log_rank_w_vectors_,
            dual_stats)) {
      return false;
    }
  }

  return true;
}

bool CoxDualRegression::ComputeDualLogRank(
    const set<int>& common_rows,
    const double& log_rank_variance_one,
    const double& log_rank_variance_two,
    const map<int, int>& orig_row_to_unsorted_model_row_one,
    const map<int, int>& orig_row_to_unsorted_model_row_two,
    const MatrixXd& w_vectors_one,
    const MatrixXd& w_vectors_two,
    DualStatistics* dual_stats) {
  double& log_rank_b_matrix = dual_stats->log_rank_b_matrix_;
  double& log_rank_dual_covariance = dual_stats->log_rank_dual_correlation_;
  MatrixXd w_vectors_sum;
  w_vectors_sum.resize(1, 1);
  w_vectors_sum.setZero();
  
  // Compute Covariance: B(0, 0).
  for (const int orig_row : common_rows) {
    if (orig_row_to_unsorted_model_row_one.find(orig_row) ==
        orig_row_to_unsorted_model_row_one.end() ||
        orig_row_to_unsorted_model_row_two.find(orig_row) ==
        orig_row_to_unsorted_model_row_two.end()) {
      cout << "ERROR: Unable to find W-Vector for row: "
           << (1 + orig_row) << endl;
      return false;
    }
    const VectorXd& w_vector_one = w_vectors_one.row(
        orig_row_to_unsorted_model_row_one.find(orig_row)->second);
    const VectorXd& w_vector_two = w_vectors_two.row(
        orig_row_to_unsorted_model_row_two.find(orig_row)->second);
    w_vectors_sum += w_vector_one * w_vector_two.transpose();
  }
  log_rank_b_matrix = w_vectors_sum(0, 0);

  // Sanity-check log_rank_variance values.
  if (log_rank_variance_one <= 0.0 || log_rank_variance_two <= 0.0) {
    cout << "ERROR in ComputeDualLogRank: negative variance. "
         << "log_rank_variance_one: " << log_rank_variance_one
         << ", log_rank_variance_two: " << log_rank_variance_two << endl;
    return false;
  }

  // Compute Correlation 'Matrix'.
  log_rank_dual_covariance =
      log_rank_b_matrix / sqrt(log_rank_variance_one * log_rank_variance_two);
  return true;
}

bool CoxDualRegression::PrintDualLogRank(
    const bool is_weighted, const DualStatistics& dual_stats,
    string* output) {
  if (output == nullptr) return false;
  // Print Covariance.
  const string prefix = is_weighted ? "Weighted" : "Unweighted";
  *output += "Covariance for " + prefix + " Log-Rank Test:\t" +
             Itoa(dual_stats.log_rank_b_matrix_, 6) + "\n";

  // Print Correlation Matrix.
  *output += "Correlation for " + prefix + "Log-Rank Test:\t" +
             Itoa(dual_stats.log_rank_dual_correlation_, 6) + "\n";

  return true;
}

bool CoxDualRegression::ComputeBMatrix(
    const set<int>& common_rows,
    const map<int, int>& orig_row_to_unsorted_model_row_one,
    const map<int, int>& orig_row_to_unsorted_model_row_two,
    const MatrixXd& w_vectors_one,
    const MatrixXd& w_vectors_two,
    MatrixXd* b_matrix) {
  for (const int orig_row : common_rows) {
    if (orig_row_to_unsorted_model_row_one.find(orig_row) ==
        orig_row_to_unsorted_model_row_one.end() ||
        orig_row_to_unsorted_model_row_two.find(orig_row) ==
        orig_row_to_unsorted_model_row_two.end()) {
      cout << "ERROR: Unable to find W-Vector for row: "
           << (1 + orig_row) << endl;
      return false;
    }
    const VectorXd& w_vector_one = w_vectors_one.row(
        orig_row_to_unsorted_model_row_one.find(orig_row)->second);
    const VectorXd& w_vector_two = w_vectors_two.row(
        orig_row_to_unsorted_model_row_two.find(orig_row)->second);
    *b_matrix += w_vector_one * w_vector_two.transpose();
  }
  return true;
}

bool CoxDualRegression::ComputeLogRank(
    const KaplanMeierEstimatorType kme_type_for_log_rank,
    vector<StratificationData>* stratification_data,
    double* log_rank_estimate, double* log_rank_variance) {
  *log_rank_estimate = 0.0;
  *log_rank_variance = 0.0;

  for (StratificationData& data : *stratification_data) {
    // Log-Rank is computed by plugging in Beta = 0.
    VectorXd exp_logistic_eq;
    exp_logistic_eq.resize(data.dep_vars_.size());
    for (int i = 0; i < data.dep_vars_.size(); ++i) {
      exp_logistic_eq(i) = 1.0;
    }

    // Compute partial sums S_i(0):
    //   := \sum_j I(T_j >= T_i)
    VectorXd partial_sums;
    if (!ComputePartialSums(
            data.transition_indices_, exp_logistic_eq, &partial_sums)) {
      cout << "ERROR: Unable to compute partial sums for Log rank. Aborting."
           << endl;
      return false;
    }
    
    // Compute K-M estimators.
    if (!ComputeKaplanMeierEstimator(
            kme_type_for_log_rank, partial_sums, data.transition_indices_,
            data.dep_vars_, &data.kaplan_meier_estimators_)) {
      return false;
    }

    // Compute Score function U(0).
    VectorXd score_function;
    if (!ComputeScoreFunction(
            exp_logistic_eq, partial_sums, data.transition_indices_,
            data.kaplan_meier_estimators_, data.indep_vars_, data.dep_vars_,
            &score_function)) {
      cout << "ERROR: Unable to compute score function for Log rank. "
           << "Aborting.\n";
      return false;
    }

    *log_rank_estimate += score_function(0);

    // For Information-Matrix, we want to use KME's squared.
    VectorXd kme_squared;
    kme_squared.resize(data.kaplan_meier_estimators_.size());
    for (int i = 0; i < data.kaplan_meier_estimators_.size(); ++i) {
      kme_squared(i) =
          data.kaplan_meier_estimators_(i) * data.kaplan_meier_estimators_(i);
    }

    // Compute Information Matrix I(0) (or I*(0)).
    MatrixXd information_matrix;
    if (!ComputeInformationMatrix(
          exp_logistic_eq, partial_sums,
          data.ties_constant_, data.transition_indices_,
          kme_squared, data.indep_vars_, data.dep_vars_,
          &information_matrix)) {
      cout << "ERROR: Unable to compute information matrix for log rank. "
           << "Aborting.\n";
      return false;
    }
    *log_rank_variance += information_matrix(0, 0);
  }

  return true;
}

bool CoxDualRegression::ComputeRegressionCoefficients(
    const int n, const int p, const KaplanMeierEstimatorType kme_type,
    const ConvergenceCriteria& convergence_criteria,
    int* num_iterations,
    VectorXd* regression_coefficients,
    MatrixXd* info_matrix_inverse,
    vector<StratificationData>* stratification_data) {
  // If no data, nothing to do.
  if (n == 0) return true;

  // First guess for \hat{\beta} is a vector of all zeros.
  VectorXd beta_hat;
  beta_hat.resize(p);
  beta_hat.setZero();

  // For \beta = Zero vector:
  //   - initial logistic_eq (\beta^T * X) is 0.0 for every sample.
  //   - initial exp_logistic_eq is 1.0 for every sample.
  //   - initial log-likelihood is -n * log(2)
  for (StratificationData& data : *stratification_data) {
    int strata_size = data.dep_vars_.size();
    data.log_likelihood_ = static_cast<double>(strata_size) * log(2.0);
    data.logistic_eq_.resize(strata_size);
    data.exp_logistic_eq_.resize(strata_size);
    for (int i = 0; i < strata_size; ++i) {
      data.logistic_eq_(i) = 0.0;
      data.exp_logistic_eq_(i) = 1.0;
    }
    if (!ComputePartialSums(data.transition_indices_, data.exp_logistic_eq_,
                            &data.partial_sums_)) {
      return false;
    }
    if (!ComputeKaplanMeierEstimator(
            kme_type, data.partial_sums_, data.transition_indices_,
            data.dep_vars_, &data.kaplan_meier_estimators_)) {
      return false;
    }
  }

  // Perform Newton-Raphson method to find y-intercept of Score function
  // (yields local maximum/minimum of log-likelihood).
  const double initial_log_likelihood = static_cast<double>(n) * log(2.0);
  if (!RunNewtonRaphson(
          initial_log_likelihood, convergence_criteria, beta_hat,
          stratification_data, regression_coefficients, num_iterations)) {
    return false;
  }

  // Compute the final log likelihood and inverse of Info Matrix.
  double final_log_likelihood = 0.0;
  for (StratificationData& data : *stratification_data) {
    // Compute the final value for exp(\beta^T * X_i).
    if (!ComputeExponentialOfLogisticEquation(
            *regression_coefficients, data.indep_vars_,
            &data.logistic_eq_, &data.exp_logistic_eq_)) {
      cout << "ERROR: Unable to compute the exponential of the RHS of the "
           << "logistic equation for beta:\n"
           << *regression_coefficients << "\nAborting.\n";
      return false;
    }

    // Compute the final partial sums S^0_i(beta_hat) for beta_hat:
    //   := \sum_j I(T_j >= T_i) * exp_logistic_eq_j
    if (!ComputePartialSums(
            data.transition_indices_, data.exp_logistic_eq_,
            &data.partial_sums_)) {
      cout << "ERROR: Unable to compute partial sums for beta:\n"
           << *regression_coefficients << "\nAborting.\n";
      return false;
    }

    // Compute final Log Likelihood of \hat{\beta}.
    if (!ComputeLogLikelihood(
            data.logistic_eq_, data.exp_logistic_eq_, data.partial_sums_,
            data.transition_indices_, data.dep_vars_, &data.log_likelihood_)) {
      cout << "ERROR: Unable to compute log-likelihood for beta:\n"
           << *regression_coefficients << "\nAborting.\n";
      return false;
    }
    final_log_likelihood += data.log_likelihood_;

    // Compute final Score function U(\hat{\beta}).
    if (!ComputeScoreFunction(
            data.exp_logistic_eq_, data.partial_sums_,
            data.transition_indices_, data.kaplan_meier_estimators_,
            data.indep_vars_, data.dep_vars_, &data.score_function_)) {
      cout << "ERROR: Unable to compute final score function for beta:\n"
           << *regression_coefficients << "\nAborting.\n";
      return false;
    }

    // Compute final Information Matrix V(\hat{\beta}).
    if (!ComputeInformationMatrix(
          data.exp_logistic_eq_, data.partial_sums_,
          data.ties_constant_, data.transition_indices_,
          data.kaplan_meier_estimators_, data.indep_vars_, data.dep_vars_,
          &data.information_matrix_)) {
      cout << "ERROR: Unable to compute final information matrix for beta:\n"
           << *regression_coefficients << "\nAborting.\n";
      return false;
    }
  }

  // Sum over Score Function and Info Matrix for each strata to get global values.
  VectorXd score_function;
  MatrixXd info_matrix;
  if (!ComputeGlobalScoreFunctionAndInfoMatrix(
        *stratification_data, &score_function, &info_matrix)) {
    return false;
  }

  // Store info_matrix_inverse.
  *info_matrix_inverse = info_matrix.inverse();
  return true;
}

bool CoxDualRegression::ComputeDualCovarianceMatrix(
    const set<int>& common_rows,
    const map<int, int>& orig_row_to_unsorted_model_row_one,
    const map<int, int>& orig_row_to_unsorted_model_row_two,
    const MatrixXd& w_vectors_one,
    const MatrixXd& w_vectors_two,
    const MatrixXd& final_info_matrix_inverse_one,
    const MatrixXd& final_info_matrix_inverse_two,
    DualStatistics* dual_stats) {
  MatrixXd* b_matrix = &(dual_stats->b_matrix_);
  MatrixXd* dual_covariance_matrix = &(dual_stats->dual_covariance_matrix_);
  // Compute R-Matrix.
  const int n = common_rows.size();
  const int p_one = w_vectors_one.cols();
  const int p_two = w_vectors_two.cols();
  b_matrix->resize(p_one, p_two);
  b_matrix->setZero();

  if (!ComputeBMatrix(
        common_rows,
        orig_row_to_unsorted_model_row_one, orig_row_to_unsorted_model_row_two,
        w_vectors_one, w_vectors_two, b_matrix)) {
    return false;
  }

  // Compute Dual Covariance Matrix (Beta vs. Gamma Regression coefficients).
  *dual_covariance_matrix = 
      final_info_matrix_inverse_one * (*b_matrix) *
      final_info_matrix_inverse_two;

  // Also compute dual_correlation_matrix_.
  dual_stats->dual_correlation_matrix_.resize(
      dual_covariance_matrix->rows(), dual_covariance_matrix->cols());
  for (int r = 0; r < dual_covariance_matrix->rows(); ++r) {
    for (int c = 0; c < dual_covariance_matrix->cols(); ++c) {
      dual_stats->dual_correlation_matrix_(r, c) =
        (*dual_covariance_matrix)(r, c) /
        (sqrt(final_info_matrix_inverse_one(r, r) *
              final_info_matrix_inverse_two(c, c)));
    }
  }

  return true;
}

bool CoxDualRegression::ComputeNewBetaHat(
    const VectorXd& beta_hat,
    const double& log_likelihood,
    VectorXd* new_beta_hat,
    double* new_log_likelihood,
    vector<StratificationData>* stratification_data) {
  // Get overall Score Function and Information Matrix.
  VectorXd score_function;
  MatrixXd info_matrix;
  if (!ComputeGlobalScoreFunctionAndInfoMatrix(
        *stratification_data, &score_function, &info_matrix)) {
    return false;
  }

  int num_halving_attempts = 0;
  bool found_better_likelihood = false;
  while (!found_better_likelihood &&
         num_halving_attempts < MAX_HALVING_ATTEMPTS) {
    if (num_halving_attempts == 0) {
      // First time through, no need to perform halving yet, just use
      // ordinary iterative formula for new beta hat.
      *new_beta_hat = beta_hat + info_matrix.inverse() * score_function;
    } else {
      // The first time(s) through failed. Go half the distance as the
      // previous attempt.
      *new_beta_hat = (beta_hat + *new_beta_hat) / 2.0;
    }

    // Compute New log-likelihood.
    *new_log_likelihood = 0.0;
    for (StratificationData& data : *stratification_data) {
      // Get \beta^T * X_i and exp(\beta^T * X_i) for new_beta_hat.
      if (!ComputeExponentialOfLogisticEquation(
              *new_beta_hat, data.indep_vars_,
              &data.logistic_eq_, &data.exp_logistic_eq_)) {
        cout << "ERROR: Unable to compute the exponential of the RHS of the "
             << "logistic equation for beta:\n"
             << *new_beta_hat << "\nAborting.\n";
        return false;
      }

      // Get partial sums S_i(new_beta_hat) for new_beta_hat:
      //   := \sum_j I(T_j >= T_i) * new_exp_logistic_eq_j
      if (!ComputePartialSums(
              data.transition_indices_, data.exp_logistic_eq_,
              &data.partial_sums_)) {
        cout << "ERROR: Unable to compute partial sums for beta:\n"
             << *new_beta_hat << "\nAborting.\n";
        return false;
      }

      // Compute Log Likelihood of new \hat{\beta}.
      if (!ComputeLogLikelihood(
              data.logistic_eq_, data.exp_logistic_eq_, data.partial_sums_,
              data.transition_indices_, data.dep_vars_, &data.log_likelihood_)) {
        cout << "ERROR: Unable to compute new log-likelihood for beta:\n"
             << *new_beta_hat << "\nAborting.\n";
        return false;
      }
      *new_log_likelihood += data.log_likelihood_;
    }

    // Check if likelihood has increased.
    if (*new_log_likelihood <= log_likelihood) {
      num_halving_attempts++;
    } else {
      found_better_likelihood = true;
    }
  }
  return true;
}

bool CoxDualRegression::RunNewtonRaphson(
    const double& log_likelihood,
    const ConvergenceCriteria& convergence_criteria,
    const VectorXd& beta_hat,
    vector<StratificationData>* stratification_data,
    VectorXd* regression_coefficients, int* iterations) {
  if (*iterations > MAX_ITERATIONS) {
    cout << "ERROR: Newton-Raphson failed to converge after "
         << MAX_ITERATIONS << " iterations. Aborting.\n";
    return false;
  }

  (*iterations)++;

  for (StratificationData& data : *stratification_data) {
    // Compute Score function U(\hat{\beta}).
    if (!ComputeScoreFunction(
            data.exp_logistic_eq_, data.partial_sums_,
            data.transition_indices_, data.kaplan_meier_estimators_,
            data.indep_vars_, data.dep_vars_,
            &data.score_function_)) {

      cout << "ERROR in RunNewtonRaphson: On iteration " << *iterations
           << ", Unable to compute the score function of beta:\n"
           << beta_hat << "\nAborting.\n";
      return false;
    }

    // Compute Information Matrix V(\hat{\beta}).
    if (!ComputeInformationMatrix(
            data.exp_logistic_eq_, data.partial_sums_,
            data.ties_constant_, data.transition_indices_,
            data.kaplan_meier_estimators_, data.indep_vars_, data.dep_vars_,
            &data.information_matrix_)) {
      cout << "ERROR in RunNewtonRaphson: On iteration " << *iterations
           << ", Unable to compute the Information Matrix of beta:\n"
           << beta_hat << "\nAborting.\n";
      return false;
    }
  }

  // Compute new \beta, \beta^T * X_i, exp(\beta^T * X_i), and Log-Likelihood.
  VectorXd new_beta_hat;
  double new_log_likelihood;
  if (!ComputeNewBetaHat(
        beta_hat, log_likelihood,
        &new_beta_hat, &new_log_likelihood, stratification_data)) {
    cout << "ERROR in ComputeNewBetaHat: On iteration " << *iterations
         << ", Unable to compute new beta for old beta:\n"
         << beta_hat << "\nAborting.\n";
    return false;
  }

  // Compare difference between old beta and new.
  bool convergence_attained = false;
  if (convergence_criteria.to_compare_ == ItemsToCompare::LIKELIHOOD) {
    convergence_attained = AbsoluteConvergenceSafe(
        log_likelihood, new_log_likelihood,
        convergence_criteria.delta_, convergence_criteria.threshold_);
  } else if (convergence_criteria.to_compare_ == ItemsToCompare::COORDINATES) {
    convergence_attained = VectorAbsoluteConvergenceSafe(
        beta_hat, new_beta_hat,
        convergence_criteria.delta_, convergence_criteria.threshold_);
  } else {
     cout << "ERROR in RunNewtonRaphson: On iteration " << *iterations
          << ", Unrecognized convergence_criteria: "
          << convergence_criteria.to_compare_ << ". Aborting.\n";
     return false;
  }
  if (!convergence_attained) {
    return RunNewtonRaphson(
        new_log_likelihood, convergence_criteria, new_beta_hat,
        stratification_data, regression_coefficients, iterations);
  } else {
    *regression_coefficients = new_beta_hat;
    return true;
  }  
}

bool CoxDualRegression::ComputeFinalValues(
    const MatrixXd& info_matrix_inverse,
    const VectorXd& regression_coefficients,
    SummaryStatistics* stats, char* error_msg) {
  // Sanity check input.
  if (stats == nullptr) {
    sprintf(error_msg, "Null pointers passed to ComputeFinalValues().");
    return false;
  }

  // Compute all statistics from info_matrix_inverse and
  // regression_coefficients, and populate vectors with the values.
  for (int i = 0; i < regression_coefficients.size(); ++i) {
    double est_val = regression_coefficients(i);
    stats->estimates_.push_back(est_val);
    stats->estimates_squared_.push_back(est_val * est_val);
    double est_var = info_matrix_inverse(i, i);
    stats->variances_.push_back(est_var);
    double est_se = -1.0;
    if (est_var >= 0.0) {
      est_se = sqrt(est_var);
      stats->standard_error_.push_back(est_se);
    } else {
      sprintf(error_msg, "Negative estimated variance (should never happen): "
                         "%.04f\n", est_var);
      return false;
    }
  }
  return true;
}

bool CoxDualRegression::ComputeTestStatistic(
    const VectorXd& beta,
    SummaryStatistics* stats, char* error_msg) {
  if (stats == nullptr) {
    sprintf(error_msg, "Null pointers passed to ComputeFinalValues().");
    return false;
  }
 
  const int n = stats->estimates_.size();
  if (beta.size() != n ||
      stats->standard_error_.size() != n) {
    sprintf(error_msg, "Mismatching sizes: beta has size %d, while number "
                       "of estimates is %d and number of SEE's is %d",
            beta.size(), n, stats->standard_error_.size());
    return false;
  }

  for (int i = 0; i < n; ++i) {
    double test_stat = -1.0;
    const double& est_val = stats->estimates_[i];
    const double& est_se = stats->standard_error_[i];
    if (est_se > 0.0) {
      test_stat = (est_val - beta(i)) / est_se;
      stats->t_statistics_.push_back(test_stat);
    } else {
      sprintf(error_msg, "Negative S.E.E. (should never happen): %.04f.\n",
              est_se);
      return false;
    }
    stats->p_values_.push_back(
        RegularizedReverseIncompleteGammaFunction(0.5, (0.5 * test_stat * test_stat)));
  }

  return true;
}

bool CoxDualRegression::ComputeTestStatistic(
    SummaryStatistics* stats, char* error_msg) {
  const int p = stats->estimates_.size();
  VectorXd zero_vector;
  zero_vector.resize(p);
  zero_vector.setZero();
  return ComputeTestStatistic(zero_vector, stats, error_msg);
}

bool CoxDualRegression::ComputeTestStatistic(
    const vector<VectorXd>& actual_beta,
    vector<SummaryStatistics>* summary_stats, char* error_msg) {
  // Sanity-check input.
  if (summary_stats == nullptr ||
      actual_beta.size() != summary_stats->size()) {
    if (error_msg != nullptr) {
      if (summary_stats == nullptr) {
        sprintf(error_msg, "Null summary_stats.");
      } else {
        sprintf(error_msg, "Number of beta values provided (%d) "
                           "Does not match number of summary stats (%d)",
                actual_beta.size(), summary_stats->size());
      }
    }
    return false;
  }

  // Compute test-statistic for each study.
  for (int i = 0; i < actual_beta.size(); ++i) {
    SummaryStatistics& stats = (*summary_stats)[i];
    const VectorXd& beta = actual_beta[i];
    if (!ComputeTestStatistic(beta, &stats, error_msg)) {
      return false;
    }
  }
  return true;
}

bool CoxDualRegression::ComputeTestStatistic(
    vector<SummaryStatistics>* summary_stats, char* error_msg) {
  for (SummaryStatistics& stats : *summary_stats) {
    if (!ComputeTestStatistic(&stats, error_msg)) {
      return false;
    }
  }
  return true;
}

void CoxDualRegression::PrintDualCovarianceMatrix(
    const set<int>& common_rows,
    const vector<vector<string>>& titles,
    const DualStatistics& dual_stats,
    string* output) {
  if (output == nullptr) return;
  // Print Dual Statistics Header.
  *output += "\n###################################################\n\n";
  *output += "Joint Statistics.\n\n";
  *output += "Rows common to both models (" +
             Itoa(static_cast<int>(common_rows.size())) + ")";
  const int kMaxPrintableRows = 1000;
  if (!kPrintRows) {
    *output += "\n";
  }
  else if (common_rows.size() > kMaxPrintableRows) {
    *output += ": Too many to print...\n";
  } else {
    *output += ":";
    int i = 0;
    for (const int common_row_itr : common_rows) {
      // Add one, as common_rows are indexed assuming first data row has index
      // '0', but when printing, it makes more sense to start indexing at '1'.
      if ((i < 1000 && (i % 20) == 0) ||
          (i >= 1000 && i < 10000 && (i % 16) == 0) ||
          (i >= 10000 && i < 100000 && (i % 12) == 0) ||
          (i >= 100000 && (i % 10) == 0)) {
        *output += "\n\t";
      }
      *output += Itoa(1 + common_row_itr);
      if (i != common_rows.size() - 1) *output += ", ";
      ++i;
    }
  }
  *output += "\n\n";

  // Print R-Matrix, Covariance Matrix, and Correlation Matrix.
  string dual_cov_output = "";
  PrintDualCovarianceMatrix(titles, dual_stats, &dual_cov_output);
  *output += dual_cov_output;
}

void CoxDualRegression::PrintDualCovarianceMatrix(
    const vector<vector<string>>& titles,
    const DualStatistics& dual_stats, string* output) {
  if (output == nullptr) return;

  const int p_one = dual_stats.dual_covariance_matrix_.rows();
  const int p_two = dual_stats.dual_covariance_matrix_.cols();

  // Get Row and Column Titles.
  if (titles.size() != 2) return;
  const vector<string>& model_one_titles = titles[0];
  const vector<string>& model_two_titles = titles[1];
  vector<string> row_names;
  if (model_one_titles.empty()) {
    for (int i = 1; i <= p_one; ++i) {
      row_names.push_back("X_1" + i);
    }
  } else {
    row_names = model_one_titles;
  }
  if (row_names.size() != p_one) {
    cout << "ERROR: Mismatching number of Row names ("
         << row_names.size() << ") and rows in Covariance Matrix ("
         << p_one << ")." << endl;
    return;
  }
  vector<string> col_names;
  if (model_two_titles.empty()) {
    for (int i = 1; i <= p_two; ++i) {
      col_names.push_back("X_2" + i);
    }
  } else {
    col_names = model_two_titles;
  }
  if (col_names.size() != p_two) {
    cout << "ERROR: Mismatching number of Col names ("
         << col_names.size() << ") and cols in Covariance Matrix ("
         << p_two << ")." << endl;
    return;
  }

  // Print R-Matrix.
  if (kPrintRMatrix) {
    *output += "\nR-Matrix (Product of W-vectors from overlapping samples "
               "in Model 1 and Model 2):\n" +
               PrintEigenMatrix(row_names, col_names, dual_stats.b_matrix_, 6) +
               "\n";
  }

  // Print Covariance Matrix.
  *output += "\nCovariance Matrix for Estimated Regression Coefficients:\n" +
             PrintEigenMatrix(row_names, col_names,
                              dual_stats.dual_covariance_matrix_, 6) + "\n";

  // Print Correlation Matrix.
  *output += "\nCorrelation Matrix for Estimated Regression Coefficients:\n" +
             PrintEigenMatrix(row_names, col_names,
                              dual_stats.dual_correlation_matrix_, 6) + "\n";
}

string CoxDualRegression::PrintEigenMatrix(
    const vector<string>& row_names, const vector<string>& col_names,
    const MatrixXd& matrix, const int precision) {
  const int num_rows = matrix.rows();
  const int num_cols = matrix.cols();

  // Sanity-check input.
  if (!row_names.empty() && row_names.size() != num_rows) {
    cout << "ERROR: Mismatching number of rows (" << num_rows
         << ") and legend (" << row_names.size() << ")" << endl;
    return "";
  }
  if (!col_names.empty() && col_names.size() != num_cols) {
    cout << "ERROR: Mismatching number of cols (" << num_cols
         << ") and legend (" << col_names.size() << ")" << endl;
    return "";
  }

  string output = "";

  // Print Names of Columns, if specified.
  if (!col_names.empty()) {
    for (int i = 0; i < num_cols; ++i) {
      output += "\t";
      output += col_names[i];
    }
    output += "\n";
  }
  for (int r = 0; r < num_rows; ++r) {
    if (!row_names.empty()) output += row_names[r];
    for (int c = 0; c < num_cols; ++c) {
      output += "\t";
      output += Itoa(matrix(r, c), precision);
    }
    output += "\n";
  }
  return output;
}

}  // namespace regression
