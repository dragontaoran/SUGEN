// Date: Feb 2016
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "statistics_utils.h"

#include "MapUtils/map_utils.h"
#include "MathUtils/constants.h"
#include "MathUtils/gamma_fns.h"
#include "MathUtils/number_comparison.h"
#include "MathUtils/quantile_fns.h"
#include "StringUtils/string_utils.h"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace map_utils;
using namespace string_utils;

namespace math_utils {

bool NegativeVariance(const MatrixXd& variance) {
  // Handle both a vector of variances, as well as a Covariance Matrix.
  if (variance.rows() == 1) {
    for (int i = 0; i < variance.cols(); ++i) {
      if (variance(0, i) < 0.0) return true;
    }
  } else {
    // Sanity-Check input matrix is square. If not, no way to return
    // an error, so just return 'true', which anyway indicates a
    // problem with the Covariance Matrix).
    if (variance.rows() != variance.cols()) return true;
    for (int i = 0; i < variance.cols(); ++i) {
      if (variance(i, i) < 0.0) return true;
    }
  }
  return false;
}

bool StandardizeMatrix(
    const MatrixXd& input_values,
    vector<pair<double, double>>* mean_and_std_dev_by_column,
    MatrixXd* output_values, string* error_msg) {
  if (mean_and_std_dev_by_column == nullptr || output_values == nullptr) {
    return false;
  }

  const int num_rows = input_values.rows();
  const int num_cols = input_values.cols();
  if (num_rows == 0 || num_cols == 0) return true;


  // Go through Matrix, finding running sum (and it's square) of each column.
  for (int col = 0; col < num_cols; ++col) {
    double col_sum = 0.0;
    double col_sum_squared = 0.0;
    for (int row = 0; row < num_rows; ++row) {
      col_sum += input_values(row, col);
      col_sum_squared += input_values(row, col) * input_values(row, col);
    }
    const double mean = col_sum / num_rows;
    const double std_dev =
        sqrt((col_sum_squared - (col_sum * col_sum) / num_rows) / (num_rows - 1));
    mean_and_std_dev_by_column->push_back(make_pair(mean, std_dev));
  }

  // Now populate output_values with the standardized values from input_values.
  output_values->resize(num_rows, num_cols);
  for (int col = 0; col < num_cols; ++col) {
    const pair<double, double>& col_stats = (*mean_and_std_dev_by_column)[col];
    for (int row = 0; row < num_rows; ++row) {
      (*output_values)(row, col) =
          (input_values(row, col) - col_stats.first) / col_stats.second;
    }
  }

  return true;
}

bool UnstandardizeVector(
    const VariableNormalization std_type,
    const VectorXd& input,
    const vector<tuple<bool, double, double>>& coordinate_mean_and_std_dev,
    VectorXd* output, string* error_msg) {
  // Return if nothing to do.
  if (std_type == VAR_NORM_NONE || coordinate_mean_and_std_dev.empty()) {
    *output = input;
    return true;
  }

  // Sanity-Check input.
  if (coordinate_mean_and_std_dev.size() != input.size()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to unstandardize vector: Mismatching "
                    "dimensions of vector (" + Itoa(static_cast<int>(input.size())) +
                    ") and number of standardization constants (" +
                    Itoa(static_cast<int>(coordinate_mean_and_std_dev.size())) + ").\n";
    }
    return false;
  }

  // Constants to help whether unstandardization is necessary.
  const bool undo_std =
      std_type == VAR_NORM_STD || std_type == VAR_NORM_STD_W_N_MINUS_ONE;
  const bool undo_non_binary_std =
      std_type == VAR_NORM_STD_NON_BINARY ||
      std_type == VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;

  // Unstandardize by dividing each coordinate i by \sigma_i.
  const int p = coordinate_mean_and_std_dev.size();
  output->resize(p);
  for (int i = 0; i < p; ++i) {
    const auto& stats = coordinate_mean_and_std_dev[i];
    const bool not_binary = get<0>(stats);
    const double std_dev = get<2>(stats);
    const double rescale_factor =
        (undo_std || (undo_non_binary_std && not_binary)) ? 1.0 / std_dev : 1.0;
    (*output)(i) = rescale_factor * input(i);
  }

  return true;
}

bool UnstandardizeMatrix(
    const VariableNormalization std_type,
    const MatrixXd& input,
    const vector<tuple<bool, double, double>>& coordinate_mean_and_std_dev,
    MatrixXd* output, string* error_msg) {
  // Return if nothing to do.
  if (std_type == VAR_NORM_NONE || coordinate_mean_and_std_dev.empty() ||
      input.rows() == 0 || input.cols() == 0) {
    *output = input;
    return true;
  }

  // Sanity-Check input.
  if (coordinate_mean_and_std_dev.size() != input.rows() ||
      coordinate_mean_and_std_dev.size() != input.cols()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to unstandardize matrix: Mismatching "
                    "dimensions of matrix (" + Itoa(static_cast<int>(input.rows())) +
                    ", " + Itoa(static_cast<int>(input.cols())) +
                    ") and number of standardization constants (" +
                    Itoa(static_cast<int>(coordinate_mean_and_std_dev.size())) + ").\n";
    }
    return false;
  }

  // Constants to help whether unstandardization is necessary.
  const bool undo_std =
      std_type == VAR_NORM_STD || std_type == VAR_NORM_STD_W_N_MINUS_ONE;
  const bool undo_non_binary_std =
      std_type == VAR_NORM_STD_NON_BINARY ||
      std_type == VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY;

  // Unstandardize by dividing coordinate (i, j) by \sigma_i * \sigma_j.
  const int p = coordinate_mean_and_std_dev.size();
  output->resize(p, p);
  for (int i = 0; i < p; ++i) {
    const auto& stats = coordinate_mean_and_std_dev[i];
    const bool not_binary = get<0>(stats);
    const double std_dev = get<2>(stats);
    const double rescale_factor =
        (undo_std || (undo_non_binary_std && not_binary)) ? 1.0 / std_dev : 1.0;
    for (int j = 0; j < p; ++j) {
      const auto& stats2 = coordinate_mean_and_std_dev[j];
      const bool not_binary2 = get<0>(stats2);
      const double std_dev2 = get<2>(stats2);
      const double rescale_factor2 =
          (undo_std || (undo_non_binary_std && not_binary2)) ?
          1.0 / std_dev2 : 1.0;
      (*output)(i, j) = input(i, j) * rescale_factor * rescale_factor2;
    }
  }

  return true;
}

bool GenerateSummaryStatistics(
    const VectorXd& estimates, const VectorXd& est_variance,
    VectorXd* z_statistics, VectorXd* p_values) {
  if (z_statistics == nullptr || p_values == nullptr) return false;

  const int p = estimates.size();
  if (est_variance.size() != p) return false;

  z_statistics->resize(p);
  p_values->resize(p);
  for (int i = 0; i < p; ++i) {
    const double& variance = est_variance(i);
    if (variance < 0.0) return false;
    const double std_error = sqrt(variance);
    const double z_stat = estimates(i) / std_error;
    (*z_statistics)(i) = z_stat;
    (*p_values)(i) =
        RegularizedReverseIncompleteGammaFunction(0.5, (0.5 * z_stat * z_stat));
  }
  return true;
}

bool GenerateSummaryStatisticsFromCovarianceMatrix(
    const VectorXd& estimates, const MatrixXd& var_cov_matrix,
    VectorXd* z_statistics, VectorXd* p_values) {
  const int p = var_cov_matrix.rows();
  if (var_cov_matrix.cols() != p) return false;

  // Select out variances as the diagonal elements of var_cov_matrix.
  VectorXd est_variance;
  est_variance.resize(p);
  for (int i = 0; i < p; ++i) {
    est_variance(i) = var_cov_matrix(i, i);
  }

  return GenerateSummaryStatistics(
      estimates, est_variance, z_statistics, p_values);
}

bool CompleteSummaryStatistics(
    const AnalysisParams& params, SummaryStatistics* summary_stats) {
  if (summary_stats == nullptr) return false;

  // Complete fields related to "standard" statistics.
  if (params.standard_analysis_ &&
      !CompleteStandardSummaryStatistics(summary_stats)) {
    return false;
  }

  // Complete fields related to "log-rank" statistics.
  if (params.log_rank_analysis_ &&
      !CompleteLogRankSummaryStatistics(summary_stats)) {
    return false;
  }

  // Complete fields related to "log-rank" statistics.
  if (params.peto_analysis_ &&
      !CompletePetoSummaryStatistics(summary_stats)) {
    return false;
  }

  return true;
}

bool CompleteStandardSummaryStatistics(SummaryStatistics* summary_stats) {
  if (summary_stats == nullptr) return false;

  const int p = summary_stats->estimates_.size();
  if (summary_stats->variances_.size() != p) return false;

  summary_stats->estimates_squared_.resize(p);
  summary_stats->standard_error_.resize(p);
  summary_stats->t_statistics_.resize(p);
  summary_stats->p_values_.resize(p);
  summary_stats->ci_width_.resize(p);
  for (int i = 0; i < p; ++i) {
    summary_stats->estimates_squared_[i] =
        summary_stats->estimates_[i] * summary_stats->estimates_[i];
    if (summary_stats->variances_[i] < 0.0) return false;
    const double est_se = sqrt(summary_stats->variances_[i]);
    summary_stats->standard_error_[i] = est_se;
    summary_stats->t_statistics_[i] = summary_stats->estimates_[i] / est_se;
    const double& z_stat = summary_stats->t_statistics_[i];
    summary_stats->p_values_[i] =
        RegularizedReverseIncompleteGammaFunction(0.5, (0.5 * z_stat * z_stat));
    summary_stats->ci_width_[i] = 2.0 * Z_SCORE_FOR_TWO_TAILED_FIVE_PERCENT * est_se;
  }
  return true;
}

bool CompleteLogRankSummaryStatistics(SummaryStatistics* summary_stats) {
  if (summary_stats == nullptr) return false;

  summary_stats->log_rank_estimate_squared_ =
      summary_stats->log_rank_estimate_ * summary_stats->log_rank_estimate_;
  if (summary_stats->log_rank_variance_ < 0.0) return false;
  summary_stats->log_rank_standard_estimate_of_error_ =
      sqrt(summary_stats->log_rank_variance_);
  return true;
}

bool CompletePetoSummaryStatistics(SummaryStatistics* summary_stats) {
  if (summary_stats == nullptr) return false;

  summary_stats->peto_estimate_squared_ =
      summary_stats->peto_estimate_ * summary_stats->peto_estimate_;
  if (summary_stats->peto_variance_ < 0.0) return false;
  summary_stats->peto_standard_estimate_of_error_ =
      sqrt(summary_stats->peto_variance_);
  return true;
}

bool SanityCheckCompatibleStandardSummaryStatistics(
    const SummaryStatistics& stats_one, const SummaryStatistics& stats_two,
    string* error_msg) {
  if (stats_one.estimates_.size() != stats_two.estimates_.size() ||
      stats_one.estimates_squared_.size() !=
      stats_two.estimates_squared_.size() ||
      stats_one.standard_error_.size() !=
      stats_two.standard_error_.size() ||
      stats_one.t_statistics_.size() != stats_two.t_statistics_.size() ||
      stats_one.p_values_.size() != stats_two.p_values_.size() ||
      stats_one.final_info_matrix_inverse_.rows() !=
      stats_two.final_info_matrix_inverse_.rows() ||
      stats_one.final_info_matrix_inverse_.cols() !=
      stats_two.final_info_matrix_inverse_.cols()) {
    *error_msg =
        (stats_one.estimates_.size() != stats_two.estimates_.size()) ?
        ("estimates size mismatch (" +
         Itoa(static_cast<int>(stats_one.estimates_.size())) +
        " vs. " + Itoa(static_cast<int>(stats_two.estimates_.size())) + ")") :
        (stats_one.estimates_squared_.size() !=
         stats_two.estimates_squared_.size()) ?
        ("estimates_squared size mismatch (" +
         Itoa(static_cast<int>(stats_one.estimates_squared_.size())) + " vs. " +
         Itoa(static_cast<int>(stats_two.estimates_squared_.size())) + ")") :
        (stats_one.standard_error_.size() !=
         stats_two.standard_error_.size()) ?
        ("SEE size mismatch (" +
         Itoa(static_cast<int>(stats_one.standard_error_.size())) +
         " vs " +
         Itoa(static_cast<int>(stats_two.standard_error_.size())) +
         ")") :
        (stats_one.t_statistics_.size() != stats_two.t_statistics_.size()) ?
        ("t-statistic size mismatch (" +
         Itoa(static_cast<int>(stats_one.t_statistics_.size())) +
         " vs " + Itoa(static_cast<int>(stats_two.t_statistics_.size())) + ")") :
        (stats_one.p_values_.size() != stats_two.p_values_.size()) ?
        ("p_values size mismatch (" +
         Itoa(static_cast<int>(stats_one.p_values_.size())) + " vs " +
         Itoa(static_cast<int>(stats_two.p_values_.size())) + ")") :
        (stats_one.final_info_matrix_inverse_.rows() !=
         stats_two.final_info_matrix_inverse_.rows()) ?
        ("Info Matrix num rows mismatch (" +
         Itoa(static_cast<int>(
             stats_one.final_info_matrix_inverse_.rows())) +
         " vs " +
         Itoa(static_cast<int>(
             stats_two.final_info_matrix_inverse_.rows())) + ")") :
        (stats_one.final_info_matrix_inverse_.cols() !=
         stats_two.final_info_matrix_inverse_.cols()) ?
        ("Info Matrix num cols mismatch (" +
         Itoa(static_cast<int>(
             stats_one.final_info_matrix_inverse_.cols())) +
         " vs " +
         Itoa(static_cast<int>(
             stats_two.final_info_matrix_inverse_.cols())) + ")") :
        "Unknown Error";
    return false;
  }
  return true;
}

bool SanityCheckCompatibleRobustSummaryStatistics(
    const SummaryStatistics& stats_one, const SummaryStatistics& stats_two,
    string* error_msg) {
  if (stats_one.w_vectors_.rows() != stats_two.w_vectors_.rows() ||
      stats_one.w_vectors_.cols() != stats_two.w_vectors_.cols() ||
      stats_one.robust_var_.rows() != stats_two.robust_var_.rows() ||
      stats_one.robust_var_.cols() != stats_two.robust_var_.cols()) {
    *error_msg =
        (stats_one.w_vectors_.rows() != stats_two.w_vectors_.rows()) ?
        ("W-Vectors num rows mismatch (" +
         Itoa(static_cast<int>(
             stats_one.w_vectors_.rows())) + " vs " +
         Itoa(static_cast<int>(
             stats_two.w_vectors_.rows())) + ")") :
        (stats_one.w_vectors_.cols() != stats_two.w_vectors_.cols()) ?
        ("W-Vectors num cols mismatch (" +
         Itoa(static_cast<int>(stats_one.w_vectors_.cols())) +
         " vs " +
         Itoa(static_cast<int>(
             stats_two.w_vectors_.cols())) + ")") :
        (stats_one.robust_var_.rows() != stats_two.robust_var_.rows()) ?
        ("Robust Variance num rows mismatch (" +
         Itoa(static_cast<int>(stats_one.robust_var_.rows())) +
         " vs " +
         Itoa(static_cast<int>(
             stats_two.robust_var_.rows())) + ")") :
        (stats_one.robust_var_.cols() != stats_two.robust_var_.cols()) ?
        ("Robust Variance num cols mismatch (" +
         Itoa(static_cast<int>(stats_one.robust_var_.cols())) +
         " vs " +
         Itoa(static_cast<int>(
             stats_two.robust_var_.cols())) + ")") :
        "Unknown Error";
    return false;
  }
  return true;
}

bool SanityCheckCompatibleLogRankAndPetoSummaryStatistics(
    const SummaryStatistics& stats_one, const SummaryStatistics& stats_two,
    string* error_msg) {
  if (stats_one.log_rank_w_vectors_.rows() !=
      stats_two.log_rank_w_vectors_.rows() ||
      stats_one.log_rank_w_vectors_.cols() !=
      stats_two.log_rank_w_vectors_.cols()) {
    *error_msg =
        (stats_one.log_rank_w_vectors_.rows() !=
         stats_two.log_rank_w_vectors_.rows()) ?
        ("Log-Rank W-Vectors num rows mismatch (" +
         Itoa(static_cast<int>(
             stats_one.log_rank_w_vectors_.rows())) +
         " vs " +
         Itoa(static_cast<int>(
             stats_two.log_rank_w_vectors_.rows())) + ")") :
        (stats_one.log_rank_w_vectors_.cols() !=
         stats_two.log_rank_w_vectors_.cols()) ?
        ("Log-Rank W-Vectors num cols mismatch (" +
         Itoa(static_cast<int>(
             stats_one.log_rank_w_vectors_.cols())) +
         " vs " +
         Itoa(static_cast<int>(
             stats_two.log_rank_w_vectors_.cols())) + ")") :
        "Unknown Error";
    return false;
  }
  return true;
}

void InitializeIterationDataForModel(
    const int p, const AnalysisParams& params,
    IterationDataForModel* stats) {
  if (params.standard_analysis_) stats->standard_counts_.Initialize(p);
  if (params.log_rank_analysis_) stats->log_rank_counts_.Initialize(p);
  if (params.peto_analysis_) stats->peto_counts_.Initialize(p);
  if (params.score_method_analysis_) stats->score_method_counts_.Initialize(p);
  if (params.satterthwaite_analysis_) stats->satterthwaite_counts_.Initialize(p);
}

void InitializeIterationDataHolder(
    const int p_one, const int p_two,
    const AnalysisParams& params_one, const AnalysisParams& params_two,
    IterationDataHolder* stats) {
  InitializeIterationDataForModel(p_one, params_one, &(stats->model_one_));
  if (p_two > 0) {
    InitializeIterationDataForModel(p_two, params_two, &(stats->model_two_));
    stats->dual_stats_.b_matrix_.resize(p_one, p_two);
    stats->dual_stats_.dual_covariance_matrix_.resize(p_one, p_two);
    if (p_one == 1 && p_two == 1) {
      stats->dual_stats_.dual_correlation_matrix_.resize(1, 1);
    }
  }
}

bool AddSummaryStatistics(
    const AnalysisParams& params, const SummaryStatistics& stats_one,
    SummaryStatistics* stats_two, string* error_msg) {
  return AddSummaryStatistics(
      params.standard_analysis_,
      params.log_rank_analysis_ || params.peto_analysis_,
      params.robust_analysis_,
      stats_one, stats_two, error_msg);
}

bool AddSummaryStatistics(
    const bool standard, const bool log_rank_or_peto, const bool robust,
    const SummaryStatistics& stats_one, SummaryStatistics* stats_two,
    string* error_msg) {
  if (standard &&
      !AddStandardSummaryStatistics(stats_one, stats_two, error_msg)) {
    return false;
  }
  if (log_rank_or_peto &&
      !AddLogRankAndPetoSummaryStatistics(stats_one, stats_two, error_msg)) {
    return false;
  }
  if (standard && robust &&
      !AddRobustSummaryStatistics(stats_one, stats_two, error_msg)) {
    return false;
  }

  // Update fraction alive.
  if (stats_one.fraction_alive_.size() != stats_two->fraction_alive_.size()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in merging fraction_alive_: Mismatching sizes (" +
                    Itoa(static_cast<int>(stats_one.fraction_alive_.size())) +
                    " vs. " +
                    Itoa(static_cast<int>(stats_two->fraction_alive_.size())) +
                    ").\n";
    }
    return false;
  }
  for (int i = 0; i < stats_one.fraction_alive_.size(); ++i) {
    if ((get<0>(stats_one.fraction_alive_[i]) !=
         get<0>(stats_two->fraction_alive_[i])) ||
        (get<1>(stats_one.fraction_alive_[i]) !=
         get<1>(stats_two->fraction_alive_[i]))) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in merging fraction_alive_: Mismatching ranges "
                      "for parition " + Itoa(i + 1) + ": [" +
                      Itoa(get<0>(stats_one.fraction_alive_[i])) + ", " +
                      Itoa(get<1>(stats_one.fraction_alive_[i])) + "] vs. [" +
                      Itoa(get<0>(stats_two->fraction_alive_[i])) + ", " +
                      Itoa(get<1>(stats_two->fraction_alive_[i])) + "].\n";
      }
      return false;
    }
    get<2>(stats_two->fraction_alive_[i]) +=
        get<2>(stats_one.fraction_alive_[i]);
  }

  // Update fraction_ties_per_var_.
  for (const pair<string, double>& ties_per_var :
       stats_one.fraction_ties_per_var_) {
    double* cumulative_fraction_ties_per_var =
        FindOrInsert(ties_per_var.first, stats_two->fraction_ties_per_var_, 0.0);
    (*cumulative_fraction_ties_per_var) += ties_per_var.second;
  }

  return true;
}

bool AddStandardSummaryStatistics(
    const SummaryStatistics& stats_one, SummaryStatistics* stats_two,
    string* error_msg) {
  // Sanity check all vectors have same length.
  if (!SanityCheckCompatibleStandardSummaryStatistics(
          stats_one, *stats_two, error_msg)) {
    return false;
  }

  for (int i = 0; i < stats_one.estimates_.size(); ++i) {
    stats_two->estimates_[i] += stats_one.estimates_[i];
  }
  for (int i = 0; i < stats_one.estimates_squared_.size(); ++i) {
    stats_two->estimates_squared_[i] += stats_one.estimates_squared_[i];
  }
  for (int i = 0; i < stats_one.variances_.size(); ++i) {
    stats_two->variances_[i] += stats_one.variances_[i];
  }
  for (int i = 0; i < stats_one.standard_error_.size(); ++i) {
    stats_two->standard_error_[i] +=
        stats_one.standard_error_[i];
  }
  for (int i = 0; i < stats_one.t_statistics_.size(); ++i) {
    stats_two->t_statistics_[i] += stats_one.t_statistics_[i];
  }
  for (int i = 0; i < stats_one.p_values_.size(); ++i) {
    stats_two->p_values_[i] += stats_one.p_values_[i];
  }
  stats_two->final_info_matrix_inverse_ += stats_one.final_info_matrix_inverse_;

  return true;
}

bool AddRobustSummaryStatistics(
    const SummaryStatistics& stats_one, SummaryStatistics* stats_two,
    string* error_msg) {
  // Sanity check all vectors have same length.
  if (!SanityCheckCompatibleRobustSummaryStatistics(
          stats_one, *stats_two, error_msg)) {
    return false;
  }

  stats_two->w_vectors_ += stats_one.w_vectors_;
  stats_two->robust_var_ += stats_one.robust_var_;
  return true;
}

bool AddLogRankAndPetoSummaryStatistics(
    const SummaryStatistics& stats_one, SummaryStatistics* stats_two,
    string* error_msg) {
  // Sanity check all vectors have same length.
  if (!SanityCheckCompatibleLogRankAndPetoSummaryStatistics(
          stats_one, *stats_two, error_msg)) {
    return false;
  }

  // Log-rank statistics.
  stats_two->log_rank_w_vectors_ += stats_one.log_rank_w_vectors_;
  stats_two->log_rank_estimate_ += stats_one.log_rank_estimate_;
  stats_two->log_rank_estimate_squared_ += stats_one.log_rank_estimate_squared_;
  stats_two->log_rank_variance_ += stats_one.log_rank_variance_;
  stats_two->log_rank_standard_estimate_of_error_ +=
      stats_one.log_rank_standard_estimate_of_error_;

  // Peto statistics.
  stats_two->peto_estimate_ += stats_one.peto_estimate_;
  stats_two->peto_estimate_squared_ += stats_one.peto_estimate_squared_;
  stats_two->peto_variance_ += stats_one.peto_variance_;
  stats_two->peto_standard_estimate_of_error_ +=
      stats_one.peto_standard_estimate_of_error_;

  return true;
}

bool AddDualRobustStatistics(
    const DualStatistics& stats_one, DualStatistics* stats_two,
    string* error_msg) {
  // Sanity-check Matrices have same size.
  if (stats_one.b_matrix_.rows() != stats_two->b_matrix_.rows() ||
      stats_one.b_matrix_.cols() != stats_two->b_matrix_.cols() ||
      stats_one.dual_covariance_matrix_.rows() !=
      stats_two->dual_covariance_matrix_.rows() ||
      stats_one.dual_covariance_matrix_.cols() !=
      stats_two->dual_covariance_matrix_.cols()) {
    if (error_msg != nullptr) {
      *error_msg =
          (stats_one.b_matrix_.rows() != stats_two->b_matrix_.rows()) ?
          ("Mismatch in num rows for B-Matrix (" +
           Itoa(static_cast<int>(stats_one.b_matrix_.rows())) +
           " vs " +
           Itoa(static_cast<int>(stats_two->b_matrix_.rows())) + ")") :
          (stats_one.b_matrix_.cols() != stats_two->b_matrix_.cols()) ?
          ("Mismatch in num cols for B-Matrix (" +
           Itoa(static_cast<int>(stats_one.b_matrix_.cols())) +
           " vs " +
           Itoa(static_cast<int>(stats_two->b_matrix_.cols())) + ")") :
          (stats_one.dual_covariance_matrix_.rows() !=
           stats_two->dual_covariance_matrix_.rows()) ?
          ("Mismatch in num rows for Dual Covariance Matrix (" +
           Itoa(static_cast<int>(
               stats_one.dual_covariance_matrix_.rows())) +
           " vs " + Itoa(static_cast<int>(
               stats_two->dual_covariance_matrix_.rows())) + ")") :
          (stats_one.dual_covariance_matrix_.cols() !=
           stats_two->dual_covariance_matrix_.cols()) ?
          ("Mismatch in num cols for Dual Covariance Matrix (" +
           Itoa(static_cast<int>(
               stats_one.dual_covariance_matrix_.cols())) +
           " vs " + Itoa(static_cast<int>(
               stats_two->dual_covariance_matrix_.cols())) + ")") :
          "Unknown Error";
    }
    return false;
  }

  stats_two->b_matrix_ += stats_one.b_matrix_;
  stats_two->dual_covariance_matrix_ += stats_one.dual_covariance_matrix_;
  stats_two->dual_correlation_matrix_ += stats_one.dual_correlation_matrix_;
  return true;
}

bool AddDualLogRankStatistics(
    const DualStatistics& stats_one, DualStatistics* stats_two) {
  stats_two->log_rank_b_matrix_ += stats_one.log_rank_b_matrix_;
  stats_two->log_rank_dual_correlation_ += stats_one.log_rank_dual_correlation_;
  return true;
}

bool PrintSummaryStatistics(
    const ModelAndDataParams& params,
    const SummaryStatistics& summary_stats,
    string* output) {
  if (output == nullptr) return false;

  if (summary_stats.num_iterations_ > 0) {
    *output += "Regression completed in " + Itoa(summary_stats.num_iterations_) +
               " iterations.\n";
  }

  *output += "\n###################################################\n";

  // Print "standard" statistics.
  if (params.analysis_params_.standard_analysis_) {
    string standard_output =
        params.model_type_ == MODEL_TYPE_LINEAR ? "Linear " :
        params.model_type_ == MODEL_TYPE_LOGISTIC ? "Logistic " :
        params.model_type_ == MODEL_TYPE_TIME_DEPENDENT_INTERVAL_CENSORED ? "NPMLE " :
        params.model_type_ == MODEL_TYPE_TIME_INDEPENDENT_INTERVAL_CENSORED ? "NPMLE " :
        params.model_type_ == MODEL_TYPE_RIGHT_CENSORED_SURVIVAL ? "Cox " : "";
    standard_output += "Statistics:\n";
    if (!GetStandardSummaryStatistics(
            params.print_options_, params.legend_,
            summary_stats, &standard_output)) {
      return false;
    }
    *output += standard_output;
  }

  // Print "log-rank" statistics.
  if (params.analysis_params_.log_rank_analysis_) {
    string log_rank_output =
        (params.kme_type_for_log_rank_ ==
         KaplanMeierEstimatorType::LEFT_CONTINUOUS) ?
        "Weighted " : "Unweighted ";
    GetLogRankSummaryStatistics(summary_stats, &log_rank_output);
    *output += "\n" + log_rank_output;
  }

  // Print "peto" statistics.
  if (params.analysis_params_.peto_analysis_) {
    string peto_output = "";
    GetPetoSummaryStatistics(summary_stats, &peto_output);
    *output += "\n" + peto_output;
  }

  // Print "Score Method" Statistics.
  if (params.analysis_params_.score_method_analysis_ ||
      params.analysis_params_.score_method_width_analysis_) {
    string score_method_output = "";
    GetScoreMethodSummaryStatistics(summary_stats, &score_method_output);
    *output += "\n" + score_method_output;
  }

  return true;
}

bool PrintSummaryStatistics(
    ofstream& outfile, const ModelAndDataParams& params,
    const SummaryStatistics& summary_stats) {
  string output = "";
  if (!PrintSummaryStatistics(params, summary_stats, &output)) {
    return false;
  }
  outfile << output;
  return true;
}

void PrintDualStatistics(
    ofstream& outfile, const DualStatistics& dual_stats) {
  outfile << "B-Matrix:\n" << dual_stats.b_matrix_ << endl << endl;
  outfile << "Dual Covariance Matrix:\n" << dual_stats.dual_covariance_matrix_
          << endl << endl;
  outfile << "Dual Correlation Matrix:\n" << dual_stats.dual_correlation_matrix_
          << endl << endl;
  if (dual_stats.log_rank_b_matrix_ > -1.0) {
    outfile << "Log-Rank B-Matrix: " << dual_stats.log_rank_b_matrix_ << endl;
  }
  if (dual_stats.log_rank_dual_correlation_ > -2.0) {
    outfile << "Log-Rank Dual Correlation: "
            << dual_stats.log_rank_dual_correlation_ << endl;
  }
  outfile << endl;
}

void GetLogRankSummaryStatistics(
    const SummaryStatistics& stats, string* output) {
  if (output == nullptr) return;

  // Log-Rank SummaryStatistics fields don't include p-value or test_stat.
  // Compute them now.
  const double test_stat =
      stats.log_rank_estimate_ / sqrt(stats.log_rank_variance_);
  const double p_value =
       RegularizedReverseIncompleteGammaFunction(0.5, (0.5 * test_stat * test_stat));

  // Print Log-Rank Summary Statistics.
  char out_line[512] = "";
  sprintf(out_line, "%0.06f\t%0.06f\t%0.06f\t%0.06f",
          stats.log_rank_estimate_, stats.log_rank_variance_,
          test_stat, p_value);
  *output += "Log-Rank Statistics:\nStatistic\tVariance\tZ-Statistic\t"
             "p-value\n" + string(out_line) + "\n";
}

void GetPetoSummaryStatistics(
    const SummaryStatistics& stats, string* output) {
  if (output == nullptr) return;

  // Peto SummaryStatistics fields don't include p-value or test_stat.
  // Compute them on the fly, and print Peto Summary Statistics.
  *output += "Peto Statistics:\nEstimate\tVariance\tSE     \t95%_CI\n" +
             Itoa(stats.peto_estimate_) + "\t" + Itoa(stats.peto_variance_) +
             "\t" + Itoa(stats.peto_standard_estimate_of_error_) + "\t[" +
             Itoa(stats.peto_estimate_ -
                      Z_SCORE_FOR_TWO_TAILED_FIVE_PERCENT *
                      stats.peto_standard_estimate_of_error_) + ", " +
             Itoa(stats.peto_estimate_ +
                      Z_SCORE_FOR_TWO_TAILED_FIVE_PERCENT *
                      stats.peto_standard_estimate_of_error_) + "]\n";
}

void GetScoreMethodSummaryStatistics(
    const SummaryStatistics& stats, string* output) {
  if (output == nullptr) return;

  *output += "Score Method 95% Confidence Interval:\n[" +
             Itoa(stats.score_method_ci_left_) + ", " +
             Itoa(stats.score_method_ci_right_) + "]\n";
}

bool GetStandardSummaryStatistics(
    const SummaryStatisticsPrintOptions& print_options,
    const vector<string>& titles,
    const SummaryStatistics& stats, string* output) {
  if (output == nullptr || stats.estimates_.empty()) return false;

  const int p = stats.estimates_.size();

  vector<string> generic_titles;
  const vector<string>* titles_ptr;
  if (titles.empty()) {
    for (int i = 0; i < p; ++i) {
      generic_titles.push_back("X_" + Itoa(i));
    }
    titles_ptr = &generic_titles;
  } else {
    titles_ptr = &titles;
  }

  if (titles_ptr->size() != p) return false;

  // Print (inverse) information matrix (Variance/Covariance), if called for.
  if (print_options.print_covariance_matrix_) {
    if (print_options.print_robust_variance_) {
      if (stats.robust_var_.rows() != p || stats.robust_var_.cols() != p) {
        return false;
      }
      *output += "Robust Variance:\n\t\t";
      *output += Join(*titles_ptr, "\t") + "\n";
      for (int r = 0; r < stats.robust_var_.rows(); ++r) {
        *output += (*titles_ptr)[r];
        for (int c = 0; c < stats.robust_var_.rows(); ++c) {
          *output += "\t" + Itoa(stats.robust_var_(r, c));
        }
        *output += "\n";
      }
      *output += "\n";
    } else {
      if (stats.final_info_matrix_inverse_.rows() != p ||
          stats.final_info_matrix_inverse_.cols() != p) {
        return false;
      }
      *output += "Variance-Covariance Matrix:\n\t\t";
      *output += Join(*titles_ptr, "\t") + "\n";
      for (int r = 0; r < stats.final_info_matrix_inverse_.rows(); ++r) {
        *output += (*titles_ptr)[r];
        for (int c = 0; c < stats.final_info_matrix_inverse_.rows(); ++c) {
          *output += "\t" + Itoa(stats.final_info_matrix_inverse_(r, c));
        }
        *output += "\n";
      }
      *output += "\n";
    }
  }

  // Print robust variance, if called for.

  // Write HEADER line.
  *output += "Variable_Name";
  if (print_options.print_estimates_) {
    *output += "\tEstimate";
  }
  if (print_options.print_variance_) {
    if (stats.variances_.size() != p) return false;
    *output += "\tVariance";
  }
  if (print_options.print_robust_variance_) {
    if (stats.robust_var_.rows() != p || stats.robust_var_.cols() != p) {
      return false;
    }
    *output += "\tRobust_Var";
  }
  if (print_options.print_se_) {
    if (stats.standard_error_.size() != p) return false;
    if (print_options.print_robust_variance_) {
      *output += "\tRobust_SE";
    } else {
      *output += "\tSE     ";
    }
  }
  if (print_options.print_t_stat_) {
    if (stats.t_statistics_.size() != p) return false;
    *output += "\tZ-Statistic";
  }
  if (print_options.print_p_value_) {
    if (stats.p_values_.size() != p) return false;
    *output += "\tp-value";
  }
  if (print_options.print_ci_width_) {
    if (stats.ci_width_.size() != p) return false;
    *output += "\tCI_Width";
  }
  *output += "\n";

  // Write Summary Statsitics.
  for (int i = 0; i < p; ++i) {
    *output += (*titles_ptr)[i];
    if (print_options.print_estimates_) {
      *output += "\t" + Itoa(stats.estimates_[i], 6);
    }
    if (print_options.print_variance_) {
      *output += "\t" + Itoa(stats.variances_[i], 6);
    }
    if (print_options.print_robust_variance_) {
      *output += "\t" + Itoa(stats.robust_var_(i, i), 6);
    }
    if (print_options.print_se_) {
      if (print_options.print_robust_variance_) {
        const double robust_var = stats.robust_var_(i, i);
        if (robust_var < 0.0) return false;
        *output += "\t" + Itoa(sqrt(robust_var), 6);
      } else {
        *output += "\t" + Itoa(stats.standard_error_[i], 6);
      }
    }
    if (print_options.print_t_stat_) {
      *output += "\t" + Itoa(stats.t_statistics_[i], 6);
    }
    if (print_options.print_p_value_) {
      *output += "\t" + Itoa(stats.p_values_[i], 6);
    }
    if (print_options.print_ci_width_) {
      *output += "\t" + Itoa(stats.ci_width_[i], 6);
    }
    *output += "\n";
  }
  *output += "\n";
}

bool ComputeMedian(const vector<double>& input, double* median) {
  if (median == nullptr || input.empty()) return false;
  // Only one replicate, just return the value of estimate/variance.
  const int k = input.size();
  if (k == 1) {
    *median = input[0];
    return true;
  }
  

  // Copy input to temporary container, so it can be sorted to find median.
  vector<double> values;
  for (const double& value : input) values.push_back(value);

  // Sort values.
  sort(values.begin(), values.end());

  // Pick out median:
  //   - Middle element, if k is odd
  //   - Mean of middle two elements, if k is even
  const int middle = k / 2;
  const bool k_is_odd = 2 * middle < k;
  *median = k_is_odd ?
      values[middle] : (values[middle - 1] + values[middle]) / 2.0;
 
  return true;
}

bool ComputeMedians(
    const vector<vector<double>>& input, vector<double>* medians) {
  if (medians == nullptr || input.empty()) return false;
  const int k = input.size();
  const int p = input[0].size();
  medians->clear();
  medians->resize(p);

  // Only one replicate, just return the value of estimate/variance.
  if (k == 1) {
    *medians = input[0];
    return true;
  }
  

  // Copy input to temporary container, so it can be sorted to find medians.
  // Also, it will be more convenient to have the inner vector represent
  // a single column; i.e. if viewing 'input' as a (k, p) matrix, then
  // 'values' will represent its transpose.
  vector<vector<double>> values(p, vector<double>(k));
  for (int orig_row = 0; orig_row < k; ++orig_row) {
    for (int orig_col = 0; orig_col < p; ++orig_col) {
      values[orig_col][orig_row] = input[orig_row][orig_col];
    }
  }

  // Note that the median is the:
  //   - Middle element, if k is odd
  //   - Mean of middle two elements, if k is even
  const int middle = k / 2;
  const bool k_is_odd = 2 * middle < k;

  // Sort values (in place), and choose middle element as the median.
  for (int i = 0; i < p; ++i) {
    vector<double>& row_values = values[i];
    sort(row_values.begin(), row_values.end());
    (*medians)[i] = k_is_odd ?
        row_values[middle] : (row_values[middle - 1] + row_values[middle]) / 2.0;
  }
 
  return true;
}

}  // namespace math_utils
