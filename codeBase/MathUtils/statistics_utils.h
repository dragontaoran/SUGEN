// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description:
//   Structure to hold basic statistics.

#include "FileReaderUtils/read_file_structures.h"
#include "MathUtils/constants.h"

#include <Eigen/Dense>
#include <fstream>
#include <vector>

#ifndef STATISTICS_UTILS_H
#define STATISTICS_UTILS_H

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace file_reader_utils;
using namespace std;

namespace math_utils {

struct Statistics {
  double mean_;
  double mean_squared_;
  double variance_;
  double variance_squared_;
  double std_error_;
  double z_stat_;
  double p_value_;

  Statistics() {
    mean_ = 0.0;
    mean_squared_ = 0.0;
    variance_ = 0.0;
    variance_squared_ = 0.0;
    std_error_ = 0.0;
    z_stat_ = 0.0;
    p_value_ = 0.0;
  }
};

// Holds final values of interest for a given model.
struct SummaryStatistics {
  int num_iterations_;

  // The following field is only relevant for Survival (Cox) models.
  // The fraction of censored (still "alive") sampled, i.e. \Delta = 0.
  // The original domain of Samples [1, n] may be partitioned (e.g. for
  // simulations, we may define different distributions for a sampling
  // parameter based on Sample index); in which case we keep track
  // seperately of the fraction alive on each partition, and the range
  // of the partition in the first two coordinates of each tuple.
  vector<tuple<int, int, double>> fraction_alive_;

  // The following field indicates how many (fraction of) ties, for each
  // variable, was in the data.
  map<string, double> fraction_ties_per_var_;

  // ======================= Fields for Standard Analysis ======================
  vector<double> estimates_;  // Final estimates for \hat{\beta}:
                              //   \hat{\beta}_1, \hat{\beta}_2,..., \hat{\beta}_p
                              // i.e. the value of \hat{\beta} that satisifies:
                              //   U(\hat{\beta}) = 0
                              // where U is the score function.
  vector<double> estimates_squared_;  // The square of (each coordinate of) the
                                     // final estimates for \hat{\beta}; useful
                                     // if doing a simulation for computing
                                     // variance of the estimates.
  vector<double> variances_;  // Diagonal of Covariance Matrix =
                              // Inverse of Information Matrix: I^-1(estimates_)
                              // (evaluated at final estimates_ values)
  vector<double> standard_error_;  // i^th coordinate is: 
                                                //   sqrt(variances_[i])
  vector<double> t_statistics_;  // i^th coordinate is:
                                 //   (estimates_[i] - actual_beta[i]) /
                                 //   standard_error_[i]
  vector<double> p_values_;      // i^th coordinate is:
                                 //  Pr[t_statistics_[i] > 1.96 | Null Hypothesis],
  vector<double> ci_width_;  // Width of confidence interval; i^th coordinate is:
                             //    2 * 1.96 * standard_error_[i] 
  MatrixXd final_info_matrix_inverse_;  // I(estimates_)
  // ===================== END Fields for Standard Analysis ====================

  // ======================= Fields for Robust Analysis ========================
  MatrixXd w_vectors_;  // W(\hat{\beta}) (See page 5 of "Simultaneous Inference
                        // on Treatment Effects in Factorial Surviavl Studies")
                        // NOTE: There are a number of ways to store W-Vectors:
                        //    1) In a matrix, where row i corresponds to the
                        //       W-vector for row i of the (sorted) data
                        //    2) In a matrix, where row i corresponds to the
                        //       W-vector for row i of the (unsorted) data
                        //       (Note: this may not be the same index
                        //       as the passed-in data, e.g. if some rows are
                        //       excluded because they're not in a subgroup,
                        //       have NA values, etc.)
                        //    3) Have a separate Matrix for each (sorted) strata
                        // This field here does (2); the 'w_vectors_' field of
                        // StratificationData does (3) (indeed, the latter is
                        // transformed into the former via CopyWVectors());
                        // it isn't really possible to do (1), since not all
                        // rows present in the original data necessarily exist
                        // after filtering (for subgroup etc). When you need the
                        // W-vector of a row that is indexed by it's original
                        // row index, use 'orig_row_to_unsorted_model_row'.
  MatrixXd robust_var_;  // The actual "robust_variance" is the diagonal of this matrix:
                         // I^-1(\hat{\beta}) *
                         // [ \sum_{strata} \sum_{samples_in_strata}
                         //     W(\hat{\beta}) * W^T(\hat{\beta}) ] *
                         // I^-1(\hat{\beta})
  // ===================== END Fields for Robust Analysis ======================

  // ======================= Fields for Log Rank Analysis ======================
  // NOTE: only valid if p = 1, and then evaluated using \hat{\beta} = 0.
  double log_rank_estimate_;                    // U(0)
  double log_rank_estimate_squared_;            // U(0)^2
  double log_rank_variance_;                    // I(0)
  double log_rank_standard_estimate_of_error_;  // sqrt(I(0))
  MatrixXd log_rank_w_vectors_;                 // W(0)
  // ===================== END Fields for Log Rank Analysis ====================

  // ======================= Fields for Peto Analysis ==========================
  // NOTE: only valid if p = 1, and then evaluated using \hat{\beta} = 0.
  // NOTE: Even though the below values are directly obtainable from the
  // log-rank fields above, we put them here anyway, because when running
  // a simulation, we use SummaryStatistics as a holder of cummulative sums,
  // and there is not a way to recover e.g. the sum of two peto_estimates_
  // from the corresponding sum of log_rank_estimates_ and log_rank_variances_.
  double peto_estimate_;                    // U(0) / I(0)
  double peto_estimate_squared_;            // (U(0) / I(0))^2
  double peto_variance_;                    // 1.0 / I(0)
  double peto_standard_estimate_of_error_;  // 1.0 / sqrt(I(0))
  // ===================== END Fields for Peto Analysis ========================

  // ======================= Fields for Score Method Analysis ==================
  double score_method_value_;
  double score_method_ci_left_; 
  double score_method_ci_right_;
  // ===================== END Fields for Score Method Analysis ================

  SummaryStatistics() {
    num_iterations_ = 0;
    log_rank_estimate_ = 0.0;
    log_rank_variance_ = 0.0;
    log_rank_standard_estimate_of_error_ = 0.0;
    peto_estimate_ = 0.0;
    peto_variance_ = 0.0;
    peto_standard_estimate_of_error_ = 0.0;
    score_method_value_ = 0.0;
    score_method_ci_left_= 0.0; 
    score_method_ci_right_ = 0.0;
  }

  void Initialize(
      const bool standard, const bool log_rank, const bool robust,
      const int n, const int p, const vector<pair<int, int>>& sample_partitions) {
    if (standard) {
      estimates_.resize(p, 0.0);
      estimates_squared_.resize(p, 0.0);
      variances_.resize(p, 0.0);
      standard_error_.resize(p, 0.0);
      t_statistics_.resize(p, 0.0);
      p_values_.resize(p, 0.0);
      final_info_matrix_inverse_.resize(p, p);
      final_info_matrix_inverse_.setZero();
    }
    if (robust) {
      w_vectors_.resize(n, p);
      w_vectors_.setZero();
      robust_var_.resize(p, p);
      robust_var_.setZero();
    }
    if (log_rank) {
      log_rank_w_vectors_.resize(n, 1);
      log_rank_w_vectors_.setZero();
    }
    for (const pair<int, int>& partition : sample_partitions) {
      fraction_alive_.push_back(make_tuple(
          partition.first, partition.second, 0.0));
    }
  }

  void Initialize(
      const bool standard, const bool log_rank, const bool robust,
      const int n, const int p) {
    Initialize(standard, log_rank, robust, n, p, vector<pair<int, int>>());
  }

  void Initialize(const int p) {
    Initialize(true, false, false, -1, p);
  }

};

// Holds "dual" summary statistics for two models.
// See "Simultaneous Inference on Treatment Effects in Factorial Surviavl
// Studies" for explanation of notation and concepts.
struct DualStatistics {
  // B(\hat{\beta}, \hat{\gamma}) =
  //     \sum_{samples_in_both_models) W(\beta) * \tilde{W}^T(\gamma)
  MatrixXd b_matrix_;  // Dim = (p_one, p_two)
  // I^-1(\hat{\beta}) * B(\hat{\beta}, \hat{\gamma}) * \tilde{I}^-1(\hat{\gamma})
  MatrixXd dual_covariance_matrix_;  // Dim = (p_one, p_two)
  // Same as above, but with an extra sqrt on bottom:
  //   B(\hat{\beta}, \hat{\gamma}) / sqrt(I(\hat{\beta}) * \tilde{I}(\hat{\gamma}))
  // Only valid if p_one = p_two = 1 (otherwise, need a different formula, as
  // sqrt of Matrix is not defined).
  MatrixXd dual_correlation_matrix_;

  // Log-rank statistics (only valid if p = 1, and then evaluated using
  // \hat{\beta} = 0 and \hat{\gamma} = 0).
  double log_rank_b_matrix_;  // B(0, 0), estimates log-rank dual covariance
  // B(0, 0) / sqrt(I(0) * \tilde{I}(0)), estimates log-rank dual correlation
  double log_rank_dual_correlation_;

  DualStatistics() {
    // Ordinary covariance is non-negative. Initialize with an invalid
    // value, to indicate this field shouldn't be used if not set.
    log_rank_b_matrix_ = -1.0;
    // Ordinary correlation is between [-1, 1]. Initialize with an invalid
    // value, to indicate this field shouldn't be used if not set.
    log_rank_dual_correlation_ = -2.0;
  }
};

// Holds cumulative values that will be useful for computing Power
// and coverage probability.
struct IterationCounts {
  vector<int> inside_count_;
  vector<int> outside_count_;
  vector<int> tstat_above_count_;
  vector<int> tstat_below_count_;
  vector<double> ci_width_;

  IterationCounts() {}

  IterationCounts(const int p) {
    inside_count_.resize(p, 0);
    outside_count_.resize(p, 0);
    tstat_above_count_.resize(p, 0);
    tstat_below_count_.resize(p, 0);
    ci_width_.resize(p, 0.0);
  }

  void Initialize(const int p) {
    inside_count_.resize(p, 0);
    outside_count_.resize(p, 0);
    tstat_above_count_.resize(p, 0);
    tstat_below_count_.resize(p, 0);
    ci_width_.resize(p, 0.0);
  }
};

struct IterationDataForModel {
  SummaryStatistics stats_;
  IterationCounts standard_counts_;
  IterationCounts log_rank_counts_;
  IterationCounts peto_counts_;
  IterationCounts score_method_counts_;
  IterationCounts satterthwaite_counts_;
};

// Used for dual regression.
struct IterationDataHolder {
  IterationDataForModel model_one_;
  IterationDataForModel model_two_;
  DualStatistics dual_stats_;
};

// Sanity-Check a vector of Covariances are all non-negative; or alternatively,
// that a Covariance Matrix has non-negative values on the diagonal.
extern bool NegativeVariance(const MatrixXd& variance);

// Takes in a Matrix of values, and returns a new Matrix of the same dimensions,
// where the new matrix has column-standardized each value: Replace each given
// value x with (x - \mu) / \sigma, where \mu is the column mean, and \sigma is
// the column standard deviation. Also returns the mean and standard deviation
// of each column (so that user can de-standardize values later).
extern bool StandardizeMatrix(
    const MatrixXd& input_values,
    vector<pair<double, double>>* mean_and_std_dev_by_column,
    MatrixXd* output_values, string* error_msg);

// Uses the input 'coordinate_mean_and_std_dev' vector as a legend to
// unstandardize the 'input' vector. If 'coordinate_mean_and_std_dev'
// is empty, just copy input vector to output.
extern bool UnstandardizeVector(
    const VariableNormalization std_type,
    const VectorXd& input,
    const vector<tuple<bool, double, double>>& coordinate_mean_and_std_dev,
    VectorXd* output, string* error_msg);
// Same as above, but for a Matrix. Assumes that the matrix is square, and in
// particular, the coordinate_mean_and_std_dev vector applies to both the rows
// and columns.
extern bool UnstandardizeMatrix(
    const VariableNormalization std_type,
    const MatrixXd& input,
    const vector<tuple<bool, double, double>>& coordinate_mean_and_std_dev,
    MatrixXd* output, string* error_msg);

// Uses the provided estimates and variances to compute z-statistics and p-values.
extern bool GenerateSummaryStatistics(
    const VectorXd& estimates, const VectorXd& est_variance,
    VectorXd* z_statistics, VectorXd* p_values);
// Uses the provided estimates and variances (from the diagonal of the provided
// Covariance matrix) to compute z-statistics and p-values.
extern bool GenerateSummaryStatisticsFromCovarianceMatrix(
    const VectorXd& estimates, const MatrixXd& var_cov_matrix,
    VectorXd* z_statistics, VectorXd* p_values);

// Looking at the fields in SummaryStatistics, some of them are naturally
// filled by the underlying algorithm (e.g. cox, linear regression, etc.):
// estimates_, variances_, final_info_matrix_, etc; while others are only
// really needed for simulations: t_statistics_, estimates_squared_, etc.
// This function uses the values of the first category of fields (those
// populated via an underlying algorithm) to populate the 2nd category.
extern bool CompleteSummaryStatistics(
    const AnalysisParams& params, SummaryStatistics* summary_stats);
// Same as above, but just for fields of SummaryStatistics related to
// "Standard" analysis.
extern bool CompleteStandardSummaryStatistics(SummaryStatistics* summary_stats);
// Same as above, but just for fields of SummaryStatistics related to
// "Log-Rank" analysis.
extern bool CompleteLogRankSummaryStatistics(SummaryStatistics* summary_stats);
// Same as above, but just for fields of SummaryStatistics related to
// "Peto" analysis.
extern bool CompletePetoSummaryStatistics(SummaryStatistics* summary_stats);

// When an IterationDataForModel model is first declared, use this
// to set sizes of its vector member fields.
extern void InitializeIterationDataForModel(
    const int p, const AnalysisParams& params,
    IterationDataForModel* stats);
extern void InitializeIterationDataHolder(
    const int p_one, const int p_two,
    const AnalysisParams& params_one, const AnalysisParams& params_two,
    IterationDataHolder* stats);

// Adds the relevant fields of stats_one (as determined by 'params') to stats_two.
extern bool AddSummaryStatistics(
    const AnalysisParams& params,
    const SummaryStatistics& stats_one, SummaryStatistics* stats_two,
    string* error_msg);
// Same as above, but with specific booleans to indicate which fields to add.
extern bool AddSummaryStatistics(
    const bool standard, const bool log_rank_or_peto, const bool robust,
    const SummaryStatistics& stats_one, SummaryStatistics* stats_two,
    string* error_msg);
// Adds all the fields of stats_one that are related to 'Standard Analysis' to stats_two.
extern bool AddStandardSummaryStatistics(
    const SummaryStatistics& stats_one, SummaryStatistics* stats_two,
    string* error_msg);
// Adds all the fields of stats_one that are related to 'Robust' to stats_two.
extern bool AddRobustSummaryStatistics(
    const SummaryStatistics& stats_one, SummaryStatistics* stats_two,
    string* error_msg);
// Adds all the fields of stats_one that are related to 'Log-Rank' to stats_two.
extern bool AddLogRankAndPetoSummaryStatistics(
    const SummaryStatistics& stats_one, SummaryStatistics* stats_two,
    string* error_msg);
// Adds all the fields of stats_one that are related to 'Dual Robust' to stats_two.
extern bool AddDualRobustStatistics(
    const DualStatistics& stats_one, DualStatistics* stats_two,
    string* error_msg);
// Adds all the fields of stats_one that are related to 'Dual Log-Rank' to stats_two.
extern bool AddDualLogRankStatistics(
    const DualStatistics& stats_one, DualStatistics* stats_two);
// Sanity check the 'Standard Analysis' fields inside the two SummaryStatistics
// objects all have the same size their counterparts in the other (so e.g. the
// two summary statistics can be combined).
extern bool SanityCheckCompatibleStandardSummaryStatistics(
    const SummaryStatistics& stats_one, const SummaryStatistics& stats_two,
    string* error_msg);
// Sanity check the 'LogRank' fields inside the two SummaryStatistics objects all have
// the same size their counterparts in the other (so e.g. the two summary
// statistics can be combined).
extern bool SanityCheckCompatibleLogRankAndPetoSummaryStatistics(
    const SummaryStatistics& stats_one, const SummaryStatistics& stats_two,
    string* error_msg);
// Sanity check the 'Robust' fields inside the two SummaryStatistics objects all have
// the same size their counterparts in the other (so e.g. the two summary
// statistics can be combined).
extern bool SanityCheckCompatibleRobustSummaryStatistics(
    const SummaryStatistics& stats_one, const SummaryStatistics& stats_two,
    string* error_msg);

// Uses 'params' to determine which fields of SummaryStatistics to print to file.
extern bool PrintSummaryStatistics(
    const ModelAndDataParams& params,
    const SummaryStatistics& summary_stats,
    string* output);
// Same as above, but prints to an outfile instead of a string.
extern bool PrintSummaryStatistics(
    ofstream& outfile, const ModelAndDataParams& params,
    const SummaryStatistics& summary_stats);
// Similar to above, for DualStatistics.
extern void PrintDualStatistics(
    ofstream& outfile, const DualStatistics& dual_stats);

extern void GetLogRankSummaryStatistics(
    const SummaryStatistics& stats, string* output);

extern void GetPetoSummaryStatistics(
    const SummaryStatistics& stats, string* output);

extern void GetScoreMethodSummaryStatistics(
    const SummaryStatistics& stats, string* output);

extern bool GetStandardSummaryStatistics(
    const SummaryStatisticsPrintOptions& print_options,
    const vector<string>& header,
    const SummaryStatistics& stats, string* output);

extern bool ComputeMedian(const vector<double>& input, double* median);
// Same as above, but for a vector of values. In particular, if k is the size of
// the outer vector and i is the size of the inner vector, we will find i medians
// by finding the center element (w.r.t. k). Stated differently, view the input
// data structure as a matrix with k rows and i columns; then we find the column
// medians.
extern bool ComputeMedians(
    const vector<vector<double>>& input, vector<double>* median);

}  // namespace math_utils

#endif
