// Date: April 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description: Tools for running Cox Proportional Hazards regression.

#ifndef COX_DUAL_REGRESSION_H
#define COX_DUAL_REGRESSION_H

#include "FileReaderUtils/read_file_structures.h"
#include "FileReaderUtils/read_file_utils.h"
#include "MathUtils/data_structures.h"
#include "MathUtils/number_comparison.h"
#include "MathUtils/statistics_utils.h"

#include <cstdlib>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace file_reader_utils;
using namespace math_utils;
using namespace std;

namespace regression {

const int MAX_ITERATIONS = 50;
const int MAX_HALVING_ATTEMPTS = 10;

// Holds the values (regression coefficients, log-likelihood, Score Function,
// Information Matrix) needed to computer the next iteration of Stratified Cox
// PH Model. There will be a corresponding StratificationData object for each,
// stratification.
struct StratificationData {
  // Holds the transition indices for this strata.
  // For example, given \tilde{T} times:
  //   1.5, 2, 2, 4, 5, 5, 5, 6, 7, 7, 7, 7
  // The transition indices would be [0, 1, 3, 4, 7, 8]
  vector<int> transition_indices_;
  double log_likelihood_;
  VectorXd logistic_eq_;
  VectorXd exp_logistic_eq_;
  VectorXd partial_sums_;
  VectorXd kaplan_meier_estimators_;  // The K-M Estimator for each row in this strata.
  // The following vector is used to compute revised information matrix
  // I*, which includes an extra multiple for handling ties. In particular,
  // the following field is a VectorXd for strata k, representing:
  //   ties_constant_(k,j) = [R_k(T_{kj}) - D_k(T_{kj})] / [R_k(T_{kj}) - 1]
  // where:
  //   R_k(T_{kj}) := S^0_k(0, T_{kj}) = \sum_i^{n_k} I(T_{ki} >= T_{kj})
  //   D_k(T_{kj}) := \sum_i^{n_k} \Delta_{ki} * I(T_{ki} = T_{kj})
  VectorXd ties_constant_;
  VectorXd score_function_;
  MatrixXd information_matrix_;
  MatrixXd indep_vars_;
  vector<CensoringData> dep_vars_;
  MatrixXd w_vectors_;
  MatrixXd log_rank_w_vectors_;
};

class CoxDualRegression {
 public:
  // Computes the Kaplan-Meier estimator for the input data:
  //   KME(t) = \Pi_{T_j < t} (1 - (D_j / S^0(0, T_j))),
  // where:
  //   - The product is taken over *distinct* times (i.e. in case of ties
  //     T_i = T_j, only one of T_i (or T_j) contributes to the product
  //   - Also, the product uses equality (T_j <= t) iff KaplanMeierEstimatorType
  //     is RIGHT_CONTINUOUS.
  //   - D_j represents \Delta_j (status of j^th sample) in the case of no-ties;
  //     in case there are multiple {T_i} = T_j, then D_j is the sum of all such
  //     \Delta_i
  //   - S^0 is the 0^th partial sum (see 'ComputePartialSums' below):
  //       S^0(\beta, t) = \sum_j (I(T_j >= t) * (exp_logistic_eq)_j
  //     where (exp_logistic_eq)_j is the exponential of the logistic equation:
  //       (exp_logistic_eq)_j = exp( <\beta | X_j> )
  static bool ComputeKaplanMeierEstimator(
      const KaplanMeierEstimatorType& kme_type,
      const VectorXd& S0, const vector<CensoringData>& dep_var,
      VectorXd* estimator);
  // Same as above, but computes S0 first.
  static bool ComputeKaplanMeierEstimator(
      const KaplanMeierEstimatorType& kme_type,
      const vector<CensoringData>& dep_var, VectorXd* estimator);
  // Same as above, but computes the estimators on a per-strata basis.
  static bool ComputeKaplanMeierEstimator(
      const KaplanMeierEstimatorType& kme_type,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      map<int, VectorXd>* strata_km_estimators);

  // Computes the Score Function U(\beta).
  static bool ComputeScoreFunction(
      const int n, const int p,
      const VectorXd& beta,
      const VectorXd& kaplan_meier_estimators,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      const vector<int>& reverse_sorted_indices,
      VectorXd* score_function);
  // DEPRECATED. Use above method, especially if it will be used iteratively,
  // e.g. when solving for Cox PH or Score Method.
  // Same as above, but without the hint of Samples sorted by order.
  static bool ComputeScoreFunction(
      const int n, const int p,
      const VectorXd& beta,
      const VectorXd& kaplan_meier_estimators,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      VectorXd* score_function);
  // Same as above, but takes in kme_type instead of already computed KME values
  // (KME values will be computed from the provided dep_var, if necessary).
  static bool ComputeScoreFunction(
      const int n, const int p, const KaplanMeierEstimatorType& kme_type,
      const VectorXd& beta,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      const vector<int>& reverse_sorted_indices,
      VectorXd* score_function);
  // DEPRECATED. Use above method, especially if it will be used iteratively,
  // e.g. when solving for Cox PH or Score Method.
  // Same as above, but without the hint of Samples sorted by order.
  static bool ComputeScoreFunction(
      const int n, const int p, const KaplanMeierEstimatorType& kme_type,
      const VectorXd& beta,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      VectorXd* score_function);
  // Same as above, but for stratified data.
  static bool ComputeScoreFunction(
      const int n, const int p,
      const VectorXd& beta,
      const map<int, VectorXd>& strata_kaplan_meier_estimators,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      map<int, VectorXd>* strata_to_score_function);
  // Same as above, but takes in kme_type instead of already computed KME values
  // (KME values will be computed from the provided dep_var, if necessary).
  static bool ComputeScoreFunction(
      const int n, const int p, const KaplanMeierEstimatorType& kme_type,
      const VectorXd& beta,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      map<int, VectorXd>* strata_to_score_function);

  // Computes the Information Matrix V(\beta).
  static bool ComputeInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const VectorXd& beta,
      const VectorXd& kaplan_meier_estimators,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      const vector<int>& reverse_sorted_indices,
      MatrixXd* info_matrix);
  // DEPRECATED. Use above method, especially if it will be used iteratively,
  // e.g. when solving for Cox PH or Score Method.
  // Same as above, but without the hint of Samples sorted by order.
  static bool ComputeInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const VectorXd& beta,
      const VectorXd& kaplan_meier_estimators,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      MatrixXd* info_matrix);
  // Same as above, but takes in kme_type instead of already computed KME values
  // (KME values will be computed from the provided dep_var, if necessary).
  static bool ComputeInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const KaplanMeierEstimatorType& kme_type,
      const VectorXd& beta,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      const vector<int>& reverse_sorted_indices,
      MatrixXd* info_matrix);
  // DEPRECATED. Use above method, especially if it will be used iteratively,
  // e.g. when solving for Cox PH or Score Method.
  // Same as above, but without the hint of Samples sorted by order.
  static bool ComputeInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const KaplanMeierEstimatorType& kme_type,
      const VectorXd& beta,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      MatrixXd* info_matrix);
  // Same as above, but for stratified data.
  static bool ComputeInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const VectorXd& beta,
      const map<int, VectorXd>& strata_kaplan_meier_estimators,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      map<int, MatrixXd>* strata_to_info_matrix);
  // Same as above, but takes in kme_type instead of already computed KME values
  // (KME values will be computed from the provided dep_var, if necessary).
  static bool ComputeInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const KaplanMeierEstimatorType& kme_type,
      const VectorXd& beta,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      map<int, MatrixXd>* strata_to_info_matrix);

  // Same as above, but combines computation of ScoreFunction and
  // Information matrix, so that the partial sums don't have to be
  // computed twice (saves time over running the above methods for
  // Score Function and Information Matrix separately).
  // There are two separate Kaplan-Meier estimators passed in, in
  // case you want to apply them different to score fn vs. info-matrix
  // (e.g. for log-rank, we'll use KME directly for score function,
  // but their squares for Info matrix).
  static bool ComputeScoreFunctionAndInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const VectorXd& beta,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      VectorXd* score_function, MatrixXd* info_matrix);
  // Same as above, but for unstratified data (will be called as a sub-routine
  // for the stratified version above).
  static bool ComputeScoreFunctionAndInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const VectorXd& beta,
      const VectorXd& kaplan_meier_estimators_for_score_fn,
      const VectorXd& kaplan_meier_estimators_for_info_matrix,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      const vector<int>& reverse_sorted_indices,
      VectorXd* score_function, MatrixXd* info_matrix);
  // DEPRECATED. Use above method, especially if it will be used iteratively,
  // e.g. when solving for Cox PH or Score Method.
  // Same as above, but without the hint of Samples sorted by order.
  static bool ComputeScoreFunctionAndInformationMatrix(
      const int n, const int p,
      const bool use_ties_constant,
      const VectorXd& beta,
      const VectorXd& kaplan_meier_estimators_for_score_fn,
      const VectorXd& kaplan_meier_estimators_for_info_matrix,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      VectorXd* score_function, MatrixXd* info_matrix);

  // Runs Dual Regression End-to-End: Parses command-line arguments of model,
  // subgroup, and strata, and runs dual regression accordingly.
  // PHB HERE. Determine which API to RunCoxRegression is used, and get rid of the other.
  // Also, get rid of passing in 'criteria' (either create this within RunCoxRegression,
  // or add it as a field of ModelAndDataParams).
  static bool RunCoxRegression(
      const VectorXd& actual_beta,
      const ModelAndDataParams& params,
      map<int, int>* orig_row_to_unsorted_model_row,
      SummaryStatistics* summary_stats);
  // Same as above, but for non-simulations (actual data).
  static bool RunCoxRegression(
      const ModelAndDataParams& params,
      map<int, int>* orig_row_to_unsorted_model_row,
      SummaryStatistics* summary_stats) {
    return RunCoxRegression(
        VectorXd(), params,
        orig_row_to_unsorted_model_row, summary_stats);
  }

  // Gets the rows (Subjects) common to both models, and Computes the
  // Dual Covariance Matrix (if Robust Analysis is being performed) and the
  // Dual Log Rank Covariance (if log-rank or peto analysis is being performed).
  static bool ComputeDualCovariance(
      const map<int, int>& orig_row_to_model_one_row,
      const map<int, int>& orig_row_to_model_two_row,
      const AnalysisParams& analysis_params,
      const SummaryStatistics& stats_one, const SummaryStatistics& stats_two,
      set<int>* common_rows, DualStatistics* dual_stats);


  // Computes Dual Covariance Matrix (between Regression Coefficients of
  // two models, intersecting over the data common to both of them):
  //  I(\hat{\beta})^-1 * B(\hat{\beta}, \hat{\gamma}) * \tilde{I}(\hat{\gamma})
  // where:
  //   B(\beta, \gamma) := \sum_i^{\tilde{n}} W_i(\beta) * \tilde{W}^T_i(\gamma)
  // (See notation on page 5 of "Simultaneous Inference on Treatment Effects
  //  in Factorial Survival Studies).
  // orig_row_to_unsorted_model_row_[one | two] is Keyed according the row's
  // index (according to the original input data), and Values are the row's
  // index among the (unsorted) model (often, the Key and Value are the same,
  // but in general, there may be gaps, for rows that were excluded due to
  // bad data, not being in a subgroup, etc).
  static bool ComputeDualCovarianceMatrix(
      const set<int>& common_rows,
      const map<int, int>& orig_row_to_unsorted_model_row_one,
      const map<int, int>& orig_row_to_unsorted_model_row_two,
      const MatrixXd& w_vectors_one,
      const MatrixXd& w_vectors_two,
      const MatrixXd& final_info_matrix_inverse_one,
      const MatrixXd& final_info_matrix_inverse_two,
      DualStatistics* dual_stats);

  // Computes Dual Log Rank, storing result in the two provided doubles.
  static bool ComputeDualLogRank(
      const set<int>& common_rows,
      const double& log_rank_variance_one,
      const double& log_rank_variance_two,
      const map<int, int>& orig_row_to_unsorted_model_row_one,
      const map<int, int>& orig_row_to_unsorted_model_row_two,
      const MatrixXd& w_vectors_one,
      const MatrixXd& w_vectors_two,
      DualStatistics* dual_stats);

  // Given the censoring data in dep_var, sorts it based on censoring time.
  static bool GetSortLegend(
      const vector<CensoringData>& dep_vars,
      map<int, int>* sorted_row_to_orig_row);

  // Given mappings from original data row to strata (for two models),
  // determines the rows in both models.
  static void GetCommonRows(
      const map<int, int>& orig_row_to_unsorted_model_row_one,
      const map<int, int>& orig_row_to_unsorted_model_row_two,
      set<int>* common_rows);

  // Prints Dual Covariance Matrix to the indicated file.
  static void PrintDualCovarianceMatrix(
      const set<int>& common_rows,
      const vector<vector<string>>& titles,
      const DualStatistics& dual_stats,
      string* output);
  // Same as above, but doesn't print rows.
  static void PrintDualCovarianceMatrix(
      const vector<vector<string>>& titles,
      const DualStatistics& dual_stats, string* output);

  // Prints dual log-rank Covariance and Correlation to 'output'.
  // 'is_weighted' means the Kaplan-Meier Estimator was used in
  // the computation of log-rank.
  static bool PrintDualLogRank(
      const bool is_weighted, const DualStatistics& dual_stats,
      string* output);

 private:
  // Computes the "S-Vectors" S^0, S^1, and S^2 (see notation on page 4 of
  // "Simulaneous Inference on Treatment Effects in Factorial Survival Studies").
  // Reverse_sorted_indices should give the order of the Samples, as sorted by
  // time, from max to min (i.e. the first element of reverse_sorted_indices
  // should be the index of the Sample has the largest time); ties are broken
  // arbitrarily.
  static bool ComputePartialSums(
      const int n, const int p,
      const VectorXd& beta,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      const vector<int>& reverse_sorted_indices,
      vector<double>* S0, vector<VectorXd>* S1, vector<MatrixXd>* S2);
  // DEPRECATED. Use above method, especially if it will be used iteratively,
  // e.g. when solving for Cox PH or Score Method.
  // Same as above, but without the hint of Samples sorted by order.
  static bool ComputePartialSums(
      const int n, const int p,
      const VectorXd& beta,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      vector<double>* S0, vector<VectorXd>* S1, vector<MatrixXd>* S2);

  // Evaluates a U^2(x)/I(x).
  static bool EvaluateScoreMethodFunction(
      const int n, const int p, const double& x,
      const VectorXd& score_function, const MatrixXd& info_matrix,
      double* value);
  // Same as above, but first computes U and I.
  static bool EvaluateScoreMethodFunction(
      const int n, const int p,
      const bool use_ties_constant,
      const double& x,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      double* value);
  // Same as above, but on unstratified data.
  static bool EvaluateScoreMethodFunction(
      const int n, const int p,
      const bool use_ties_constant,
      const double& x,
      const VectorXd& kaplan_meier_estimators_for_score_fn,
      const VectorXd& kaplan_meier_estimators_for_info_matrix,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      const vector<int>& reverse_sorted_indices,
      double* value) {
    // Create containers that wrap a single strata.
    map<int, pair<vector<CensoringData>, MatrixXd>> strata_vars;
    strata_vars.insert(make_pair(0, make_pair(dep_var, indep_vars)));
    map<int, VectorXd> strata_kaplan_meier_estimators_for_score_fn;
    strata_kaplan_meier_estimators_for_score_fn.insert(
        make_pair(0, kaplan_meier_estimators_for_score_fn));
    map<int, VectorXd> strata_kaplan_meier_estimators_for_info_matrix;
    strata_kaplan_meier_estimators_for_info_matrix.insert(
        make_pair(0, kaplan_meier_estimators_for_info_matrix));
    map<int, vector<int>> strata_reverse_sorted_indices;
    strata_reverse_sorted_indices.insert(
        make_pair(0, reverse_sorted_indices));
    return EvaluateScoreMethodFunction(
        n, p, use_ties_constant, x, strata_kaplan_meier_estimators_for_score_fn,
        strata_kaplan_meier_estimators_for_info_matrix, strata_vars,
        strata_reverse_sorted_indices, value);
  }

  // Computes the width of the Score Method's confidence interval, by
  // finding the roots of U^2(x)/I(x) = 1.96^2.
  static bool ComputeScoreMethodCi(
      const int n, const int p,
      const bool use_ties_constant,
      const double& beta, const double& hat_std_err,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      double* score_method_ci_left, double* score_method_ci_right);
  // Same as above, for unstratified data.
  static bool ComputeScoreMethodCi(
      const int n, const int p,
      const bool use_ties_constant,
      const double& beta, const double& hat_std_err,
      const VectorXd& kaplan_meier_estimators_for_score_fn,
      const VectorXd& kaplan_meier_estimators_for_info_matrix,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      const vector<int>& reverse_sorted_indices,
      double* score_method_ci_left, double* score_method_ci_right) {
    // Create containers that wrap a single strata.
    map<int, pair<vector<CensoringData>, MatrixXd>> strata_vars;
    strata_vars.insert(make_pair(0, make_pair(dep_var, indep_vars)));
    map<int, VectorXd> strata_kaplan_meier_estimators_for_score_fn;
    strata_kaplan_meier_estimators_for_score_fn.insert(
        make_pair(0, kaplan_meier_estimators_for_score_fn));
    map<int, VectorXd> strata_kaplan_meier_estimators_for_info_matrix;
    strata_kaplan_meier_estimators_for_info_matrix.insert(
        make_pair(0, kaplan_meier_estimators_for_info_matrix));
    map<int, vector<int>> strata_reverse_sorted_indices;
    strata_reverse_sorted_indices.insert(
        make_pair(0, reverse_sorted_indices));
    return ComputeScoreMethodCi(
        n, p, use_ties_constant, beta, hat_std_err,
        strata_kaplan_meier_estimators_for_score_fn,
        strata_kaplan_meier_estimators_for_info_matrix,
        strata_vars, strata_reverse_sorted_indices,
        score_method_ci_left, score_method_ci_right);
  }

  // Given a linear model described by values in params.dep_vars_.dep_vars_cox_
  // and params.linear_term_values_, performs cox (dual) regression to compute
  // an estimate of the coefficient for each linear term, as well as the
  // Variance, SEE, t-statistic, and p-value for each of these Estimates.
  // Uses convergence_criteria to determine stopping condition for N-R.
  // Stores values in stats (if not null), and also prints values to
  // 'output_filename' (if not null).
  static bool Compute(
      const ModelAndDataParams& params,
      const string* output_filename,
      const map<int, int>& row_to_strata,
      map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row,
      SummaryStatistics* stats);

  // Same as above, for printing values to output_filename (all other pointers
  // are null).
  static bool Compute(
      const ModelAndDataParams& params,
      const string* output_filename,
      const map<int, int>& row_to_strata,
      map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row) {
    return Compute(
        params, output_filename, row_to_strata,
        unsorted_model_row_to_strata_index_and_row, nullptr);
  }
  // Same as above, but populates the given input vectors instead of printing
  // to file.
  static bool Compute(
      const ModelAndDataParams& params,
      const map<int, int>& row_to_strata,
      map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row,
      SummaryStatistics* stats) {
    return Compute(
        params, nullptr, row_to_strata,
        unsorted_model_row_to_strata_index_and_row, stats);
  }

  // Private interface for RunCoxRegression().
  static bool RunCoxRegression(
      const ModelAndDataParams& params,
      map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row,
      SummaryStatistics* summary_stats);

  // The following functions are used to compute the Score Method width.
  // The strategy to find such a value will be as follows:
  //   0) Define f(x) = U^2(x)/I(x). The goal is to find the two values of
  //      'x' such that f(x) = 3.8415.
  //   1) First, we search for an 'x' such that f(x) < 3.8415.
  //      a) Try f(beta). Usually this will already produce a value
  //         less than 3.8415, since f(\hat{beta}) = 0 (by construction of
  //         \hat{beta}, and \hat{beta} is an approximation of \beta.
  //      b) If (1) fails, then we want to find points L1 to the left and R1 right
  //         of beta such that f(L1) > f(beta) and f(R1) > f(beta).
  //         This will guarantee that the min of f(x) is between those two (since
  //         f(x) is parabolic).
  //         Find L1 and R1 as follows: one of them will be immediate (by
  //         parabolic nature of f(x), either f(L1) > f(beta) or f(R1) > f(beta) (or both).
  //         So in case one of them evaluates to less than f(beta), keep moving away
  //         until you evaluate to a value that is bigger than f(beta).
  //      c) Once we've identified L1 and R1, we take the midpoint M1. If f(M1) < 3.8415,
  //         then done. Otherwise, there are two cases:
  //           i)  f(M1) is less than BOTH f(L1) and f(R1)
  //           ii) f(M1) is bigger than one of them, smaller than the other
  //         Note that by the fact that f(x) is parabolic, it is not possible that
  //         f(M1) is LARGER than BOTH f(L1) AND f(R1). The next iteration will
  //         either be on the interval [L1, M1] or [M1, R1], depending on which side
  //         of the min M1 lies on. If we're in case (3ii), then it is easy to know
  //         which side the min is on (it must be closer to whichever point {L1, R1}
  //         that evaluates to a smaller value than M1 does). Otherwise, we'll have
  //         to keep bisecting (in both directions) until we find a point P1 that
  //         evaluates to smaller than f(M1). WLOG suppose L1 < P1 < M1. Then the next
  //         iteration would be on [L1, M1].
  //   2) Search for a root to the right of the value found in step 1 (a good
  //      starting guess is \hat{\beta} + 1.96 * \hat{SE}).
  //   3) Search for a root to the left of the value found in step 1 (a good
  //      starting guess is \hat{\beta} - 1.96 * \hat{SE}).
  // Step (1a) is done in FindScoreMethodCiNegValue, Step (1b) is done in
  // DoScoreMethodStepTwo, and Step (1c) is done in DoScoreMethodStepThree.
  // Steps (2) and (3) are done in FindScoreMethodCiRoot.
  static bool FindScoreMethodCiNegValue(
      const int n, const int p,
      const bool use_ties_constant,
      const double& beta,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      double* close_to_min);
  // Same as above, for unstratified data.
  static bool FindScoreMethodCiNegValue(
      const int n, const int p,
      const bool use_ties_constant,
      const double& beta,
      const VectorXd& kaplan_meier_estimators_for_score_fn,
      const VectorXd& kaplan_meier_estimators_for_info_matrix,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      const vector<int>& reverse_sorted_indices,
      double* close_to_min) {
    // Create containers that wrap a single strata.
    map<int, pair<vector<CensoringData>, MatrixXd>> strata_vars;
    strata_vars.insert(make_pair(0, make_pair(dep_var, indep_vars)));
    map<int, VectorXd> strata_kaplan_meier_estimators_for_score_fn;
    strata_kaplan_meier_estimators_for_score_fn.insert(
        make_pair(0, kaplan_meier_estimators_for_score_fn));
    map<int, VectorXd> strata_kaplan_meier_estimators_for_info_matrix;
    strata_kaplan_meier_estimators_for_info_matrix.insert(
        make_pair(0, kaplan_meier_estimators_for_info_matrix));
    map<int, vector<int>> strata_reverse_sorted_indices;
    strata_reverse_sorted_indices.insert(
        make_pair(0, reverse_sorted_indices));
    return FindScoreMethodCiNegValue(
        n, p, use_ties_constant, beta,
        strata_kaplan_meier_estimators_for_score_fn,
        strata_kaplan_meier_estimators_for_info_matrix,
        strata_vars, strata_reverse_sorted_indices, close_to_min);
  }
  static bool DoScoreMethodStepTwo(
      const int num_iterations,
      const int n, const int p,
      const bool use_ties_constant,
      const double& beta, const double& f_beta,
      const double& guess_left, const double& guess_right,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      double* L1, double* R1);
  static bool DoScoreMethodStepThree(
      const int num_iterations,
      const int n, const int p,
      const bool use_ties_constant,
      const double& left, const double& right, const double& midpoint,
      const double& guess_left, const double& guess_right,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      double* close_to_min);

  // Finds a root of U^2(x)/I(x) = 3.8416.
  static bool FindScoreMethodCiRoot(
      const int num_iterations,
      const int n, const int p,
      const bool use_ties_constant,
      const double& prev_neg, const double& prev_pos,
      const double& beta, const double& guess,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_score_fn,
      const map<int, VectorXd>& strata_kaplan_meier_estimators_for_info_matrix,
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      const map<int, vector<int>>& strata_reverse_sorted_indices,
      double* root);

  static bool MatrixHasNanTerm(const MatrixXd& mat);

  // km_estimate has already been populated with the relevant KME values
  // for each sample. However, if we want to use the right-continuous
  // version of the K-M curve (see kUseRightContinuousKme in the .cpp file),
  // we need to use the estimator of the previous (w.r.t. time) sample.
  // This method finds the index of dep_var that has time closest to (below)
  // data.time_, and looks of the km_estimate for that index.
  static double GetKmeOfPreviousTimepoint(
      const CensoringData& data, const vector<CensoringData>& dep_var,
      const VectorXd& km_estimate);

  // Computes the fraction of Subjects that have Status = 0 (Alive).
  static void GetFractionAlive(
      const vector<CensoringData>& dep_var, double* fraction_alive);
  // Same as above, but for Simulations (so computes Fraction alive on
  // each partition, where a partition is defined as the largest
  // contiguous set of Subjects that have consistent Variable
  // Sampling Parameters.
  static bool GetFractionAlive(
      const vector<pair<int, int>>& sample_partitions,
      const map<int, set<int>>* subgroup_index_to_its_rows,
      const vector<CensoringData>& dep_var,
      vector<double>* fraction_alive);

  // Reads the input dimensions (number samples, number of terms on RHS of
  // logistic equation).
  static bool GetDimensions(const vector<VectorXd>& indep_vars,
                            int* n, int* p) {
    if (indep_vars.empty()) return false;
    *p = indep_vars[0].size();
    *n = indep_vars.size();
    return true;
  }
  // Same as above, for different API (vector<VectorXd> vs. MatrixXd).
  static bool GetDimensions(const MatrixXd& indep_vars,
                            int* n, int* p) {
    if (indep_vars.size() == 0) return false;
    *p = indep_vars.cols();
    *n = indep_vars.rows();
    return true;
  }

  static bool InitializeStratificationData(
      const int p,
      const vector<CensoringData>& dep_vars,
      const MatrixXd& linear_term_values,
      const map<int, int>& row_to_strata,
      map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row,
      vector<StratificationData>* stratification_data);

  // Creates a new StratificationData object, pushing it to the end
  // of the provided vector. Initializes the Vector/Matrix fields
  // of the new object, to have the indicated dimensions.
  static void PushBackStratificationIndices(
      const int n, const int p,
      vector<StratificationData>* stratification_data);

  // Fill stratification_data with the appropriate number of strata,
  // setting each StratificationData object's 'stratification_indices' field.
  // Also initializes the Vector/Matrix fields of each StratificationData.
  static void PopulateStratificationIndices(
      const int p, const map<int, int>& row_to_strata,
      vector<StratificationData>* stratification_data);

  // Populates stratification_data->ties_constant_ (see comments about this
  // field where it is declared above in StratificationData struct).
  static void SetTiesConstant(
      const vector<CensoringData>& dep_vars, VectorXd* ties_constant);
  // Same as above, but does it for all strata.
  static void SetTiesConstant(vector<StratificationData>* stratification_data);

  // Populates transition_indices, which (for each strata) keeps track of the 
  // (sorted) indices {i} for which T_i > T_{i-1} (as opposed to equality).
  static bool GetTransitionIndices(
      const vector<CensoringData>& dep_vars, vector<int>* transition_indices);

  // Return the data at index, converting it first to CensoringData format,
  // if necessary.
  static CensoringData GetDepVar(
      const vector<CensoringData>& dep_vars, const int index);

  // Sorts the indep_vars based on min(survival_time, censoring_time), and
  // places the resulting sorted data into stratification_data. Also populates
  // transition_indices with the indices (with respect to the sorted data)
  // for which T_i \neq T_{i-1}. Note that T_1 is always a member of the set,
  // and e.g. if all times were equal, it would be the only element of the set.
  static bool SortInputByTime(
      const vector<CensoringData>& dep_vars,
      const MatrixXd& linear_term_values,
      const map<int, int>& row_to_strata,
      map<int, pair<int, int>>* unsorted_model_row_to_strata_index_and_row,
      vector<StratificationData>* stratification_data);

  // Same as ComputeKaplanMeierEstimator in the public sectionabove,
  // but takes advantage of values already computed in
  // the process of computing Regression coefficients.
  static bool ComputeKaplanMeierEstimator(
      const KaplanMeierEstimatorType& kme_type,
      const VectorXd& partial_sums, const vector<int>& transition_indices,
      const vector<CensoringData>& dep_var,
      VectorXd* estimator);

  // Given a mapping from original data row to strata index, partitions the
  // dependent and independent variables into separate data structures for
  // each strata.
  static bool GetStrataVars(
      const map<int, pair<int, int>>& row_to_strata,
      const vector<CensoringData>& dep_var,
      const MatrixXd& indep_vars,
      map<int, pair<vector<CensoringData>, MatrixXd>>* strata_vars);

  // Uses the min(survival, censoring) times within each strata to sort the
  // Samples within each strata based on time.
  static bool GetStrataSortedIndices(
      const map<int, pair<vector<CensoringData>, MatrixXd>>& strata_vars,
      map<int, vector<int>>* strata_reverse_sorted_indices);

  // Populates orig_row_to_unsorted_model_row.
  // Discussion: Data from the input file goes through a series of transitions:
  //   1) Read input vector<vector<DataHolder>>, the 'data_values'
  //   2) Remove NA rows
  //   3) Remove rows not in a subgroup
  //   4) Sort based on time
  //   5) Break into Strata
  // Note that we don't have access to the data after (1) or (2); after (3), the
  // data is stored in ModelAndDataParams.linear_term_values_; we don't have
  // access to data after (4); after (5), the data is stored in
  // StratificationData.indep_vars_.
  // When doing Dual comparison, we need to only use data (Subjects) that are
  // present in both models. Since the two models may remove different rows
  // in Step (3) above, we need a way to get the common rows (Subjects), and
  // then know which row (say w.r.t. data in format after Step (1)) each
  // common row index corresponds to (say w.r.t. data in format after Step(3))
  // for each Model.
  static bool ConstructMappingFromOrigRowToUnsortedModelRow(
    const ModelAndDataParams& params,
    map<int, int>* orig_row_to_unsorted_model_row);

  // Stores the partial sums:
  //   S^0_i(\beta) := \sum_{j \in [1..n]} I(T_j >= T_i) * exp(\beta * X_j)
  // Note that we don't have to store this for all i, just at the i's in
  // 'transition_indices'; thus partial_sums will be a vector of same length
  // as transition_indices; with the first element corresponding to S_1
  // (note '1' is always transition_indices[0]), the second element is S_K
  // (where K := transition_indices[1]), ..., the final element is S_L
  // (where L := transition_indices[end]). 
  static bool ComputePartialSums(
      const vector<int>& transition_indices,
      const VectorXd& exp_logistic_eq,
      VectorXd* partial_sums);

  // The term exp(\beta^T * X_i) appears in many variables. We compute it once
  // here, storing the result in exp_logistic_eq.
  static bool ComputeExponentialOfLogisticEquation(
      const VectorXd& beta_hat,
      const vector<VectorXd>& indep_vars,
      VectorXd* logistic_eq,
      VectorXd* exp_logistic_eq);
  // Same as above, with different API: indep_vars is Eigen Matrix, not vector.
  static bool ComputeExponentialOfLogisticEquation(
      const VectorXd& beta_hat,
      const MatrixXd& indep_vars,
      VectorXd* logistic_eq,
      VectorXd* exp_logistic_eq);

  // Compute the log-likelihood l(\beta).
  static bool ComputeLogLikelihood(
      const VectorXd& logistic_eq,
      const VectorXd& exp_logistic_eq,
      const VectorXd& partial_sums,
      const vector<int>& transition_indices,
      const vector<CensoringData>& dep_var,
      double* log_likelihood);

  // Computes the global (sum over all strata) Score Function and Information
  // Matrix.
  static bool ComputeGlobalScoreFunctionAndInfoMatrix(
      const vector<StratificationData>& stratification_data,
      VectorXd* score_function, MatrixXd* info_matrix);

  // Computes the Score Function U(\beta).
  static bool ComputeScoreFunction(
      const VectorXd& exp_logistic_eq,
      const VectorXd& partial_sums,
      const vector<int>& transition_indices,
      const VectorXd& kaplan_meier_estimators,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      VectorXd* score_function);

  // Computes the Information Matrix V(\beta).
  static bool ComputeInformationMatrix(
      const VectorXd& exp_logistic_eq,
      const VectorXd& partial_sums,
      const VectorXd& ties_constant,
      const vector<int>& transition_indices,
      const VectorXd& kaplan_meier_estimators,
      const MatrixXd& indep_vars,
      const vector<CensoringData>& dep_var,
      MatrixXd* info_matrix);

  // Uses Newton-Rhapson algorithm to compute the next iteration to find
  // the next guess at y-intercept (beta_hat) from the old value and
  // its derivatives (Score Function and Information Matrix).
  // Performs halving if necessary (up to MAX_HALVING_ATTEMPTS times).
  // Also computes the log-likelihood of the new beta, as well as
  // new_logistic_eq and new_exp_logistic_eq.
  static bool ComputeNewBetaHat(
      const VectorXd& beta_hat,
      const double& log_likelihood,
      VectorXd* new_beta_hat,
      double* new_log_likelihood,
      vector<StratificationData>* stratification_data);

  // Run the NewtonRaphson Method.
  static bool RunNewtonRaphson(
      const double& log_likelihood,
      const ConvergenceCriteria& convergence_criteria,
      const VectorXd& beta_hat,
      vector<StratificationData>* stratification_data,
      VectorXd* regression_coefficients, int* iterations);

  // Compute Log-Rank statisitics, which is simply Score Function and
  // Info matrix, using regresion coefficient "0".
  static bool ComputeLogRank(
      const KaplanMeierEstimatorType kme_type_for_log_rank,
      vector<StratificationData>* stratification_data,
      double* log_rank_score_function, double* log_rank_info_matrix);

  // Given input indep_vars and dep_var, computes the Vector of Regression
  // Coefficients (Beta Hat) using the standard method of Maximum Likelihood
  // Estimation (MLE), with stopping criterion:
  //   |log(L(B^new)) - log(L(B^old))| / log(L(B^old)) < 10^{-6}
  // This function uses Armadillo (a C++ wrapper for LAPACK and BLAS) to do
  // matrix computations.
  static bool ComputeRegressionCoefficients(
      const int n, const int p, const KaplanMeierEstimatorType kme_type,
      const ConvergenceCriteria& convergence_criteria,
      int* num_iterations,
      VectorXd* regression_coefficients,
      MatrixXd* info_matrix_inverse,
      vector<StratificationData>* stratification_data);

  static bool ComputeSTerms(
      const vector<int>& transition_indices,
      const VectorXd& regression_coefficients,
      const MatrixXd& indep_vars, const vector<CensoringData>& dep_vars,
      VectorXd* s_zero, MatrixXd* s_one);

  static VectorXd WVectorFirstTerm(
      const double& kaplan_meier_multiplier,
      const VectorXd& row, const CensoringData& dep_var,
      const double& s_zero, const VectorXd& s_one);

  static VectorXd WVectorSecondTerm(
      const double& exp_logistic_eq, const int stop_index,
      const vector<CensoringData>& dep_vars,
      const VectorXd& kaplan_meier_estimators,
      const VectorXd& row, const VectorXd& s_zero, const MatrixXd& s_one);

  static bool ComputeWVectors(
      const vector<int>& transition_indices,
      const VectorXd& kaplan_meier_estimators,
      const VectorXd& regression_coefficients,
      const MatrixXd& indep_vars, const vector<CensoringData>& dep_vars,
      MatrixXd* w_vectors);

  static bool ComputeWVectors(
      const int p,
      const VectorXd& regression_coefficients,
      vector<StratificationData>* stratification_data);

  static void ComputeRobustVariance(
      const MatrixXd& info_matrix_inverse,
      const vector<StratificationData>& stratification_data,
      MatrixXd* robust_var);

  // NOTE: Each row has a corresponding w-vector; but the way this W-vector
  // is stored depends on context:
  //   (i)  There is the row's index with respect to it's strata
  //   (ii) There is the row's index with respect to the original data.
  // This method copies W-vectors that are indexed as in (i) (there is a
  // separate Matrix of W-vectors for each strata) into a single
  // Matrix; organizing them so that the order in the final matrix matches
  // the original index of the row as in (ii).
  static void CopyWVectors(
      const bool is_log_rank, const int p,
      const map<int, pair<int, int>>& unsorted_model_row_to_strata_index_and_row,
      const vector<StratificationData>& stratification_data,
      MatrixXd* w_vectors);

  // Computes W-Vectors for each strata, as well as the subgroup, populating the
  // 'w_vectors_' field of each StratificationData object with the relevant
  // W-Vectors for that strata/subgroup.
  // Also populates the robust variances (probably not needed by user, except
  // to sanity check W-Vectors).
  static bool ComputeWVectorsAndRobustVariance(
      const int p,
      const VectorXd& regression_coefficients,
      const MatrixXd& info_matrix_inverse,
      const map<int, pair<int, int>>& unsorted_model_row_to_strata_index_and_row,
      vector<StratificationData>* stratification_data,
      MatrixXd* robust_var, SummaryStatistics* stats);

  // Computes Dual Log Rank: B(0, 0).
  static bool ComputeLogRankWVectors(
      const map<int, pair<int, int>>& unsorted_model_row_to_strata_index_and_row,
      vector<StratificationData>* stratification_data,
      SummaryStatistics* stats);

  static bool ComputeBMatrix(
      const set<int>& common_rows,
      const map<int, int>& orig_row_to_unsorted_model_row_one,
      const map<int, int>& orig_row_to_unsorted_model_row_two,
      const MatrixXd& w_vectors_one,
      const MatrixXd& w_vectors_two,
      MatrixXd* b_matrix);

  // Computes estimated statistics to the input vectors.
  static bool ComputeFinalValues(
      const MatrixXd& info_matrix_inverse,
      const VectorXd& regression_coefficients,
      SummaryStatistics* stats, char* error_msg);

  // Computes the Z-score test-statistic:
  //   Z_i = (\hat{\beta}_i - \beta_i) / \hat{SE}
  // Where \hat{\beta} is the estimated value, and \beta is the actual value.
  static bool ComputeTestStatistic(
    const VectorXd& actual_beta,
    SummaryStatistics* stats, char* error_msg);
  // Same as above, but uses '0' as the actual \beta value.
  static bool ComputeTestStatistic(
    SummaryStatistics* stats, char* error_msg);
  // Same as above, but does this for each SummaryStatistic in summary_stats.
  static bool ComputeTestStatistic(
    const vector<VectorXd>& actual_beta,
    vector<SummaryStatistics>* summary_stats, char* error_msg);
  // Same as above, but uses '0' as the actual \beta value.
  static bool ComputeTestStatistic(
    vector<SummaryStatistics>* summary_stats, char* error_msg);

  // Prints Eigen Matrix (for when we can't use 'cout << EigenMatrix', because
  // we are printing to a string and not an output stream).
  static string PrintEigenMatrix(
      const vector<string>& row_names, const vector<string>& col_names,
      const MatrixXd& matrix, const int precision);
  // Same as above, without specifying row/column names.
  static string PrintEigenMatrix(const MatrixXd& matrix, const int precision) {
    return PrintEigenMatrix(vector<string>(), vector<string>(), matrix, precision);
  }
  // Same as above, using default precision of 6 decimal places.
  static string PrintEigenMatrix(const MatrixXd& matrix) {
    return PrintEigenMatrix(matrix, 6);
  }
  // Same as above, but with different API.
  static void PrintEigenMatrix(
      const vector<string>& row_names, const vector<string>& col_names,
      const MatrixXd& matrix, const int precision, string* output) {
    *output = PrintEigenMatrix(row_names, col_names, matrix, precision);
  }
  // Same as above, without specifying row/column names.
  static void PrintEigenMatrix(
      const MatrixXd& matrix, const int precision, string* output) {
    *output = PrintEigenMatrix(matrix, precision);
  }
  // Same as above, using default precision of 6 decimal places.
  static void PrintEigenMatrix(const MatrixXd& matrix, string* output) {
    PrintEigenMatrix(matrix, 6, output);
  }

};

}  // namespace regression

#endif
