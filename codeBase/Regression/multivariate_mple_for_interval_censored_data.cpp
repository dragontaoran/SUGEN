// Date: Dec 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "multivariate_mple_for_interval_censored_data.h"

#include "FileReaderUtils/read_file_utils.h"
#include "FileReaderUtils/read_time_dep_interval_censored_data.h"
#include "Regression/mple_for_interval_censored_data.h"
#include "MapUtils/map_utils.h"
#include "MathUtils/constants.h"
#include "MathUtils/data_structures.h"
#include "MathUtils/eq_solver.h"
#include "MathUtils/gamma_fns.h"
#include "MathUtils/gaussian_quadrature.h"
#include "MathUtils/number_comparison.h"
#include "MathUtils/statistics_utils.h"
#include "StringUtils/string_utils.h"
#include "TestUtils/test_utils.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using Eigen::FullPivLU;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using test_utils::Timer;
using namespace file_reader_utils;
using namespace map_utils;
using namespace math_utils;
using namespace string_utils;
using namespace std;

namespace regression {

bool MultivariateMpleForIntervalCensoredData::logging_on_;
bool MultivariateMpleForIntervalCensoredData::no_use_pos_def_variance_;
bool MultivariateMpleForIntervalCensoredData::force_one_right_censored_;
bool MultivariateMpleForIntervalCensoredData::PHB_no_variance_;
bool MultivariateMpleForIntervalCensoredData::PHB_use_exp_lambda_convergence_;
int MultivariateMpleForIntervalCensoredData::num_failed_variance_computations_;
vector<int> MultivariateMpleForIntervalCensoredData::p_;
vector<Timer> MultivariateMpleForIntervalCensoredData::timers_;
const int kNumberGaussianHermitePoints = 40;
const int kNumberGaussianLaguerrePoints = 40;

// Dummy functions for debugging.
namespace {

void PrintIntermediateValues(
    const int K, const int iteration_index,
    const bool print_first_constants,
    const bool print_first_beta_lambda, const bool print_all_beta_lambda,
    const vector<DependentCovariateIntermediateValues>& intermediate_values) {
  // Nothing to do if all print flags are false.
  if (!print_first_constants && !print_first_beta_lambda &&
      !print_all_beta_lambda) {
    return;
  }

  if (iteration_index == 1 &&
      (print_first_beta_lambda || print_first_constants)) {
    for (int k = 0; k < K; ++k) {
      if (print_first_beta_lambda) {
        Eigen::IOFormat format(Eigen::FullPrecision);
        cout << "\nAfter first iteration, beta_" << k + 1 << ":\n"
             << intermediate_values[k].beta_.transpose().format(format) << endl;
        cout << "Lambda_" << k + 1 << ":\n"
             << intermediate_values[k].lambda_.transpose() << endl;
      }
      if (print_first_constants) {
        cout << "Iteration " << iteration_index << " values for k = " << k + 1 << ":";
        cout << "\n\tbeta_" << k + 1 << ": " << intermediate_values[k].beta_
             << "\n\tlambda_" << k + 1 << ": " << intermediate_values[k].lambda_
             << "\n\ta_" << k + 1 << ": " << intermediate_values[k].a_is_
             << "\n\tc_" << k + 1 << ": " << intermediate_values[k].c_is_
             << "\n\td_" << k + 1 << ": " << intermediate_values[k].d_is_
             << "\n\tf_" << k + 1 << ": " << intermediate_values[k].f_i_
             << "\n\tS_L_" << k + 1 << ": " << intermediate_values[k].S_L_
             << "\n\tS_U_" << k + 1 << ": " << intermediate_values[k].S_U_
             << "\n\texp_neg_g_S_L_" << k + 1 << ": "
             << intermediate_values[k].exp_neg_g_S_L_
             << "\n\texp_neg_g_S_U_" << k + 1 << ": "
             << intermediate_values[k].exp_neg_g_S_U_;
        for (int i = 0; i < intermediate_values[k].exp_beta_x_plus_b_.size(); ++i) {
          cout << "\n\texp_beta_x_plus_b_" << k + 1 << "," << i << ": "
               << intermediate_values[k].exp_beta_x_plus_b_[i];
        }
      }
    }
    cout << endl
         << "################################################################"
         << endl;
  } else if (print_all_beta_lambda) {
    for (int k = 0; k < K; ++k) {
      Eigen::IOFormat format(Eigen::FullPrecision);
      cout << "\nAfter iteration " << iteration_index + 1
           << ", beta_" << k + 1 << ":\n"
           << intermediate_values[k].beta_.transpose().format(format) << endl;
      cout << "Lambda_" << k + 1 << ":\n"
           << intermediate_values[k].lambda_.transpose() << endl;
      cout << endl
           << "################################################################"
           << endl;
    }
  }
}

void PrintFinalValues(
    const double& b_variance,
    const vector<DependentCovariateEstimates>& estimates) {
  const int K = estimates.size();
  for (int k = 0; k < K; ++k) {
    Eigen::IOFormat format(Eigen::FullPrecision);
    cout << "\nFinal beta_" << k + 1 << ":\n"
         << estimates[k].beta_.transpose().format(format) << endl;
    cout << "Final Lambda_" << k + 1 << ":\n"
         << estimates[k].lambda_.transpose().format(format) << endl;
  }
  cout << "\nFinal sigma^2: " << setprecision(17) << b_variance << endl;
}

void PrintProfileLikelihoods(
    const double& pl_at_beta,
    const VectorXd& pl_toggle_one_dim,
    const MatrixXd& pl_toggle_two_dim) {
  Eigen::IOFormat format(Eigen::FullPrecision);
  cout << "pl_at_beta: " << setprecision(17)
       << pl_at_beta << endl << "pl_toggle_one_dim:\n"
       << pl_toggle_one_dim.format(format) << endl << "pl_toggle_two_dim:\n"
       << pl_toggle_two_dim.format(format) << endl;
}

}  // namespace

void MultivariateMpleForIntervalCensoredData::InitializeTimers() {
  const int num_timers = 0;
  for (int i = 0; i < num_timers; ++i) {
    timers_.push_back(Timer());
  }
}

void MultivariateMpleForIntervalCensoredData::PrintTimers() {
  for (int i = 0; i < timers_.size(); ++i) {
    const Timer& t = timers_[i];
    cout << "Timer " << i << ": " << test_utils::GetElapsedTime(t) << endl;
  }
}

bool MultivariateMpleForIntervalCensoredData::InitializeInput(
    const vector<double>& r, const TimeDepIntervalCensoredData& data,
    vector<DependentCovariateInfo>* input) {
  if (input == nullptr) return false;
  input->clear();

  const int K = r.size();
  const int n = data.subject_info_.size();
  if (data.distinct_times_.size() != K || data.legend_.size() != K || n == 0 ||
      data.subject_info_[0].times_.size() != K ||
      data.subject_info_[0].linear_term_values_.size() != K ||
      data.subject_info_[0].is_time_indep_.size() != K) {
    cout << "ERROR in Preparing Data: " << K << " r-values were provided, but "
         << "only found " << data.distinct_times_.size()
         << " sets of distinct times, " << n
         << " subjects, and " << data.legend_.size()
         << " model legends." << endl;
    return false;
  }
  p_.clear();

  // Number of gaussian-laguerre points will be set to a constant,
  // independent of k.
  // TODO(PHB): Allow N to be N_k, dependent on k. 
  const int N = kNumberGaussianLaguerrePoints;

  // Convert the (independent variable) values in each data.subject_info_
  // to the appropriate Matrix.
  if (logging_on_) {
    time_t current_time = time(nullptr);
    cout << endl << asctime(localtime(&current_time))
         << "Preparing data structures to run E-M algorithm.\n";
  }

  for (int k = 0; k < K; ++k) {
    input->push_back(DependentCovariateInfo());
    DependentCovariateInfo& dep_cov_info = input->back();

    // Set distinct times.
    // If force_one_right_censored_ is false, just copy the input set of
    // distinct times. Otherwise, we'll need to recompute the set of distinct
    // times.
    double max_left_time_k = -1.0;
    set<double> times_made_right_censored_k;
    if (!force_one_right_censored_) {
      dep_cov_info.distinct_times_ = data.distinct_times_[k];
    } else {
      // First go through, and find the maximum L-time.
      for (const SubjectInfo& info : data.subject_info_) {
        const EventTimeAndCause& time_info_k = info.times_[k];
        if (time_info_k.lower_ > max_left_time_k) {
          max_left_time_k = time_info_k.lower_;
        }
      }

      // Now Shift any Right-Time that is bigger than 'max_left_time_k' to
      // be right-censored (i.e. inf).
      // Also, update distinct_times with all non-negative (shouldn't be any)
      // and non-infinity values.
      for (const SubjectInfo& info : data.subject_info_) {
        const EventTimeAndCause& time_info_k = info.times_[k];
        if (time_info_k.lower_ > 0.0) {
          dep_cov_info.distinct_times_.insert(time_info_k.lower_);
        }
        const double& upper = time_info_k.upper_;
        if (upper == numeric_limits<double>::infinity()) {
          // Nothing to do.
        } else if (upper <= max_left_time_k) {
          dep_cov_info.distinct_times_.insert(upper);
        } else {
          // Keep track of times that were shifted to be infinity.
          times_made_right_censored_k.insert(upper);
        }
      }
    }

    // Set lower and upper times for each Subject, and all of that
    // Subject's values at all distinct times.
    vector<double>& lower_time_bounds = dep_cov_info.lower_time_bounds_;
    vector<double>& upper_time_bounds = dep_cov_info.upper_time_bounds_;
    vector<pair<VectorXd, MatrixXd>>& x = dep_cov_info.x_;
    for (int i = 0; i < n; ++i) {
      const SubjectInfo& subject_info_i = data.subject_info_[i];
      dep_cov_info.time_indep_vars_.push_back(subject_info_i.is_time_indep_[k]);
      if (times_made_right_censored_k.empty() ||
          subject_info_i.linear_term_values_[k].second.cols() == 0) {
        x.push_back(subject_info_i.linear_term_values_[k]);
      } else {
        x.push_back(pair<VectorXd, MatrixXd>());
        VectorXd& time_indep_values = x.back().first;
        time_indep_values = subject_info_i.linear_term_values_[k].first;
        const MatrixXd& values_to_copy = 
            subject_info_i.linear_term_values_[k].second;
        const int num_cols_to_skip = times_made_right_censored_k.size();
        const int num_cols_to_copy = values_to_copy.cols() - num_cols_to_skip;
        MatrixXd& time_dep_values = x.back().second;
        time_dep_values =
            values_to_copy.block(0, 0, values_to_copy.rows(), num_cols_to_copy);
      }
      lower_time_bounds.push_back(subject_info_i.times_[k].lower_);
      if (!force_one_right_censored_ ||
          subject_info_i.times_[k].upper_ <= max_left_time_k) {
        upper_time_bounds.push_back(subject_info_i.times_[k].upper_);
      } else {
        upper_time_bounds.push_back(numeric_limits<double>::infinity());
      }
    }

    // Set r_star_: I(t_k <= R*_i), where R*_i is R_i if R_i \neq \infty,
    // otherwise it is L_i.
    if (!MpleForIntervalCensoredData::ComputeRiStarIndicator(
            dep_cov_info.distinct_times_, lower_time_bounds,
            upper_time_bounds, &dep_cov_info.r_star_)) {
      return false;
    }

    // Compute X * X^T once.
    if (!ComputeXTimesXTranspose(
            dep_cov_info.time_indep_vars_, x, &dep_cov_info.x_x_transpose_)) {
      return false;
    }

    // Set r values.
    dep_cov_info.r_ = r[k];

    // Set transformaation G.
    if (!ConstructTransformation(
            dep_cov_info.r_, &dep_cov_info.transformation_G_)) {
      return false;
    }

    // Set gaussian-laguerre points.
    if (dep_cov_info.r_ != 0.0 &&
        !ComputeGaussianLaguerrePoints(
            N, dep_cov_info.r_, &dep_cov_info.points_and_weights_)) {
      return false;
    }

    // Compute \sum_q points * weights (used in computation of posterior means).
    dep_cov_info.sum_points_times_weights_ = 0.0;
    if (dep_cov_info.r_ != 0.0) {
      for (int q = 0; q < N; ++q) {
        const GaussianQuadratureTuple& current_point =
            dep_cov_info.points_and_weights_[q];
        dep_cov_info.sum_points_times_weights_ +=
            current_point.weight_ * current_point.abscissa_;
      }
    }

    // Get number of linear terms for each dependent variable (event type).
    p_.push_back(data.legend_[k].size());
  }

  if (logging_on_) {
    time_t current_time = time(nullptr);
    cout << asctime(localtime(&current_time))
         << "Done preparing input for E-M algorithm.\n";
  }

  return true;
}

MpleReturnValue
MultivariateMpleForIntervalCensoredData::PerformEmAlgorithmForParameterEstimation(
    const double& convergence_threshold,
    const int h_n_constant, const int max_itr,
    const vector<DependentCovariateInfo>& input,
    int* num_iterations, double* log_likelihood, double* b_variance,
    vector<DependentCovariateEstimates>* estimates, MatrixXd* variance) {
  // Sanity-Check input.
  if (b_variance == nullptr || estimates == nullptr) {
    cout << "ERROR in Performing EM Algorithm: Null input."
         << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  // Sanity-check dimensions match.
  const int K = input.size();
  if (K == 0 || input[0].x_.empty() ||
      (!estimates->empty() && estimates->size() != K)) {
    cout << "ERROR in Performing EM Algorithm: Mismatching dimensions on inputs: "
         << "input.size(): " << input.size() << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }
  const int n = input[0].x_.size();
  if (n == 0 || input[0].time_indep_vars_.size() != n) {
    cout << "ERROR in Performing EM Algorithm: No subjects (" << n
         << ") or mismatching number of subjects ("
         << input[0].time_indep_vars_.size() << ")" << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  // Check if there is a single dependent covariate.
  if (K == 1) {
    // Univariate MPLE. Use the Univariate class to solve.
    DependentCovariateEstimates* final_estimates;
    if (estimates->empty()) {
      *b_variance = 0.0;
      estimates->push_back(DependentCovariateEstimates());
      final_estimates = &(estimates->back());
    } else {
      final_estimates = &((*estimates)[0]);
    }
    MpleForIntervalCensoredData::SetLoggingOn(logging_on_);
    MpleForIntervalCensoredData::SetForceOneRightCensored(force_one_right_censored_);
    MpleForIntervalCensoredData::SetNoUsePositiveDefiniteVariance(
        no_use_pos_def_variance_);
    return MpleForIntervalCensoredData::PerformEmAlgorithmForParameterEstimation(
        input[0].r_, convergence_threshold, h_n_constant, max_itr,
        input[0].distinct_times_, input[0].lower_time_bounds_,
        input[0].upper_time_bounds_, input[0].time_indep_vars_, input[0].x_,
        num_iterations, log_likelihood,
        &final_estimates->beta_, &final_estimates->lambda_, variance);
  }

  // Check input has the requisite fields set.
  if ((input[0].points_and_weights_.empty() && input[0].r_ != 0.0) ||
      input[0].x_x_transpose_.empty() ||
      input[0].r_star_.empty()) {
    cout << "ERROR: Missing input to Perform EM Algorithm." << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  // Temproary debugging flags.
  const bool PHB_abort_after_one_iteration = false;
  const bool PHB_print_input = false;
  const bool PHB_print_constants_first_iteration = false;
  const bool PHB_print_first_beta_lambda = false;
  const bool PHB_print_all_beta_lambda = false;
  const bool PHB_print_final_beta_lambda = false;

  const int N_s = kNumberGaussianHermitePoints;
  double sum_hermite_weights = 0.0;
  vector<GaussianQuadratureTuple> gaussian_hermite_points;
  if (!ComputeGaussianHermitePointsAndSumWeights(
          N_s, &sum_hermite_weights, &gaussian_hermite_points)) {
    return MpleReturnValue::FAILED_PRELIMINARY_COMPUTATION;
  }

  // The following data structure will hold all the intermediate values
  // that are generated at each step of the E-M algorithm.
  vector<DependentCovariateIntermediateValues> intermediate_values;
  intermediate_values.resize(K);

  // Initialize first guess for beta, lambda, and sigma.
  double old_sigma = estimates->empty() ? 1.0 : sqrt(*b_variance);
  for (int k = 0; k < K; ++k) {
    if (estimates->empty()) {
      // Initial input to E-M algorithm will be with \beta = 0.
      VectorXd& old_beta = intermediate_values[k].beta_;
      const int p_k = input[k].time_indep_vars_[0].size();
      if (p_k == 0) {
        cout << "ERROR in Performing EM Algorithm: "
             << "p is zero for k = " << k + 1 << endl;
        return MpleReturnValue::FAILED_PRELIMINARY_COMPUTATION;
      }
      old_beta.resize(p_k);
      old_beta.setZero();

      // Initial input to E-M algorithm will be with \lambda = 1 / M_k.
      VectorXd& old_lambda = intermediate_values[k].lambda_;
      const int M_k = input[k].distinct_times_.size();
      if (M_k == 0) {
        cout << "ERROR in Performing EM Algorithm: "
             << "M_k is zero for k = " << k + 1 << endl;
        return MpleReturnValue::FAILED_PRELIMINARY_COMPUTATION;
      }
      old_lambda.resize(M_k);
      for (int m = 0; m < M_k; ++m) {
        old_lambda(m) = 1.0 / M_k;
      }
    } else {
      // Use the passed-in values to kick-off the E-M algorithm.
      intermediate_values[k].beta_ = (*estimates)[k].beta_;
      intermediate_values[k].lambda_ = (*estimates)[k].lambda_;
    }
  }

  // Determine how often to print status updates, based on how much data there
  // is (and thus, how long each iteration may take).
  int complexity = 0;
  for (int k = 0; k < K; ++k) {
    const int M_k = input[k].distinct_times_.size();
    const int p_k = input[k].time_indep_vars_[0].size();
    complexity += n * M_k * p_k;
  }
  const int print_modulus =
      complexity < 10000 ? 1000 :
      complexity < 1000000 ? 100 :
      complexity < 10000000 ? 50 :
      complexity < 100000000 ? 25 :
      complexity < 1000000000 ? 10 :
      complexity < 10000000000 ? 5 : 2;

  // Run E-M algorithm until convergence, or max_itr.
  const bool PHB_print_final_beta_on_failure = false;
  const bool PHB_print_final_lamba_on_failure = false;
  int iteration_index = 1;
  time_t current_time_start = time(nullptr);
  double current_difference = numeric_limits<double>::infinity();
  if (logging_on_) {
    cout << endl << asctime(localtime(&current_time_start))
         << "Beginning E-M algorithm...\n\n";
  }
  for (; iteration_index < max_itr; ++iteration_index) {
    if (logging_on_ && (iteration_index % print_modulus == 1)) {
      time_t current_time = time(nullptr);
      cout << asctime(localtime(&current_time)) << "On iteration: "
           << iteration_index;
      if (current_difference != numeric_limits<double>::infinity()) {
        cout << ". Current iteration difference: "
             << current_difference << " (will terminate when < "
             << convergence_threshold << ").";
      }
      cout << endl;
    }

    // For convenience, compute:
    //   b_s := \sqrt{2} * \sigma * y_s
    // once per iteration.
    VectorXd b;
    b.resize(N_s);
    for (int s = 0; s < N_s; ++s) {
      b(s) = old_sigma * sqrt(2.0) * gaussian_hermite_points[s].abscissa_;
    }

    // For convenience, compute:
    //   exp(\beta_k^T * X_kim + \sqrt{2} * \sigma * y_s)
    // once per iteration.
    for (int k = 0; k < K; ++k) {
      if (!ComputeExpBetaXPlusB(
              intermediate_values[k].beta_, b,
              input[k].time_indep_vars_, input[k].x_,
              &(intermediate_values[k].exp_beta_x_),
              &(intermediate_values[k].exp_beta_x_plus_b_))) {
        cout << "ERROR: Failed in Computing exp(beta^T * X + b) for k = "
             << k + 1 << " of iteration " << iteration_index << endl;
        return MpleReturnValue::FAILED_PRELIMINARY_COMPUTATION;
      }
    }

    // Run E-Step.
    vector<VectorXd> posterior_means;
    vector<MatrixXd> weights;
    VectorXd phi;
    if (!DoEStep(old_sigma, sum_hermite_weights, b, input, gaussian_hermite_points,
                 &intermediate_values, &weights, &posterior_means, &phi)) {
      cout << "ERROR: Failed in EStep of iteration " << iteration_index << endl;
      return MpleReturnValue::FAILED_E_STEP;
    }
    
    // Run M-Step.
    if (!DoMStep(iteration_index, input, intermediate_values, phi,
                 posterior_means, weights, b_variance, estimates)) {
      cout << "ERROR: Failed in MStep of iteration " << iteration_index << endl;
      return MpleReturnValue::FAILED_M_STEP;
    }

    // Check Convergence Criterion.
    if (EmAlgorithmHasConverged(
            convergence_threshold, old_sigma * old_sigma, *b_variance,
            intermediate_values, *estimates, &current_difference)) {
      *num_iterations = iteration_index;
      break;
    }

    // Copy current values to intermediate values.
    old_sigma = sqrt(*b_variance);

    for (int k = 0; k < K; ++k) {
      intermediate_values[k].beta_ = (*estimates)[k].beta_;
      intermediate_values[k].lambda_ = (*estimates)[k].lambda_;
    }
    if (iteration_index == 1 && PHB_abort_after_one_iteration) break;

    PrintIntermediateValues(
        K, iteration_index, PHB_print_constants_first_iteration,
        PHB_print_first_beta_lambda, PHB_print_all_beta_lambda,
        intermediate_values);
  }

  // Abort if we failed to converge after max_itr.
  if (iteration_index >= max_itr) {
    cout << "ERROR in Performing EM Algorithm: "
         << "E-M algorithm exceeded maximum number of allowed "
         << "iterations: " << max_itr << endl;
    // At least print out the final beta values.
    if (PHB_print_final_beta_on_failure) {
      Eigen::IOFormat format(Eigen::FullPrecision);
      for (int k = 0; k < K; ++k) {
        cout << "\nFinal beta_" << k + 1 << ":\n"
             << (*estimates)[k].beta_.transpose().format(format) << endl;
      }
    }
    if (PHB_print_final_lamba_on_failure) {
      Eigen::IOFormat format(Eigen::FullPrecision);
      for (int k = 0; k < K; ++k) {
        cout << "Final Lambda_" << k + 1 << ":\n"
             << (*estimates)[k].lambda_.transpose().format(format) << endl;    
      }
    }
    return MpleReturnValue::FAILED_MAX_ITR;
  }

  // Print final beta, lambda, and sigma values.
  if (PHB_print_final_beta_lambda) PrintFinalValues(*b_variance, *estimates);

  // Compute sqrt(2 * \sigma) * \nu_s, which will be needed in Likelihood and
  // Variance computations below.
  VectorXd b;
  b.resize(N_s);
  for (int s = 0; s < N_s; ++s) {
    b(s) = sqrt(*b_variance * 2.0) * gaussian_hermite_points[s].abscissa_;
  }

  // If running in 'PHB_no_variance' mode, print likelihood and return.
  if (log_likelihood != nullptr) {
    vector<VectorXd> final_beta(K);
    vector<VectorXd> final_lambda(K);
    for (int k = 0; k < K; ++k) {
      final_beta[k] = (*estimates)[k].beta_;
      final_lambda[k] = (*estimates)[k].lambda_;
    }
    // Temporarily override no_use_pos_def_variance_ option, so we can
    // compute the log-likelihood at the final estimates.
    const bool orig_no_use_pos_def_variance = no_use_pos_def_variance_;
    no_use_pos_def_variance_ = true;
    if (!ComputeProfileLikelihood(
            max_itr, convergence_threshold, sum_hermite_weights,
            gaussian_hermite_points, b, input,
            final_beta, final_lambda, log_likelihood, nullptr)) {
      cout << "ERROR: Failed to compute Profile "
           << "Likelihood at final beta." << endl;
      return MpleReturnValue::FAILED_VARIANCE;
    }
    // Restore no_use_pos_def_variance_ to its proper value.
    no_use_pos_def_variance_ = orig_no_use_pos_def_variance;
  }

  // Compute variance, and return.
  if (!PHB_no_variance_ && variance != nullptr) {
    // Time-stamp progress.
    if (logging_on_) {
      time_t current_time = time(nullptr);
      cout << endl << asctime(localtime(&current_time))
           << "Finished E-M algorithm "
           << "to compute beta and lambda in " << iteration_index
           << " iterations.\nComputing Covariance Matrix..." << endl;
    }
    VectorXd b;
    b.resize(N_s);
    for (int s = 0; s < N_s; ++s) {
      b(s) = sqrt(*b_variance * 2.0) * gaussian_hermite_points[s].abscissa_;
    }
    
    MpleReturnValue var_result = ComputeVariance(
        input[0].x_.size(), p_, h_n_constant, max_itr,
        convergence_threshold, sqrt(*b_variance),
        sum_hermite_weights, gaussian_hermite_points, b, input, *estimates,
        variance);
    if (var_result != MpleReturnValue::SUCCESS) {
      return var_result;
    }

    if (logging_on_) {
      time_t current_time = time(nullptr);
      cout << endl << asctime(localtime(&current_time))
           << "Done computing Covariance Matrix." << endl;
    }
  }

  return MpleReturnValue::SUCCESS;
}

bool MultivariateMpleForIntervalCensoredData::ComputeGaussianLaguerrePoints(
    const int n, const double& r,
    vector<GaussianQuadratureTuple>* gaussian_laguerre_points) {
  // Sanity-Check input.
  if (gaussian_laguerre_points == nullptr) {
    cout << "ERROR in ComputeGaussianLaguerrePoints: Null input." << endl;
    return false;
  }
  if (r <= 0.0) {
    cout << "ERROR: Bad value for r: " << r << " (must have r >= 0)" << endl;
    return false;
  }
  return ComputeGaussLaguerreQuadrature(
      n, (-1.0 + 1.0 / r), 0.0  /* a */, 1.0  /* b */, gaussian_laguerre_points);
}

bool MultivariateMpleForIntervalCensoredData::ComputeGaussianHermitePointsAndSumWeights(
    const int N_s,
    double* sum_hermite_weights,
    vector<GaussianQuadratureTuple>* gaussian_hermite_points) {
  if (sum_hermite_weights == nullptr || gaussian_hermite_points == nullptr) {
    return false;
  }
  if (!ComputeGaussianHermitePoints(N_s, gaussian_hermite_points)) {
    return false;
  }
  for (int s = 0; s < N_s; ++s) {
    *sum_hermite_weights += (*gaussian_hermite_points)[s].weight_;
  }
  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeGaussianHermitePoints(
    const int n, vector<GaussianQuadratureTuple>* gaussian_hermite_points) {
  // Sanity-Check input.
  if (gaussian_hermite_points == nullptr) {
    cout << "ERROR in ComputeGaussianHermitePoints: Null input." << endl;
    return false;
  }
  return ComputeGaussHermiteQuadrature(
      n, 0.0  /* alpha */, 0.0  /* a */, 1.0  /* b */, gaussian_hermite_points);
}

bool MultivariateMpleForIntervalCensoredData::ConstructTransformation(
    const double& r_k, Expression* transformation_G_k) {
  if (transformation_G_k == nullptr) {
    cout << "ERROR in ConstructTransformation: Null input." << endl;
    return false;
  }

  const string r_str = Itoa(r_k);
  const string g_str = r_k == 0.0 ? "x" : "log(1+" + r_str + "x)/" + r_str;
  if (!ParseExpression(g_str, transformation_G_k)) {
    cout << "ERROR: Unable to parse '"
         << g_str << "' as an Expression." << endl;
    return false;
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeXTimesXTranspose(
    const vector<vector<bool>>& time_indep_vars_k,
    const vector<pair<VectorXd, MatrixXd>>& x_k,
    vector<vector<MatrixXd>>* x_k_x_k_transpose) {
  if (x_k_x_k_transpose == nullptr || x_k.empty() ||
      (x_k[0].first.size() == 0 &&
       (x_k[0].second.rows() == 0 || x_k[0].second.cols() == 0))) {
    cout << "ERROR in ComputeXTimesXTranspose: Null input." << endl;
    return false;
  }

  const int n = x_k.size();

  x_k_x_k_transpose->clear();
  for (int i = 0; i < n; ++i) {
    x_k_x_k_transpose->push_back(vector<MatrixXd>());
    vector<MatrixXd>& ith_entry = x_k_x_k_transpose->back();
    // Only need one distinct time, if all covariates are time-independent.
    const int p_k_dep = x_k[i].second.rows();
    const int M_k = p_k_dep == 0 ? 1 : x_k[i].second.cols();
    for (int m = 0; m < M_k; ++m) {
      ith_entry.push_back(MatrixXd());
      MatrixXd& kth_entry = ith_entry.back();
      VectorXd x_kim;
      if (!MpleForIntervalCensoredData::GetXim(
              m, time_indep_vars_k[i], x_k[i], &x_kim)) {
        return false;
      }
      kth_entry = x_kim * x_kim.transpose();  // kth_entry is p_k x p_k.
    }
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeExpBetaXPlusB(
    const VectorXd& beta_k,
    const VectorXd& b,
    const vector<vector<bool>>& time_indep_vars_k,
    const vector<pair<VectorXd, MatrixXd>>& x_k,
    vector<vector<double>>* exp_beta_x_k,
    vector<MatrixXd>* exp_beta_x_plus_b_k) {
  // Sanity-check input.
  if (exp_beta_x_plus_b_k == nullptr) {
    cout << "ERROR in ComputeExpBetaXPlusB: Null input." << endl;
    return false;
  }

  const int n = x_k.size();
  const int p_k = beta_k.size();
  const int p_k_indep = x_k[0].first.size();
  const int p_k_dep = x_k[0].second.rows();
  const int N_s = b.size();
  if (n == 0 || p_k == 0 || N_s == 0 ||
      time_indep_vars_k.size() != n || time_indep_vars_k[0].size() != p_k ||
      p_k_dep + p_k_indep != p_k || (p_k_dep > 0 && x_k[0].second.cols() == 0)) {
    cout << "ERROR in ComputeExpBetaXPlusB: Mismatching dimensions on inputs: "
         << "beta_k.size(): " << p_k << ", x_k.size(): " << n
         << ", b.size(): " << N_s << endl;
    return false;
  }

  exp_beta_x_k->resize(n);
  exp_beta_x_plus_b_k->resize(n);
  for (int i = 0; i < n; ++i) {
    // Only need one distinct time, if all covariates are time-independent.
    const int p_k_i_dep = x_k[i].second.rows();
    const int M_k = p_k_i_dep == 0 ? 1 : x_k[i].second.cols();
    (*exp_beta_x_k)[i].clear();
    (*exp_beta_x_plus_b_k)[i].resize(N_s, M_k);
    for (int m = 0; m < M_k; ++m) {
      double dot_product;
      if (!MpleForIntervalCensoredData::ComputeBetaDotXim(
              m, time_indep_vars_k[i], beta_k, x_k[i], &dot_product)) {
        return false;
      }
      (*exp_beta_x_k)[i].push_back(exp(dot_product));
      for (int s = 0; s < N_s; ++s) {
        (*exp_beta_x_plus_b_k)[i](s, m) = exp(dot_product + b(s));
      }
    }
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::EmAlgorithmHasConverged(
    const double& convergence_threshold,
    const double& b_variance_old, const double& b_variance_new,
    const vector<DependentCovariateIntermediateValues>& intermediate_values,
    const vector<DependentCovariateEstimates>& estimates,
    double* current_difference) {
  const int K = intermediate_values.size();
  if (K == 0 || estimates.size() != K) {
    cout << "ERROR in EmAlgorithmHasConverged: Mismatching dimensions.\n";
    return false;
  }

  // Compute Cumulative Lambda.
  vector<VectorXd> cumulative_lambda_old, cumulative_lambda_new;
  vector<VectorXd> exp_neg_cum_lambda_old, exp_neg_cum_lambda_new;
  for (int k = 0; k < K; ++k) {
    const VectorXd& old_lambda_k = intermediate_values[k].lambda_;
    const VectorXd& new_lambda_k = estimates[k].lambda_;
    cumulative_lambda_old.push_back(VectorXd());
    cumulative_lambda_new.push_back(VectorXd());
    exp_neg_cum_lambda_old.push_back(VectorXd());
    exp_neg_cum_lambda_new.push_back(VectorXd());
    VectorXd& cumulative_lambda_old_k = cumulative_lambda_old.back();
    VectorXd& cumulative_lambda_new_k = cumulative_lambda_new.back();
    VectorXd& exp_neg_cum_lambda_old_k = exp_neg_cum_lambda_old.back();
    VectorXd& exp_neg_cum_lambda_new_k = exp_neg_cum_lambda_new.back();
    cumulative_lambda_old_k.resize(old_lambda_k.size());
    cumulative_lambda_new_k.resize(new_lambda_k.size());
    exp_neg_cum_lambda_old_k.resize(old_lambda_k.size());
    exp_neg_cum_lambda_new_k.resize(new_lambda_k.size());
    cumulative_lambda_old_k.setZero();
    cumulative_lambda_new_k.setZero();
    if (old_lambda_k.size() != new_lambda_k.size()) {
      cout << "ERROR: Mismatching sizes for lambda "
           << "for k = " << k + 1 << endl;
      return false;
    }
    for (int i = 0; i < old_lambda_k.size(); ++i) {
      if (i != 0) {
        cumulative_lambda_old_k(i) = cumulative_lambda_old_k(i - 1);
        cumulative_lambda_new_k(i) = cumulative_lambda_new_k(i - 1);
      }
      cumulative_lambda_old_k(i) += old_lambda_k(i);
      cumulative_lambda_new_k(i) += new_lambda_k(i);
      exp_neg_cum_lambda_old_k(i) = exp(-1.0 * cumulative_lambda_old_k(i));
      exp_neg_cum_lambda_new_k(i) = exp(-1.0 * cumulative_lambda_new_k(i));
    }
  }

  // TODO(PHB): Make this a global constant.
  const double delta = 0.01;
  double max_beta = 0.0;
  double max_lambda = 0.0;
  for (int k = 0; k < K; ++k) {
    // Compute Max(Beta).
    const VectorXd& old_beta_k = intermediate_values[k].beta_;
    const VectorXd& new_beta_k = estimates[k].beta_;
    if (old_beta_k.size() != new_beta_k.size()) {
      cout << "ERROR: Mismatching sizes for beta "
           << "for k = " << k + 1 << endl;
      return false;
    }
    const double beta_diff_k =
        VectorAbsoluteDifferenceSafe(old_beta_k, new_beta_k, delta);
    if (beta_diff_k > max_beta) {
      max_beta = beta_diff_k;
    }

    // Compute Max(Lambda).
    const VectorXd& old_lambda_k = cumulative_lambda_old[k];
    const VectorXd& new_lambda_k = cumulative_lambda_new[k];
    const VectorXd& old_exp_neg_lambda_k = exp_neg_cum_lambda_old[k];
    const VectorXd& new_exp_neg_lambda_k = exp_neg_cum_lambda_new[k];
    if (old_lambda_k.size() != new_lambda_k.size()) {
      cout << "ERROR: Mismatching sizes for lambda "
           << "for k = " << k + 1 << endl;
      return false;
    }
    const double lambda_diff_k = PHB_use_exp_lambda_convergence_ ?
        VectorAbsoluteDifferenceSafe(old_exp_neg_lambda_k, new_exp_neg_lambda_k, delta) :
        VectorAbsoluteDifferenceSafe(old_lambda_k, new_lambda_k, delta);
    if (lambda_diff_k > max_lambda) {
      max_lambda = lambda_diff_k;
    }
  }
  const double b_diff =
      AbsoluteDifferenceSafe(b_variance_old, b_variance_new, delta);

  const double curr_diff = b_diff + max_beta + max_lambda;

  if (current_difference != nullptr) {
    *current_difference = curr_diff;
  }
 
  return curr_diff < convergence_threshold;
}

bool MultivariateMpleForIntervalCensoredData::ProfileEmAlgorithmHasConverged(
    const double& convergence_threshold, const vector<VectorXd>& old_lambda,
    const vector<DependentCovariateIntermediateValues>& intermediate_values) {
  if (old_lambda.size() != intermediate_values.size()) {
    cout << "ERROR in ProfileEmAlgorithmHasConverged: Mismatching dimensions: "
         << "old_lambda (" << old_lambda.size() << ") vs. intermediate_values ("
         << intermediate_values.size() << ")." << endl;
    return false;
  }

  // Compute cumulative lambda for each of the K dep vars.
  vector<VectorXd> cumulative_lambda_old, cumulative_lambda_new;
  vector<VectorXd> exp_neg_cum_lambda_old, exp_neg_cum_lambda_new;
  const int K = old_lambda.size();
  for (int k = 0; k < K; ++k) {
    const VectorXd& new_lambda_k = intermediate_values[k].lambda_;
    const VectorXd& old_lambda_k = old_lambda[k];
    cumulative_lambda_old.push_back(VectorXd());
    cumulative_lambda_new.push_back(VectorXd());
    exp_neg_cum_lambda_old.push_back(VectorXd());
    exp_neg_cum_lambda_new.push_back(VectorXd());
    VectorXd& cumulative_lambda_old_k = cumulative_lambda_old.back();
    VectorXd& cumulative_lambda_new_k = cumulative_lambda_new.back();
    VectorXd& exp_neg_cum_lambda_old_k = exp_neg_cum_lambda_old.back();
    VectorXd& exp_neg_cum_lambda_new_k = exp_neg_cum_lambda_new.back();
    cumulative_lambda_old_k.resize(old_lambda_k.size());
    cumulative_lambda_new_k.resize(new_lambda_k.size());
    exp_neg_cum_lambda_old_k.resize(old_lambda_k.size());
    exp_neg_cum_lambda_new_k.resize(new_lambda_k.size());
    cumulative_lambda_old_k.setZero();
    cumulative_lambda_new_k.setZero();
    if (old_lambda_k.size() != new_lambda_k.size()) {
      cout << "ERROR: Mismatching sizes for lambda "
           << "for k = " << k + 1 << endl;
      return false;
    }
    for (int i = 0; i < old_lambda_k.size(); ++i) {
      if (i != 0) {
        cumulative_lambda_old_k(i) = cumulative_lambda_old_k(i - 1);
        cumulative_lambda_new_k(i) = cumulative_lambda_new_k(i - 1);
      }
      cumulative_lambda_old_k(i) += old_lambda_k(i);
      cumulative_lambda_new_k(i) += new_lambda_k(i);
      exp_neg_cum_lambda_old_k(i) = exp(-1.0 * cumulative_lambda_old_k(i));
      exp_neg_cum_lambda_new_k(i) = exp(-1.0 * cumulative_lambda_new_k(i));
    }
  }

  // TODO(PHB): Make delta a global constant.
  const double delta = 0.01;
  double max_lambda = 0.0;
  for (int k = 0; k < K; ++k) {
    const VectorXd& old_lambda_k = cumulative_lambda_old[k];
    const VectorXd& new_lambda_k = cumulative_lambda_new[k];
    const VectorXd& old_exp_neg_lambda_k = exp_neg_cum_lambda_old[k];
    const VectorXd& new_exp_neg_lambda_k = exp_neg_cum_lambda_new[k];
    if (old_lambda_k.size() != new_lambda_k.size()) {
      cout << "ERROR: Mismatching sizes for lambda "
           << "for k = " << k + 1 << endl;
      return false;
    }
    const double lambda_diff_k = PHB_use_exp_lambda_convergence_ ?
        VectorAbsoluteDifferenceSafe(old_exp_neg_lambda_k, new_exp_neg_lambda_k, delta) :
        VectorAbsoluteDifferenceSafe(old_lambda_k, new_lambda_k, delta);
    if (lambda_diff_k > max_lambda) {
      max_lambda = lambda_diff_k;
    }
  }

  return max_lambda < convergence_threshold;
}

bool MultivariateMpleForIntervalCensoredData::DoEStep(
    const double& current_sigma, const double& sum_hermite_weights,
    const VectorXd& b,
    const vector<DependentCovariateInfo>& input,
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    vector<DependentCovariateIntermediateValues>* intermediate_values,
    vector<MatrixXd>* weights,
    vector<VectorXd>* posterior_means,
    VectorXd* phi) {
  // Sanity-check input.
  if (weights == nullptr || posterior_means == nullptr || phi == nullptr) {
    cout << "ERROR in DoEStep: Null input." << endl;
    return false;
  }

  // Sanity-check dimensions match.
  const int K = input.size();
  const int N_s = gaussian_hermite_points.size();
  if (N_s == 0 || K == 0 || intermediate_values->size() != K) {
    cout << "ERROR in DoEStep: Mismatching dimensions on inputs: "
         << "input.size(): " << input.size()
         << ", gaussian_hermite_points.size(): " << N_s
         << ", intermediate_values.size(): "
         << intermediate_values->size() << endl;
    return false;
  }

  // Initialize (set size of) weights and posterior_means.
  weights->clear();
  weights->resize(K);
  posterior_means->clear();
  posterior_means->resize(K);

  // Compute and popoulate the fields of intermediate_values.
  for (int k = 0; k < K; ++k) {
    // Compute S_L and S_U.
    if (!ComputeS(input[k].distinct_times_, input[k].lower_time_bounds_,
                  (*intermediate_values)[k].lambda_,
                  (*intermediate_values)[k].exp_beta_x_plus_b_, 
                  &((*intermediate_values)[k].S_L_))) {
      cout << "ERROR in Computing S^L for k = " << k + 1 << endl;
      return false;
    }
    if (!ComputeS(input[k].distinct_times_, input[k].upper_time_bounds_,
                  (*intermediate_values)[k].lambda_,
                  (*intermediate_values)[k].exp_beta_x_plus_b_, 
                  &((*intermediate_values)[k].S_U_))) {
      cout << "ERROR in Computing S^U for k = " << k + 1 << endl;
      return false;
    }

    // Compute exp(-G(S_L)) and exp(-G(S_U)).
    if (!ComputeExpTransformation(
            input[k].transformation_G_,
            (*intermediate_values)[k].S_L_,
            &((*intermediate_values)[k].exp_neg_g_S_L_))) {
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }
    if (!ComputeExpTransformation(
            input[k].transformation_G_,
            (*intermediate_values)[k].S_U_,
            &((*intermediate_values)[k].exp_neg_g_S_U_))) {
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }
  }

  // Compute e_is.
  MatrixXd e;
  if (!ComputeEis(*intermediate_values, &e)) {
    return false;
  }

  // Compute a_kis, c_kis, d_kis, and f_ki.
  for (int k = 0; k < K; ++k) {
    if (!ComputeConstants(
            (input[k].r_ == 0.0 ? 0.0 : (1.0 / input[k].r_)),
            sum_hermite_weights, gaussian_hermite_points,
            input[k].upper_time_bounds_, e,
            (*intermediate_values)[k].S_L_,
            (*intermediate_values)[k].S_U_,
            (*intermediate_values)[k].exp_neg_g_S_L_,
            (*intermediate_values)[k].exp_neg_g_S_U_,
            &((*intermediate_values)[k].a_is_),
            &((*intermediate_values)[k].c_is_),
            &((*intermediate_values)[k].d_is_),
            &((*intermediate_values)[k].f_i_))) {
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }

    // Compute Posterior Means v_i.
    if (!ComputePosteriorMeans(
            input[k].r_,
            input[k].sum_points_times_weights_,
            b, gaussian_hermite_points,
            input[k].upper_time_bounds_, e,
            (*intermediate_values)[k].a_is_,
            (*intermediate_values)[k].c_is_,
            (*intermediate_values)[k].d_is_,
            (*intermediate_values)[k].f_i_,
            &((*posterior_means)[k]))) {
      cout << "ERROR in Computing Posterior Means for k = " << k + 1 << endl;
      return false;
    }

    // Compute Weights w_ki.
    if (!ComputeWeights(
            input[k].r_,
            gaussian_hermite_points, input[k].points_and_weights_,
            input[k].distinct_times_,
            input[k].lower_time_bounds_, input[k].upper_time_bounds_,
            (*intermediate_values)[k].exp_beta_x_plus_b_, 
            (*intermediate_values)[k].lambda_, e,
            (*intermediate_values)[k].S_L_,
            (*intermediate_values)[k].S_U_,
            (*intermediate_values)[k].a_is_,
            (*intermediate_values)[k].c_is_,
            (*intermediate_values)[k].d_is_,
            (*intermediate_values)[k].f_i_,
            (*posterior_means)[k],
            &((*weights)[k]))) {
      cout << "ERROR in Computing Weights for k = " << k + 1 << endl;
      return false;
    }
  }

  // Compute \phi_i := (\sum_s \mu_s * e_is * 2 * \sigma^2 * y_s^2) /
  //                   (\sum_s \mu_s * e_is)
  const int n = e.rows();
  phi->resize(n);
  for (int i = 0; i < n; ++i) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int s = 0; s < N_s; ++s) {
      const double mu_times_e = gaussian_hermite_points[s].weight_ * e(i, s);
      denominator += mu_times_e;
      numerator +=
        mu_times_e * 2.0 * current_sigma * current_sigma *
        gaussian_hermite_points[s].abscissa_ *
        gaussian_hermite_points[s].abscissa_;
    }
    if (denominator == 0.0) {
      cout << "ERROR in DoEStep: zero denominator for i = " << i << endl;
      return false;
    }
    (*phi)(i) = numerator / denominator;
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeS(
    const set<double>& t_k,
    const vector<double>& time_bounds_k,
    const VectorXd& lambda_k,
    const vector<MatrixXd>& exp_beta_x_plus_b_k,
    MatrixXd* S_k) {
  // Sanity-check input.
  if (S_k == nullptr) {
    cout << "ERROR in ComputeS: Null input." << endl;
    return false;
  }

  const int n = time_bounds_k.size();
  const int M_k = t_k.size();
  if (n == 0 || M_k == 0 ||
      lambda_k.size() != M_k || exp_beta_x_plus_b_k.size() != n ||
      exp_beta_x_plus_b_k[0].rows() == 0 || exp_beta_x_plus_b_k[0].cols() == 0) {
    cout << "ERROR in ComputeS: Mismatching dimensions on inputs: "
         << "t_k.size(): " << t_k.size()
         << ", time_bounds_k.size(): " << time_bounds_k.size()
         << ", lambda_k.size(): " << lambda_k.size()
         << ", exp_beta_x_plus_b_k.size(): " << exp_beta_x_plus_b_k.size() << endl;
    return false;
  }
  const int N_s = exp_beta_x_plus_b_k[0].rows();

  // Go through each b value, computing S_kis(b_s) for each.
  S_k->resize(n, N_s);
  S_k->setZero();
  for (int i = 0; i < n; ++i) {
    // Check if R_ki is infinity for this subject. If so set S_k(i, s)
    // to infinity (for each s in [1..N_s])
    if (time_bounds_k[i] == numeric_limits<double>::infinity()) {
      for (int s = 0; s < N_s; ++s) {
        // NOTE: The following line is the only one that needs to be
        // updated in order to match Donglin's results:
        //(*S_k)(i, s) = 99999.0;
        (*S_k)(i, s) = numeric_limits<double>::infinity();
      }
      continue;
    }

    // Loop through each time point m in [1..M_k], checking if distinct_time
    // t_k[m] is less than or equal to L_ki (resp. R_ki). If not, skip this
    // time (and all later times, since they will automatically be later, since
    // distinct_times_k is sorted). If so, add to running total.
    int m = 0;
    const bool covariates_all_time_indep =
        exp_beta_x_plus_b_k[i].cols() == 1;
    for (const double& time : t_k) {
      const int col_to_use = covariates_all_time_indep ? 0 : m;
      if (time <= time_bounds_k[i]) {
        for (int s = 0; s < N_s; ++s) {
          (*S_k)(i, s) +=
              lambda_k(m) * exp_beta_x_plus_b_k[i](s, col_to_use);
        }
      } else {
        break;
      }
      ++m;
    }
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeExpTransformation(
    const Expression& transformation_G_k,
    const MatrixXd& S_k,
    MatrixXd* exp_neg_g_S_k) {
  if (exp_neg_g_S_k == nullptr || S_k.rows() == 0 || S_k.cols() == 0) {
    cout << "ERROR in ComputeExpTransformation: Null input." << endl;
    return false;
  }

  const int n = S_k.rows();
  const int N_s = S_k.cols();

  exp_neg_g_S_k->resize(n, N_s);
  for (int s = 0; s < N_s; ++s) {
    for (int i = 0; i < n; ++i) {
      if (S_k(i, s) == numeric_limits<double>::infinity()) {
        (*exp_neg_g_S_k)(i, s) = 0.0;
        continue;
      }
      double g_value;
      string error_msg;
      if (!EvaluateExpression(transformation_G_k, "x", S_k(i, s), &g_value, &error_msg)) {
        cout << "ERROR:\n\t" << error_msg << endl;
        return false;
      }
      (*exp_neg_g_S_k)(i, s) = exp(-1.0 * g_value);
    }
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeEis(
    const vector<DependentCovariateIntermediateValues>& intermediate_values,
    MatrixXd* e) {
  const int K = intermediate_values.size();
  if (e == nullptr || K == 0) {
    cout << "ERROR in ComputeEis: Null input.\n";
    return false;
  }

  const int n = intermediate_values[0].exp_neg_g_S_L_.rows();
  const int N_s = intermediate_values[0].exp_neg_g_S_L_.cols();
  if (n == 0 || N_s == 0) {
    cout << "ERROR in ComputeEis: empty intermediate values." << endl;
    return false;
  }

  e->resize(n, N_s);

  for (int k = 0; k < K; ++k) {
    const MatrixXd& exp_neg_g_S_L = intermediate_values[k].exp_neg_g_S_L_;  // Dim (n, N_s)
    const MatrixXd& exp_neg_g_S_U = intermediate_values[k].exp_neg_g_S_U_;  // Dim (n, N_s)
    if (exp_neg_g_S_L.rows() != n || exp_neg_g_S_L.cols() != N_s ||
        exp_neg_g_S_U.rows() != n || exp_neg_g_S_U.cols() != N_s) {
      cout << "ERROR in ComputeEis: Inconsistent dimensions." << endl;
      return false;
    }
    for (int i = 0; i < n; ++i) {
      for (int s = 0; s < N_s; ++s) {
        if (k == 0) {
          (*e)(i, s) = 1.0;
        }
        (*e)(i, s) *= (exp_neg_g_S_L(i, s) - exp_neg_g_S_U(i, s));
      }
    }
  }
  
  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeConstants(
      const double& r_k_inverse, const double& sum_hermite_weights,
      const vector<GaussianQuadratureTuple>& hermite_points_and_weights,
      const vector<double>& upper_time_bounds_k,
      const MatrixXd& e,
      const MatrixXd& S_L_k,
      const MatrixXd& S_U_k,
      const MatrixXd& exp_neg_g_S_L_k,
      const MatrixXd& exp_neg_g_S_U_k,
      MatrixXd* a_k, MatrixXd* c_k, MatrixXd* d_k,
      VectorXd* f_k) {
  if (a_k == nullptr || c_k == nullptr || d_k == nullptr || f_k == nullptr) {
    cout << "ERROR in ComputeConstants: Null Input." << endl;
    return false;
  }
  // Sanity-check dimensions match.
  const int n = upper_time_bounds_k.size();
  const int N_s = hermite_points_and_weights.size();
  if (n == 0 || N_s == 0 || e.rows() != n || e.cols() != N_s ||
      S_L_k.rows() != n || S_L_k.cols() != N_s ||
      S_U_k.rows() != n || S_U_k.cols() != N_s ||
      exp_neg_g_S_L_k.rows() != n || exp_neg_g_S_L_k.cols() != N_s ||
      exp_neg_g_S_U_k.rows() != n || exp_neg_g_S_U_k.cols() != N_s) {
    cout << "ERROR in ComputeConstants: Mismatching dimensions: "
         << "n: " << n << ", N_s: " << N_s
         << "e.rows(): " << e.rows()
         << "e.cols(): " << e.cols()
         << "S_L_k.rows(): " << S_L_k.rows()
         << "S_L_k.cols(): " << S_L_k.cols()
         << "S_U_k.rows(): " << S_U_k.rows()
         << "S_U_k.cols(): " << S_U_k.cols()
         << "exp_neg_g_S_L_k.rows(): " << exp_neg_g_S_L_k.rows()
         << "exp_neg_g_S_L_k.cols(): " << exp_neg_g_S_L_k.cols()
         << "exp_neg_g_S_U_k.rows(): " << exp_neg_g_S_U_k.rows()
         << "exp_neg_g_S_U_k.cols(): " << exp_neg_g_S_U_k.cols() << endl;
    return false;
  }

  a_k->resize(n, N_s);
  c_k->resize(n, N_s);
  d_k->resize(n, N_s);
  f_k->resize(n);

  if (r_k_inverse == 0.0) {
    // The values for a_k, c_k, d_k, and f_k are not needed when r = 0.
    // Set them to zero (for no good reason, other than it feels wrong
    // not to set them, and perhaps setting them to zero will throw an
    // error if they accidentally try to get used).
    a_k->setZero();
    c_k->setZero();
    d_k->setZero();
    f_k->setZero();
    return true;
  }

  // Compute a_k, c_k, d_k, and f_k.
  for (int i = 0; i < n; ++i) {
    double f_second_sum = 0.0;
    for (int s = 0; s < N_s; ++s) {
      (*a_k)(i, s) = r_k_inverse + S_L_k(i, s);
      (*c_k)(i, s) = r_k_inverse + S_U_k(i, s);
      (*d_k)(i, s) = e(i, s) / (exp_neg_g_S_L_k(i, s) - exp_neg_g_S_U_k(i, s));
      const double f_second_sum_second_term = pow((*a_k)(i, s), -1.0 * r_k_inverse) -
          (upper_time_bounds_k[i] == numeric_limits<double>::infinity() ?
           0.0 : pow((*c_k)(i, s), -1.0 * r_k_inverse));
      f_second_sum +=
          hermite_points_and_weights[s].weight_ * (*d_k)(i, s) *
          f_second_sum_second_term;
    }
    (*f_k)(i) = sum_hermite_weights * f_second_sum;
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputePosteriorMeans(
    const double& r_k,
    const double& sum_points_times_weights,
    const VectorXd& b,
    const vector<GaussianQuadratureTuple>& hermite_points_and_weights,
    const vector<double>& upper_time_bounds_k,
    const MatrixXd& e,
    const MatrixXd& a_k,
    const MatrixXd& c_k,
    const MatrixXd& d_k,
    const VectorXd& f_k,
    VectorXd* v_k) {
  // Sanity-check input.
  if (v_k == nullptr) {
    cout << "ERROR in ComputePosteriorMeans: Null input." << endl;
    return false;
  }

  // Sanity-check dimensions match.
  const int n = upper_time_bounds_k.size();
  const int N_s = hermite_points_and_weights.size();
  if (n == 0 || N_s == 0 || f_k.size() != n || b.size() != N_s ||
      a_k.rows() != n || a_k.cols() != N_s ||
      c_k.rows() != n || c_k.cols() != N_s ||
      d_k.rows() != n || d_k.cols() != N_s) {
    cout << "ERROR in ComputeConstants: Mismatching dimensions: "
         << "n: " << n << ", N_s: " << N_s
         << ", b.size(): " << b.size()
         << ", f_k.size(): " << f_k.size()
         << ", a_k.rows(): " << a_k.rows()
         << ", a_k.cols(): " << a_k.cols()
         << ", c_k.rows(): " << c_k.rows()
         << ", c_k.cols(): " << c_k.cols()
         << ", d_k.rows(): " << d_k.rows()
         << ", d_k.cols(): " << d_k.cols() << endl;
    return false;
  }

  v_k->resize(n);

  // Separate formula for posterior mean when r is zero.
  if (r_k == 0.0) {
    for (int i = 0; i < n; ++i) {
      double numerator = 0.0;
      double denominator = 0.0;
      for (int s = 0; s < N_s; ++s) {
        const double mu_times_e = 
          hermite_points_and_weights[s].weight_ * e(i, s);
        denominator += mu_times_e;
        numerator += mu_times_e * exp(b(s));
      }
      if (denominator == 0.0) {
        cout << "ERROR in Computing Posterior Mean: Zero denominator for i = "
             << i << endl;
        return false;
      }
      (*v_k)(i) = numerator / denominator;
    }
    return true;
  }

  // Compute term \mu_s * exp(sqrt(2) * sigma * y_s); we could do
  // this within the loop below, but since this does not depend on i,
  // we do it here to avoid extraneous computations.
  VectorXd hermite_factor;
  hermite_factor.resize(N_s);
  for (int s = 0; s < N_s; ++s) {
    hermite_factor(s) = hermite_points_and_weights[s].weight_ * exp(b(s));
  }

  // Compute posterior mean.
  for (int i = 0; i < n; ++i) {
    double second_sum = 0.0;
    for (int s = 0; s < N_s; ++s) {
      const double second_sum_second_term = pow(a_k(i, s), -1.0 - (1.0 / r_k)) -
          (upper_time_bounds_k[i] == numeric_limits<double>::infinity() ?
           0.0 : pow(c_k(i, s), -1.0 - (1.0 / r_k)));
      second_sum +=
          hermite_factor(s) * d_k(i, s) * second_sum_second_term;
    }
    if (f_k(i) == 0.0) {
      cout << "ERROR in Computing Posterior Mean: f_k is 0.0 for i = " << i + 1
           << ". Note:\n\ta_k.row(i): " << a_k.row(i) << endl << "\tc_k.row(i): "
           << c_k.row(i) << "\n\td_k.row(i): " << d_k.row(i) << "\n\te.row(i): "
           << e.row(i) << endl;
      return false;
    }
    (*v_k)(i) = (1.0 / f_k(i)) * sum_points_times_weights * second_sum;
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeWeights(
    const double& r_k,
    const vector<GaussianQuadratureTuple>& hermite_points_and_weights,
    const vector<GaussianQuadratureTuple>& laguerre_points_and_weights_k,
    const set<double>& distinct_times_k,
    const vector<double>& lower_time_bounds_k,
    const vector<double>& upper_time_bounds_k,
    const vector<MatrixXd>& exp_beta_x_plus_b_k,
    const VectorXd& lambda_k,
    const MatrixXd& e,
    const MatrixXd& S_L_k,
    const MatrixXd& S_U_k,
    const MatrixXd& a_k,
    const MatrixXd& c_k,
    const MatrixXd& d_k,
    const VectorXd& f_k,
    const VectorXd& v_k,
    MatrixXd* weights_k) {
  // Sanity-Check input.
  if (weights_k == nullptr) {
    cout << "ERROR in Computing Weights: Null input." << endl;
    return false;
  }

  const int n = lower_time_bounds_k.size();
  const int N_s = e.cols();
  const int N_k = laguerre_points_and_weights_k.size();
  const int M_k = distinct_times_k.size();
  if (n == 0 || N_s == 0 || (N_k == 0 && r_k != 0.0) || M_k == 0 ||
      exp_beta_x_plus_b_k.size() != n || exp_beta_x_plus_b_k[0].rows() != N_s ||
      (exp_beta_x_plus_b_k[0].cols() != M_k && exp_beta_x_plus_b_k[0].cols() != 1) ||
      e.rows() != n || upper_time_bounds_k.size() != n || lambda_k.size() != M_k ||
      S_L_k.rows() != n || S_L_k.cols() != N_s ||
      S_U_k.rows() != n || S_U_k.cols() != N_s ||
      a_k.rows() != n || a_k.cols() != N_s ||
      c_k.rows() != n || c_k.cols() != N_s ||
      d_k.rows() != n || d_k.cols() != N_s ||
      f_k.size() != n || v_k.size() != n) {
    cout << "ERROR in ComputeWeights: Mismatching dimensions on inputs: "
         << ", distinct_times_k.size(): " << distinct_times_k.size()
         << ", lower_time_bounds_k.size(): " << lower_time_bounds_k.size()
         << ", upper_time_bounds_k.size(): " << upper_time_bounds_k.size()
         << ", lambda_k.size(): " << lambda_k.size()
         << ", exp_beta_x_plus_b_k.size(): " << exp_beta_x_plus_b_k.size()
         << ", v_k.size(): " << v_k.size()
         << ", f_k.size(): " << f_k.size()
         << ", a_k.rows(): " << a_k.rows()
         << ", a_k.cols(): " << a_k.cols()
         << ", c_k.rows(): " << c_k.rows()
         << ", c_k.cols(): " << c_k.cols()
         << ", d_k.rows(): " << d_k.rows()
         << ", d_k.cols(): " << d_k.cols()
         << ", e.rows(): " << e.rows()
         << ", e.cols(): " << e.cols()
         << ", S_L_k.rows(): " << S_L_k.rows()
         << ", S_L_k.cols(): " << S_L_k.cols()
         << ", S_U_k.rows(): " << S_U_k.rows()
         << ", S_U_k.cols(): " << S_U_k.cols() << endl;
    return false;
  }

  weights_k->resize(n, M_k);
  const double neg_one_minus_r_k_inverse =
      r_k == 0.0 ? 0.0 : (-1.0 - (1.0 / r_k));

  // First, compute the "Hermite Sum" for each (i, s). Here, the "hermite sum"
  // represents the second two lines of w_kim (see paper), and we compute it
  // here because it does *not* depend on m.
  MatrixXd hermite_sums;
  hermite_sums.resize(n, N_s);
  if (r_k != 0.0) {
    for (int i = 0; i < n; ++i) {
      const double u_i = upper_time_bounds_k[i];
      for (int s = 0; s < N_s; ++s) {
        double first_laguerre_sum = 0.0;
        double second_laguerre_sum = 0.0;
        const double S_difference_kis = S_L_k(i, s) - S_U_k(i, s);
        const double first_laguerre_sum_power =
            S_difference_kis / a_k(i, s);
        const double second_laguerre_sum_power =
            (u_i == numeric_limits<double>::infinity() ?
             0.0 : S_difference_kis / c_k(i, s));
        for (int q = 0; q < N_k; ++q) {
          const GaussianQuadratureTuple& points_and_weights_q =
              laguerre_points_and_weights_k[q];
          const double points_times_weights_q =
              points_and_weights_q.weight_ * points_and_weights_q.abscissa_;
          first_laguerre_sum +=
              points_times_weights_q /
              (1.0 - exp(first_laguerre_sum_power * points_and_weights_q.abscissa_));
          if (u_i != numeric_limits<double>::infinity()) {
            second_laguerre_sum +=
                points_times_weights_q /
                (1.0 - exp(second_laguerre_sum_power * points_and_weights_q.abscissa_));
          }
        }
        hermite_sums(i, s) =
            (pow(a_k(i, s), neg_one_minus_r_k_inverse) * first_laguerre_sum) -
             (u_i == numeric_limits<double>::infinity() ? 0.0 :
              (pow(c_k(i, s), neg_one_minus_r_k_inverse) * second_laguerre_sum));
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    const bool covariates_all_time_indep = exp_beta_x_plus_b_k[i].cols() == 1;
    const double l_i = lower_time_bounds_k[i];
    const double u_i = upper_time_bounds_k[i];
    const double f_ki_inverse = 1.0 / f_k(i);
    int m = 0;
    for (const double& time : distinct_times_k) {
      if (time <= l_i) {
        (*weights_k)(i, m) = 0.0;
      } else if (u_i != numeric_limits<double>::infinity() && time <= u_i) {
        const int col_to_use = covariates_all_time_indep ? 0 : m;
        if (r_k == 0.0) {
          double numerator = 0.0;
          double denominator = 0.0;
          for (int s = 0; s < N_s; ++s) {
            const GaussianQuadratureTuple& points_and_weights =
                hermite_points_and_weights[s]; 
            const double mu_times_e = points_and_weights.weight_ * e(i, s);
            denominator += mu_times_e;
            const double numerator_numerator =
                lambda_k(m) * exp_beta_x_plus_b_k[i](s, col_to_use);
            const double numerator_denominator =
                1.0 - exp(S_L_k(i, s) - S_U_k(i, s));
            numerator += mu_times_e * numerator_numerator / numerator_denominator;
          }
          if (denominator == 0.0) {
            cout << "ERROR in Computing Weights for i = " << i << " and time = "
                 << time << ": Zero denominator." << endl;
            return false;
          }
          (*weights_k)(i, m) = numerator / denominator;
        } else {
          double hermite_sum = 0.0;
          for (int s = 0; s < N_s; ++s) {
            hermite_sum +=
                hermite_points_and_weights[s].weight_ * d_k(i, s) *
                lambda_k(m) * exp_beta_x_plus_b_k[i](s, col_to_use) * hermite_sums(i, s);
          }
          (*weights_k)(i, m) = f_ki_inverse * hermite_sum;
        }
      } else {
        // These weights will not be used; just set a dummy value.
        (*weights_k)(i, m) = 0.0;
        //PHB_OLD(*weights_k)(i, m) = lambda_k(m) * exp(beta_k.dot(x_k[i].col(m))) * v_k(i);
      }
      ++m;
    }
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::DoMStep(
    const int itr_index,
    const vector<DependentCovariateInfo>& input,
    const vector<DependentCovariateIntermediateValues>& intermediate_values,
    const VectorXd& phi,
    const vector<VectorXd>& posterior_means,
    const vector<MatrixXd>& weights,
    double* new_sigma_squared,
    vector<DependentCovariateEstimates>* new_estimates) {
  // Sanity-Check input.
  if (new_sigma_squared == nullptr || new_estimates == nullptr) {
    cout << "ERROR in DoMStep: Null input." << endl;
    return false;
  }

  const int n = phi.size();
  const int K = input.size();
  if (n == 0 || K == 0 || intermediate_values.size() != K ||
      posterior_means.size() != K || posterior_means[0].size() != n ||
      weights.size() != K || weights[0].rows() != n || weights[0].cols() == 0) {
    cout << "ERROR in DoMStep: Mismatching dimensions on inputs: "
         << "input.size(): " << K
         << ", phi.size(): " << n
         << ", intermediate_values.size(): " << intermediate_values.size()
         << ", posterior_means.size(): " << posterior_means.size()
         << ", weights.size(): " << weights.size() << endl;
    return false;
  }

  new_estimates->resize(K);
  for (int k = 0; k < K; ++k) {
    // Compute v_ki * exp(beta_k^T * X_kim), for each i in [1..n] and
    // m in [1..M_k] (these values will be used over and over in formula
    // for new beta).
    vector<VectorXd> v_exp_beta_x_k;
    if (!ComputeSummandTerm(
            posterior_means[k], intermediate_values[k].exp_beta_x_,
            &v_exp_beta_x_k)) {
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }

    // Compute constants that will be used to compute new beta.
    vector<double> S0_k;
    vector<VectorXd> S1_k;
    vector<MatrixXd> S2_k;
    if (!ComputeSValues(v_exp_beta_x_k, input[k].time_indep_vars_,
                        input[k].x_, input[k].x_x_transpose_,
                        input[k].r_star_, &S0_k, &S1_k, &S2_k)) {
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }

    // Compute term 'Sigma' that will be used to compute new beta.
    MatrixXd Sigma_k;
    if (!ComputeSigma(weights[k], S0_k, S1_k, S2_k, input[k].r_star_, &Sigma_k)) {
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }

    // Compute new Beta.
    if (!ComputeNewBeta(
            itr_index, intermediate_values[k].beta_, Sigma_k, weights[k],
            input[k].time_indep_vars_, input[k].x_, S0_k, S1_k, input[k].r_star_,
            &((*new_estimates)[k].beta_))) {
      cout << "ERROR in Computing new Beta for k = " << k + 1 << endl;
      return false;
    }

    // Compute the denominator used for new lambda.
    vector<double> S0_new_k;
    if (!ComputeS0((*new_estimates)[k].beta_, posterior_means[k],
                   input[k].time_indep_vars_, input[k].x_,
                   input[k].r_star_, &S0_new_k)) {
      cout << "ERROR in ComputeS0 for k = " << k + 1 << endl;
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }

    // Update Lambda.
    if (!ComputeNewLambda(weights[k], S0_new_k, input[k].r_star_,
                          &((*new_estimates)[k].lambda_))) {
      cout << "ERROR in Computing new Lambda for k = " << k + 1 << endl;
      return false;
    }
  }

  // Update sigma.
  double numerator = 0.0;
  for (int i = 0; i < n; ++i) {
    numerator += phi(i);
  }
  *new_sigma_squared = numerator / static_cast<double>(n);

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeSummandTerm(
    const VectorXd& v_k,
    const vector<vector<double>>& exp_beta_x_k,
    vector<VectorXd>* v_exp_beta_x_k) {
  // Sanity-check input.
  if (v_exp_beta_x_k == nullptr) {
    cout << "ERROR in ComputeSummandTerm: Null input." << endl;
    return false;
  }

  const int n = v_k.size();
  if (n == 0 || exp_beta_x_k.size() != n) {
    cout << "ERROR in ComputeSummandTerm: Mismatching dimensions on inputs." << endl;
    return false;
  }

  v_exp_beta_x_k->resize(n);

  for (int i = 0; i < n; ++i) {
    const int M_ki = exp_beta_x_k[i].size();
    (*v_exp_beta_x_k)[i].resize(M_ki);
    for (int m = 0; m < M_ki; ++m) {
      (*v_exp_beta_x_k)[i](m) = v_k(i) * exp_beta_x_k[i][m];
    }
  }
  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeSValues(
    const vector<VectorXd>& v_exp_beta_x_k,
    const vector<vector<bool>>& time_indep_vars_k,
    const vector<pair<VectorXd, MatrixXd>>& x_k,
    const vector<vector<MatrixXd>>& x_k_x_k_transpose,
    const vector<vector<bool>>& r_star_k,
    vector<double>* S0_k, vector<VectorXd>* S1_k, vector<MatrixXd>* S2_k) {
  // Sanity-check input.
  if (S0_k == nullptr || S1_k == nullptr || S2_k == nullptr) {
    cout << "ERROR in ComputeSValues: Null input." << endl;
    return false;
  }

  const int n = v_exp_beta_x_k.size();
  const int M_k = r_star_k[0].size();
  const int p_k = x_k[0].first.size() + x_k[0].second.rows();
  if (n == 0 || x_k.size() != n || p_k == 0 ||
      M_k == 0 || r_star_k.size() != n) {
    cout << "ERROR in ComputeSValues: Mismatching dimensions on inputs: "
         << "v_exp_beta_x_k.rows(): " << v_exp_beta_x_k.size()
         << ", x_k.size(): " << x_k.size()
         << ", r_star_k.size(): " << r_star_k.size() << endl;
    return false;
  }

  S0_k->resize(M_k, 0.0);
  S1_k->resize(M_k, VectorXd());
  S2_k->resize(M_k, MatrixXd());
  for (int m = 0; m < M_k; ++m) {
    double& S0_km = (*S0_k)[m];
    VectorXd& S1_km = (*S1_k)[m];
    S1_km.resize(p_k);
    S1_km.setZero();

    MatrixXd& S2_km = (*S2_k)[m];
    S2_km.resize(p_k, p_k);
    S2_km.setZero();

    for (int i = 0; i < n; ++i) {
      if (r_star_k[i][m]) {
        // The number of times that we need to store S values is either M (if at least
        // one covariate is time-dep), or 1 (if all covariates are time-indep). We
        // check which case we're in by checking if there are any dependent covariates.
        const int col_to_use = x_k[i].second.rows() == 0 ? 0 : m;
        S0_km += v_exp_beta_x_k[i](col_to_use);
        if (!MpleForIntervalCensoredData::AddConstantTimesXim(
                m, time_indep_vars_k[i],
                v_exp_beta_x_k[i](col_to_use), x_k[i], &S1_km)) {
          return false;
        }
        const int x_x_transpose_col_to_use = x_k_x_k_transpose[i].size() > 1 ? m : 0;
        S2_km += v_exp_beta_x_k[i](col_to_use) *
                 x_k_x_k_transpose[i][x_x_transpose_col_to_use];
      }
    }
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeS0(
    const VectorXd& beta_k,
    const VectorXd& posterior_means_k,
    const vector<vector<bool>>& time_indep_vars_k,
    const vector<pair<VectorXd, MatrixXd>>& x_k,
    const vector<vector<bool>>& r_star_k,
    vector<double>* S0_k) {
  // Sanity-check input.
  if (S0_k == nullptr) {
    cout << "ERROR in ComputeS0: Null input." << endl;
    return false;
  }

  const int p_k_indep = x_k[0].first.size();
  const int p_k_dep = x_k[0].second.rows();
  const int p_k = beta_k.size();
  const int n = posterior_means_k.size();
  if (p_k == 0 || p_k_indep + p_k_dep != p_k || n == 0 || x_k.size() != n ||
      (p_k_dep > 0 && x_k[0].second.cols() == 0)) {
    cout << "ERROR in ComputeS0: Mismatching dimensions on inputs: "
         << "beta_k.size(): " << beta_k.size()
         << ", posterior_means_k.size(): " << posterior_means_k.size()
         << ", x_k.size(): " << x_k.size() << endl;
    return false;
  }

  const int M_k = r_star_k[0].size();

  S0_k->resize(M_k, 0.0);

  for (int m = 0; m < M_k; ++m) {
    for (int i = 0; i < n; ++i) {
      if (r_star_k[i][m]) {
        double dot_product;
        if (!MpleForIntervalCensoredData::ComputeBetaDotXim(
                m, time_indep_vars_k[i], beta_k, x_k[i], &dot_product)) {
          return false;
        }
        (*S0_k)[m] += posterior_means_k(i) * exp(dot_product);
      }
    }
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeSigma(
    const MatrixXd& weights_k,
    const vector<double>& S0_k,
    const vector<VectorXd>& S1_k,
    const vector<MatrixXd>& S2_k,
    const vector<vector<bool>>& r_star_k,
    MatrixXd* Sigma_k) {
  // Sanity-Check input.
  if (Sigma_k == nullptr) {
    cout << "ERROR in ComputeSigma: Null input." << endl;
    return false;
  }

  const int n = weights_k.rows();
  const int M_k = weights_k.cols();
  if (n == 0 || M_k == 0 ||
      S0_k.size() != M_k || S1_k.size() != M_k || S2_k.size() != M_k ||
      r_star_k.size() != n || r_star_k[0].size() != M_k ||
      S1_k[0].size() != S2_k[0].rows() || S1_k[0].size() != S2_k[0].cols()) {
    cout << "ERROR in ComputeSigma: Mismatching dimensions on inputs: "
         << "S0_k.size(): " << S0_k.size() << ", S1_k.size(): "
         << S1_k.size() << ", S2_k.size(): " << S2_k.size()
         << ", weights_k.cols(): " << weights_k.cols()
         << ", r_star_k.size(): " << r_star_k.size() << endl;
    return false;
  }

  const int p_k = S1_k[0].size();

  Sigma_k->resize(p_k, p_k);
  Sigma_k->setZero();
  for (int m = 0; m < M_k; ++m) {
    const double& S0_km = S0_k[m];
    const VectorXd& S1_km = S1_k[m];  // p_k x 1
    const MatrixXd& S2_km = S2_k[m];  // p_k x p_k
    const MatrixXd factor_km =
        (S1_km / S0_km) * (S1_km.transpose() / S0_km) - (S2_km / S0_km);
    for (int i = 0; i < n; ++i) {
      if (r_star_k[i][m]) {
        (*Sigma_k) += weights_k(i, m) * factor_km;
      }
    }
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeNewBeta(
    const int itr_index,
    const VectorXd& old_beta_k,
    const MatrixXd& Sigma_k,
    const MatrixXd& weights_k,
    const vector<vector<bool>>& time_indep_vars_k,
    const vector<pair<VectorXd, MatrixXd>>& x_k,
    const vector<double>& S0_k,
    const vector<VectorXd>& S1_k,
    const vector<vector<bool>>& r_star_k,
    VectorXd* new_beta_k) {
  // Sanity-Check input.
  if (new_beta_k == nullptr) {
    cout << "ERROR in ComputeNewBeta: Null input." << endl;
    return false;
  }

  const int p_k_indep = x_k[0].first.size();
  const int p_k_dep = x_k[0].second.rows();
  const int p_k = old_beta_k.size();
  const int n = weights_k.rows();
  const int M_k = weights_k.cols();
  if (p_k == 0 || p_k_indep + p_k_dep != p_k || n == 0 || M_k == 0 ||
      x_k.size() != n ||
      (S0_k.size() != M_k && S0_k.size() != 1) ||
      (S1_k.size() != M_k && S1_k.size() != 1) ||
      r_star_k.size() != n || r_star_k[0].size() != M_k ||
      Sigma_k.rows() != p_k || Sigma_k.cols() != p_k || S1_k[0].size() != p_k ||
      (p_k_dep > 0 && x_k[0].second.cols() != M_k)) {
    cout << "ERROR in ComputeNewBeta: Mismatching dimensions on inputs: "
         << "S0_k.size(): " << S0_k.size() << ", S1_k.size(): "
         << S1_k.size() << ", x_k.size(): " << x_k.size()
         << ", old_beta_k.size(): " << old_beta_k.size()
         << ", Sigma_k.rows(): " << Sigma_k.rows()
         << ", Sigma_k.cols(): " << Sigma_k.cols()
         << ", weights_k.rows(): " << weights_k.rows()
         << ", weights_k.cols(): " << weights_k.cols() << endl;
    return false;
  }

  // Sanity-check Sigma_k is invertible.
  FullPivLU<MatrixXd> lu = Sigma_k.fullPivLu();
  if (!lu.isInvertible()) {
    cout << "ERROR: E-M algorithm failed on iteration " << itr_index
         << " due to singular information matrix.\n";
    return false;
  }

  VectorXd sum;
  sum.resize(p_k);
  sum.setZero();
  for (int m = 0; m < M_k; ++m) {
    const double& S0_km = S0_k[m];
    const VectorXd& S1_km = S1_k[m];  // p_k x 1
    const VectorXd quotient_km = S1_km / S0_km;  // p_k x 1
    for (int i = 0; i < n; ++i) {
      if (r_star_k[i][m] &&
          !MpleForIntervalCensoredData::AddConstantTimesXimMinusVector(
              m, time_indep_vars_k[i], weights_k(i, m), quotient_km, x_k[i], &sum)) {
        return false;
      }
    }
  }

  *new_beta_k = old_beta_k - Sigma_k.inverse() * sum;

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeNewLambda(
    const MatrixXd& weights_k,
    const vector<double>& S0_new_k,
    const vector<vector<bool>>& r_star_k,
    VectorXd* new_lambda_k) {
  // Sanity-Check input.
  if (new_lambda_k == nullptr) {
    cout << "ERROR in ComputeNewLambda: Null input." << endl;
    return false;
  }

  const int n = weights_k.rows();
  const int M_k = weights_k.cols();
  if (n == 0 || M_k == 0 ||
      (S0_new_k.size() != M_k && S0_new_k.size() != 1)) {
    cout << "ERROR in ComputeNewLambda: Mismatching dimensions on inputs: "
         << "S0_new_k.size(): " << S0_new_k.size()
         << ", weights_k.rows(): " << weights_k.rows()
         << ", weights_k.cols(): " << weights_k.cols() << endl;
    return false;
  }

  new_lambda_k->resize(M_k);
  for (int m = 0; m < M_k; ++m) {
    double numerator_km = 0.0;
    for (int i = 0; i < n; ++i) {
      if (r_star_k[i][m]) {
        numerator_km += weights_k(i, m);
      }
    }
    (*new_lambda_k)(m) = numerator_km / S0_new_k[m];
  }

  return true;
}

// Public version.
MpleReturnValue MultivariateMpleForIntervalCensoredData::ComputeVariance(
    const int n, const vector<int>& p, const int h_n_constant, const int max_itr,
    const double& convergence_threshold,
    const double& final_sigma,
    const vector<DependentCovariateInfo>& input,
    const vector<DependentCovariateEstimates>& estimates,
    MatrixXd* variance) {
  const int N_s = kNumberGaussianHermitePoints;
  double sum_hermite_weights = 0.0;
  vector<GaussianQuadratureTuple> gaussian_hermite_points;
  if (!ComputeGaussianHermitePointsAndSumWeights(
          N_s, &sum_hermite_weights, &gaussian_hermite_points)) {
    return MpleReturnValue::FAILED_PRELIMINARY_COMPUTATION;
  }

  VectorXd b;
  b.resize(N_s);
  for (int s = 0; s < N_s; ++s) {
    b(s) = final_sigma * sqrt(2.0) * gaussian_hermite_points[s].abscissa_;
  }

  return ComputeVariance(
      n, p, h_n_constant, max_itr, convergence_threshold, final_sigma,
      sum_hermite_weights, gaussian_hermite_points, b,
      input, estimates, variance);
}

// Private version.
MpleReturnValue MultivariateMpleForIntervalCensoredData::ComputeVariance(
    const int n, const vector<int>& p, const int h_n_constant, const int max_itr,
    const double& convergence_threshold,
    const double& final_sigma, const double& sum_hermite_weights,
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const VectorXd& b,
    const vector<DependentCovariateInfo>& input,
    const vector<DependentCovariateEstimates>& estimates,
    MatrixXd* variance) {
  if (variance == nullptr) {
    cout << "ERROR in ComputeVariance: Null Input." << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  double pl_at_beta;
  VectorXd pl_toggle_one_dim;
  MatrixXd pl_toggle_two_dim;
  if (!ComputeProfileLikelihoods(
          n, p, h_n_constant, max_itr, convergence_threshold, final_sigma,
          sum_hermite_weights, gaussian_hermite_points, b, input, estimates,
          &pl_at_beta, &pl_toggle_one_dim, &pl_toggle_two_dim)) {
    return MpleReturnValue::FAILED_VARIANCE;
  }

  if (!no_use_pos_def_variance_) {
    if (!ComputeAlternateVarianceFromProfileLikelihoods(
            n, h_n_constant, pl_toggle_one_dim, pl_toggle_two_dim, variance)) {
      return MpleReturnValue::FAILED_VARIANCE;
    }
  } else {
    if (!ComputeVarianceFromProfileLikelihoods(
            n, h_n_constant, pl_at_beta, pl_toggle_one_dim, pl_toggle_two_dim,
            variance)) {
      return MpleReturnValue::FAILED_VARIANCE;
    }
  }

  if (NegativeVariance(*variance)) {
    return MpleReturnValue::FAILED_NEGATIVE_VARIANCE;
  }

  return MpleReturnValue::SUCCESS;
}

bool MultivariateMpleForIntervalCensoredData::ComputeProfileLikelihoods(
    const int n, const vector<int>& p, const int h_n_constant, const int max_itr,
    const double& convergence_threshold,
    const double& final_sigma, const double& sum_hermite_weights,
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const VectorXd& b,
    const vector<DependentCovariateInfo>& input,
    const vector<DependentCovariateEstimates>& estimates,
    double* pl_at_beta,
    VectorXd* pl_toggle_one_dim,
    MatrixXd* pl_toggle_two_dim) {
  const int K = estimates.size();
  if (pl_at_beta == nullptr || pl_toggle_one_dim == nullptr ||
      pl_toggle_two_dim == nullptr || K == 0) {
    cout << "ERROR in ComputeProfileLikelihoods: Null Input." << endl;
    return false;
  }

  // Make a copy of final beta values, so that we can toggle one coordinate
  // at a time. Also, initialize this and final_lambda based on passed-in
  // values from estimates.
  vector<VectorXd> beta_plus_e_i(K);
  vector<VectorXd> final_lambda(K);
  for (int k = 0; k < K; ++k) {
    beta_plus_e_i[k] = estimates[k].beta_;
    final_lambda[k] = estimates[k].lambda_;
  }

  // Covariance Matrix will have Dim (1 + \sum p_k, 1 + \sum p_k): Each of the K
  // covariates \beta_k has size p_k, and an extra covariate for \sigma^2.
  int sum_p = 0;
  for (const int p_k : p) {
    sum_p += p_k;
  }
  const int cov_size = 1 + sum_p;
  if (!no_use_pos_def_variance_) {
    pl_toggle_one_dim->resize(n);
    pl_toggle_one_dim->setZero();
    pl_toggle_two_dim->resize(n, cov_size);
    pl_toggle_two_dim->setZero();
  } else {
    pl_toggle_one_dim->resize(cov_size);
    pl_toggle_two_dim->resize(cov_size, cov_size);
  }

  // First compute pl_n(final_beta).
  if (!ComputeProfileLikelihood(
          max_itr, convergence_threshold, sum_hermite_weights,
          gaussian_hermite_points, b, input,
          beta_plus_e_i, final_lambda,
          no_use_pos_def_variance_ ? pl_at_beta : nullptr,
          no_use_pos_def_variance_ ? nullptr : pl_toggle_one_dim)) {
    cout << "ERROR: Failed to compute Profile "
         << "Likelihood at final beta." << endl;
    return false;
  }

  // Now compute the profile likelihoods at the final beta that has
  // "one-dimension toggled"; do this for each of the p_k dimensions.
  // NOTE: In terms of what constant to use for h_n_constant_factor; for
  // univariate case, we used 5; in email with Donglin, he said:
  // "there is no firm rule--we need to empirically study the performance of each choice"
  const double h_n = static_cast<double>(h_n_constant) /
                     sqrt(static_cast<double>(n));

  const double final_sigma_toggle_one = sqrt(final_sigma * final_sigma + h_n);
  const double final_sigma_toggle_two =
     sqrt(final_sigma * final_sigma + 2.0 * h_n);

  // Will hold values for "b", for when sigma is the covariate being toggled.
  VectorXd b_toggle, b_double_toggle;
  b_toggle.resize(b.size());
  b_double_toggle.resize(b.size());
  for (int s = 0; s < b.size(); ++s) {
    b_toggle(s) =
        sqrt(2.0) * final_sigma_toggle_one * gaussian_hermite_points[s].abscissa_;
    b_double_toggle(s) =
        sqrt(2.0) * final_sigma_toggle_two * gaussian_hermite_points[s].abscissa_;
  }

  int k_coordinate = -1;
  int p_coordinate = -1;
  // The total number of toggle computations is:
  //   - Number one-dimension toggled:   1 + sum_p
  //   - Number of 2-dimensions toggled: (1 + sum_p) * (1 + sum_p + 1) / 2,
  // where for the toggle 2-dimensions, we don't need to compute all of the
  // (1 + sum_p)^2 coordinates since it is symmetric, just need to do e.g.
  // upper-triangle part, i.e. n (n + 1) / 2 where n = 1 + sum_p.
  const int num_toggles = (sum_p * sum_p + 5 * sum_p + 4) / 2;
  for (int i = 0; i < cov_size; ++i) {
    if (i == cov_size - 1) {
      k_coordinate = -1;
      p_coordinate = -1;
    } else {
      k_coordinate = 0;
      int sum_p_through_k = 0;
      while (true) {
        if (sum_p_through_k + p[k_coordinate] > i) break;
        sum_p_through_k += p[k_coordinate];
        k_coordinate++;
      }
      //PHBk_coordinate = i / p;
      //PHBp_coordinate = i % p;
      p_coordinate = i - sum_p_through_k;
      beta_plus_e_i[k_coordinate](p_coordinate) += h_n;
    }
    VectorXd col_i;
    col_i.resize(n);
    col_i.setZero();
    if (!ComputeProfileLikelihood(
            max_itr, convergence_threshold, sum_hermite_weights,
            gaussian_hermite_points,
            i == cov_size - 1 ? b_toggle : b,
            input, beta_plus_e_i, final_lambda,
            no_use_pos_def_variance_ ? &((*pl_toggle_one_dim)(i)) : nullptr,
            no_use_pos_def_variance_ ? nullptr : &col_i)) {
      cout << "ERROR: Failed to compute Profile "
           << "Likelihood for i = " << i + 1 << "." << endl;
      return false;
    }
    // Annoying copy, since Eigen doesn't allow passing a column by reference
    // (in particular, passing &(pl_toggle_two_dim->col(i)) as an argument to
    // ComputeProfileLikelihood above doesn't work.
    // TODO(PHB): If we end up keeping the positive-definite variance computation,
    // figure out a better way to do this, that avoids this copy (e.g. try the
    // Eigen swap() function, or look at how I handled this for other MatrixXd
    // objects above).
    if (!no_use_pos_def_variance_) pl_toggle_two_dim->col(i) = col_i;

    // Reset beta_plus_e_i to return to final_beta.
    if (k_coordinate >= 0 && p_coordinate >= 0) {
      beta_plus_e_i[k_coordinate](p_coordinate) -= h_n;
    }
    if (logging_on_) {
      time_t current_time = time(nullptr);
      cout << asctime(localtime(&current_time))
           << "Finished " << (i + 1) << " of "
           << (no_use_pos_def_variance_ ? num_toggles : cov_size)
           << " computations for covariance matrix." << endl;
    }
  }

  // No need to compute the 'toggle-two coordinates' values if just using
  // the positive-definite version of Covariance computation.
  if (!no_use_pos_def_variance_) return true;

  // Now compute the profile likelihoods at the final beta that has
  // "two-dimensions toggled"; do this for each of the
  // (cov_size_C_2 = cov_size(cov_size + 1) / 2)
  // pairs of the cov_size dimensions.
  pl_toggle_two_dim->setZero();
  int k_coordinate_one = -1;
  int k_coordinate_two = -1;
  int p_coordinate_one = -1;
  int p_coordinate_two = -1;
  for (int i = 0; i < cov_size; ++i) {
    if (i == cov_size - 1) {
      k_coordinate_one = -1;
      p_coordinate_one = -1;
    } else {
      k_coordinate_one = 0;
      int sum_p_through_k_one = 0;
      while (true) {
        if (sum_p_through_k_one + p[k_coordinate_one] > i) break;
        sum_p_through_k_one += p[k_coordinate_one];
        k_coordinate_one++;
      }
      p_coordinate_one = i - sum_p_through_k_one;
      beta_plus_e_i[k_coordinate_one](p_coordinate_one) += h_n;
    }
    VectorXd e_i;
    e_i.resize(cov_size);
    e_i.setZero();
    e_i(i) = h_n;
    for (int j = 0; j <= i; ++j) {
      if (j == cov_size - 1) {
        k_coordinate_two = -1;
        p_coordinate_two = -1;
      } else {
        k_coordinate_two = 0;
        int sum_p_through_k_two = 0;
        while (true) {
          if (sum_p_through_k_two + p[k_coordinate_two] > j) break;
          sum_p_through_k_two += p[k_coordinate_two];
          k_coordinate_two++;
        }
        p_coordinate_two = j - sum_p_through_k_two;
        beta_plus_e_i[k_coordinate_two](p_coordinate_two) += h_n;
      }
      if (!ComputeProfileLikelihood(
              max_itr, convergence_threshold, sum_hermite_weights,
              gaussian_hermite_points,
              (i == cov_size - 1 ?
               (j == cov_size - 1 ? b_double_toggle : b_toggle) :
               (j == cov_size - 1 ? b_toggle : b)),
              input, beta_plus_e_i, final_lambda,
              &((*pl_toggle_two_dim)(i, j)), nullptr)) {
        cout << "ERROR: Failed to compute Profile "
             << "Likelihood for coordinate (i, j) = (" << i + 1
             << ", " << j + 1 << ")." << endl;
        return false;
      }
      // Untoggle beta_plus_e_i.
      if (k_coordinate_two >= 0 && p_coordinate_two >= 0) {
        beta_plus_e_i[k_coordinate_two](p_coordinate_two) -= h_n;
      }
      if (logging_on_) {
        time_t current_time = time(nullptr);
        cout << asctime(localtime(&current_time))
             << "Finished " << (cov_size + j + 1 + (i * i + i) / 2)
             << " of " << num_toggles << " computations for Covariance Matrix."
             << endl;
      }
    }
    // Reset beta_plus_e_i to return to final_beta.
    if (k_coordinate_one >= 0 && p_coordinate_one >= 0) {
      beta_plus_e_i[k_coordinate_one](p_coordinate_one) -= h_n;
    }
  }

  const bool PHB_print_profile_likelihoods = false;
  if (logging_on_ && PHB_print_profile_likelihoods) {
    PrintProfileLikelihoods(*pl_at_beta, *pl_toggle_one_dim, *pl_toggle_two_dim);
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeProfileLikelihood(
    const int max_itr,
    const double& convergence_threshold,
    const double& sum_hermite_weights,
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const VectorXd& b,
    const vector<DependentCovariateInfo>& input,
    const vector<VectorXd>& beta,
    const vector<VectorXd>& lambda,
    double* pl, VectorXd* pl_alternate) {
  if (pl == nullptr && pl_alternate == nullptr) {
    cout << "ERROR in ComputeProfileLikelihood: Null Input." << endl;
    return false;
  }

  const int K = input.size();
  const int N_s = gaussian_hermite_points.size();
  if (K == 0  || N_s == 0 || beta.size() != K || lambda.size() != K || b.size() != N_s) {
    cout << "ERROR in ComputeProfileLikelihood: Empty input: "
         << "input.size(): " << K
         << ", beta.size(): "  << beta.size()
         << ", lambda.size(): "  << lambda.size()
         << ", gaussian_hermite_points.size(): " << gaussian_hermite_points.size()
         << ", b.size(): " << b.size() << endl;
    return false;
  }

  // The following data structure will hold all the intermediate values
  // that are generated at each step of the E-M algorithm.
  vector<DependentCovariateIntermediateValues> intermediate_values;
  intermediate_values.resize(K);

  vector<VectorXd> old_lambda;
  old_lambda.resize(K);

  // Since \theta = (\beta_1, ..., \beta_K, \sigma^2) will not change in the
  // E-M algorithm to find the maximizing \lambda, we compute here (outside
  // of the E-M iterations) the fields of 'intermediate_values' that will not
  // change: beta_, and exp_beta_x_plus_b_.
  // Also, initialize old_lambda[k] and intermediate_values[k].lambda_
  for (int k = 0; k < K; ++k) {
    intermediate_values[k].beta_ = beta[k];
    intermediate_values[k].lambda_ = lambda[k];
    old_lambda[k] = lambda[k];
    // Compute exp(\beta_k^T * X_kim + \sqrt{2} * \sigma * y_s)
    if (!ComputeExpBetaXPlusB(
            intermediate_values[k].beta_, b, input[k].time_indep_vars_,
            input[k].x_,
            &(intermediate_values[k].exp_beta_x_),
            &(intermediate_values[k].exp_beta_x_plus_b_))) {
      cout << "ERROR: Failed in Computing exp(beta^T * X + b) for k = "
           << k + 1 << " while Computing Profile Likelihood.\n";
      return false;
    }
  }

  // Run (profile, i.e. only maximimizing one parameter \lambda while
  // leaving \beta parameter fixed) E-M algorithm to find maximizing \lambda.
  int iteration_index = 1;
  for (; iteration_index < max_itr; ++iteration_index) {
    // Run E-Step.
    vector<VectorXd> posterior_means;
    vector<MatrixXd> weights;
    VectorXd phi;
    if (!DoEStep(1.0 /* Not Used */, sum_hermite_weights, b, input,
                 gaussian_hermite_points,
                 &intermediate_values, &weights, &posterior_means, &phi)) {
      cout << "ERROR: Failed in EStep of iteration " << iteration_index << endl;
      return false;
    }

    // No need to do M-step; just the part where \lambda is updated.
    for (int k = 0; k < K; ++k) {
      vector<double> S0_new_k;
      if (!ComputeS0(beta[k], posterior_means[k],
                     input[k].time_indep_vars_, input[k].x_,
                     input[k].r_star_, &S0_new_k)) {
        cout << "ERROR in computation for k = " << k + 1 << endl;
        return false;
      }

      // Update Lambda.
      if (!ComputeNewLambda(weights[k], S0_new_k, input[k].r_star_,
                            &(intermediate_values[k].lambda_))) {
        cout << "ERROR in Computing new Lambda for k = " << k + 1 << endl;
        return false;
      }
    }

    // Check Convergence Criterion.
    if (ProfileEmAlgorithmHasConverged(
            convergence_threshold, old_lambda, intermediate_values)) {
      bool PHB_print_each_pl_convergence_itr_num = true;
      if (logging_on_ && PHB_print_each_pl_convergence_itr_num) {
        cout << "PL converged in: " << iteration_index << endl;
      }
      break;
    }

    // Update old_lambda.
    for (int k = 0; k < K; ++k) {
      old_lambda[k] = intermediate_values[k].lambda_;
    }
  }

  // Abort if we failed to converge after max_itr.
  if (iteration_index >= max_itr) {
    cout << "ERROR in ComputeProfileLikelihood: "
         << "E-M algorithm exceeded maximum number of allowed "
         << "iterations: " << max_itr << endl;
    return false;
  }

  if (!EvaluateLogLikelihoodFunctionAtBetaLambda(
          gaussian_hermite_points, input,
          &intermediate_values, pl, pl_alternate)) {
    return false;
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::EvaluateLogLikelihoodFunctionAtBetaLambda(
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const vector<DependentCovariateInfo>& input,
    vector<DependentCovariateIntermediateValues>* intermediate_values,
    double* likelihood, VectorXd* e_i_likelihoods) {
  if (likelihood == nullptr && e_i_likelihoods == nullptr) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Null Input." << endl;
    return false;
  }

  const int K = intermediate_values->size();
  const int N_s = gaussian_hermite_points.size();
  if (N_s == 0 || K == 0  || input.size() != K) {
    cout << "ERROR in ComputeProfileLikelihood: Empty input: "
         << "intermediate_values.size(): " << K
         << ", input.size(): " << input.size() << endl;
    return false;
  }

  for (int k = 0; k < K; ++k) {
    // No need to re-compute exp(\beta_k^T * X_kim + \sqrt{2} * \sigma * y_s),
    // since this was computed once (outside the E-M algorithm) and won't have
    // changed since \beta and \sigma are not changing.

    // Compute S_L and S_U.
    if (!ComputeS(input[k].distinct_times_, input[k].lower_time_bounds_,
                  (*intermediate_values)[k].lambda_,
                  (*intermediate_values)[k].exp_beta_x_plus_b_, 
                  &((*intermediate_values)[k].S_L_))) {
      cout << "ERROR in Computing S^L for k = " << k + 1 << endl;
      return false;
    }
    if (!ComputeS(input[k].distinct_times_, input[k].upper_time_bounds_,
                  (*intermediate_values)[k].lambda_,
                  (*intermediate_values)[k].exp_beta_x_plus_b_, 
                  &((*intermediate_values)[k].S_U_))) {
      cout << "ERROR in Computing S^U for k = " << k + 1 << endl;
      return false;
    }

    // Compute exp(-G(S_L)) and exp(-G(S_U)).
    if (!ComputeExpTransformation(
            input[k].transformation_G_,
            (*intermediate_values)[k].S_L_,
            &((*intermediate_values)[k].exp_neg_g_S_L_))) {
      cout << "ERROR in Computing exp(G(S^L) for k = " << k + 1 << endl;
      return false;
    }
    if (!ComputeExpTransformation(
            input[k].transformation_G_,
            (*intermediate_values)[k].S_U_,
            &((*intermediate_values)[k].exp_neg_g_S_U_))) {
      cout << "ERROR in Computing exp(G(S^U) for k = " << k + 1 << endl;
      return false;
    }
  }

  // Compute Likelihood L_n.
  return no_use_pos_def_variance_ ?
    EvaluateLogLikelihoodFunctionAtBetaLambda(
      gaussian_hermite_points, *intermediate_values, likelihood) :
    EvaluateAlternateLogLikelihoodFunctionAtBetaLambda(
      gaussian_hermite_points, *intermediate_values, e_i_likelihoods);
}

bool MultivariateMpleForIntervalCensoredData::EvaluateLogLikelihoodFunctionAtBetaLambda(
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const vector<DependentCovariateIntermediateValues>& intermediate_values,
    double* likelihood) {
  if (likelihood == nullptr) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Null Input."
         << endl;
    return false;
  }
  const int K = intermediate_values.size();
  const int N_s = gaussian_hermite_points.size();
  if (N_s == 0 || K == 0) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Empty Input.\n";
    return false;
  }

  const int n = intermediate_values[0].exp_beta_x_plus_b_.size();
  if (n == 0) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: n = 0.\n";
    return false;
  }

  *likelihood = 0.0;
  for (int i = 0; i < n; ++i) {
    double inner_summand = 0.0;
    for (int s = 0; s < N_s; ++s) {
      double product_term = 1.0;
      for (int k = 0; k < K; ++k) {
        const MatrixXd& exp_neg_g_S_L_k = intermediate_values[k].exp_neg_g_S_L_;
        const MatrixXd& exp_neg_g_S_U_k = intermediate_values[k].exp_neg_g_S_U_;
        if (i == 0 && s == 0 &&  // Only need to check dimensions on first pass.
            (exp_neg_g_S_L_k.cols() != N_s || exp_neg_g_S_U_k.cols() != N_s ||
             exp_neg_g_S_L_k.rows() != n || exp_neg_g_S_U_k.rows() != n)) {
          cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: "
               << "Mismatching dimensions: "
               << "exp_neg_g_S_L[k].rows(): " << exp_neg_g_S_L_k.rows()
               << "exp_neg_g_S_L[k].cols(): " << exp_neg_g_S_L_k.cols()
               << "exp_neg_g_S_U[k].rows(): " << exp_neg_g_S_U_k.rows()
               << "exp_neg_g_S_U[k].cols(): " << exp_neg_g_S_U_k.cols() << endl;
          return false;
        }
        product_term *= (exp_neg_g_S_L_k(i, s) - exp_neg_g_S_U_k(i, s));
      }
      const double mu_s = gaussian_hermite_points[s].weight_;
      inner_summand += mu_s * product_term;
    }
    *likelihood += log(inner_summand / sqrt(PI));
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::EvaluateAlternateLogLikelihoodFunctionAtBetaLambda(
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const vector<DependentCovariateIntermediateValues>& intermediate_values,
    VectorXd* e_i_likelihoods) {
  if (e_i_likelihoods == nullptr) {
    cout << "ERROR in EvaluateAltLogLikelihoodFunctionAtBetaLambda: Null Input."
         << endl;
    return false;
  }

  const int K = intermediate_values.size();
  const int N_s = gaussian_hermite_points.size();
  if (N_s == 0 || K == 0) {
    cout << "ERROR in EvaluateAltLogLikelihoodFunctionAtBetaLambda: Empty Input.\n";
    return false;
  }

  const int n = intermediate_values[0].exp_beta_x_plus_b_.size();
  if (n == 0) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: n = 0.\n";
    return false;
  }

  for (int i = 0; i < n; ++i) {
    double inner_summand = 0.0;
    for (int s = 0; s < N_s; ++s) {
      double product_term = 1.0;
      for (int k = 0; k < K; ++k) {
        const MatrixXd& exp_neg_g_S_L_k = intermediate_values[k].exp_neg_g_S_L_;
        const MatrixXd& exp_neg_g_S_U_k = intermediate_values[k].exp_neg_g_S_U_;
        if (i == 0 && s == 0 &&  // Only need to check dimensions on first pass.
            (exp_neg_g_S_L_k.cols() != N_s || exp_neg_g_S_U_k.cols() != N_s ||
             exp_neg_g_S_L_k.rows() != n || exp_neg_g_S_U_k.rows() != n)) {
          cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: "
               << "Mismatching dimensions: "
               << "exp_neg_g_S_L[k].rows(): " << exp_neg_g_S_L_k.rows()
               << "exp_neg_g_S_L[k].cols(): " << exp_neg_g_S_L_k.cols()
               << "exp_neg_g_S_U[k].rows(): " << exp_neg_g_S_U_k.rows()
               << "exp_neg_g_S_U[k].cols(): " << exp_neg_g_S_U_k.cols() << endl;
          return false;
        }
        product_term *= (exp_neg_g_S_L_k(i, s) - exp_neg_g_S_U_k(i, s));
      }
      const double mu_s = gaussian_hermite_points[s].weight_;
      inner_summand += mu_s * product_term;
    }
    (*e_i_likelihoods)(i) = log(inner_summand / sqrt(PI));
  }

  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeVarianceFromProfileLikelihoods(
      const int n, const int h_n_constant, const double& pl_at_theta,
      const VectorXd& pl_toggle_one_dim, const MatrixXd& pl_toggle_two_dim,
      MatrixXd* variance) {
  if (variance == nullptr) {
    cout << "ERROR in ComputeVarianceFromProfileLikelihoods: Null input."
         << endl;
    return false;
  }

  const int cov_dim = pl_toggle_one_dim.size();
  if (cov_dim == 0 || pl_toggle_two_dim.rows() != cov_dim ||
      pl_toggle_two_dim.cols() != cov_dim) {
    cout << "ERROR in ComputeVarianceFromProfileLikelihoods: Mismatching "
         << "dimensions: pl_toggle_one_dim.size(): " << cov_dim
         << ", pl_toggle_two_dim.rows(): " << pl_toggle_two_dim.rows()
         << ", pl_toggle_two_dim.cols(): " << pl_toggle_two_dim.cols()
         << endl;
    return false;
  }

  const double h_n_squared =
      pow(static_cast<double>(h_n_constant), 2.0) /
      static_cast<double>(n);

  MatrixXd variance_inverse;
  variance_inverse.resize(cov_dim, cov_dim);
  for (int i = 0; i < cov_dim; ++i) {
    for (int j = 0; j <= i; ++j) {
      variance_inverse(i, j) =
          (pl_at_theta - pl_toggle_one_dim(i) -
           pl_toggle_one_dim(j) + pl_toggle_two_dim(i, j)) / h_n_squared;
      // Variance-Covariance matrix is symmetric; set the upper-triangular
      // coordinates from the lower-triangular ones (rather than recomputing).
      if (j < i) {
        variance_inverse(j, i) = variance_inverse(i, j);
      }
    }
  }

  // Sanity-check varinace inverse is invertible.
  FullPivLU<MatrixXd> lu = variance_inverse.fullPivLu();
  if (!lu.isInvertible()) {
    Eigen::IOFormat format(Eigen::FullPrecision);
    cout << "ERROR: variance matrix is not invertible:" << endl
         << variance_inverse.format(format) << endl;
    return false;
  }
  
  *variance = -1.0 * variance_inverse.inverse();
 
  return true;
}

bool MultivariateMpleForIntervalCensoredData::ComputeAlternateVarianceFromProfileLikelihoods(
      const int n, const int h_n_constant,
      const VectorXd& pl_toggle_none,
      const MatrixXd& pl_toggle_one_dim,
      MatrixXd* variance) {
  if (variance == nullptr) {
    cout << "ERROR in ComputeAltVarianceFromProfileLikelihoods: Null input."
         << endl;
    return false;
  }

  const int cov_dim = pl_toggle_one_dim.cols();
  if (n == 0 || pl_toggle_one_dim.rows() != n ||
      cov_dim == 0 || pl_toggle_none.size() != n) {
    cout << "ERROR in ComputeAltVarianceFromProfileLikelihoods: empty matrix "
         << "pl_toggle_one_dim.rows(): " << pl_toggle_one_dim.rows()
         << ", pl_toggle_one_dim.cols(): " << pl_toggle_one_dim.cols()
         << ", pl_toggle_none.size(): " << pl_toggle_none.size()
         << endl;
    return false;
  }

  const double h_n = static_cast<double>(h_n_constant) /
                     sqrt(static_cast<double>(n));

  MatrixXd variance_inverse;
  variance_inverse.resize(cov_dim, cov_dim);
  variance_inverse.setZero();
  for (int i = 0; i < n; ++i) {
    const double& toggle_none_i = pl_toggle_none(i);
    VectorXd ps_i;
    ps_i.resize(cov_dim);
    for (int index = 0; index < cov_dim; ++index) {
      ps_i(index) = (pl_toggle_one_dim(i, index) - toggle_none_i) / h_n; 
    }
    variance_inverse += ps_i * ps_i.transpose();
  }

  // Sanity-check varinace inverse is invertible.
  FullPivLU<MatrixXd> lu = variance_inverse.fullPivLu();
  if (!lu.isInvertible()) {
    Eigen::IOFormat format(Eigen::FullPrecision);
    cout << "ERROR: variance matrix is not invertible:" << endl
         << variance_inverse.format(format) << endl;
    return false;
  }
  
  *variance = variance_inverse.inverse();
 
  return true;
}

}  // namespace regression
