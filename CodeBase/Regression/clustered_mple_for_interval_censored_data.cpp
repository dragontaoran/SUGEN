// Date: Dec 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "clustered_mple_for_interval_censored_data.h"

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
#include <iomanip>  // std::setprecision
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

bool ClusteredMpleForIntervalCensoredData::logging_on_;
bool ClusteredMpleForIntervalCensoredData::no_use_pos_def_variance_;
bool ClusteredMpleForIntervalCensoredData::force_one_right_censored_;
bool ClusteredMpleForIntervalCensoredData::PHB_use_exp_lambda_convergence_;
int ClusteredMpleForIntervalCensoredData::num_failed_variance_computations_;
vector<Timer> ClusteredMpleForIntervalCensoredData::timers_;
const int kNumberGaussianHermitePoints = 40;
const int kNumberGaussianLaguerrePoints = 40;

// Dummy functions for debugging.
namespace {

void PrintIntermediateValues(
    const int iteration_index,
    const bool print_first_constants,
    const bool print_first_beta_lambda, const bool print_all_beta_lambda,
    const IntermediateValues& intermediate_values) {
  // Nothing to do if all print flags are false.
  if (!print_first_constants && !print_first_beta_lambda &&
      !print_all_beta_lambda) {
    return;
  }

  if (iteration_index == 1 &&
      (print_first_beta_lambda || print_first_constants)) {
    if (print_first_beta_lambda) {
      Eigen::IOFormat format(Eigen::FullPrecision);
      cout << "\nAfter first iteration, beta:\n"
           << intermediate_values.beta_.transpose().format(format) << endl;
      cout << "Lambda:\n"
           << intermediate_values.lambda_.transpose() << endl;
    }
    if (print_first_constants) {
      cout << "\n\tbeta: " << intermediate_values.beta_
           << "\n\tlambda: " << intermediate_values.lambda_ << endl;
      for (int k = 0; k < intermediate_values.family_values_.size(); ++k) {
        const FamilyIntermediateValues& family_values_k =
            intermediate_values.family_values_[k];
        cout << "Iteration " << iteration_index << " values for k = " << k + 1 << ":"
             << "\n\ta_" << k + 1 << ": " << family_values_k.a_is_
             << "\n\tc_" << k + 1 << ": " << family_values_k.c_is_
             << "\n\td_" << k + 1 << ": " << family_values_k.d_is_
             << "\n\tf_" << k + 1 << ": " << family_values_k.f_i_
             << "\n\tS_L_" << k + 1 << ": " << family_values_k.S_L_
             << "\n\tS_U_" << k + 1 << ": " << family_values_k.S_U_
             << "\n\texp_neg_g_S_L_" << k + 1 << ": "
             << family_values_k.exp_neg_g_S_L_
             << "\n\texp_neg_g_S_U_" << k + 1 << ": "
             << family_values_k.exp_neg_g_S_U_;
        for (int m = 0; m < family_values_k.exp_beta_x_plus_b_.size(); ++m) {
          cout << "\n\texp_beta_x_plus_b_" << k + 1 << "," << m << ": "
               << family_values_k.exp_beta_x_plus_b_[m];
        }
      }
    }
    cout << endl
         << "################################################################"
         << endl;
  } else if (print_all_beta_lambda) {
    Eigen::IOFormat format(Eigen::FullPrecision);
    cout << "\nAfter iteration " << iteration_index + 1
         << ", beta:\n"
         << intermediate_values.beta_.transpose().format(format) << endl;
    cout << "Lambda:\n"
         << intermediate_values.lambda_.transpose() << endl;
    cout << endl
         << "################################################################"
         << endl;
  }
}

void PrintFinalValues(
    const double& b_variance,
    const DependentCovariateEstimates& estimates) {
  Eigen::IOFormat format(Eigen::FullPrecision);
  cout << "\nFinal beta:\n"
       << estimates.beta_.transpose().format(format) << endl;
  cout << "Final Lambda:\n"
       << estimates.lambda_.transpose().format(format) << endl;
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

void ClusteredMpleForIntervalCensoredData::InitializeTimers() {
  const int num_timers = 0;
  for (int i = 0; i < num_timers; ++i) {
    timers_.push_back(Timer());
  }
}

void ClusteredMpleForIntervalCensoredData::PrintTimers() {
  for (int i = 0; i < timers_.size(); ++i) {
    const Timer& t = timers_[i];
    cout << "Timer " << i << ": " << test_utils::GetElapsedTime(t) << endl;
  }
}

bool ClusteredMpleForIntervalCensoredData::InitializeInput(
    const double& r, const TimeDepIntervalCensoredData& data,
    InputValues* input) {
  if (input == nullptr) return false;

  if (logging_on_) {
    time_t current_time = time(nullptr);
    cout << endl << asctime(localtime(&current_time))
         << "Preparing data structures to run E-M algorithm.\n";
  }

  // Number of gaussian-laguerre points will be set to a constant,
  // independent of k.
  // TODO(PHB): Allow N to be N_k, dependent on k. 
  const int N = kNumberGaussianLaguerrePoints;

  // Set the family-independent fields of input.
  //   - r_
  input->r_ = r;
  //   - transformation_G_
  if (!ConstructTransformation(r, &input->transformation_G_)) {
    return false;
  }
  //   - points_and_weights_
  if (r != 0.0 &&
      !ComputeGaussianLaguerrePoints(N, r, &input->points_and_weights_)) {
    return false;
  }
  //   - sum_points_times_weights_
  input->sum_points_times_weights_ = 0.0;
  if (r != 0.0) {
    for (int q = 0; q < N; ++q) {
      const GaussianQuadratureTuple& current_point =
          input->points_and_weights_[q];
      input->sum_points_times_weights_ +=
          current_point.weight_ * current_point.abscissa_;
    }
  }
  //   - distinct_times_
  // If force_one_right_censored_ is false, just copy the input set of distinct
  // times. Otherwise, we'll need to recompute the set of distinct times.
  double max_left_time = -1.0;
  set<double> times_made_right_censored;
  if (!force_one_right_censored_) {
    input->distinct_times_ = data.distinct_times_[0];
  } else {
    // First go through, and find the maximum L-time.
    for (const SubjectInfo& info : data.subject_info_) {
      const EventTimeAndCause& time_info = info.times_[0];
      if (time_info.lower_ > max_left_time) {
        max_left_time = time_info.lower_;
      }
    }

    // Now Shift any Right-Time that is bigger than 'max_left_time' to
    // be right-censored (i.e. inf).
    // Also, update distinct_times with all non-negative (shouldn't be any)
    // and non-infinity values.
    for (const SubjectInfo& info : data.subject_info_) {
      const EventTimeAndCause& time_info = info.times_[0];
      if (time_info.lower_ > 0.0) {
        input->distinct_times_.insert(time_info.lower_);
      }
      const double& upper = time_info.upper_;
      if (upper == numeric_limits<double>::infinity()) {
        // Nothing to do.
      } else if (upper <= max_left_time) {
        input->distinct_times_.insert(upper);
      } else {
        // Keep track of times that were shifted to be infinity.
        times_made_right_censored.insert(upper);
      }
    }
  }
  //   - family_values_
  const int K = data.family_index_to_id_.size();
  input->family_values_.resize(K);
  // Store the (independent variable) values in each data.subject_info_
  // to the appropriate pair<VectorXd, MatrixXd>.
  for (const SubjectInfo& subject_info_i : data.subject_info_) {
    const int family_index = subject_info_i.family_index_;
    if (family_index < 0 || family_index >= K) {
      cout << "ERROR: Invalid Family Index (" << family_index
           << ") for Subject." << endl;
      return false;
    }
    // Sanity-check there is a single event-type and model.
    if (subject_info_i.times_.size() != 1 ||
        subject_info_i.linear_term_values_.size() != 1 ||
        subject_info_i.is_time_indep_.size() != 1) {
      cout << "ERROR in Performing EM Algorithm: Expected "
           << "a single dependent covariate for subject, but found "
           << subject_info_i.times_.size() << " sets of distinct times, "
           << subject_info_i.linear_term_values_.size()
           << " sets of covariate values, and "
           << subject_info_i.is_time_indep_.size()
           << " sets of other event information." << endl;
      return false;
    }
    FamilyInputValues& dep_cov_info = input->family_values_[family_index];
    // Set time_indep_vars_.
    dep_cov_info.time_indep_vars_.push_back(subject_info_i.is_time_indep_[0]);
    // Set lower and upper times for each Subject, and all of that
    // Subjects values at all distinct times.
    vector<double>& lower_time_bounds = dep_cov_info.lower_time_bounds_;
    lower_time_bounds.push_back(subject_info_i.times_[0].lower_);
    vector<double>& upper_time_bounds = dep_cov_info.upper_time_bounds_;
    if (!force_one_right_censored_ || subject_info_i.times_[0].upper_ <= max_left_time) {
      upper_time_bounds.push_back(subject_info_i.times_[0].upper_);
    } else {
      upper_time_bounds.push_back(numeric_limits<double>::infinity());
    }
    vector<pair<VectorXd, MatrixXd>>& x = dep_cov_info.x_;
    if (times_made_right_censored.empty() ||
        subject_info_i.linear_term_values_[0].second.cols() == 0) {
      x.push_back(subject_info_i.linear_term_values_[0]);
    } else {
      x.push_back(pair<VectorXd, MatrixXd>());
      VectorXd& time_indep_values = x.back().first;
      time_indep_values = subject_info_i.linear_term_values_[0].first;
      const MatrixXd& values_to_copy = 
          subject_info_i.linear_term_values_[0].second;
      const int num_cols_to_skip = times_made_right_censored.size();
      const int num_cols_to_copy = values_to_copy.cols() - num_cols_to_skip;
      MatrixXd& time_dep_values = x.back().second;
      time_dep_values =
          values_to_copy.block(0, 0, values_to_copy.rows(), num_cols_to_copy);
    }
    
    // Set r_star_: I(t_k <= R*_i), where R*_i is R_i if R_i \neq \infty,
    // otherwise it is L_i.
    dep_cov_info.r_star_.push_back(vector<bool>());
    if (!ComputeRiStarIndicator(
            input->distinct_times_, lower_time_bounds.back(),
            upper_time_bounds.back(), &dep_cov_info.r_star_.back())) {
      return false;
    }

    // Compute X * X^T once.
    dep_cov_info.x_x_transpose_.push_back(vector<MatrixXd>());
    if (!ComputeXTimesXTranspose(
            dep_cov_info.time_indep_vars_.back(), x.back(),
            &dep_cov_info.x_x_transpose_.back())) {
      return false;
    }
  }

  if (logging_on_) {
    time_t current_time = time(nullptr);
    cout << asctime(localtime(&current_time))
         << "Done preparing input for E-M algorithm.\n";
  }

  return true;
}

MpleReturnValue
ClusteredMpleForIntervalCensoredData::PerformEmAlgorithmForParameterEstimation(
    const double& convergence_threshold,
    const int h_n_constant, const int max_itr,
    const InputValues& input,
    int* num_iterations, double* log_likelihood, double* b_variance,
    DependentCovariateEstimates* estimates, MatrixXd* variance) {
  // Sanity-Check input.
  if (b_variance == nullptr || estimates == nullptr) {
    cout << "ERROR in Performing EM Algorithm: Null input."
         << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  // Sanity-check dimensions match.
  const int K = input.family_values_.size();
  if (K == 0 || input.family_values_[0].x_.empty()) {
    cout << "ERROR in Performing EM Algorithm: Mismatching dimensions on inputs: "
         << "input.family_values_.size(): " << input.family_values_.size() << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }
  const int M = input.distinct_times_.size();
  const int p_indep = input.family_values_[0].x_[0].first.size();
  const int p_dep = input.family_values_[0].x_[0].second.rows();
  const int p = p_indep + p_dep;
  if (p == 0 || M == 0) {
    cout << "ERROR in Performing EM Algorithm: "
         << "p (" << p << ") or M (" << M << ") is zero." << endl;
    return MpleReturnValue::FAILED_PRELIMINARY_COMPUTATION;
  }

  // Check if there is a single family.
  if (K == 1) {
    MpleForIntervalCensoredData::SetLoggingOn(logging_on_);
    MpleForIntervalCensoredData::SetForceOneRightCensored(force_one_right_censored_);
    MpleForIntervalCensoredData::SetNoUsePositiveDefiniteVariance(
        no_use_pos_def_variance_);
    return MpleForIntervalCensoredData::PerformEmAlgorithmForParameterEstimation(
        input.r_, convergence_threshold, h_n_constant, max_itr,
        input.distinct_times_,
        input.family_values_[0].lower_time_bounds_,
        input.family_values_[0].upper_time_bounds_,
        input.family_values_[0].time_indep_vars_,
        input.family_values_[0].x_, num_iterations, log_likelihood,
        &estimates->beta_, &estimates->lambda_, variance);
  }

  // Check input has the requisite fields set.
  if ((input.points_and_weights_.empty() && input.r_ != 0.0) ||
      input.family_values_[0].x_x_transpose_.empty() ||
      input.family_values_[0].r_star_.empty()) {
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
  IntermediateValues intermediate_values;
  intermediate_values.family_values_.resize(K);

  // Initialize first guess for beta, lambda, and sigma.
  double old_sigma = estimates->beta_.size() != p ? 1.0 : sqrt(*b_variance);
  if (estimates->beta_.size() != p) {
    // Initial input to E-M algorithm will be with \beta = 0.
    VectorXd& old_beta = intermediate_values.beta_;
    old_beta.resize(p);
    old_beta.setZero();

    // Initial input to E-M algorithm will be with \lambda = 1 / M_k.
    VectorXd& old_lambda = intermediate_values.lambda_;
    old_lambda.resize(M);
    for (int m = 0; m < M; ++m) {
      old_lambda(m) = 1.0 / M;
    }
  } else {
    // Use the passed-in values to kick-off the E-M algorithm.
    intermediate_values.beta_ = estimates->beta_;
    intermediate_values.lambda_ = estimates->lambda_;
  }

  // Determine how often to print status updates, based on how much data there
  // is (and thus, how long each iteration may take).
  int complexity = 0;
  for (int k = 0; k < K; ++k) {
    const int n_k = input.family_values_[k].lower_time_bounds_.size();
    complexity += n_k * M * p;
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
              intermediate_values.beta_, b,
              input.family_values_[k].time_indep_vars_,
              input.family_values_[k].x_,
              &(intermediate_values.family_values_[k].exp_beta_x_),
              &(intermediate_values.family_values_[k].exp_beta_x_plus_b_))) {
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
    if (!DoMStep(iteration_index, input, phi, posterior_means, weights,
                 b_variance, &intermediate_values, estimates)) {
      cout << "ERROR: Failed in MStep of iteration " << iteration_index << endl;
      return MpleReturnValue::FAILED_M_STEP;
    }

    // Check Convergence Criterion.
    if (EmAlgorithmHasConverged(
            convergence_threshold, old_sigma * old_sigma, *b_variance,
            intermediate_values.beta_, estimates->beta_,
            intermediate_values.lambda_, estimates->lambda_,
            &current_difference)) {
      *num_iterations = iteration_index;
      break;
    }

    // Copy current values to intermediate values.
    old_sigma = sqrt(*b_variance);

    intermediate_values.beta_ = estimates->beta_;
    intermediate_values.lambda_ = estimates->lambda_;

    if (iteration_index == 1 && PHB_abort_after_one_iteration) break;

    PrintIntermediateValues(
        iteration_index, PHB_print_constants_first_iteration,
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
      cout << "\nFinal beta:\n"
           << estimates->beta_.transpose().format(format) << endl;
    }
    if (PHB_print_final_lamba_on_failure) {
      Eigen::IOFormat format(Eigen::FullPrecision);
      cout << "Final Lambda:\n"
           << estimates->lambda_.transpose().format(format) << endl;    
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

  // Compute Likelihood at final values.
  if (log_likelihood != nullptr) {
    // Temporarily override no_use_pos_def_variance_ option, so we can
    // compute the log-likelihood at the final estimates.
    const bool orig_no_use_pos_def_variance = no_use_pos_def_variance_;
    no_use_pos_def_variance_ = true;
    if (!ComputeProfileLikelihood(
            max_itr, convergence_threshold, sum_hermite_weights,
            gaussian_hermite_points, b, input,
            estimates->beta_, estimates->lambda_, log_likelihood, nullptr)) {
      return MpleReturnValue::FAILED_VARIANCE;
    }
    // Restore no_use_pos_def_variance_ to its proper value.
    no_use_pos_def_variance_ = orig_no_use_pos_def_variance;
  }

  // Compute variance.
  if (variance != nullptr) {
    // Time-stamp progress.
    if (logging_on_) {
      time_t current_time = time(nullptr);
      cout << endl << asctime(localtime(&current_time))
           << "Finished E-M algorithm "
           << "to compute beta and lambda in " << iteration_index
           << " iterations.\nComputing Covariance Matrix..." << endl;
    }
    
    int n = 0;
    for (const FamilyInputValues& family_info : input.family_values_) {
      n += family_info.lower_time_bounds_.size();
    }
    MpleReturnValue var_result = ComputeVariance(
        n, p, h_n_constant, max_itr,
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

bool ClusteredMpleForIntervalCensoredData::ComputeGaussianLaguerrePoints(
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

bool ClusteredMpleForIntervalCensoredData::ComputeGaussianHermitePointsAndSumWeights(
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

bool ClusteredMpleForIntervalCensoredData::ComputeGaussianHermitePoints(
    const int n, vector<GaussianQuadratureTuple>* gaussian_hermite_points) {
  // Sanity-Check input.
  if (gaussian_hermite_points == nullptr) {
    cout << "ERROR in ComputeGaussianHermitePoints: Null input." << endl;
    return false;
  }
  return ComputeGaussHermiteQuadrature(
      n, 0.0  /* alpha */, 0.0  /* a */, 1.0  /* b */, gaussian_hermite_points);
}

bool ClusteredMpleForIntervalCensoredData::ConstructTransformation(
    const double& r, Expression* transformation_G_k) {
  if (transformation_G_k == nullptr) {
    cout << "ERROR in ConstructTransformation: Null input." << endl;
    return false;
  }

  const string r_str = Itoa(r);
  const string g_str = r == 0.0 ? "x" : "log(1+" + r_str + "x)/" + r_str;
  if (!ParseExpression(g_str, transformation_G_k)) {
    cout << "ERROR: Unable to parse '"
         << g_str << "' as an Expression." << endl;
    return false;
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeRiStarIndicator(
    const set<double>& distinct_times,
    const double& lower_time, const double& upper_time,
    vector<bool>* r_i_star_indicator) {
  // Sanity-check input.
  if (r_i_star_indicator == nullptr) {
    cout << "ERROR in ComputeRiStarIndicator: Null input." << endl;
    return false;
  }

  const unsigned int M = distinct_times.size();
  if (M == 0) {
    cout << "ERROR in ComputeRiStarIndicator: Empty distinct times." << endl;
    return false;
  }

  r_i_star_indicator->resize(M);
  const double r_star_i =
      upper_time == numeric_limits<double>::infinity() ?
      lower_time : upper_time;
  int m = 0;
  for (const double& time : distinct_times) {
    (*r_i_star_indicator)[m] = r_star_i >= time;
    ++m;
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeXTimesXTranspose(
    const vector<bool>& time_indep_vars_i,
    const pair<VectorXd, MatrixXd>& x_k,
    vector<MatrixXd>* x_k_x_k_transpose) {
  if (x_k_x_k_transpose == nullptr ||
      (x_k.first.size() == 0 &&
       (x_k.second.rows() == 0 || x_k.second.cols() == 0))) {
    cout << "ERROR in ComputeXTimesXTranspose: Null input." << endl;
    return false;
  }

  // Only need one distinct time, if all covariates are time-independent.
  const int p_dep = x_k.second.rows();
  const int M = p_dep == 0 ? 1 : x_k.second.cols();
  x_k_x_k_transpose->clear();

  for (int m = 0; m < M; ++m) {
    x_k_x_k_transpose->push_back(MatrixXd());
    MatrixXd& m_th_entry = x_k_x_k_transpose->back();
    VectorXd x_kim;
    if (!MpleForIntervalCensoredData::GetXim(
            m, time_indep_vars_i, x_k, &x_kim)) {
      return false;
    }
    m_th_entry = x_kim * x_kim.transpose();  // m_th_entry is p x p.
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeExpBetaXPlusB(
    const VectorXd& beta,
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

  const int n_k = x_k.size();
  const int p = beta.size();
  const int p_indep = x_k[0].first.size();
  const int p_dep = x_k[0].second.rows();
  const int N_s = b.size();
  if (n_k == 0 || p == 0 || N_s == 0 ||
      time_indep_vars_k.size() != n_k || time_indep_vars_k[0].size() != p ||
      p_dep + p_indep != p || (p_dep > 0 && x_k[0].second.cols() == 0)) {
    cout << "ERROR in ComputeExpBetaXPlusB: Mismatching dimensions on inputs: "
         << "beta.size(): " << p << ", p_dep: " << p_dep << ", p_indep: "
         << p_indep << ", time_indep_vars_k.size(): " << time_indep_vars_k.size()
         << ", x_k.size(): " << n_k << ", b.size(): " << N_s << endl;
    return false;
  }

  exp_beta_x_k->resize(n_k);
  exp_beta_x_plus_b_k->resize(n_k);
  for (int i = 0; i < n_k; ++i) {
    // Only need one distinct time, if all covariates are time-independent.
    const int p_dep_i = x_k[i].second.rows();
    const int M_i = p_dep_i == 0 ? 1 : x_k[i].second.cols();
    (*exp_beta_x_k)[i].clear();
    (*exp_beta_x_plus_b_k)[i].resize(N_s, M_i);
    for (int m = 0; m < M_i; ++m) {
      double dot_product;
      if (!MpleForIntervalCensoredData::ComputeBetaDotXim(
              m, time_indep_vars_k[i], beta, x_k[i], &dot_product)) {
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

bool ClusteredMpleForIntervalCensoredData::EmAlgorithmHasConverged(
    const double& convergence_threshold,
    const double& b_variance_old, const double& b_variance_new,
    const VectorXd& old_beta, const VectorXd& new_beta,
    const VectorXd& old_lambda, const VectorXd& new_lambda,
    double* current_difference) {
  if (old_beta.size() != new_beta.size()) {
    cout << "ERROR in EmAlgorithmHasConverged: Mismatching dimensions: "
         << "old_beta (" << old_beta.size() << ") vs. new_beta ("
         << new_beta.size() << ")." << endl;
    return false;
  }
  if (old_lambda.size() != new_lambda.size()) {
    cout << "ERROR in EmAlgorithmHasConverged: Mismatching dimensions: "
         << "old_lambda (" << old_lambda.size() << ") vs. new_lambda ("
         << new_lambda.size() << ")." << endl;
    return false;
  }

  // Compute Cumulative Lambda.
  VectorXd cumulative_lambda_old, cumulative_lambda_new;
  cumulative_lambda_old.resize(old_lambda.size());
  cumulative_lambda_old.setZero();
  cumulative_lambda_new.resize(new_lambda.size());
  cumulative_lambda_new.setZero();
  double PHB_lambda = 0.0;
  for (int i = 0; i < old_lambda.size(); ++i) {
    if (i != 0) {
      cumulative_lambda_old(i) = cumulative_lambda_old(i - 1);
      cumulative_lambda_new(i) = cumulative_lambda_new(i - 1);
    }
    cumulative_lambda_old(i) += old_lambda(i);
    cumulative_lambda_new(i) += new_lambda(i);
    PHB_lambda += abs(old_lambda(i) - new_lambda(i));
  }

  // PHB TEMP.
  // Compute Exp Neg Cumulative Lambda.
  VectorXd exp_neg_cum_lambda_old, exp_neg_cum_lambda_new;
  if (PHB_use_exp_lambda_convergence_) {
    exp_neg_cum_lambda_old.resize(old_lambda.size());
    exp_neg_cum_lambda_new.resize(new_lambda.size());
    for (int i = 0; i < old_lambda.size(); ++i) {
      exp_neg_cum_lambda_old(i) = exp(-1.0 * cumulative_lambda_old(i));
      exp_neg_cum_lambda_new(i) = exp(-1.0 * cumulative_lambda_new(i));
    }
  }

  double PHB_beta = 0.0;
  for (int i = 0; i < old_beta.size(); ++i) {
    PHB_beta += abs(old_beta(i) - new_beta(i));
  }
  // END PHB TEMP.

  // Compute difference between previous and current iterations.
  // TODO(PHB): Make delta a global constant.
  const double delta = 0.01;
  const double max_beta =
      VectorAbsoluteDifferenceSafe(old_beta, new_beta, delta);
  const double max_cumulative_lambda = PHB_use_exp_lambda_convergence_ ?
      VectorAbsoluteDifferenceSafe(
          exp_neg_cum_lambda_old, exp_neg_cum_lambda_new, delta) :
      VectorAbsoluteDifferenceSafe(
          cumulative_lambda_old, cumulative_lambda_new, delta);
  const double b_diff =
      AbsoluteDifferenceSafe(b_variance_old, b_variance_new, delta);

  const double curr_diff = b_diff + max_beta + max_cumulative_lambda;

  if (current_difference != nullptr) {
    *current_difference = curr_diff;
  }
 
  const bool PHB_junk = false;
  if (PHB_junk) {
    const double PHB_diff =
        PHB_beta + PHB_lambda + abs(b_variance_old - b_variance_new);
    *current_difference = PHB_diff;
    return PHB_diff < convergence_threshold;
  }
  return curr_diff < convergence_threshold;
}

bool ClusteredMpleForIntervalCensoredData::ProfileEmAlgorithmHasConverged(
    const double& convergence_threshold,
    const VectorXd& old_lambda, const VectorXd& new_lambda) {
  if (old_lambda.size() != new_lambda.size()) {
    cout << "ERROR in ProfileEmAlgorithmHasConverged: Mismatching dimensions: "
         << "old_lambda (" << old_lambda.size() << ") vs. new_lambda ("
         << new_lambda.size() << ")." << endl;
    return false;
  }

  // Construct cumulative Lambda.
  VectorXd cumulative_old, cumulative_new;
  cumulative_old.resize(old_lambda.size());
  cumulative_new.resize(new_lambda.size());
  cumulative_old.setZero();
  cumulative_new.setZero();
  double PHB_lambda = 0.0;
  for (int i = 0; i < old_lambda.size(); ++i) {
    if (i != 0) {
      cumulative_old(i) = cumulative_old(i - 1);
      cumulative_new(i) = cumulative_new(i - 1);
    }
    cumulative_old(i) += old_lambda(i);
    cumulative_new(i) += new_lambda(i);
    PHB_lambda += abs(old_lambda(i) - new_lambda(i));
  }

  // PHB TEMP.
  // Compute Exp Neg Cumulative Lambda.
  VectorXd exp_neg_cum_old, exp_neg_cum_new;
  if (PHB_use_exp_lambda_convergence_) {
    exp_neg_cum_old.resize(old_lambda.size());
    exp_neg_cum_new.resize(new_lambda.size());
    for (int i = 0; i < old_lambda.size(); ++i) {
      exp_neg_cum_old(i) = exp(-1.0 * cumulative_old(i));
      exp_neg_cum_new(i) = exp(-1.0 * cumulative_new(i));
    }
  }
  // PHB TEMP.

  // TODO(PHB): Make delta a global constant.
  const double delta = 0.01;
  const double max_cumulative = PHB_use_exp_lambda_convergence_ ?
      VectorAbsoluteDifferenceSafe(exp_neg_cum_old, exp_neg_cum_new, delta) :
      VectorAbsoluteDifferenceSafe(cumulative_old, cumulative_new, delta);

  const bool PHB_junk = false;
  if (PHB_junk) {
    return PHB_lambda < convergence_threshold;
  }
  return max_cumulative < convergence_threshold;
}

bool ClusteredMpleForIntervalCensoredData::DoEStep(
    const double& current_sigma, const double& sum_hermite_weights,
    const VectorXd& b,
    const InputValues& input,
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    IntermediateValues* intermediate_values,
    vector<MatrixXd>* weights,
    vector<VectorXd>* posterior_means,
    VectorXd* phi) {
  // Sanity-check input.
  if (weights == nullptr || posterior_means == nullptr) {
    cout << "ERROR in DoEStep: Null input." << endl;
    return false;
  }

  // Sanity-check dimensions match.
  const int K = input.family_values_.size();
  const int N_s = gaussian_hermite_points.size();
  if (N_s == 0 || K == 0 || intermediate_values->family_values_.size() != K) {
    cout << "ERROR in DoEStep: Mismatching dimensions on inputs: "
         << "input.size(): " << input.family_values_.size()
         << ", gaussian_hermite_points.size(): " << N_s
         << ", intermediate_values.size(): "
         << intermediate_values->family_values_.size() << endl;
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
    if (!ComputeS(input.distinct_times_, input.family_values_[k].lower_time_bounds_,
                  intermediate_values->lambda_,
                  intermediate_values->family_values_[k].exp_beta_x_plus_b_, 
                  &(intermediate_values->family_values_[k].S_L_))) {
      cout << "ERROR in Computing S^L for k = " << k + 1 << endl;
      return false;
    }
    if (!ComputeS(input.distinct_times_, input.family_values_[k].upper_time_bounds_,
                  intermediate_values->lambda_,
                  intermediate_values->family_values_[k].exp_beta_x_plus_b_, 
                  &(intermediate_values->family_values_[k].S_U_))) {
      cout << "ERROR in Computing S^U for k = " << k + 1 << endl;
      return false;
    }

    // Compute exp(-G(S_L)) and exp(-G(S_U)).
    if (!ComputeExpTransformation(
            input.transformation_G_,
            intermediate_values->family_values_[k].S_L_,
            &(intermediate_values->family_values_[k].exp_neg_g_S_L_))) {
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }
    if (!ComputeExpTransformation(
            input.transformation_G_,
            intermediate_values->family_values_[k].S_U_,
            &(intermediate_values->family_values_[k].exp_neg_g_S_U_))) {
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }
  }

  // Compute a_kis, c_kis, d_kis, and f_ki.
  for (int k = 0; k < K; ++k) {
    if (!ComputeConstants(
            (input.r_ == 0.0 ? 0.0 : (1.0 / input.r_)),
            sum_hermite_weights, gaussian_hermite_points,
            input.family_values_[k].upper_time_bounds_,
            intermediate_values->family_values_[k].S_L_,
            intermediate_values->family_values_[k].S_U_,
            intermediate_values->family_values_[k].exp_neg_g_S_L_,
            intermediate_values->family_values_[k].exp_neg_g_S_U_,
            &(intermediate_values->family_values_[k].a_is_),
            &(intermediate_values->family_values_[k].c_is_),
            &(intermediate_values->family_values_[k].d_is_),
            &(intermediate_values->family_values_[k].e_s_),
            &(intermediate_values->family_values_[k].f_i_))) {
      cout << "ERROR in performing computation for k = " << k + 1 << endl;
      return false;
    }

    // Compute Posterior Means v_i.
    if (!ComputePosteriorMeans(
            input.r_,
            input.sum_points_times_weights_,
            b, gaussian_hermite_points,
            input.family_values_[k].upper_time_bounds_,
            intermediate_values->family_values_[k].a_is_,
            intermediate_values->family_values_[k].c_is_,
            intermediate_values->family_values_[k].d_is_,
            intermediate_values->family_values_[k].e_s_,
            intermediate_values->family_values_[k].f_i_,
            &((*posterior_means)[k]))) {
      cout << "ERROR in Computing Posterior Means for k = " << k + 1 << endl;
      return false;
    }

    // Compute Weights w_ki.
    if (!ComputeWeights(
            input.r_,
            gaussian_hermite_points, input.points_and_weights_,
            input.distinct_times_,
            input.family_values_[k].lower_time_bounds_,
            input.family_values_[k].upper_time_bounds_,
            intermediate_values->family_values_[k].exp_beta_x_plus_b_, 
            intermediate_values->lambda_,
            intermediate_values->family_values_[k].S_L_,
            intermediate_values->family_values_[k].S_U_,
            intermediate_values->family_values_[k].a_is_,
            intermediate_values->family_values_[k].c_is_,
            intermediate_values->family_values_[k].d_is_,
            intermediate_values->family_values_[k].e_s_,
            intermediate_values->family_values_[k].f_i_,
            (*posterior_means)[k],
            &((*weights)[k]))) {
      cout << "ERROR in Computing Weights for k = " << k + 1 << endl;
      return false;
    }
  }

  // Compute \phi_k := (\sum_s \mu_s * e_ks * 2 * \sigma^2 * y_s^2) /
  //                   (\sum_s \mu_s * e_ks)
  // NOTE: Profile-Likelihood computations don't recompute \phi; such
  // use-cases are distinguished by having a nullptr passed to DoEStep().
  if (phi != nullptr) {
    phi->resize(K);
    for (int k = 0; k < K; ++k) {
      double numerator = 0.0;
      double denominator = 0.0;
      for (int s = 0; s < N_s; ++s) {
        const double mu_times_e =
            gaussian_hermite_points[s].weight_ *
            intermediate_values->family_values_[k].e_s_(s);
        denominator += mu_times_e;
        numerator +=
          mu_times_e * 2.0 * current_sigma * current_sigma *
          gaussian_hermite_points[s].abscissa_ *
          gaussian_hermite_points[s].abscissa_;
      }
      if (denominator == 0.0) {
        cout << "ERROR in DoEStep: zero denominator for k = " << k << endl;
        return false;
      }
      (*phi)(k) = numerator / denominator;
    }
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeS(
    const set<double>& distinct_times,
    const vector<double>& time_bounds_k,
    const VectorXd& lambda,
    const vector<MatrixXd>& exp_beta_x_plus_b_k,
    MatrixXd* S_k) {
  // Sanity-check input.
  if (S_k == nullptr) {
    cout << "ERROR in ComputeS: Null input." << endl;
    return false;
  }

  const int n_k = time_bounds_k.size();
  const int M = distinct_times.size();
  if (n_k == 0 || M == 0 ||
      lambda.size() != M || exp_beta_x_plus_b_k.size() != n_k ||
      exp_beta_x_plus_b_k[0].rows() == 0 || exp_beta_x_plus_b_k[0].cols() == 0) {
    cout << "ERROR in ComputeS: Mismatching dimensions on inputs: "
         << "distinct_times.size(): " << distinct_times.size()
         << ", time_bounds_k.size(): " << time_bounds_k.size()
         << ", lambda.size(): " << lambda.size()
         << ", exp_beta_x_plus_b_k.size(): " << exp_beta_x_plus_b_k.size() << endl;
    return false;
  }
  const int N_s = exp_beta_x_plus_b_k[0].rows();

  // Go through each b value, computing S_kis(b_s) for each.
  S_k->resize(n_k, N_s);
  S_k->setZero();
  for (int i = 0; i < n_k; ++i) {
    // Check if R_ki is infinity for this subject. If so set S_k(i, s)
    // to infinity (for each s in [1..N_s])
    if (time_bounds_k[i] == numeric_limits<double>::infinity()) {
      for (int s = 0; s < N_s; ++s) {
        (*S_k)(i, s) = numeric_limits<double>::infinity();
        // NOTE: The following line is the only one that needs to be
        // updated in order to match Donglin's results:
        const bool PHB_junk = false;
        if (PHB_junk) {
          (*S_k)(i, s) = 99999.0;
        }
      }
      continue;
    }

    // Loop through each time point m in [1..M], checking if distinct_time
    // distinct_times[m] is less than or equal to L_ki (resp. R_ki). If not, skip this
    // time (and all later times, since they will automatically be later, since
    // distinct_times is sorted). If so, add to running total.
    int m = 0;
    const bool covariates_all_time_indep =
        exp_beta_x_plus_b_k[i].cols() == 1;
    for (const double& time : distinct_times) {
      const int col_to_use = covariates_all_time_indep ? 0 : m;
      if (time <= time_bounds_k[i]) {
        for (int s = 0; s < N_s; ++s) {
          (*S_k)(i, s) +=
              lambda(m) * exp_beta_x_plus_b_k[i](s, col_to_use);
        }
      } else {
        break;
      }
      ++m;
    }
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeExpTransformation(
    const Expression& transformation_G_k,
    const MatrixXd& S_k,
    MatrixXd* exp_neg_g_S_k) {
  if (exp_neg_g_S_k == nullptr || S_k.rows() == 0 || S_k.cols() == 0) {
    cout << "ERROR in ComputeExpTransformation: Null input." << endl;
    return false;
  }

  const int n_k = S_k.rows();
  const int N_s = S_k.cols();

  exp_neg_g_S_k->resize(n_k, N_s);
  for (int s = 0; s < N_s; ++s) {
    for (int i = 0; i < n_k; ++i) {
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

bool ClusteredMpleForIntervalCensoredData::ComputeEis(
    const MatrixXd& exp_neg_g_S_L_k,
    const MatrixXd& exp_neg_g_S_U_k,
    VectorXd* e_k) {
  if (e_k == nullptr) {
    cout << "ERROR in ComputeEis: Null input.\n";
    return false;
  }

  const int n_k = exp_neg_g_S_L_k.rows();
  const int N_s = exp_neg_g_S_L_k.cols();
  if (n_k == 0 || N_s == 0 || e_k->size() != N_s) {
    cout << "ERROR in ComputeEis: empty intermediate values." << endl;
    return false;
  }

  if (exp_neg_g_S_U_k.rows() != n_k || exp_neg_g_S_U_k.cols() != N_s) {
    cout << "ERROR in ComputeEis: Inconsistent dimensions." << endl;
    return false;
  }
  for (int s = 0; s < N_s; ++s) {
    (*e_k)(s) = 1.0;
    for (int i = 0; i < n_k; ++i) {
      (*e_k)(s) *= (exp_neg_g_S_L_k(i, s) - exp_neg_g_S_U_k(i, s));
    }
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeConstants(
      const double& r_k_inverse, const double& sum_hermite_weights,
      const vector<GaussianQuadratureTuple>& hermite_points_and_weights,
      const vector<double>& upper_time_bounds_k,
      const MatrixXd& S_L_k,
      const MatrixXd& S_U_k,
      const MatrixXd& exp_neg_g_S_L_k,
      const MatrixXd& exp_neg_g_S_U_k,
      MatrixXd* a_k, MatrixXd* c_k, MatrixXd* d_k,
      VectorXd* e_k,
      VectorXd* f_k) {
  if (a_k == nullptr || c_k == nullptr || d_k == nullptr ||
      e_k == nullptr || f_k == nullptr) {
    cout << "ERROR in ComputeConstants: Null Input." << endl;
    return false;
  }
  // Sanity-check dimensions match.
  const int n_k = upper_time_bounds_k.size();
  const int N_s = hermite_points_and_weights.size();
  if (n_k == 0 || N_s == 0 ||
      S_L_k.rows() != n_k || S_L_k.cols() != N_s ||
      S_U_k.rows() != n_k || S_U_k.cols() != N_s ||
      exp_neg_g_S_L_k.rows() != n_k || exp_neg_g_S_L_k.cols() != N_s ||
      exp_neg_g_S_U_k.rows() != n_k || exp_neg_g_S_U_k.cols() != N_s) {
    cout << "ERROR in ComputeConstants: Mismatching dimensions: "
         << "n_k: " << n_k << ", N_s: " << N_s
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

  a_k->resize(n_k, N_s);
  c_k->resize(n_k, N_s);
  d_k->resize(n_k, N_s);
  e_k->resize(N_s);
  f_k->resize(n_k);

  // Compute e_is separately (it's the only one that has a value, even when
  // r_k_inverse is zero).
  if (!ComputeEis(exp_neg_g_S_L_k, exp_neg_g_S_U_k, e_k)) {
    return false;
  }

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
  for (int i = 0; i < n_k; ++i) {
    double f_second_sum = 0.0;
    for (int s = 0; s < N_s; ++s) {
      (*a_k)(i, s) = r_k_inverse + S_L_k(i, s);
      (*c_k)(i, s) = r_k_inverse + S_U_k(i, s);
      (*d_k)(i, s) = (*e_k)(s) / (exp_neg_g_S_L_k(i, s) - exp_neg_g_S_U_k(i, s));
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

bool ClusteredMpleForIntervalCensoredData::ComputePosteriorMeans(
    const double& r,
    const double& sum_points_times_weights,
    const VectorXd& b,
    const vector<GaussianQuadratureTuple>& hermite_points_and_weights,
    const vector<double>& upper_time_bounds_k,
    const MatrixXd& a_k,
    const MatrixXd& c_k,
    const MatrixXd& d_k,
    const VectorXd& e_k,
    const VectorXd& f_k,
    VectorXd* v_k) {
  // Sanity-check input.
  if (v_k == nullptr) {
    cout << "ERROR in ComputePosteriorMeans: Null input." << endl;
    return false;
  }

  // Sanity-check dimensions match.
  const int n_k = upper_time_bounds_k.size();
  const int N_s = hermite_points_and_weights.size();
  if (n_k == 0 || N_s == 0 || f_k.size() != n_k || b.size() != N_s ||
      a_k.rows() != n_k || a_k.cols() != N_s ||
      c_k.rows() != n_k || c_k.cols() != N_s ||
      d_k.rows() != n_k || d_k.cols() != N_s || e_k.size() != N_s) {
    cout << "ERROR in ComputeConstants: Mismatching dimensions: "
         << "n_k: " << n_k << ", N_s: " << N_s
         << ", b.size(): " << b.size()
         << ", f_k.size(): " << f_k.size()
         << ", a_k.rows(): " << a_k.rows()
         << ", a_k.cols(): " << a_k.cols()
         << ", c_k.rows(): " << c_k.rows()
         << ", c_k.cols(): " << c_k.cols()
         << ", d_k.rows(): " << d_k.rows()
         << ", d_k.cols(): " << d_k.cols()
         << ", e_k.size(): " << e_k.size() << endl;
    return false;
  }

  v_k->resize(n_k);

  // Separate formula for posterior mean when r is zero.
  if (r == 0.0) {
    for (int i = 0; i < n_k; ++i) {
      double numerator = 0.0;
      double denominator = 0.0;
      for (int s = 0; s < N_s; ++s) {
        const double mu_times_e = 
          hermite_points_and_weights[s].weight_ * e_k(s);
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
  for (int i = 0; i < n_k; ++i) {
    double second_sum = 0.0;
    for (int s = 0; s < N_s; ++s) {
      const double second_sum_second_term = pow(a_k(i, s), -1.0 - (1.0 / r)) -
          (upper_time_bounds_k[i] == numeric_limits<double>::infinity() ?
           0.0 : pow(c_k(i, s), -1.0 - (1.0 / r)));
      second_sum +=
          hermite_factor(s) * d_k(i, s) * second_sum_second_term;
    }
    if (f_k(i) == 0.0) {
      cout << "ERROR in Computing Posterior Mean: f_k is 0.0 for i = " << i + 1
           << ". Note:\n\ta_k.row(i): " << a_k.row(i) << endl << "\tc_k.row(i): "
           << c_k.row(i) << "\n\td_k.row(i): " << d_k.row(i) << "\n\te_k: "
           << e_k << endl;
      return false;
    }
    (*v_k)(i) = (1.0 / f_k(i)) * sum_points_times_weights * second_sum;
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeWeights(
    const double& r,
    const vector<GaussianQuadratureTuple>& hermite_points_and_weights,
    const vector<GaussianQuadratureTuple>& laguerre_points_and_weights_k,
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds_k,
    const vector<double>& upper_time_bounds_k,
    const vector<MatrixXd>& exp_beta_x_plus_b_k,
    const VectorXd& lambda,
    const MatrixXd& S_L_k,
    const MatrixXd& S_U_k,
    const MatrixXd& a_k,
    const MatrixXd& c_k,
    const MatrixXd& d_k,
    const VectorXd& e_k,
    const VectorXd& f_k,
    const VectorXd& v_k,
    MatrixXd* weights_k) {
  // Sanity-Check input.
  if (weights_k == nullptr) {
    cout << "ERROR in Computing Weights: Null input." << endl;
    return false;
  }

  const int n_k = lower_time_bounds_k.size();
  const int N_s = e_k.size();
  const int N_k = laguerre_points_and_weights_k.size();
  const int M = distinct_times.size();
  if (n_k == 0 || N_s == 0 || (N_k == 0 && r != 0.0) || M == 0 ||
      exp_beta_x_plus_b_k.size() != n_k || exp_beta_x_plus_b_k[0].rows() != N_s ||
      (exp_beta_x_plus_b_k[0].cols() != M && exp_beta_x_plus_b_k[0].cols() != 1) ||
      upper_time_bounds_k.size() != n_k || lambda.size() != M ||
      S_L_k.rows() != n_k || S_L_k.cols() != N_s ||
      S_U_k.rows() != n_k || S_U_k.cols() != N_s ||
      a_k.rows() != n_k || a_k.cols() != N_s ||
      c_k.rows() != n_k || c_k.cols() != N_s ||
      d_k.rows() != n_k || d_k.cols() != N_s ||
      f_k.size() != n_k || v_k.size() != n_k) {
    cout << "ERROR in ComputeWeights: Mismatching dimensions on inputs: "
         << ", distinct_times.size(): " << distinct_times.size()
         << ", lower_time_bounds_k.size(): " << lower_time_bounds_k.size()
         << ", upper_time_bounds_k.size(): " << upper_time_bounds_k.size()
         << ", lambda.size(): " << lambda.size()
         << ", exp_beta_x_plus_b_k.size(): " << exp_beta_x_plus_b_k.size()
         << ", v_k.size(): " << v_k.size()
         << ", f_k.size(): " << f_k.size()
         << ", a_k.rows(): " << a_k.rows()
         << ", a_k.cols(): " << a_k.cols()
         << ", c_k.rows(): " << c_k.rows()
         << ", c_k.cols(): " << c_k.cols()
         << ", d_k.rows(): " << d_k.rows()
         << ", d_k.cols(): " << d_k.cols()
         << ", e_k.size(): " << e_k.size()
         << ", S_L_k.rows(): " << S_L_k.rows()
         << ", S_L_k.cols(): " << S_L_k.cols()
         << ", S_U_k.rows(): " << S_U_k.rows()
         << ", S_U_k.cols(): " << S_U_k.cols() << endl;
    return false;
  }

  weights_k->resize(n_k, M);
  const double neg_one_minus_r_k_inverse =
      r == 0.0 ? 0.0 : (-1.0 - (1.0 / r));

  // First, compute the "Hermite Sum" for each (i, s). Here, the "hermite sum"
  // represents the second two lines of w_kim (see paper), and we compute it
  // here because it does *not* depend on m.
  MatrixXd hermite_sums;
  hermite_sums.resize(n_k, N_s);
  if (r != 0.0) {
    for (int i = 0; i < n_k; ++i) {
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

  for (int i = 0; i < n_k; ++i) {
    const bool covariates_all_time_indep = exp_beta_x_plus_b_k[i].cols() == 1;
    const double l_i = lower_time_bounds_k[i];
    const double u_i = upper_time_bounds_k[i];
    const double f_ki_inverse = 1.0 / f_k(i);
    int m = 0;
    for (const double& time : distinct_times) {
      if (time <= l_i) {
        (*weights_k)(i, m) = 0.0;
      } else if (u_i != numeric_limits<double>::infinity() && time <= u_i) {
        const int col_to_use = covariates_all_time_indep ? 0 : m;
        if (r == 0.0) {
          double numerator = 0.0;
          double denominator = 0.0;
          for (int s = 0; s < N_s; ++s) {
            const GaussianQuadratureTuple& points_and_weights =
                hermite_points_and_weights[s]; 
            const double mu_times_e = points_and_weights.weight_ * e_k(s);
            denominator += mu_times_e;
            const double numerator_numerator =
                lambda(m) * exp_beta_x_plus_b_k[i](s, col_to_use);
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
                lambda(m) * hermite_sums(i, s) *
                exp_beta_x_plus_b_k[i](s, col_to_use);
          }
          (*weights_k)(i, m) = f_ki_inverse * hermite_sum;
        }
      } else {
        // These weights will not be used; just set a dummy value.
        (*weights_k)(i, m) = 0.0;
      }
      ++m;
    }
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::DoMStep(
    const int itr_index,
    const InputValues& input,
    const VectorXd& phi,
    const vector<VectorXd>& posterior_means,
    const vector<MatrixXd>& weights,
    double* new_sigma_squared,
    IntermediateValues* intermediate_values,
    DependentCovariateEstimates* new_estimates) {
  // Sanity-Check input.
  if (new_sigma_squared == nullptr || new_estimates == nullptr) {
    cout << "ERROR in DoMStep: Null input." << endl;
    return false;
  }

  const int K = input.family_values_.size();
  if (K == 0 || phi.size() != K || intermediate_values->family_values_.size() != K ||
      posterior_means.size() != K || posterior_means[0].size() == 0 ||
      weights.size() != K || weights[0].rows() == 0 || weights[0].cols() == 0) {
    cout << "ERROR in DoMStep: Mismatching dimensions on inputs: "
         << "input.size(): " << K
         << ", phi.size(): " << phi.size()
         << ", intermediate_values.size(): " << intermediate_values->family_values_.size()
         << ", posterior_means.size(): " << posterior_means.size()
         << ", weights.size(): " << weights.size() << endl;
    return false;
  }

  for (int k = 0; k < K; ++k) {
    // Compute v_ki * exp(beta_k^T * X_kim), for each i in [1..n_k] and
    // m in [1..M] (these values will be used over and over in formula
    // for new beta).
    if (!ComputeSummandTerm(
            posterior_means[k], intermediate_values->family_values_[k].exp_beta_x_,
            &(intermediate_values->family_values_[k].v_exp_beta_x_))) {
      cout << "ERROR in performing computation at E-M step " << itr_index << endl;
      return false;
    }
  }

  // Compute constants that will be used to compute new beta.
  vector<double> S0;
  vector<VectorXd> S1;
  vector<MatrixXd> S2;
  if (!ComputeSValues(input, *intermediate_values, &S0, &S1, &S2)) {
    cout << "ERROR in performing computation at E-M step " << itr_index << endl;
    return false;
  }

  // Compute term 'Sigma' that will be used to compute new beta.
  MatrixXd Sigma;
  if (!ComputeSigma(input, weights, S0, S1, S2, &Sigma)) {
    cout << "ERROR in performing computation at E-M step " << itr_index << endl;
    return false;
  }

  // Compute new Beta.
  if (!ComputeNewBeta(
          itr_index, input, intermediate_values->beta_, Sigma, weights, S0, S1,
          &(new_estimates->beta_))) {
    cout << "ERROR in Computing new Beta at E-M step " << itr_index << endl;
    return false;
  }

  // Compute the denominator used for new lambda.
  vector<double> S0_new;
  if (!ComputeS0(input, new_estimates->beta_, posterior_means, &S0_new)) {
    cout << "ERROR in performing computation at E-M step " << itr_index << endl;
    return false;
  }

  // Update Lambda.
  if (!ComputeNewLambda(input, weights, S0_new, &(new_estimates->lambda_))) {
    cout << "ERROR in Computing new Lambda at E-M step " << itr_index << endl;
    return false;
  }

  // Update sigma.
  double numerator = 0.0;
  for (int k = 0; k < K; ++k) {
    numerator += phi(k);
  }
  *new_sigma_squared = numerator / static_cast<double>(K);

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeSummandTerm(
    const VectorXd& v_k,
    const vector<vector<double>>& exp_beta_x_k,
    vector<VectorXd>* v_exp_beta_x_k) {
  // Sanity-check input.
  if (v_exp_beta_x_k == nullptr) {
    cout << "ERROR in ComputeSummandTerm: Null input." << endl;
    return false;
  }

  const int n_k = v_k.size();
  if (n_k == 0 || exp_beta_x_k.size() != n_k) {
    cout << "ERROR in ComputeSummandTerm: Mismatching dimensions on inputs."
         << endl;
    return false;
  }

  v_exp_beta_x_k->resize(n_k);

  for (int i = 0; i < n_k; ++i) {
    const int M = exp_beta_x_k[i].size();
    (*v_exp_beta_x_k)[i].resize(M);
    for (int m = 0; m < M; ++m) {
      (*v_exp_beta_x_k)[i](m) = v_k(i) * exp_beta_x_k[i][m];
    }
  }
  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeSValues(
    const InputValues& input,
    const IntermediateValues& intermediate_values,
    vector<double>* S0, vector<VectorXd>* S1, vector<MatrixXd>* S2) {
  // Sanity-check input.
  if (S0 == nullptr || S1 == nullptr || S2 == nullptr) {
    cout << "ERROR in ComputeSValues: Null input." << endl;
    return false;
  }

  const int K = input.family_values_.size();
  const int p = intermediate_values.beta_.size();
  if (K == 0 || K != intermediate_values.family_values_.size() || p == 0) {
    return false;
  }
  const int M = input.distinct_times_.size();

  S0->resize(M, 0.0);
  S1->resize(M, VectorXd());
  S2->resize(M, MatrixXd());
  for (int m = 0; m < M; ++m) {
    double& S0_m = (*S0)[m];
    VectorXd& S1_m = (*S1)[m];
    S1_m.resize(p);
    S1_m.setZero();

    MatrixXd& S2_m = (*S2)[m];
    S2_m.resize(p, p);
    S2_m.setZero();

    for (int k = 0; k < K; ++k) {
      const vector<pair<VectorXd, MatrixXd>>& x_k = input.family_values_[k].x_;
      const vector<vector<MatrixXd>>& x_k_x_k_transpose =
          input.family_values_[k].x_x_transpose_;
      const vector<vector<bool>>& r_star_k =
          input.family_values_[k].r_star_;
      const vector<VectorXd>& v_exp_beta_x_k =
          intermediate_values.family_values_[k].v_exp_beta_x_;
      const int n_k = v_exp_beta_x_k.size();
      if (n_k == 0 || M == 0 || x_k.size() != n_k ||
          r_star_k.size() != n_k || r_star_k[0].size() != M) {
        cout << "ERROR in ComputeSValues: Mismatching dimensions on inputs: "
             << ", x_k.size(): " << x_k.size()
             << ", r_star_k.size(): " << r_star_k.size()
             << ", r_star_k[0].size(): " << r_star_k[0].size()
             << ", M: " << M << ", n_k: " << n_k << endl;
        return false;
      }
      for (int i = 0; i < n_k; ++i) {
        // The number of times that we need to store S values is either M (if at least
        // one covariate is time-dep), or 1 (if all covariates are time-indep). We
        // check which case we're in by checking if there are any dependent covariates.
        const int col_to_use = x_k[i].second.rows() == 0 ? 0 : m;
        if (r_star_k[i][m]) {
          S0_m += v_exp_beta_x_k[i](col_to_use);
          if (!MpleForIntervalCensoredData::AddConstantTimesXim(
                  m, input.family_values_[k].time_indep_vars_[i],
                  v_exp_beta_x_k[i](col_to_use), x_k[i], &S1_m)) {
            return false;
          }
          const int x_x_transpose_col_to_use = x_k_x_k_transpose[i].size() > 1 ? m : 0;
          S2_m += v_exp_beta_x_k[i](col_to_use) *
                  x_k_x_k_transpose[i][x_x_transpose_col_to_use];
        }
      }
    }
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeS0(
    const InputValues& input,
    const VectorXd& beta,
    const vector<VectorXd>& posterior_means,
    vector<double>* S0) {
  // Sanity-check input.
  if (S0 == nullptr) {
    cout << "ERROR in ComputeS0: Null input." << endl;
    return false;
  }

  const int p_indep = input.family_values_[0].x_[0].first.size();
  const int p_dep = input.family_values_[0].x_[0].second.rows();
  const int M = input.distinct_times_.size();
  const int p = beta.size();
  const int K = posterior_means.size();
  if (p == 0 || p_dep + p_indep != p || M == 0 || K == 0) {
    cout << "ERROR in ComputeS0: Mismatching dimensions on inputs: "
         << "beta.size(): " << beta.size()
         << ", posterior_means.size(): " << posterior_means.size() << endl;
    return false;
  }

  // The number of times that we need to store S0 values is either M (if at least
  // one covariate is time-dep), or 1 (if all covariates are time-indep). We
  // check which case we're in by checking if there are any dependent covariates.

  S0->resize(M, 0.0);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      const vector<pair<VectorXd, MatrixXd>>& x_k = input.family_values_[k].x_;
      const vector<vector<bool>>& r_star_k = input.family_values_[k].r_star_;
      const VectorXd& posterior_means_k = posterior_means[k];
      const int n_k = posterior_means_k.size();
      if (n_k == 0 || x_k.size() != n_k ||
          (x_k[0].second.cols() != 0 && x_k[0].second.cols() != M) ||
          r_star_k.size() != n_k || r_star_k[0].size() != M) {
        cout << "ERROR in ComputeS0: Mismatching dimensions on inputs: "
             << ", x_k.size(): " << x_k.size()
             << ", r_star_k.size(): " << r_star_k.size() << endl;
        return false;
      }

      for (int i = 0; i < n_k; ++i) {
        if (r_star_k[i][m]) {
          double dot_product;
          if (!MpleForIntervalCensoredData::ComputeBetaDotXim(
                  m, input.family_values_[k].time_indep_vars_[i],
                  beta, x_k[i], &dot_product)) {
            return false;
          }
          (*S0)[m] += posterior_means_k(i) * exp(dot_product);
        }
      }
    }
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeSigma(
    const InputValues& input,
    const vector<MatrixXd>& weights,
    const vector<double>& S0,
    const vector<VectorXd>& S1,
    const vector<MatrixXd>& S2,
    MatrixXd* Sigma) {
  // Sanity-Check input.
  if (Sigma == nullptr) {
    cout << "ERROR in ComputeSigma: Null input." << endl;
    return false;
  }

  // Sanity-Check dimensions.
  const int K = input.family_values_.size();
  const int p_indep = input.family_values_[0].x_[0].first.size();
  const int p_dep = input.family_values_[0].x_[0].second.rows();
  const int p = p_dep + p_indep;
  const int M = input.distinct_times_.size();
  if (K == 0 || M == 0 ||
      (S0.size() != M && S0.size() != 1) ||
      (S1.size() != M && S1.size() != 1) ||
      (S2.size() != M && S2.size() != 1) ||
      S1[0].size() != S2[0].rows() || S1[0].size() != S2[0].cols()) {
    cout << "ERROR in ComputeSigma: Mismatching dimensions on inputs: "
         << "S0.size(): " << S0.size() << ", S1.size(): "
         << S1.size() << ", S2.size(): " << S2.size() << endl;
    return false;
  }

  Sigma->resize(p, p);
  Sigma->setZero();
  for (int m = 0; m < M; ++m) {
    const double& S0_m = S0[m];
    const VectorXd& S1_m = S1[m];  // p x 1
    const MatrixXd& S2_m = S2[m];  // p x p
    const MatrixXd factor_m =
        (S1_m / S0_m) * (S1_m.transpose() / S0_m) - (S2_m / S0_m);
    for (int k = 0; k < K; ++k) {
      const vector<vector<bool>>& r_star_k = input.family_values_[k].r_star_;
      const MatrixXd& weights_k = weights[k];
      const int n_k = weights_k.rows();
      if (n_k == 0 || weights_k.cols() != M ||
          r_star_k.size() != n_k || r_star_k[0].size() != M) {
        cout << "ERROR in ComputeSigma: Mismatching dimensions on inputs: "
             << "weights_k.cols(): " << weights_k.cols()
             << ", r_star_k.size(): " << r_star_k.size() << endl;
        return false;
      }

      for (int i = 0; i < n_k; ++i) {
        if (r_star_k[i][m]) {
          (*Sigma) += weights_k(i, m) * factor_m;
        }
      }
    }
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeNewBeta(
    const int itr_index,
    const InputValues& input,
    const VectorXd& old_beta,
    const MatrixXd& Sigma,
    const vector<MatrixXd>& weights,
    const vector<double>& S0,
    const vector<VectorXd>& S1,
    VectorXd* new_beta) {
  // Sanity-Check input.
  if (new_beta == nullptr) {
    cout << "ERROR in ComputeNewBeta: Null input." << endl;
    return false;
  }

  // Sanity-check Sigma is invertible.
  FullPivLU<MatrixXd> lu = Sigma.fullPivLu();
  if (!lu.isInvertible()) {
    cout << "ERROR: E-M algorithm failed on iteration " << itr_index
         << " due to singular information matrix.\n";
    return false;
  }

  const int p_indep = input.family_values_[0].x_[0].first.size();
  const int p_dep = input.family_values_[0].x_[0].second.rows();
  const int p = p_dep + p_indep;
  const int M = input.distinct_times_.size();
  const int K = weights.size();
  if (p == 0 || old_beta.size() != p || M == 0 || K == 0 ||
      (S0.size() != M && S0.size() != 1) ||
      (S1.size() != M && S1.size() != 1) ||
      Sigma.rows() != p || Sigma.cols() != p || S1[0].size() != p) {
    cout << "ERROR in ComputeNewBeta: Mismatching dimensions on inputs: "
         << "S0.size(): " << S0.size() << ", S1.size(): "
         << S1.size()
         << ", old_beta.size(): " << old_beta.size()
         << ", Sigma.rows(): " << Sigma.rows()
         << ", Sigma.cols(): " << Sigma.cols() << endl;
    return false;
  }

  // Compute the second sum in expression for \beta^new.
  VectorXd sum;
  sum.resize(p);
  sum.setZero();
  for (int m = 0; m < M; ++m) {
    const double& S0_m = S0[m];
    const VectorXd& S1_m = S1[m];  // p x 1
    const VectorXd quotient_m = S1_m / S0_m;  // p x 1

    for (int k = 0; k < K; ++k) {
      const vector<vector<bool>>& r_star_k = input.family_values_[k].r_star_;
      const MatrixXd& weights_k = weights[k];
      const int n_k = weights_k.rows();
      const vector<pair<VectorXd, MatrixXd>>& x_k = input.family_values_[k].x_;
      if (n_k == 0 || weights_k.cols() != M ||
          x_k.size() != n_k ||
          (x_k[0].second.cols() != 0 && x_k[0].second.cols() != M) ||
          r_star_k.size() != n_k || r_star_k[0].size() != M) {
        cout << "ERROR in ComputeNewBeta: Mismatching dimensions on inputs: "
             << "weights_k.rows(): " << weights_k.rows()
             << ", weights_k.cols(): " << weights_k.cols() << endl;
        return false;
      }
      for (int i = 0; i < n_k; ++i) {
        if (r_star_k[i][m] &&
            !MpleForIntervalCensoredData::AddConstantTimesXimMinusVector(
                m, input.family_values_[k].time_indep_vars_[i],
                weights_k(i, m), quotient_m, x_k[i], &sum)) {
          return false;
        }
      }
    }
  }

  *new_beta = old_beta - Sigma.inverse() * sum;

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeNewLambda(
    const InputValues& input,
    const vector<MatrixXd>& weights,
    const vector<double>& S0_new,
    VectorXd* new_lambda) {
  // Sanity-Check input.
  if (new_lambda == nullptr) {
    cout << "ERROR in ComputeNewLambda: Null input." << endl;
    return false;
  }

  const int K = weights.size();
  const int M = S0_new.size();
  if (M == 0) {
    cout << "ERROR in ComputeNewLambda: Mismatching dimensions on inputs: "
         << "S0_new.size(): " << S0_new.size() << endl;
    return false;
  }

  new_lambda->resize(M);

  // The size S0 is either M (if at least one covariate is time-dep), or 1 (if
  // all covariates are time-indep).
  for (int m = 0; m < M; ++m) {
    double numerator_m = 0.0;

    for (int k = 0; k < K; ++k) {
      const vector<vector<bool>>& r_star_k = input.family_values_[k].r_star_;
      const int n_k = r_star_k.size();
      const MatrixXd& weights_k = weights[k];
      if (n_k == 0 || weights_k.rows() != n_k || weights_k.cols() != M) {
        cout << "ERROR in ComputeNewLambda: Mismatching dimensions on inputs: "
             << "S0_new.size(): " << S0_new.size()
             << ", weights_k.rows(): " << weights_k.rows()
             << ", weights_k.cols(): " << weights_k.cols() << endl;
        return false;
      }
      for (int i = 0; i < n_k; ++i) {
        if (r_star_k[i][m]) {
          numerator_m += weights_k(i, m);
        }
      }
    }
    (*new_lambda)(m) = numerator_m / S0_new[m];
  }

  return true;
}

// Public version.
MpleReturnValue ClusteredMpleForIntervalCensoredData::ComputeVariance(
    const int h_n_denominator, const int p, const int h_n_constant, const int max_itr,
    const double& convergence_threshold,
    const double& final_sigma,
    const InputValues& input,
    const DependentCovariateEstimates& estimates,
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
      h_n_denominator, p, h_n_constant, max_itr, convergence_threshold, final_sigma,
      sum_hermite_weights, gaussian_hermite_points, b,
      input, estimates, variance);
}

// Private version.
MpleReturnValue ClusteredMpleForIntervalCensoredData::ComputeVariance(
    const int h_n_denominator, const int p, const int h_n_constant, const int max_itr,
    const double& convergence_threshold,
    const double& final_sigma, const double& sum_hermite_weights,
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const VectorXd& b,
    const InputValues& input,
    const DependentCovariateEstimates& estimates,
    MatrixXd* variance) {
  if (variance == nullptr) {
    cout << "ERROR in ComputeVariance: Null Input." << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  double pl_at_beta;
  VectorXd pl_toggle_one_dim;
  MatrixXd pl_toggle_two_dim;
  if (!ComputeProfileLikelihoods(
          h_n_denominator, p, h_n_constant, max_itr, convergence_threshold, final_sigma,
          sum_hermite_weights, gaussian_hermite_points, b, input, estimates,
          &pl_at_beta, &pl_toggle_one_dim, &pl_toggle_two_dim)) {
    return MpleReturnValue::FAILED_VARIANCE;
  }

  if (!no_use_pos_def_variance_) {
    if (!ComputeAlternateVarianceFromProfileLikelihoods(
            h_n_denominator, h_n_constant, pl_toggle_one_dim, pl_toggle_two_dim, variance)) {
      return MpleReturnValue::FAILED_VARIANCE;
    }
  } else {
    if (!ComputeVarianceFromProfileLikelihoods(
            h_n_denominator, h_n_constant, pl_at_beta, pl_toggle_one_dim, pl_toggle_two_dim,
            variance)) {
      return MpleReturnValue::FAILED_VARIANCE;
    }
  }

  if (NegativeVariance(*variance)) {
    return MpleReturnValue::FAILED_NEGATIVE_VARIANCE;
  }

  return MpleReturnValue::SUCCESS;
}

bool ClusteredMpleForIntervalCensoredData::ComputeProfileLikelihoods(
    const int h_n_denominator, const int p, const int h_n_constant, const int max_itr,
    const double& convergence_threshold,
    const double& final_sigma, const double& sum_hermite_weights,
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const VectorXd& b,
    const InputValues& input,
    const DependentCovariateEstimates& estimates,
    double* pl_at_beta,
    VectorXd* pl_toggle_one_dim,
    MatrixXd* pl_toggle_two_dim) {
  if (pl_at_beta == nullptr || pl_toggle_one_dim == nullptr ||
      pl_toggle_two_dim == nullptr) {
    cout << "ERROR in ComputeProfileLikelihoods: Null Input." << endl;
    return false;
  }

  const VectorXd& final_beta = estimates.beta_;
  const VectorXd& final_lambda = estimates.lambda_;
  const int M = final_lambda.size();
  const int K = input.family_values_.size();
  if (p == 0 || M == 0  || h_n_denominator == 0 || K == 0 ||
      input.distinct_times_.size() != M ||
      final_beta.size() != p) {
    cout << "ERROR in ComputeProfileLikelihoods: Empty input: "
         << "final_beta.size(): " << p << ", final_lambda.size(): "
         << M << ", number clusters: " << K
         << ", distinct_times.size(): "  << input.distinct_times_.size()
         << endl;
    return false;
  }

  // Make a copy of final beta values, so that we can toggle one coordinate
  // at a time. Also, initialize this and final_lambda based on passed-in
  // values from estimates.
  VectorXd beta_plus_e_i = final_beta;

  // Covariance Matrix will have Dim (1 + p, 1 + p): p from the dependent
  // variables, and an extra covariate for \sigma^2.
  const int cov_size = p + 1;
  if (!no_use_pos_def_variance_) {
    pl_toggle_one_dim->resize(input.family_values_.size());
    pl_toggle_one_dim->setZero();
    pl_toggle_two_dim->resize(input.family_values_.size(), cov_size);
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
  // "one-dimension toggled"; do this for each of the p dimensions.
  // NOTE: In terms of what constant to use for h_n_constant_factor; for
  // univariate case, we used 5; in email with Donglin, he said:
  // "there is no firm rule--we need to empirically study the performance of each choice"
  const double h_n = static_cast<double>(h_n_constant) /
                     sqrt(static_cast<double>(h_n_denominator));

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

  int p_coordinate = -1;
  const int num_toggles = (p * p + 5 * p + 4) / 2;
  for (int i = 0; i < cov_size; ++i) {
    if (i == cov_size - 1) {
      p_coordinate = -1;
    } else {
      p_coordinate = i;
      beta_plus_e_i(p_coordinate) += h_n;
    }
    VectorXd col_i;
    col_i.resize(K);
    col_i.setZero();
    if (!ComputeProfileLikelihood(
            max_itr, convergence_threshold,
            sum_hermite_weights,
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
    if (p_coordinate >= 0) {
      beta_plus_e_i(p_coordinate) -= h_n;
    }
    if (logging_on_) {
      time_t current_time = time(nullptr);
      cout << asctime(localtime(&current_time))
           << "Finished " << (i + 1) << " of "
           << (no_use_pos_def_variance_ ? num_toggles : cov_size)
           << " computations for Covariance Matrix." << endl;
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
  int p_coordinate_one = -1;
  int p_coordinate_two = -1;
  for (int i = 0; i < cov_size; ++i) {
    if (i == cov_size - 1) {
      p_coordinate_one = -1;
    } else {
      p_coordinate_one = i;
      beta_plus_e_i(p_coordinate_one) += h_n;
    }
    VectorXd e_i;
    e_i.resize(cov_size);
    e_i.setZero();
    e_i(i) = h_n;
    for (int j = 0; j <= i; ++j) {
      if (j == cov_size - 1) {
        p_coordinate_two = -1;
      } else {
        p_coordinate_two = j;
        beta_plus_e_i(p_coordinate_two) += h_n;
      }
      if (!ComputeProfileLikelihood(
              max_itr, convergence_threshold,
              sum_hermite_weights,
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
      if (p_coordinate_two >= 0) {
        beta_plus_e_i(p_coordinate_two) -= h_n;
      }
      if (logging_on_) {
        time_t current_time = time(nullptr);
        cout << asctime(localtime(&current_time))
             << "Finished " << (1 + p + j + 1 + (i * i + i) / 2) << " of "
             << num_toggles << " computations for Covariance Matrix." << endl;
      }
    }
    // Reset beta_plus_e_i to return to final_beta.
    if (p_coordinate_one >= 0) {
      beta_plus_e_i(p_coordinate_one) -= h_n;
    }
  }

  const bool PHB_print_profile_likelihoods = true;
  if (logging_on_ && PHB_print_profile_likelihoods) {
    PrintProfileLikelihoods(*pl_at_beta, *pl_toggle_one_dim, *pl_toggle_two_dim);
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeProfileLikelihood(
    const int max_itr,
    const double& convergence_threshold,
    const double& sum_hermite_weights,
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const VectorXd& b,
    const InputValues& input,
    const VectorXd& beta,
    const VectorXd& lambda,
    double* pl, VectorXd* pl_alternate) {
  if (pl == nullptr && pl_alternate == nullptr) {
    cout << "ERROR in ComputeProfileLikelihood: Null Input." << endl;
    return false;
  }

  const int N_s = gaussian_hermite_points.size();
  if (N_s == 0 || b.size() != N_s) {
    cout << "ERROR in ComputeProfileLikelihood: Empty input: "
         << ", gaussian_hermite_points.size(): " << gaussian_hermite_points.size()
         << ", b.size(): " << b.size() << endl;
    return false;
  }

  // The following data structure will hold all the intermediate values
  // that are generated at each step of the E-M algorithm.
  IntermediateValues intermediate_values;

  // Since \theta = (\beta, \sigma^2) will not change in the
  // E-M algorithm to find the maximizing \lambda, we compute here (outside
  // of the E-M iterations) the fields of 'intermediate_values' that will not
  // change: beta_, and exp_beta_x_plus_b_.
  intermediate_values.beta_ = beta;
  intermediate_values.lambda_ = lambda;
  intermediate_values.family_values_.resize(input.family_values_.size());
  VectorXd old_lambda = lambda;
  // Compute exp(\beta_k^T * X_kim + \sqrt{2} * \sigma * y_s)
  for (int k = 0; k < input.family_values_.size(); ++k) {
    if (!ComputeExpBetaXPlusB(
            beta, b,
            input.family_values_[k].time_indep_vars_,
            input.family_values_[k].x_,
            &(intermediate_values.family_values_[k].exp_beta_x_),
            &(intermediate_values.family_values_[k].exp_beta_x_plus_b_))) {
      cout << "ERROR: Failed in Computing exp(beta^T * X + b) for k = "
           << k + 1 << " while Computing Profile Likelihood.\n";
      return false;
    }
  }

  // Run Profile (i.e. only maximimizing one parameter \lambda while leaving
  // (\beta, \sigma^2) parameters fixed) E-M algorithm to find maximizing \lambda.
  int iteration_index = 1;
  for (; iteration_index < max_itr; ++iteration_index) {
    // Run E-Step.
    vector<VectorXd> posterior_means;
    vector<MatrixXd> weights;
    if (!DoEStep(1.0 /* Not Used */, sum_hermite_weights, b, input,
                 gaussian_hermite_points,
                 &intermediate_values, &weights, &posterior_means, nullptr)) {
      cout << "ERROR: Failed in EStep of iteration " << iteration_index << endl;
      return false;
    }

    // No need to do M-step; just the part where \lambda is updated.
    vector<double> S0_new;
    if (!ComputeS0(input, beta, posterior_means, &S0_new)) {
      cout << "ERROR in computation of profile likelihood." << endl;
      return false;
    }

    // Update Lambda.
    if (!ComputeNewLambda(
            input, weights, S0_new, &(intermediate_values.lambda_))) {
      cout << "ERROR in Computing new Lambda." << endl;
      return false;
    }

    // Check Convergence Criterion.
    if (ProfileEmAlgorithmHasConverged(
            convergence_threshold, old_lambda, intermediate_values.lambda_)) {
      bool PHB_print_each_pl_convergence_itr_num = true;
      if (logging_on_ && PHB_print_each_pl_convergence_itr_num) {
        cout << "PL converged in: " << iteration_index << endl;
      }
      break;
    }

    // Update old_lambda.
    old_lambda = intermediate_values.lambda_;
  }

  // Abort if we failed to converge after max_itr.
  if (iteration_index >= max_itr) {
    cout << "ERROR in ComputeProfileLikelihood: "
         << "E-M algorithm exceeded maximum number of allowed "
         << "iterations: " << max_itr << endl;
    return false;
  }

  if (!EvaluateLogLikelihoodFunctionAtBetaLambda(
          gaussian_hermite_points, input, &intermediate_values,
          pl, pl_alternate)) {
    return false;
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::EvaluateLogLikelihoodFunctionAtBetaLambda(
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const InputValues& input,
    IntermediateValues* intermediate_values,
    double* likelihood, VectorXd* e_k_likelihoods) {
  if (likelihood == nullptr && e_k_likelihoods == nullptr) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Null Input." << endl;
    return false;
  }

  const int K = intermediate_values->family_values_.size();
  const int N_s = gaussian_hermite_points.size();
  if (N_s == 0 || K == 0  || input.family_values_.size() != K) {
    cout << "ERROR in ComputeProfileLikelihood: Empty input: "
         << "intermediate_values.family_values_.size(): " << K
         << ", input.size(): " << input.family_values_.size() << endl;
    return false;
  }

  for (int k = 0; k < K; ++k) {
    // No need to re-compute exp(\beta_k^T * X_kim + \sqrt{2} * \sigma * y_s),
    // since this was computed once (outside the E-M algorithm) and won't have
    // changed since \beta and \sigma are not changing.

    // Compute S_L and S_U.
    if (!ComputeS(input.distinct_times_, input.family_values_[k].lower_time_bounds_,
                  intermediate_values->lambda_,
                  intermediate_values->family_values_[k].exp_beta_x_plus_b_, 
                  &(intermediate_values->family_values_[k].S_L_))) {
      cout << "ERROR in Computing S^L for k = " << k + 1 << endl;
      return false;
    }
    if (!ComputeS(input.distinct_times_, input.family_values_[k].upper_time_bounds_,
                  intermediate_values->lambda_,
                  intermediate_values->family_values_[k].exp_beta_x_plus_b_, 
                  &(intermediate_values->family_values_[k].S_U_))) {
      cout << "ERROR in Computing S^U for k = " << k + 1 << endl;
      return false;
    }

    // Compute exp(-G(S_L)) and exp(-G(S_U)).
    if (!ComputeExpTransformation(
            input.transformation_G_,
            intermediate_values->family_values_[k].S_L_,
            &(intermediate_values->family_values_[k].exp_neg_g_S_L_))) {
      cout << "ERROR in Computing exp(G(S^L) for k = " << k + 1 << endl;
      return false;
    }
    if (!ComputeExpTransformation(
            input.transformation_G_,
            intermediate_values->family_values_[k].S_U_,
            &(intermediate_values->family_values_[k].exp_neg_g_S_U_))) {
      cout << "ERROR in Computing exp(G(S^U) for k = " << k + 1 << endl;
      return false;
    }
  }

  // Compute Likelihood L_n.
  return no_use_pos_def_variance_ ?
    EvaluateLogLikelihoodFunctionAtBetaLambda(
      gaussian_hermite_points, *intermediate_values, likelihood) :
    EvaluateAlternateLogLikelihoodFunctionAtBetaLambda(
      gaussian_hermite_points, *intermediate_values, e_k_likelihoods);
}

bool ClusteredMpleForIntervalCensoredData::EvaluateLogLikelihoodFunctionAtBetaLambda(
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const IntermediateValues& intermediate_values,
    double* likelihood) {
  if (likelihood == nullptr) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Null Input."
         << endl;
    return false;
  }
  const int K = intermediate_values.family_values_.size();
  const int N_s = gaussian_hermite_points.size();
  if (N_s == 0 || K == 0) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Empty Input.\n";
    return false;
  }

  *likelihood = 0.0;
  for (int k = 0; k < K; ++k) {
    const FamilyIntermediateValues& family_values_k =
        intermediate_values.family_values_[k];
    const MatrixXd& exp_neg_g_S_L_k = family_values_k.exp_neg_g_S_L_;
    const MatrixXd& exp_neg_g_S_U_k = family_values_k.exp_neg_g_S_U_;
    const int n_k = exp_neg_g_S_L_k.rows();
    if (n_k == 0 || n_k != exp_neg_g_S_U_k.rows() ||
        exp_neg_g_S_L_k.cols() != N_s || exp_neg_g_S_U_k.cols() != N_s) {
      cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: "
           << "Mismatching dimensions: "
           << "exp_neg_g_S_L[k].rows(): " << exp_neg_g_S_L_k.rows()
           << "exp_neg_g_S_L[k].cols(): " << exp_neg_g_S_L_k.cols()
           << "exp_neg_g_S_U[k].rows(): " << exp_neg_g_S_U_k.rows()
           << "exp_neg_g_S_U[k].cols(): " << exp_neg_g_S_U_k.cols() << endl;
      return false;
    }

    double inner_summand = 0.0;
    for (int s = 0; s < N_s; ++s) {
      double product_term = 1.0;
      for (int i = 0; i < n_k; ++i) {
        product_term *= (exp_neg_g_S_L_k(i, s) - exp_neg_g_S_U_k(i, s));
      }
      const double mu_s = gaussian_hermite_points[s].weight_;
      inner_summand += mu_s * product_term;
    }
    *likelihood += log(inner_summand / sqrt(PI));
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::EvaluateAlternateLogLikelihoodFunctionAtBetaLambda(
    const vector<GaussianQuadratureTuple>& gaussian_hermite_points,
    const IntermediateValues& intermediate_values,
    VectorXd* e_k_likelihoods) {
  if (e_k_likelihoods == nullptr) {
    cout << "ERROR in EvaluateAltLogLikelihoodFunctionAtBetaLambda: Null Input."
         << endl;
    return false;
  }
  const int K = intermediate_values.family_values_.size();
  const int N_s = gaussian_hermite_points.size();
  if (N_s == 0 || K == 0) {
    cout << "ERROR in EvaluateAltLogLikelihoodFunctionAtBetaLambda: Empty Input.\n";
    return false;
  }

  for (int k = 0; k < K; ++k) {
    const FamilyIntermediateValues& family_values_k =
        intermediate_values.family_values_[k];
    const MatrixXd& exp_neg_g_S_L_k = family_values_k.exp_neg_g_S_L_;
    const MatrixXd& exp_neg_g_S_U_k = family_values_k.exp_neg_g_S_U_;
    const int n_k = exp_neg_g_S_L_k.rows();
    if (n_k == 0 || n_k != exp_neg_g_S_U_k.rows() ||
        exp_neg_g_S_L_k.cols() != N_s || exp_neg_g_S_U_k.cols() != N_s) {
      cout << "ERROR in EvaluateAltLogLikelihoodFunctionAtBetaLambda: "
           << "Mismatching dimensions: "
           << "exp_neg_g_S_L[k].rows(): " << exp_neg_g_S_L_k.rows()
           << "exp_neg_g_S_L[k].cols(): " << exp_neg_g_S_L_k.cols()
           << "exp_neg_g_S_U[k].rows(): " << exp_neg_g_S_U_k.rows()
           << "exp_neg_g_S_U[k].cols(): " << exp_neg_g_S_U_k.cols() << endl;
      return false;
    }

    double inner_summand = 0.0;
    for (int s = 0; s < N_s; ++s) {
      double product_term = 1.0;
      for (int i = 0; i < n_k; ++i) {
        product_term *= (exp_neg_g_S_L_k(i, s) - exp_neg_g_S_U_k(i, s));
      }
      const double mu_s = gaussian_hermite_points[s].weight_;
      inner_summand += mu_s * product_term;
    }
    (*e_k_likelihoods)(k) = log(inner_summand / sqrt(PI));
  }

  return true;
}

bool ClusteredMpleForIntervalCensoredData::ComputeVarianceFromProfileLikelihoods(
      const int h_n_denominator, const int h_n_constant, const double& pl_at_theta,
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
      pow(static_cast<double>(h_n_constant), 2.0) / static_cast<double>(h_n_denominator);

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

bool ClusteredMpleForIntervalCensoredData::ComputeAlternateVarianceFromProfileLikelihoods(
      const int h_n_denominator, const int h_n_constant,
      const VectorXd& pl_toggle_none,
      const MatrixXd& pl_toggle_one_dim,
      MatrixXd* variance) {
  if (variance == nullptr) {
    cout << "ERROR in ComputeAltVarianceFromProfileLikelihoods: Null input."
         << endl;
    return false;
  }

  const int K = pl_toggle_one_dim.rows();
  const int cov_dim = pl_toggle_one_dim.cols();
  if (K == 0 || cov_dim == 0 || pl_toggle_none.size() != K) {
    cout << "ERROR in ComputeAltVarianceFromProfileLikelihoods: empty matrix "
         << "pl_toggle_one_dim.rows(): " << pl_toggle_one_dim.rows()
         << ", pl_toggle_one_dim.cols(): " << pl_toggle_one_dim.cols()
         << ", pl_toggle_none.size(): " << pl_toggle_none.size()
         << endl;
    return false;
  }

  const double h_n = static_cast<double>(h_n_constant) /
                     sqrt(static_cast<double>(h_n_denominator));

  MatrixXd variance_inverse;
  variance_inverse.resize(cov_dim, cov_dim);
  variance_inverse.setZero();
  for (int k = 0; k < K; ++k) {
    const double& toggle_none_k = pl_toggle_none(k);
    VectorXd ps_k;
    ps_k.resize(cov_dim);
    for (int p = 0; p < cov_dim; ++p) {
      ps_k(p) = (pl_toggle_one_dim(k, p) - toggle_none_k) / h_n; 
    }
    variance_inverse += ps_k * ps_k.transpose();
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
