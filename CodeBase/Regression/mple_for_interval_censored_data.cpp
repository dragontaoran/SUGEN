// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "mple_for_interval_censored_data.h"

#include "FileReaderUtils/read_file_utils.h"
#include "FileReaderUtils/read_time_dep_interval_censored_data.h"
#include "FileReaderUtils/read_time_indep_interval_censored_data.h"
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
#include <iomanip>      // std::setprecision
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
  
double MpleForIntervalCensoredData::convergence_threshold_;
int MpleForIntervalCensoredData::max_itr_;
bool MpleForIntervalCensoredData::logging_on_;
bool MpleForIntervalCensoredData::no_use_pos_def_variance_;
bool MpleForIntervalCensoredData::force_one_right_censored_;
vector<Timer> MpleForIntervalCensoredData::timers_;

void PHBGetFinalBetaLambda(
    VectorXd* final_beta, VectorXd* final_lambda) {
  final_lambda->resize(519);
  final_beta->resize(5);
  (*final_beta)(0) = -0.225523801201534;  // Age
  (*final_beta)(1) = 0.4801642507546311;  // Sex
  (*final_beta)(2) = 0.2441143248959863;  // Needle
  (*final_beta)(3) = 0.4975903901648592;  // Jail
  (*final_beta)(4) = 0.3360799253175714;  // Inject

  ifstream file("OutputFiles/mple_for_interval_censored_data_lambda.txt");
  if (!file.is_open()) {
    cout << "Couldn't open lambda file." << endl;
    return;
  }
  int line_num = 0;
  string line;
  while(getline(file, line)) {
    if (line[line.length() - 1] == 13) {
      line = line.substr(0, line.length() - 1);
    }
    if (!Stod(line, &((*final_lambda)(line_num)))) {
      cout << "ERROR: Unable to parse '" << line << "' as a numeric value." << endl;
      return;
    }
    line_num++;
  }
}

void MpleForIntervalCensoredData::InitializeTimers() {
  const int num_timers = 0;
  for (int i = 0; i < num_timers; ++i) {
    timers_.push_back(Timer());
  }
}

void MpleForIntervalCensoredData::PrintTimers() {
  for (int i = 0; i < timers_.size(); ++i) {
    const Timer& t = timers_[i];
    cout << "Timer " << i << ": " << test_utils::GetElapsedTime(t) << endl;
  }
}

bool MpleForIntervalCensoredData::InitializeData(
    const TimeDepIntervalCensoredData& data) {
  distinct_times_.clear();
  lower_time_bounds_.clear();
  upper_time_bounds_.clear();
  x_.clear();
  time_indep_vars_.clear();

  // Set distinct times.
  // If force_one_right_censored_ is false, just copy the input set of
  // distinct times. Otherwise, we'll need to recompute the set of distinct
  // times.
  double max_left_time = -1.0;
  set<double> times_made_right_censored;
  if (!force_one_right_censored_) {
    distinct_times_ = data.distinct_times_[0];
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
        distinct_times_.insert(time_info.lower_);
      }
      const double& upper = time_info.upper_;
      if (upper == numeric_limits<double>::infinity()) {
        // Nothing to do.
      } else if (upper <= max_left_time) {
        distinct_times_.insert(upper);
      } else {
        // Keep track of times that were shifted to be infinity.
        times_made_right_censored.insert(upper);
      }
    }
  }

  for (const auto& subject_info_i : data.subject_info_) {
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
    time_indep_vars_.push_back(subject_info_i.is_time_indep_[0]);
    lower_time_bounds_.push_back(subject_info_i.times_[0].lower_);
    if (!force_one_right_censored_ || subject_info_i.times_[0].upper_ <= max_left_time) {
      upper_time_bounds_.push_back(subject_info_i.times_[0].upper_);
    } else {
      upper_time_bounds_.push_back(numeric_limits<double>::infinity());
    }
    if (times_made_right_censored.empty() ||
        subject_info_i.linear_term_values_[0].second.cols() == 0) {
      x_.push_back(subject_info_i.linear_term_values_[0]);
    } else {
      x_.push_back(pair<VectorXd, MatrixXd>());
      VectorXd& time_indep_values = x_.back().first;
      time_indep_values = subject_info_i.linear_term_values_[0].first;
      const MatrixXd& values_to_copy = 
          subject_info_i.linear_term_values_[0].second;
      const int num_cols_to_skip = times_made_right_censored.size();
      const int num_cols_to_copy = values_to_copy.cols() - num_cols_to_skip;
      MatrixXd& time_dep_values = x_.back().second;
      time_dep_values =
          values_to_copy.block(0, 0, values_to_copy.rows(), num_cols_to_copy);
    }
  }
  
  return true;
}

bool MpleForIntervalCensoredData::InitializeInput(
    const int num_gauss_laguerre_points,
    const double& r, 
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    double* integral_constant_factor,
    vector<GaussianQuadratureTuple>* gaussian_laguerre_points,
    Expression* transformation_G, Expression* transformation_G_prime,
    vector<vector<MatrixXd>>* x_x_transpose,
    vector<vector<bool>>* r_i_star_indicator) {
  // Compute constants that will be used throughout the E-M algorithm.
  //   - Gets Gaussian-Laguerre Points and Weights.
  if (r != 0.0 &&
      !ComputeGaussianLaguerrePoints(
          num_gauss_laguerre_points, r, gaussian_laguerre_points)) {
    return false;
  }
  //   - Stores X * X^T
  if (!ComputeXTimesXTranspose(time_indep_vars, x, x_x_transpose)) {
    return false;
  }
  //   - Sets r_i_star_indicator: I(t_k <= R*_i), where R*_i is
  //     R_i if R_i \neq \infty, otherwise it is L_i.
  if (!ComputeRiStarIndicator(
          distinct_times, lower_time_bounds, upper_time_bounds,
          r_i_star_indicator)) {
    return false;
  }

  return InitializeInput(
      r, integral_constant_factor, transformation_G, transformation_G_prime);
}

bool MpleForIntervalCensoredData::InitializeInput(
    const double& r, 
    double* integral_constant_factor,
    Expression* transformation_G, Expression* transformation_G_prime) {
  // Compute constants that will be used throughout the E-M algorithm.
  //   - Gamma(1/r) * r^(1/r)
  *integral_constant_factor = 0.0;
  if (r != 0.0 && !ComputeIntegralConstantFactor(r, integral_constant_factor)) {
    return false;
  }
  //   - Defines G and G'
  if (!ConstructTransformation(r, transformation_G, transformation_G_prime)) {
    return false;
  }

  return true;
}

// Private, non-static version.
bool MpleForIntervalCensoredData::InitializeInput() {
  if (distinct_times_.empty() || lower_time_bounds_.empty() ||
      upper_time_bounds_.empty() || x_.empty()) {
    return false;
  }
  // Make sure member fields have already been set.
  const int M = distinct_times_.size();
  const int n = lower_time_bounds_.size();
  if (r_ < 0.0 || M == 0 || n == 0 || upper_time_bounds_.size() !=n ||
      x_.size() != n ||
      (x_[0].second.rows() > 0 && x_[0].second.cols() != M)) {
    return false;
  }

  return InitializeInput(
      num_gaussian_laguerre_points_, r_,
      distinct_times_, lower_time_bounds_, upper_time_bounds_,
      time_indep_vars_, x_,
      &integral_constant_factor_, &gaussian_laguerre_points_,
      &transformation_G_, &transformation_G_prime_,
      &x_x_transpose_, &r_i_star_indicator_);
}

// Public, static version.
MpleReturnValue MpleForIntervalCensoredData::PerformEmAlgorithmForParameterEstimation(
    const double& r, const double& convergence_threshold,
    const int h_n_constant, const int max_itr,
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    int* num_iterations, double* log_likelihood,
    VectorXd* final_beta, VectorXd* final_lambda, MatrixXd* variance) {
  double integral_constant_factor;
  vector<GaussianQuadratureTuple> gaussian_laguerre_points;
  Expression transformation_G;
  Expression transformation_G_prime;
  vector<vector<MatrixXd>> x_x_transpose;
  vector<vector<bool>> r_i_star_indicator;
  if (!InitializeInput(
          40, r, distinct_times, lower_time_bounds, upper_time_bounds,
          time_indep_vars, x,
          &integral_constant_factor, &gaussian_laguerre_points,
          &transformation_G, &transformation_G_prime,
          &x_x_transpose, &r_i_star_indicator)) {
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  return PerformEmAlgorithmForParameterEstimation(
      r, convergence_threshold, integral_constant_factor,
      h_n_constant, max_itr, gaussian_laguerre_points,
      transformation_G, transformation_G_prime,
      distinct_times, lower_time_bounds, upper_time_bounds,
      time_indep_vars, x, x_x_transpose, r_i_star_indicator,
      num_iterations, log_likelihood, final_beta, final_lambda, variance);
}

// Public, non-static version.
MpleReturnValue MpleForIntervalCensoredData::PerformEmAlgorithmForParameterEstimation(
    int* num_iterations, double* log_likelihood,
    VectorXd* final_beta, VectorXd* final_lambda, MatrixXd* variance) {
  if (!InitializeInput()) return MpleReturnValue::FAILED_BAD_INPUT;

  return PerformEmAlgorithmForParameterEstimation(
      r_, convergence_threshold_, integral_constant_factor_,
      h_n_constant_, max_itr_, gaussian_laguerre_points_,
      transformation_G_, transformation_G_prime_,
      distinct_times_, lower_time_bounds_, upper_time_bounds_,
      time_indep_vars_, x_, x_x_transpose_, r_i_star_indicator_,
      num_iterations, log_likelihood, final_beta, final_lambda, variance);
}

// Private, (static + non-static) version
MpleReturnValue MpleForIntervalCensoredData::PerformEmAlgorithmForParameterEstimation(
    const double& r, const double& convergence_threshold,
    const double& integral_constant_factor,
    const int h_n_constant, const int max_itr,
    const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
    const Expression& transformation_G,
    const Expression& transformation_G_prime,
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    const vector<vector<MatrixXd>>& x_x_transpose,
    const vector<vector<bool>>& r_i_star_indicator,
    int* num_iterations, double* log_likelihood,
    VectorXd* final_beta, VectorXd* final_lambda, MatrixXd* variance) {
  // Sanity-Check input.
  if (final_beta == nullptr || final_lambda == nullptr) {
    cout << "ERROR in Performing EM Algorithm: Null input."
         << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  // Sanity-check dimensions match.
  const unsigned int n = lower_time_bounds.size();
  const unsigned int M = distinct_times.size();
  const int p_indep = x[0].first.size();
  const int p_dep = x[0].second.rows();
  const int p = p_indep + p_dep;
  if (n == 0 || M == 0 || p == 0 || upper_time_bounds.size() != n || x.size() != n ||
      x[0].second.rows() != p_dep || (p_dep > 0 && x[0].second.cols() != M)) {
    cout << "ERROR in Performing EM Algorithm: Mismatching dimensions on inputs: "
         << "distinct_times.size(): " << distinct_times.size()
         << ", lower_bounds.size(): " << lower_time_bounds.size()
         << ", upper_bounds.size(): " << upper_time_bounds.size()
         << ", x.size(): " << x.size();
    if (!x.empty()) {
      cout << ", x[0].cols(): " << x[0].second.cols()
           << ", x[0].rows(): " << x[0].second.rows();
    }
    cout << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  VectorXd old_beta, old_lambda;
  if (final_beta->size() > 0 && final_lambda->size() > 0) {
    // Use passed-in values as initial guess.
    old_beta = *final_beta;
    old_lambda = *final_lambda;
  } else {
    old_beta.resize(p);
    old_lambda.resize(M);
    // Initial input to E-M algorithm will be with \beta = 0.
    old_beta.setZero();
    // Initial input to E-M algorithm will be with \lambda = 1 / M.
    for (unsigned int m = 0; m < M; ++m) {
      old_lambda(m) = 1.0 / M;
    }
  }

  // Determine how often to print status updates, based on how much data there
  // is (and thus, how long each iteration may take).
  const int complexity = n * M * p;
  const int print_modulus =
      complexity < 10000 ? 1000 :
      complexity < 1000000 ? 100 :
      complexity < 10000000 ? 50 :
      complexity < 100000000 ? 25 :
      complexity < 1000000000 ? 10 :
      complexity < 10000000000 ? 5 : 2;

  // Run E-M algorithm until convergence, or max_itr.
  int iteration_index = 1;
  const bool PHB_abort_after_one_iteration = false;
  const bool PHB_print_first_iteration_values = false;
  const bool PHB_print_final_beta_lambda = false;
  const bool PHB_print_final_beta_on_failure = false;
  const bool PHB_print_final_lamba_on_failure = false;
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

    // For convenience, compute: exp(\beta^T * X_ik) once.
    vector<VectorXd> exp_beta_x;  // Outer vector size n,
                                  // VectorXd size M (or 1 if all cov's are time-indep)
    if (!ComputeExpBetaX(old_beta, time_indep_vars, x, &exp_beta_x)) {
      cout << "ERROR: Failed in Computing exp(beta^T * X) of iteration "
           << iteration_index << endl;
      return MpleReturnValue::FAILED_PRELIMINARY_COMPUTATION;
    }

    // Run E-Step.
    MatrixXd weights;
    VectorXd posterior_means;
    if (!DoEStep(transformation_G, transformation_G_prime, distinct_times,
                 lower_time_bounds, upper_time_bounds, old_beta, old_lambda,
                 exp_beta_x, r, integral_constant_factor, gaussian_laguerre_points,
                 &weights, &posterior_means)) {
      cout << "ERROR: Failed in EStep of iteration " << iteration_index << endl;
      return MpleReturnValue::FAILED_E_STEP;
    }

    // Run M-Step.
    if (!DoMStep(iteration_index, old_beta, posterior_means, weights,
                 exp_beta_x, time_indep_vars, x, x_x_transpose,
                 r_i_star_indicator, final_beta, final_lambda)) {
      cout << "ERROR: Failed in MStep of iteration " << iteration_index << endl;
      return MpleReturnValue::FAILED_M_STEP;
    }

    // Check Convergence Criterion.
    if (EmAlgorithmHasConverged(
            convergence_threshold, old_beta, *final_beta,
            old_lambda, *final_lambda, &current_difference)) {
      *num_iterations = iteration_index;
      break;
    }

    old_beta = *final_beta;
    old_lambda = *final_lambda;

    if (iteration_index == 1 && PHB_print_first_iteration_values) {
      Eigen::IOFormat format(Eigen::FullPrecision);
      cout << "\nAfter first iteration, beta:\n"
           << old_beta.format(format) << endl;
    }
    if (PHB_abort_after_one_iteration) {
      cout << "Lambda:\n" << old_lambda << endl;
      break;
    }
  }

  // Abort if we failed to converge after max_itr.
  if (iteration_index >= max_itr) {
    if (logging_on_) {
      cout << "ERROR in Performing EM Algorithm: "
           << "E-M algorithm exceeded maximum number of allowed "
           << "iterations: " << max_itr << endl;
    }
    // At least print out the final beta values.
    if (PHB_print_final_beta_on_failure) {
      Eigen::IOFormat format(Eigen::FullPrecision);
      cout << "\nFinal beta:\n"
           << final_beta->transpose().format(format) << endl;
    }
    if (PHB_print_final_lamba_on_failure) {
      Eigen::IOFormat format(Eigen::FullPrecision);
      cout << "Final Lambda:\n"
           << final_lambda->transpose().format(format) << endl;    
    }
    return MpleReturnValue::FAILED_MAX_ITR;
  }

  if (PHB_abort_after_one_iteration) {
    PHBGetFinalBetaLambda(final_beta, final_lambda);
  }

  if (PHB_print_final_beta_lambda) {
    Eigen::IOFormat format(Eigen::FullPrecision);
    cout << "\nFinal beta:\n"
         << final_beta->transpose().format(format) << endl;
    cout << "Final Lambda:\n"
         << final_lambda->transpose().format(format) << endl;    
  }

  // Compute Likelihood at final values.
  if (log_likelihood != nullptr) {
    // Temporarily override no_use_pos_def_variance_ option, so we can
    // compute the log-likelihood at the final estimates.
    const bool orig_no_use_pos_def_variance = no_use_pos_def_variance_;
    no_use_pos_def_variance_ = true;
    if (!ComputeProfileLikelihood(
            transformation_G, transformation_G_prime, r, integral_constant_factor,
            convergence_threshold, max_itr, gaussian_laguerre_points,
            distinct_times, lower_time_bounds, upper_time_bounds, time_indep_vars,
            x, r_i_star_indicator, *final_beta, *final_lambda, log_likelihood, nullptr)) {
      return MpleReturnValue::FAILED_VARIANCE;
    }
    // Restore no_use_pos_def_variance_ to its proper value.
    no_use_pos_def_variance_ = orig_no_use_pos_def_variance;
  }

  // Compute Covariance.
  if (variance != nullptr) {
    // Time-stamp progress.
    time_t current_time = time(nullptr);
    if (logging_on_) {
      cout << endl << asctime(localtime(&current_time)) << "Finished E-M algorithm "
           << "to compute beta and lambda in " << iteration_index
           << " iterations.\nComputing Covariance Matrix..." << endl;
    }

    MpleReturnValue var_result = ComputeVariance(
        transformation_G, transformation_G_prime, r, integral_constant_factor,
        convergence_threshold, h_n_constant, max_itr, gaussian_laguerre_points,
        distinct_times, lower_time_bounds, upper_time_bounds, time_indep_vars,
        x, r_i_star_indicator, *final_beta, *final_lambda, variance);
    if (var_result != MpleReturnValue::SUCCESS) {
      // At least print out the final beta values.
      if (PHB_print_final_beta_on_failure) {
        Eigen::IOFormat format(Eigen::FullPrecision);
        cout << "\nFinal beta:\n"
             << final_beta->transpose().format(format) << endl;
      }
      if (PHB_print_final_lamba_on_failure) {
        Eigen::IOFormat format(Eigen::FullPrecision);
        cout << "Final Lambda:\n"
             << final_lambda->transpose().format(format) << endl;    
      }
      return var_result;
    }

    if (logging_on_) {
      current_time = time(nullptr);
      cout << endl << asctime(localtime(&current_time))
           << "Done computing Covariance Matrix." << endl;
    }
  }

  return MpleReturnValue::SUCCESS;
}

bool MpleForIntervalCensoredData::ComputeGaussianLaguerrePoints(
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

bool MpleForIntervalCensoredData::ComputeIntegralConstantFactor(
    const double& r, double* integral_constant_factor) {
  if (integral_constant_factor == nullptr) {
    cout << "ERROR in ComputeIntegralConstantFactor: Null input." << endl;
    return false;
  }
  if (r <= 0.0) {
    cout << "ERROR: Bad value for r: " << r << " (must have r >= 0)" << endl;
    return false;
  }

  *integral_constant_factor = Gamma(1.0 / r) * pow(r, 1.0 / r);
  return true;
}

bool MpleForIntervalCensoredData::ConstructTransformation(
    const double& r,
    Expression* transformation_G,
    Expression* transformation_G_prime) {
  if (transformation_G == nullptr || transformation_G_prime == nullptr) {
    cout << "ERROR in ConstructTransformation: Null input." << endl;
    return false;
  }

  string g_str = "";
  string g_prime_str = "";
  const string r_str = Itoa(r);
  if (r == 0.0) {
    g_str = "x";
    g_prime_str = "1";
  } else {
    g_str = "log(1+" + r_str + "x)/" + r_str;
    g_prime_str = "1/(1+" + r_str + "x)";
  }
  if (!ParseExpression(g_str, transformation_G)) {
    cout << "ERROR: Unable to parse '"
         << g_str << "' as an Expression." << endl;
    return false;
  }
  if (!ParseExpression(g_prime_str, transformation_G_prime)) {
    cout << "ERROR: Unable to parse '"
         << g_prime_str << "' as an Expression." << endl;
    return false;
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeXTimesXTranspose(
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    vector<vector<MatrixXd>>* x_x_transpose) {
  if (x_x_transpose == nullptr || time_indep_vars.empty() || x.empty() ||
      (x[0].first.size() == 0 &&
       (x[0].second.rows() == 0 || x[0].second.cols() == 0))) {
    cout << "ERROR in ComputeXTimesXTranspose: Null input." << endl;
    return false;
  }

  const int n = x.size();
  x_x_transpose->clear();

  for (int i = 0; i < n; ++i) {
    const pair<VectorXd, MatrixXd>& x_i = x[i];
    x_x_transpose->push_back(vector<MatrixXd>());
    vector<MatrixXd>& ith_entry = x_x_transpose->back();
    // Only need one distinct time, if all covariates are time-independent.
    const int p_dep = x_i.second.rows();
    const int M = p_dep == 0 ? 1 : x_i.second.cols();
    for (int m = 0; m < M; ++m) {
      ith_entry.push_back(MatrixXd());
      MatrixXd& m_th_entry = ith_entry.back();;
      VectorXd x_im;
      if (!GetXim(m, time_indep_vars[i], x_i, &x_im)) return false;
      m_th_entry = x_im * x_im.transpose();  // m_th_entry is p x p.
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeRiStarIndicator(
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    vector<vector<bool>>* r_i_star_indicator) {
  // Sanity-check input.
  if (r_i_star_indicator == nullptr) {
    cout << "ERROR in ComputeRiStarIndicator: Null input." << endl;
    return false;
  }

  const unsigned int n = lower_time_bounds.size();
  const unsigned int M = distinct_times.size();
  if (n == 0 || M == 0 || upper_time_bounds.size() != n) {
    cout << "ERROR in ComputeRiStarIndicator: Mismatching dimensions on inputs: "
         << "distinct_times.size(): " << distinct_times.size()
         << ", lower_time_bounds.size(): " << lower_time_bounds.size()
         << ", upper_time_bounds.size(): " << upper_time_bounds.size() << endl;
    return false;
  }

  r_i_star_indicator->resize(n);
  for (unsigned int i = 0; i < n; ++i) {
    (*r_i_star_indicator)[i].resize(M);
    const double r_star_i =
        upper_time_bounds[i] == numeric_limits<double>::infinity() ?
        lower_time_bounds[i] : upper_time_bounds[i];
    int m = 0;
    for (const double& time : distinct_times) {
      (*r_i_star_indicator)[i][m] = r_star_i >= time;
      ++m;
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::EmAlgorithmHasConverged(
    const double& convergence_threshold,
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
  cumulative_lambda_new.resize(old_lambda.size());
  cumulative_lambda_new.setZero();
  for (int i = 0; i < old_lambda.size(); ++i) {
    if (i != 0) {
      cumulative_lambda_old(i) = cumulative_lambda_old(i - 1);
      cumulative_lambda_new(i) = cumulative_lambda_new(i - 1);
    }
    cumulative_lambda_old(i) += old_lambda(i);
    cumulative_lambda_new(i) += new_lambda(i);
  }

  // Compute difference between previous and current iterations.
  // TODO(PHB): Make delta a global constant.
  const double delta = 0.01;
  const double max_beta =
      VectorAbsoluteDifferenceSafe(old_beta, new_beta, delta);
  const double max_cumulative_lambda =
      VectorAbsoluteDifferenceSafe(
          cumulative_lambda_old, cumulative_lambda_new, delta);
  const double curr_diff = max_beta + max_cumulative_lambda;
  if (current_difference != nullptr) *current_difference = curr_diff;

  return curr_diff < convergence_threshold;
}

bool MpleForIntervalCensoredData::ProfileEmAlgorithmHasConverged(
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
  cumulative_new.resize(old_lambda.size());
  cumulative_old.setZero();
  cumulative_new.setZero();
  for (int i = 0; i < old_lambda.size(); ++i) {
    if (i != 0) {
      cumulative_old(i) = cumulative_old(i - 1);
      cumulative_new(i) = cumulative_new(i - 1);
    }
    cumulative_old(i) += old_lambda(i);
    cumulative_new(i) += new_lambda(i);
  }

  // TODO(PHB): Make delta a global constant.
  const double delta = 0.01;
  const double max_cumulative =
      VectorAbsoluteDifferenceSafe(cumulative_old, cumulative_new, delta);

  return max_cumulative < convergence_threshold;
}

bool MpleForIntervalCensoredData::DoEStep(
    const Expression& transformation_G,
    const Expression& transformation_G_prime,
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    const VectorXd& beta,
    const VectorXd& lambda,
    const vector<VectorXd>& exp_beta_x,
    const double& r,
    const double& integral_constant_factor,
    const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
    MatrixXd* weights,
    VectorXd* posterior_means) {
  // Sanity-check input.
  if (weights == nullptr || posterior_means == nullptr) {
    cout << "ERROR in DoEStep: Null input." << endl;
    return false;
  }

  // Sanity-check dimensions match.
  const unsigned int p = beta.size();
  const unsigned int n = lower_time_bounds.size();
  const unsigned int M = distinct_times.size();
  if (p == 0 || n == 0 || M == 0 || lambda.size() != M ||
      upper_time_bounds.size() != n ||
      exp_beta_x.size() != n ||
      (exp_beta_x[0].size() != M && exp_beta_x[0].size() != 1)) {
    cout << "ERROR in DoEStep: Mismatching dimensions on inputs: "
         << "distinct_times.size(): " << distinct_times.size()
         << ", lower_bounds.size(): " << lower_time_bounds.size()
         << ", upper_bounds.size(): " << upper_time_bounds.size()
         << ", beta.size(): " << beta.size()
         << ", lambda.size(): " << lambda.size()
         << ", exp_beta_x.rows(): " << exp_beta_x.size() << endl;
    return false;
  }

  // Compute S_L and S_U.
  VectorXd S_L, S_U;
  if (!ComputeS(distinct_times, lower_time_bounds, lambda, exp_beta_x, &S_L)) {
    return false;
  }
  if (!ComputeS(distinct_times, upper_time_bounds, lambda, exp_beta_x, &S_U)) {
    return false;
  }

  // Compute exp(-G(S_L)) and exp(-G(S_U)).
  VectorXd exp_neg_g_S_L, exp_neg_g_S_U;  // Size n.
  if (!ComputeExpTransformation(transformation_G, S_L, &exp_neg_g_S_L)) {
    return false;
  }
  if (!ComputeExpTransformation(transformation_G, S_U, &exp_neg_g_S_U)) {
    return false;
  }

  // Compute G'(S_L), and G'(S_U).
  VectorXd g_prime_S_L, g_prime_S_U;  // Size n.
  if (!ComputeTransformationDerivative(
          transformation_G_prime, S_L, &g_prime_S_L)) {
    return false;
  }
  if (!ComputeTransformationDerivative(
          transformation_G_prime, S_U, &g_prime_S_U)) {
    return false;
  }

  // Compute Posterior Means v_i.
  if (!ComputePosteriorMeans(
          exp_neg_g_S_L, exp_neg_g_S_U, g_prime_S_L, g_prime_S_U, posterior_means)) {
    return false;
  }

  // Compute Weights w_ik.
  if (!ComputeWeights(
          distinct_times, lower_time_bounds, upper_time_bounds, lambda,
          S_L, S_U, exp_neg_g_S_L, exp_neg_g_S_U, *posterior_means, exp_beta_x,
          r, integral_constant_factor, gaussian_laguerre_points, weights)) {
    return false;
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeS(
    const set<double>& distinct_times,
    const vector<double>& time_bounds,
    const VectorXd& lambda,
    const vector<VectorXd>& exp_beta_x,
    VectorXd* S) {
  // Sanity-check input.
  if (S == nullptr) {
    cout << "ERROR in ComputeS: Null input." << endl;
    return false;
  }

  const unsigned int n = time_bounds.size();
  const unsigned int M = distinct_times.size();
  if (n == 0 || M == 0 || lambda.size() != M ||
      exp_beta_x.size() != n ||
      (exp_beta_x[0].size() != M && exp_beta_x[0].size() != 1)) {
    cout << "ERROR in ComputeS: Mismatching dimensions on inputs: "
         << "distinct_times.size(): " << distinct_times.size()
         << ", time_bounds.size(): " << time_bounds.size()
         << ", lambda.size(): " << lambda.size()
         << ", exp_beta_x.rows(): " << exp_beta_x.size() << endl;
    return false;
  }

  S->resize(n);
  S->setZero();
  for (unsigned int i = 0; i < n; ++i) {
    if (time_bounds[i] == numeric_limits<double>::infinity()) {
      (*S)(i) = numeric_limits<double>::infinity();
      // NOTE(PHB): If comparing to Donglin, temporarily use 99999 instead of
      // \infty, to match his output.
      //(*S)(i) = 99999.0;
      continue;
    }
    const bool covariates_all_time_indep = exp_beta_x[i].size() == 1;
    int m = 0;
    for (const double& time : distinct_times) {
      if (time <= time_bounds[i]) {
        const int col_to_use = covariates_all_time_indep ? 0 : m;
        (*S)(i) += lambda(m) * exp_beta_x[i](col_to_use);
      }
      ++m;
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::GetXim(
    const int m, const vector<bool>& time_indep_vars_i,
    const pair<VectorXd, MatrixXd>& x_i,
    VectorXd* x_im) {
  if (x_im == nullptr) return false;

  const int p_indep = x_i.first.size();
  const int p_dep = x_i.second.rows();
  const int p = p_indep + p_dep;

  if (time_indep_vars_i.size() != p) return false;

  // If all variables are time-independent, then the values of all covariates
  // at distinct time 'm' is simply the values at all times, which is simply
  // x_i.first.
  if (p_dep == 0) {
    *x_im = x_i.first;
    return true;
  }

  // If all variable are time-dependent, then the x_i.second holds the values
  // of all covariates (at all times); simply return the m^th column.
  if (p_indep == 0) {
    *x_im = x_i.second.col(m);
    return true;
  }

  x_im->resize(p);
  int current_dep_row_index = 0;
  int current_indep_row_index = 0;
  for (int cov_index = 0; cov_index < p; ++cov_index) {
    if (time_indep_vars_i[cov_index]) {
      (*x_im)(cov_index) = x_i.first(current_indep_row_index);
      current_indep_row_index++;
    } else {
      (*x_im)(cov_index) = x_i.second(current_dep_row_index, m);
      current_dep_row_index++;
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeBetaDotXim(
    const int m, const vector<bool>& time_indep_vars_i,
    const VectorXd& beta, const pair<VectorXd, MatrixXd>& x_i,
    double* dot_product) {
  if (dot_product == nullptr || beta.size() != time_indep_vars_i.size()) {
    return false;
  }

  *dot_product = 0.0;
  int vec_row = 0;
  int mat_row = 0;
  for (int cov_index = 0; cov_index < beta.size(); ++cov_index) {
    if (time_indep_vars_i[cov_index]) {
      *dot_product += beta(cov_index) * x_i.first(vec_row);
      vec_row++;
    } else {
      *dot_product += beta(cov_index) * x_i.second(mat_row, m);
      mat_row++;
    }
  }
  
  return true;
}

bool MpleForIntervalCensoredData::AddConstantTimesXim(
    const int m, const vector<bool>& time_indep_vars_i,
    const double& constant, const pair<VectorXd, MatrixXd>& x_i,
    VectorXd* input) {
  if (input == nullptr || input->size() != time_indep_vars_i.size()) {
    return false;
  }

  int vec_row = 0;
  int mat_row = 0;
  for (int cov_index = 0; cov_index < input->size(); ++cov_index) {
    if (time_indep_vars_i[cov_index]) {
      (*input)(cov_index) += constant * x_i.first(vec_row);
      vec_row++;
    } else {
      (*input)(cov_index) += constant * x_i.second(mat_row, m);
      mat_row++;
    }
  }
  
  return true;
}

bool MpleForIntervalCensoredData::AddConstantTimesXimMinusVector(
    const int m, const vector<bool>& time_indep_vars_i,
    const double& constant, const VectorXd& v,
    const pair<VectorXd, MatrixXd>& x_i,
    VectorXd* input) {
  const int p = time_indep_vars_i.size();
  if (input == nullptr || input->size() != p || v.size() != p) {
    return false;
  }

  int vec_row = 0;
  int mat_row = 0;
  for (int cov_index = 0; cov_index < p; ++cov_index) {
    if (time_indep_vars_i[cov_index]) {
      (*input)(cov_index) += constant * (x_i.first(vec_row) - v(cov_index));
      vec_row++;
    } else {
      (*input)(cov_index) += constant * (x_i.second(mat_row, m) - v(cov_index));
      mat_row++;
    }
  }
  
  return true;
}

bool MpleForIntervalCensoredData::ComputeExpBetaX(
    const VectorXd& beta,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    vector<VectorXd>* exp_beta_x) {
  // Sanity-check input.
  if (exp_beta_x == nullptr) {
    cout << "ERROR in ComputeExpBetaX: Null input." << endl;
    return false;
  }

  const unsigned int p = beta.size();
  const unsigned int n = x.size();
  if (n == 0 || p == 0) {
    cout << "ERROR in ComputeExpBetaX: Mismatching dimensions on inputs: "
         << "beta.size(): " << p << ", x.size(): " << n << endl;
    return false;
  }

  exp_beta_x->resize(n);
  for (unsigned int i = 0; i < n; ++i) {
    const int p_indep = x[i].first.size();
    const int p_dep = x[i].second.rows();
    // Only need one distinct time, if all covariates are time-independent.
    const unsigned int M = p_dep == 0 ? 1 : x[i].second.cols();
    (*exp_beta_x)[i].resize(M);
    (*exp_beta_x)[i].setZero();
    for (unsigned int m = 0; m < M; ++m) {
      double dot_product;
      if (!ComputeBetaDotXim(m, time_indep_vars[i], beta, x[i], &dot_product)) {
        return false;
      }
      (*exp_beta_x)[i](m) = exp(dot_product);
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeExpTransformation(
    const Expression& transformation_G,
    const VectorXd& S,
    VectorXd* exp_g_S) {
  if (exp_g_S == nullptr || S.size() == 0) {
    cout << "ERROR in ComputeExpTransformation: Null input." << endl;
    return false;
  }

  const int n = S.size();
  exp_g_S->resize(n);
  for (int i = 0; i < n; ++i) {
    if (S(i) == numeric_limits<double>::infinity()) {
      (*exp_g_S)(i) = 0.0;
      continue;
    }
    double g_value;
    string error_msg;
    if (!EvaluateExpression(transformation_G, "x", S(i), &g_value, &error_msg)) {
      cout << "ERROR: Unable to evaluate:\n\tG(x) = "
           << GetExpressionString(transformation_G) << endl
           << "at x = " << S(i) << ". Error Message:\n\t" << error_msg << endl;
      return false;
    }
    (*exp_g_S)(i) = exp(-1.0 * g_value);
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeTransformationDerivative(
    const Expression& transformation_G_prime,
    const VectorXd& S,
    VectorXd* g_prime_S) {
  if (g_prime_S == nullptr || S.size() == 0) {
    cout << "ERROR in ComputeExpTransformation: Null input." << endl;
    return false;
  }

  const int n = S.size();
  g_prime_S->resize(n);
  for (int i = 0; i < n; ++i) {
    if (S(i) == numeric_limits<double>::infinity()) {
      (*g_prime_S)(i) = 0.0;
      continue;
    }
    string error_msg;
    if (!EvaluateExpression(
            transformation_G_prime, "x", S(i), &((*g_prime_S)(i)), &error_msg)) {
      cout << "ERROR:\n\t" << error_msg << endl;
      return false;
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputePosteriorMeans(
    const VectorXd& exp_neg_g_S_L,
    const VectorXd& exp_neg_g_S_U,
    const VectorXd& g_prime_S_L,
    const VectorXd& g_prime_S_U,
    VectorXd* v) {
  // Sanity-check input.
  if (v == nullptr) {
    cout << "ERROR in ComputePosteriorMeans: Null input." << endl;
    return false;
  }

  const int n = exp_neg_g_S_L.size();
  if (exp_neg_g_S_U.size() != n || g_prime_S_L.size() != n || g_prime_S_U.size() != n) {
    cout << "ERROR in ComputePosteriorMeans: Mismatching dimensions on inputs: "
         << "exp_neg_g_S_L.size(): " << exp_neg_g_S_L.size()
         << ", exp_neg_g_S_U.size(): " << exp_neg_g_S_U.size()
         << ", g_prime_S_L.size(): " << g_prime_S_L.size()
         << ", g_prime_S_U.size(): " << g_prime_S_U.size() << endl;
    return false;
  }

  v->resize(n);
  for (int i = 0; i < n; ++i) {
    (*v)(i) =
        (exp_neg_g_S_L(i) * g_prime_S_L(i) - (exp_neg_g_S_U(i) * g_prime_S_U(i))) /
        (exp_neg_g_S_L(i) - exp_neg_g_S_U(i));
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeWeights(
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    const VectorXd& lambda,
    const VectorXd& S_L,
    const VectorXd& S_U,
    const VectorXd& exp_neg_g_S_L,
    const VectorXd& exp_neg_g_S_U,
    const VectorXd& v,
    const vector<VectorXd>& exp_beta_x,
    const double& r,
    const double& integral_constant_factor,
    const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
    MatrixXd* weights) {
  // Sanity-Check input.
  if (weights == nullptr) {
    cout << "ERROR in ComputeWeights: Null input." << endl;
    return false;
  }

  const int n = S_L.size();
  const int M = distinct_times.size();
  if (n == 0 || M == 0 || lower_time_bounds.size() != n ||
      upper_time_bounds.size() != n || lambda.size() != M ||
      S_U.size() != n || exp_neg_g_S_U.size() != n || exp_neg_g_S_L.size() != n ||
      v.size() != n || exp_beta_x.size() != n ||
      (exp_beta_x[0].size() != M && exp_beta_x[0].size() != 1)) {
    cout << "ERROR in ComputeWeights: Mismatching dimensions on inputs: "
         << "S_L.size(): " << S_L.size()
         << ", S_U.size(): " << S_U.size()
         << ", exp_neg_g_S_L.size(): " << exp_neg_g_S_L.size()
         << ", exp_neg_g_S_U.size(): " << exp_neg_g_S_U.size()
         << ", distinct_times.size(): " << distinct_times.size()
         << ", lower_time_bounds.size(): " << lower_time_bounds.size()
         << ", upper_time_bounds.size(): " << upper_time_bounds.size()
         << ", lambda.size(): " << lambda.size()
         << ", v.size(): " << v.size()
         << ", exp_beta_x.rows(): " << exp_beta_x.size() << endl;
    return false;
  }


  weights->resize(n, M);
  for (int i = 0; i < n; ++i) {
    const bool covariates_all_time_indep = exp_beta_x[i].size() == 1;
    const double l_i = lower_time_bounds[i];
    const double u_i = upper_time_bounds[i];
    int m = 0;
    for (const double& time : distinct_times) {
      if (time <= l_i) {
        (*weights)(i, m) = 0.0;
      } else if (u_i != numeric_limits<double>::infinity() && time <= u_i) {
        const int col_to_use = covariates_all_time_indep ? 0 : m;
        if (r == 0.0) {
          (*weights)(i, m) =
              lambda(m) * exp_beta_x[i](col_to_use) / (1.0 - exp(S_L(i) - S_U(i)));
        } else {
          double integral_value;
          if (!GetIntegralForWeightForTimeWithinLU(
                  S_L(i), S_U(i), r, gaussian_laguerre_points, &integral_value)) {
            cout << "ERROR: Failed to compute weight for subject " << i + 1
                 << " at distinct time " << m + 1 << ": Negative weight."
                 << endl << "If this error occured during computation of "
                 << "Covariance matrix, try modifying --spacing." << endl;
            return false;
          }
          (*weights)(i, m) =
              lambda(m) * exp_beta_x[i](col_to_use) * integral_value /
              ((exp_neg_g_S_L(i) - exp_neg_g_S_U(i)) * integral_constant_factor);
        }
      } else {
        // These weights will not be used anyway; just set a dummy value.
        (*weights)(i, m) = 0.0;
      }
      ++m;
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::GetIntegralForWeightForTimeWithinLU(
    const double& S_L_i, const double& S_U_i, const double& r,
    const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
    double* integral_value) {
  // Sanity-check input.
  if (integral_value == nullptr) {
    cout << "ERROR in GetIntegralForWeightForTimeWithinLU: Null input." << endl;
    return false;
  }
  if (S_L_i >= S_U_i) {
    cout << "ERROR: Invalid (L_i, U_i) "
         << "values: " << S_L_i << ", " << S_U_i << endl;
    return false;
  }

  const double first_term_exp_factor =
      -1.0 * (S_U_i - S_L_i) / (S_L_i + 1.0 / r);
  const double second_term_exp_factor =
      -1.0 * (S_U_i - S_L_i) / (S_U_i + 1.0 / r);
  double first_term = 0.0;
  double second_term = 0.0;
  for (const GaussianQuadratureTuple& point : gaussian_laguerre_points) {
    if (point.abscissa_ == 0.0) {
      cout << "ERROR: Gaussian-Laguerre point is zero." << endl;
      return false;
    }
    first_term += point.weight_ * point.abscissa_ /
                  (1.0 - exp(point.abscissa_ * first_term_exp_factor));
    second_term += point.weight_ * point.abscissa_ /
                   (1.0 - exp(point.abscissa_ * second_term_exp_factor));
  }

  const double first_term_factor = pow(S_L_i + 1.0 / r, -1.0 * (1.0 + 1.0 / r));
  const double second_term_factor = pow(S_U_i + 1.0 / r, -1.0 * (1.0 + 1.0 / r));
  *integral_value =
      first_term_factor * first_term - second_term_factor * second_term;

  if (*integral_value < 0.0) {
    return false;
  }

  return true;
}

bool MpleForIntervalCensoredData::DoMStep(
    const int itr_index,
    const VectorXd& beta,
    const VectorXd& posterior_means,
    const MatrixXd& weights,
    const vector<VectorXd>& exp_beta_x,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    const vector<vector<MatrixXd>>& x_x_transpose,
    const vector<vector<bool>>& r_i_star_indicator,
    VectorXd* new_beta,
    VectorXd* new_lambda) {
  // Sanity-Check input.
  if (new_beta == nullptr || new_lambda == nullptr) {
    cout << "ERROR in DoMStep: Null input." << endl;
    return false;
  }
  const int p = beta.size();
  const int M = weights.cols();
  const int n = weights.rows();
  if (p == 0 || n == 0 || M == 0 || x.size() != n ||
      (x[0].second.rows() > 0 && x[0].second.cols() != M) ||
      posterior_means.size() != n) {
    cout << "ERROR in DoMStep: Mismatching dimensions on inputs: "
         << "x.size(): " << x.size()
         << ", beta.size(): " << p
         << ", posterior_means.size(): " << posterior_means.size()
         << ", weights.rows(): " << weights.rows()
         << ", weights.cols(): " << weights.cols() << endl;
    return false;
  }

  vector<VectorXd> v_exp_beta_x;  // Outer vector size n,
                                  // VectorXd size M (or 1 if all cov's are time-indep)
  if (!ComputeSummandTerm(exp_beta_x, posterior_means, &v_exp_beta_x)) {
    return false;
  }

  vector<double> S0;
  vector<VectorXd> S1;
  vector<MatrixXd> S2;
  if (!ComputeSValues(v_exp_beta_x, time_indep_vars, x, x_x_transpose,
                      r_i_star_indicator, &S0, &S1, &S2)) {
    return false;
  }

  MatrixXd Sigma;
  if (!ComputeSigma(weights, S0, S1, S2, r_i_star_indicator, &Sigma)) return false;

  if (!ComputeNewBeta(
          itr_index, beta, Sigma, weights, time_indep_vars, x,
          S0, S1, r_i_star_indicator, new_beta)) {
    return false;
  }

  vector<double> S0_new;
  if (!ComputeS0(*new_beta, posterior_means, time_indep_vars, x,
                 r_i_star_indicator, &S0_new)) {
    return false;
  }

  if (!ComputeNewLambda(weights, S0_new, r_i_star_indicator, new_lambda)) return false;

  return true;
}

bool MpleForIntervalCensoredData::ComputeSummandTerm(
    const vector<VectorXd>& exp_beta_x,
    const VectorXd& posterior_means,
    vector<VectorXd>* v_exp_beta_x) {
  // Sanity-check input.
  if (v_exp_beta_x == nullptr) {
    cout << "ERROR in ComputeSummandTerm: Null input." << endl;
    return false;
  }
  if (exp_beta_x.size() != posterior_means.size()) {
    cout << "ERROR in ComputeSummandTerm: Mismatching dimensions on inputs: "
         << "exp_beta_x.rows(): " << exp_beta_x.size()
         << ", posterior_means.size(): " << posterior_means.size() << endl;
    return false;
  }

  const int n = exp_beta_x.size();

  v_exp_beta_x->resize(n);

  for (int i = 0; i < n; ++i) {
    (*v_exp_beta_x)[i] = posterior_means(i) * exp_beta_x[i];
  }
  return true;
}

bool MpleForIntervalCensoredData::ComputeSValues(
    const vector<VectorXd>& v_exp_beta_x,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    const vector<vector<MatrixXd>>& x_x_transpose,
    const vector<vector<bool>>& r_i_star_indicator,
    vector<double>* S0, vector<VectorXd>* S1, vector<MatrixXd>* S2) {
  // Sanity-check input.
  if (S0 == nullptr || S1 == nullptr || S2 == nullptr) {
    cout << "ERROR in ComputeSValues: Null input." << endl;
    return false;
  }
  const int n = x.size();
  const int M = r_i_star_indicator[0].size();
  if (n == 0 || v_exp_beta_x.size() != n || r_i_star_indicator.size() != n ||
      (x[0].second.rows() != 0 && x[0].second.cols() != M)) {
    cout << "ERROR in ComputeSValues: Mismatching dimensions on inputs: "
         << "v_exp_beta_x.rows(): " << v_exp_beta_x.size()
         << ", r_i_star_indicator.size(): " << r_i_star_indicator.size()
         << ", x.size(): " << x.size() << endl;
    return false;
  }

  S0->resize(M, 0.0);
  S1->resize(M, VectorXd());
  S2->resize(M, MatrixXd());

  const int p = x[0].first.size() + x[0].second.rows();

  for (int m = 0; m < M; ++m) {

    double& S0_m = (*S0)[m];

    VectorXd& S1_m = (*S1)[m];
    S1_m.resize(p);
    S1_m.setZero();

    MatrixXd& S2_m = (*S2)[m];
    S2_m.resize(p, p);
    S2_m.setZero();

    for (int i = 0; i < n; ++i) {
      const int p_indep = x[i].first.size();
      const int p_dep = x[i].second.rows();
      // The number of times that we need to store S values is either M (if at least
      // one covariate is time-dep), or 1 (if all covariates are time-indep). We
      // check which case we're in by checking if there are any dependent covariates.
      const int col_to_use = p_dep == 0 ? 0 : m;
      if (r_i_star_indicator[i][m]) {
        S0_m += v_exp_beta_x[i](col_to_use);
        if (!AddConstantTimesXim(
                m, time_indep_vars[i], v_exp_beta_x[i](col_to_use), x[i], &S1_m)) {
          return false;
        }
        const int x_x_transpose_col_to_use = x_x_transpose[i].size() > 1 ? m : 0;
        S2_m += v_exp_beta_x[i](col_to_use) * x_x_transpose[i][x_x_transpose_col_to_use];
      }
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeS0(
    const VectorXd& beta,
    const VectorXd& posterior_means,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    const vector<vector<bool>>& r_i_star_indicator,
    vector<double>* S0) {
  // Sanity-check input.
  if (S0 == nullptr) {
    cout << "ERROR in ComputeS0: Null input." << endl;
    return false;
  }

  const int p_indep = x[0].first.size();
  const int p_dep = x[0].second.rows();
  const int p = beta.size();
  const int n = posterior_means.size();
  if (p == 0 || p_indep + p_dep != p || n == 0 || x.size() != n ||
      (p_dep > 0 && x[0].second.cols() == 0)) {
    cout << "ERROR in ComputeS0: Mismatching dimensions on inputs: "
         << "beta.size(): " << beta.size()
         << ", posterior_means.size(): " << posterior_means.size()
         << ", x.size(): " << x.size() << endl;
    return false;
  }
  const int M = r_i_star_indicator[0].size();

  S0->resize(M, 0.0);
  for (int m = 0; m < M; ++m) {
    for (int i = 0; i < n; ++i) {
      if (r_i_star_indicator[i][m]) {
        double dot_product;
        if (!ComputeBetaDotXim(m, time_indep_vars[i], beta, x[i], &dot_product)) {
          return false;
        }
        (*S0)[m] += posterior_means(i) * exp(dot_product);
      }
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeSigma(
    const MatrixXd& weights,
    const vector<double>& S0,
    const vector<VectorXd>& S1,
    const vector<MatrixXd>& S2,
    const vector<vector<bool>>& r_i_star_indicator,
    MatrixXd* Sigma) {
  // Sanity-Check input.
  if (Sigma == nullptr) {
    cout << "ERROR in ComputeSigma: Null input." << endl;
    return false;
  }

  const int M = r_i_star_indicator[0].size();
  const int n = weights.rows();
  if (n == 0 || M == 0 ||
      n != r_i_star_indicator.size() || M != r_i_star_indicator[0].size() ||
      M != weights.cols() ||
      S0.size() != M ||S0.size() != S1.size() || S0.size() != S2.size() ||
      S1[0].size() != S2[0].rows() || S1[0].size() != S2[0].cols()) {
    cout << "ERROR in ComputeSigma: Mismatching dimensions on inputs: "
         << "S0.size(): " << S0.size() << ", S1.size(): "
         << S1.size() << ", S2.size(): " << S2.size()
         << ", weights.rows(): " << weights.rows()
         << ", weights.cols(): " << weights.cols()
         << ", r_i_star_indicator.size() " << r_i_star_indicator.size() << endl;
    return false;
  }

  const int p = S1[0].size();

  Sigma->resize(p, p);
  Sigma->setZero();
  for (int m = 0; m < M; ++m) {
    const double& S0_m = S0[m];
    const VectorXd& S1_m = S1[m];  // p x 1
    const MatrixXd& S2_m = S2[m];  // p x p
    const MatrixXd factor_m =  // p x p
        (S1_m / S0_m) * (S1_m.transpose() / S0_m) - (S2_m / S0_m);
    for (int i = 0; i < n; ++i) {
      if (r_i_star_indicator[i][m]) {
        (*Sigma) += weights(i, m) * factor_m;
      }
    }
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeNewBeta(
    const int itr_index,
    const VectorXd& old_beta,
    const MatrixXd& Sigma,
    const MatrixXd& weights,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    const vector<double>& S0,
    const vector<VectorXd>& S1,
    const vector<vector<bool>>& r_i_star_indicator,
    VectorXd* new_beta) {
  // Sanity-Check input.
  if (new_beta == nullptr) {
    cout << "ERROR in ComputeNewBeta: Null input." << endl;
    return false;
  }

  const int p_indep = x[0].first.size();
  const int p_dep = x[0].second.rows();
  const int p = old_beta.size();
  const int M = weights.cols();
  const int n = weights.rows();
  if (p == 0 || p_dep + p_indep != p ||
      x.empty() || x.size() != n || (p_dep > 0 && x[0].second.cols() != M) ||
      n != r_i_star_indicator.size() || M != r_i_star_indicator[0].size() ||
      S0.size() != M || S1.size() != M ||
      Sigma.rows() != p || Sigma.cols() != p) {
    cout << "ERROR in ComputeNewBeta: Mismatching dimensions on inputs: "
         << "S0.size(): " << S0.size() << ", S1.size(): "
         << S1.size() << ", x.size(): " << x.size()
         << ", old_beta.size(): " << old_beta.size()
         << ", Sigma.rows(): " << Sigma.rows()
         << ", Sigma.cols(): " << Sigma.cols()
         << ", r_i_star_indicator.size() " << r_i_star_indicator.size()
         << ", weights.rows(): " << weights.rows()
         << ", weights.cols(): " << weights.cols() << endl;
    return false;
  }

  // Sanity-check Sigma is invertible.
  FullPivLU<MatrixXd> lu = Sigma.fullPivLu();
  if (!lu.isInvertible()) {
    cout << "ERROR: E-M algorithm failed on iteration " << itr_index
         << " due to singular information matrix.\n";
    return false;
  }

  VectorXd sum;
  sum.resize(p);
  sum.setZero();
  for (int m = 0; m < M; ++m) {
    const double& S0_m = S0[m];
    const VectorXd& S1_m = S1[m];  // p x 1
    const VectorXd quotient_m = S1_m / S0_m;  // p x 1
    for (int i = 0; i < n; ++i) {
      if (r_i_star_indicator[i][m] &&
          !AddConstantTimesXimMinusVector(
              m, time_indep_vars[i], weights(i, m), quotient_m, x[i], &sum)) {
        return false;
      }
    }
  }

  *new_beta = old_beta - Sigma.inverse() * sum;

  return true;
}

bool MpleForIntervalCensoredData::ComputeNewLambda(
    const MatrixXd& weights,
    const vector<double>& S0_new,
    const vector<vector<bool>>& r_i_star_indicator,
    VectorXd* new_lambda) {
  // Sanity-Check input.
  if (new_lambda == nullptr) {
    cout << "ERROR in ComputeNewLambda: Null input." << endl;
    return false;
  }
  const int M = weights.cols();
  const int n = weights.rows();
  if (S0_new.empty() || S0_new.size() != M || n == 0) {
    cout << "ERROR in ComputeNewLambda: Mismatching dimensions on inputs: "
         << "S0_new.size(): " << S0_new.size()
         << ", weights.rows(): " << weights.rows()
         << ", weights.cols(): " << weights.cols() << endl;
    return false;
  }
    
  new_lambda->resize(M);
  for (int m = 0; m < M; ++m) {
    double numerator_k = 0.0;
    for (int i = 0; i < n; ++i) {
      if (r_i_star_indicator[i][m]) {
        numerator_k += weights(i, m);
      }
    }
    (*new_lambda)(m) = numerator_k / S0_new[m];
  }

  return true;
}

// Private, (static + non-static) version.
MpleReturnValue MpleForIntervalCensoredData::ComputeVariance(
    const Expression& transformation_G,
    const Expression& transformation_G_prime,
    const double& r,
    const double& integral_constant_factor,
    const double& convergence_threshold,
    const int h_n_constant, const int max_itr,
    const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    const vector<vector<bool>>& r_i_star_indicator,
    const VectorXd& final_beta,
    const VectorXd& final_lambda,
    MatrixXd* variance) {
  if (variance == nullptr) {
    cout << "ERROR in ComputeVariance: Null Input." << endl;
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  double pl_at_beta;
  VectorXd pl_toggle_one_dim;
  MatrixXd pl_toggle_two_dim;
  if (!ComputeProfileLikelihoods(
          transformation_G, transformation_G_prime, r, integral_constant_factor,
          convergence_threshold, h_n_constant, max_itr, gaussian_laguerre_points,
          distinct_times, lower_time_bounds, upper_time_bounds, time_indep_vars, x,
          r_i_star_indicator, final_beta, final_lambda,
          &pl_at_beta, &pl_toggle_one_dim, &pl_toggle_two_dim)) {
    return MpleReturnValue::FAILED_VARIANCE;
  }

  if (!no_use_pos_def_variance_) {
    if (!ComputeAlternateVarianceFromProfileLikelihoods(
            x.size(), h_n_constant,
            pl_toggle_one_dim, pl_toggle_two_dim, variance)) {
      return MpleReturnValue::FAILED_VARIANCE;
    }
  } else {
    if (!ComputeVarianceFromProfileLikelihoods(
            x.size(), h_n_constant, pl_at_beta,
            pl_toggle_one_dim, pl_toggle_two_dim, variance)) {
      return MpleReturnValue::FAILED_VARIANCE;
    }
  }

  if (NegativeVariance(*variance)) {
    return MpleReturnValue::FAILED_NEGATIVE_VARIANCE;
  }

  return MpleReturnValue::SUCCESS;
}

// Public, non-static version.
MpleReturnValue MpleForIntervalCensoredData::ComputeVariance(
    const VectorXd& beta, const VectorXd& lambda,
    MatrixXd* variance) {
  if (distinct_times_.empty() || lower_time_bounds_.empty() ||
      upper_time_bounds_.empty() || x_.empty()) {
    return MpleReturnValue::FAILED_BAD_INPUT;
  }

  return ComputeVariance(
      transformation_G_, transformation_G_prime_, r_, integral_constant_factor_,
      convergence_threshold_, h_n_constant_, max_itr_, gaussian_laguerre_points_,
      distinct_times_, lower_time_bounds_, upper_time_bounds_, time_indep_vars_,
      x_, r_i_star_indicator_, beta, lambda, variance);
}

bool MpleForIntervalCensoredData::ComputeProfileLikelihoods(
    const Expression& transformation_G,
    const Expression& transformation_G_prime,
    const double& r,
    const double& integral_constant_factor,
    const double& convergence_threshold,
    const int h_n_constant, const int max_itr,
    const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    const vector<vector<bool>>& r_i_star_indicator,
    const VectorXd& final_beta,
    const VectorXd& final_lambda,
    double* pl_at_beta,
    VectorXd* pl_toggle_one_dim,
    MatrixXd* pl_toggle_two_dim) {
  if (pl_at_beta == nullptr || pl_toggle_one_dim == nullptr ||
      pl_toggle_two_dim == nullptr) {
    cout << "ERROR in ComputeProfileLikelihoods: Null Input." << endl;
    return false;
  }

  const int p = final_beta.size();
  const int M = final_lambda.size();
  const int n = x.size();
  if (p == 0 || M == 0  || n == 0 || distinct_times.size() != M ||
      lower_time_bounds.size() != n || upper_time_bounds.size() != n ||
      (x[0].second.rows() > 0 && x[0].second.cols() != M)) {
    cout << "ERROR in ComputeProfileLikelihoods: Empty input: "
         << "final_beta.size(): " << p << ", final_lambda.size(): "
         << M << ", x.size(): " << n
         << ", distinct_times.size(): "  << distinct_times.size()
         << ", lower_time_bounds.size(): " << lower_time_bounds.size()
         << ", upper_time_bounds.size(): " << upper_time_bounds.size()
         << endl;
    return false;
  }

  if (!no_use_pos_def_variance_) {
    pl_toggle_one_dim->resize(n);
    pl_toggle_one_dim->setZero();
    pl_toggle_two_dim->resize(n, p);
    pl_toggle_two_dim->setZero();
  } else {
    pl_toggle_one_dim->resize(p);
    pl_toggle_two_dim->resize(p, p);
  }

  // First compute pl_n(final_beta).
  if (!ComputeProfileLikelihood(
          transformation_G, transformation_G_prime, r, integral_constant_factor,
          convergence_threshold, max_itr, gaussian_laguerre_points, distinct_times,
          lower_time_bounds, upper_time_bounds, time_indep_vars, x,
          r_i_star_indicator, final_beta, final_lambda,
          no_use_pos_def_variance_ ? pl_at_beta : nullptr,
          no_use_pos_def_variance_ ? nullptr : pl_toggle_one_dim)) {
    cout << "ERROR: Failed to compute Profile "
         << "Likelihood at final beta." << endl;
    return false;
  }

  // Now compute the profile likelihoods at the final beta that has
  // "one-dimension toggled"; do this for each of the p dimensions.
  const double h_n = static_cast<double>(h_n_constant) /
                     sqrt(static_cast<double>(n));
  const int num_toggles = (p * p + 3 * p) / 2;
  for (int i = 0; i < p; ++i) {
    VectorXd e_i;
    e_i.resize(p);
    e_i.setZero();
    e_i(i) = h_n;
    VectorXd col_i;
    col_i.resize(n);
    col_i.setZero();
    if (!ComputeProfileLikelihood(
            transformation_G, transformation_G_prime, r, integral_constant_factor,
            convergence_threshold, max_itr, gaussian_laguerre_points, distinct_times,
            lower_time_bounds, upper_time_bounds, time_indep_vars, x,
            r_i_star_indicator, final_beta + e_i, final_lambda,
            no_use_pos_def_variance_ ? &((*pl_toggle_one_dim)(i)) : nullptr,
            no_use_pos_def_variance_ ? nullptr : &col_i)) {
      cout << "ERROR: Failed to compute Profile "
           << "Likelihood for toggling coordinate i = " << i + 1 << "." << endl;
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
    if (logging_on_) {
      time_t current_time = time(nullptr);
      cout << asctime(localtime(&current_time))
           << "Finished " << (i + 1) << " of "
           << (no_use_pos_def_variance_ ? num_toggles : p)
           << " computations for Covariance Matrix." << endl;
    }
  }

  // No need to compute the 'toggle-two coordinates' values if just using
  // the positive-definite version of Covariance computation.
  if (!no_use_pos_def_variance_) return true;

  // Now compute the profile likelihoods at the final beta that has
  // "two-dimensions toggled"; do this for each of the (p_C_2 = p(p + 1) / 2)
  // pairs of the p dimensions.
  pl_toggle_two_dim->setZero();
  for (int i = 0; i < p; ++i) {
    VectorXd e_i;
    e_i.resize(p);
    e_i.setZero();
    e_i(i) = h_n;
    for (int j = 0; j <= i; ++j) {
      VectorXd e_j;
      e_j.resize(p);
      e_j.setZero();
      e_j(j) = h_n;
      if (!ComputeProfileLikelihood(
              transformation_G, transformation_G_prime, r, integral_constant_factor,
              convergence_threshold, max_itr, gaussian_laguerre_points, distinct_times,
              lower_time_bounds, upper_time_bounds, time_indep_vars, x,
              r_i_star_indicator, final_beta + e_i + e_j, final_lambda,
              &((*pl_toggle_two_dim)(i, j)), nullptr)) {
        cout << "ERROR: Failed to compute Profile "
             << "Likelihood for toggling coordinate (i, j) = ("
             << i + 1 << ", " << j + 1 << ")." << endl;
        return false;
      }
      if (logging_on_) {
        time_t current_time = time(nullptr);
        cout << asctime(localtime(&current_time))
             << "Finished " << (p + j + 1 + (i * i + i) / 2)
             << " of " << num_toggles
             << " computations for Covariance Matrix." << endl;
      }
    }
  }

  bool PHB_print_pl = false;
  if (PHB_print_pl) {
    Eigen::IOFormat format(Eigen::FullPrecision);
    cout << "pl_at_beta: " << setprecision(17)
         << *pl_at_beta << endl << "pl_toggle_one_dim:\n"
         << (*pl_toggle_one_dim).format(format) << endl << "pl_toggle_two_dim:\n"
         << (*pl_toggle_two_dim).format(format) << endl;
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeProfileLikelihood(
    const Expression& transformation_G,
    const Expression& transformation_G_prime,
    const double& r,
    const double& integral_constant_factor,
    const double& convergence_threshold,
    const int max_itr,
    const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    const vector<vector<bool>>& time_indep_vars,
    const vector<pair<VectorXd, MatrixXd>>& x,
    const vector<vector<bool>>& r_i_star_indicator,
    const VectorXd& beta,
    const VectorXd& final_lambda,
    double* pl, VectorXd* pl_alternate) {
  if (pl == nullptr && pl_alternate == nullptr) {
    cout << "ERROR in ComputeProfileLikelihood: Null Input." << endl;
    return false;
  }

  const int p = beta.size();
  const int M = final_lambda.size();
  const int n = x.size();
  if (p == 0 || M == 0  || n == 0 || distinct_times.size() != M ||
      lower_time_bounds.size() != n || upper_time_bounds.size() != n ||
      (x[0].second.rows() >0 &&  x[0].second.cols() != M)) {
    cout << "ERROR in ComputeProfileLikelihood: Empty input: "
         << "beta.size(): " << p << ", final_lambda.size(): "
         << M << ", x.size(): " << n
         << ", distinct_times.size(): "  << distinct_times.size()
         << ", lower_time_bounds.size(): " << lower_time_bounds.size()
         << ", upper_time_bounds.size(): " << upper_time_bounds.size()
         << endl;
    return false;
  }

  // Since \beta will not change in the E-M algorithm to find the 
  // maximizing \lambda, we compute exp(\beta^T * X) once here.
  vector<VectorXd> exp_beta_x;  // Outer vector size n,
                                // VectorXd size M (or 1 if all cov's are time-indep)
  if (!ComputeExpBetaX(beta, time_indep_vars, x, &exp_beta_x)) {
    cout << "ERROR: Failed in Computing "
         << "exp(beta^T * X) for beta: " << beta << endl;
    return false;
  }

  // Run (profile, i.e. only maximimizing one parameter \lambda while
  // leaving \beta parameter fixed) E-M algorithm to find maximizing \lambda.
  int iteration_index = 1;
  VectorXd old_lambda = final_lambda;
  VectorXd maximizing_lambda;
  for (; iteration_index < max_itr; ++iteration_index) {
    // Run E-Step.
    VectorXd posterior_means;
    MatrixXd weights;
    if (!DoEStep(transformation_G, transformation_G_prime, distinct_times,
                 lower_time_bounds, upper_time_bounds, beta, old_lambda,
                 exp_beta_x, r, integral_constant_factor, gaussian_laguerre_points,
                 &weights, &posterior_means)) {
      cout << "ERROR: Failed in EStep of iteration " << iteration_index << endl;
      return false;
    }

    // No need to do M-step; just the part where \lambda is updated.
    vector<double> S0_new;
    if (!ComputeS0(beta, posterior_means, time_indep_vars, x,
                   r_i_star_indicator, &S0_new)) {
      return false;
    }
    if (!ComputeNewLambda(
            weights, S0_new, r_i_star_indicator, &maximizing_lambda)) {
      return false;
    }

    // Check Convergence Criterion.
    if (ProfileEmAlgorithmHasConverged(
            convergence_threshold, old_lambda, maximizing_lambda)) {
      bool PHB_print_each_pl_convergence_itr_num = true;
      if (logging_on_ && PHB_print_each_pl_convergence_itr_num) {
        cout << "PL converged in: " << iteration_index << endl;
      }
      break;
    }

    old_lambda = maximizing_lambda;
  }

  // Abort if we failed to converge after max_itr.
  if (iteration_index >= max_itr) {
    cout << "ERROR in ComputeProfileLikelihood: "
         << "E-M algorithm exceeded maximum number of allowed "
         << "iterations: " << max_itr << endl;
    return false;
  }

  const bool PHB_print_num_pl_itrs = false;
  if (PHB_print_num_pl_itrs) {
    cout << "\nSearch for Maximizing lambda concluded in "
         << iteration_index << " iterations." << endl;
  }

  if (!EvaluateLogLikelihoodFunctionAtBetaLambda(
          transformation_G, distinct_times, lower_time_bounds,
          upper_time_bounds, maximizing_lambda, exp_beta_x,
          pl, pl_alternate)) {
    return false;
  }

  return true;
}

bool MpleForIntervalCensoredData::EvaluateLogLikelihoodFunctionAtBetaLambda(
    const Expression& transformation_G,
    const set<double>& distinct_times,
    const vector<double>& lower_time_bounds,
    const vector<double>& upper_time_bounds,
    const VectorXd& lambda,
    const vector<VectorXd>& exp_beta_x,
    double* likelihood, VectorXd* e_i_likelihoods) {
  if (likelihood == nullptr && e_i_likelihoods == nullptr) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Null Input." << endl;
    return false;
  }

  const int M = lambda.size();
  const int n = exp_beta_x.size();
  if (M == 0  || n == 0 || distinct_times.size() != M ||
      lower_time_bounds.size() != n || upper_time_bounds.size() != n ||
      (exp_beta_x[0].size() != M && exp_beta_x[0].size() != 1)) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Empty input: "
         << "lambda.size(): " << M << ", exp_beta_x.rows(): " << n
         << ", distinct_times.size(): "  << distinct_times.size()
         << ", lower_time_bounds.size(): " << lower_time_bounds.size()
         << ", upper_time_bounds.size(): " << upper_time_bounds.size()
         << endl;
    return false;
  }

  // Compute L_n(\beta, \Lambda) at the max \Lambda found above.
  // First need to compute S_L and S_U for final \Lambda vaule.
  VectorXd S_L, S_U;
  if (!ComputeS(distinct_times, lower_time_bounds, lambda, exp_beta_x, &S_L)) {
    return false;
  }
  if (!ComputeS(distinct_times, upper_time_bounds, lambda, exp_beta_x, &S_U)) {
    return false;
  }

  // Compute exp(-G(S_L)) and exp(-G(S_U)).
  VectorXd exp_neg_g_S_L, exp_neg_g_S_U;  // Size n.
  if (!ComputeExpTransformation(transformation_G, S_L, &exp_neg_g_S_L)) {
    return false;
  }
  if (!ComputeExpTransformation(transformation_G, S_U, &exp_neg_g_S_U)) {
    return false;
  }

  // Compute Likelihood L_n.
  return no_use_pos_def_variance_ ?
    EvaluateLogLikelihoodFunctionAtBetaLambda(
      exp_neg_g_S_L, exp_neg_g_S_U, likelihood) :
    EvaluateAlternateLogLikelihoodFunctionAtBetaLambda(
      exp_neg_g_S_L, exp_neg_g_S_U, e_i_likelihoods);
}

bool MpleForIntervalCensoredData::EvaluateLogLikelihoodFunctionAtBetaLambda(
    const VectorXd& exp_neg_g_S_L,
    const VectorXd& exp_neg_g_S_U,
    double* likelihood) {
  if (likelihood == nullptr) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Null Input."
         << endl;
    return false;
  }
  const int n = exp_neg_g_S_L.size();
  if (n == 0 || exp_neg_g_S_U.size() != n) {
    cout << "ERROR in EvaluateLogLikelihoodFunctionAtBetaLambda: Mismatching "
         << "dimensions: exp_neg_g_S_L.size(): " << exp_neg_g_S_L.size()
         << ", exp_neg_g_S_U.size(): " << exp_neg_g_S_U.size() << endl;
    return false;
  }

  // There are two ways to compute L_n:
  //   (1) Compute the product, then take log
  //   (2) take log first, so that product becomes a sum of logs
  // We do (2), but (1) is easy too and is commented out below.
  *likelihood = 0.0;
  for (int i = 0; i < n; ++i) {
    *likelihood += log(exp_neg_g_S_L(i) - exp_neg_g_S_U(i));
  }
  /* Computing L_n as in (1):
  double temp_likelihood = 1.0;
  for (int i = 0; i < n; ++i) {
    temp_likelihood *= (exp_neg_g_S_L(i) - exp_neg_g_S_U(i));
  }
  *likelihood = log(temp_likelihood);
  */

  return true;
}

bool MpleForIntervalCensoredData::EvaluateAlternateLogLikelihoodFunctionAtBetaLambda(
    const VectorXd& exp_neg_g_S_L,
    const VectorXd& exp_neg_g_S_U,
    VectorXd* e_i_likelihoods) {
  if (e_i_likelihoods == nullptr) {
    cout << "ERROR in EvaluateAlternateLogLikelihoodFunctionAtBetaLambda: Null Input."
         << endl;
    return false;
  }
  const int n = exp_neg_g_S_L.size();
  if (n == 0 || exp_neg_g_S_U.size() != n) {
    cout << "ERROR in EvaluateAlternateLogLikelihoodFunctionAtBetaLambda: Mismatching "
         << "dimensions: exp_neg_g_S_L.size(): " << exp_neg_g_S_L.size()
         << ", exp_neg_g_S_U.size(): " << exp_neg_g_S_U.size() << endl;
    return false;
  }

  for (int i = 0; i < n; ++i) {
    (*e_i_likelihoods)(i) = log(exp_neg_g_S_L(i) - exp_neg_g_S_U(i));
  }

  return true;
}

bool MpleForIntervalCensoredData::ComputeVarianceFromProfileLikelihoods(
      const int n, const int h_n_constant, const double& pl_at_beta,
      const VectorXd& pl_toggle_one_dim, const MatrixXd& pl_toggle_two_dim,
      MatrixXd* variance) {
  if (variance == nullptr) {
    cout << "ERROR in ComputeVarianceFromProfileLikelihoods: Null input."
         << endl;
    return false;
  }

  const int p = pl_toggle_one_dim.size();
  if (p == 0 || pl_toggle_two_dim.rows() != p || pl_toggle_two_dim.cols() != p) {
    cout << "ERROR in ComputeVarianceFromProfileLikelihoods: Mismatching "
         << "dimensions: pl_toggle_one_dim.size(): " << p
         << ", pl_toggle_two_dim.rows(): " << pl_toggle_two_dim.rows()
         << ", pl_toggle_two_dim.cols(): " << pl_toggle_two_dim.cols()
         << endl;
    return false;
  }

  const double h_n_squared =
      pow(static_cast<double>(h_n_constant), 2.0) / static_cast<double>(n);

  MatrixXd variance_inverse;
  variance_inverse.resize(p, p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j <= i; ++j) {
      variance_inverse(i, j) =
          (pl_at_beta - pl_toggle_one_dim(i) -
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
    cout << "ERROR: variance_inverse is not invertible:" << endl
         << variance_inverse << endl;
    return false;
  }
  
  *variance = -1.0 * variance_inverse.inverse();
 
  return true;
}

bool MpleForIntervalCensoredData::ComputeAlternateVarianceFromProfileLikelihoods(
      const int n, const int h_n_constant,
      const VectorXd& pl_toggle_none,
      const MatrixXd& pl_toggle_one_dim,
      MatrixXd* variance) {
  if (variance == nullptr) {
    cout << "ERROR in ComputeVarianceAlternateFromProfileLikelihoods: Null input."
         << endl;
    return false;
  }

  const int cov_dim = pl_toggle_one_dim.cols();
  if (n == 0 || pl_toggle_one_dim.rows() != n ||
      cov_dim == 0 || pl_toggle_none.size() != n) {
    cout << "ERROR in ComputeAltVarianceFromProfileLikelihoods: n: " << n
         << ", pl_toggle_one_dim.rows(): " << pl_toggle_one_dim.rows()
         << ", pl_toggle_one_dim.cols(): " << pl_toggle_one_dim.cols()
         << ", pl_toggle_none.size(): " << pl_toggle_none.size()
         << endl;
    return false;
  }

  const double h_n =
      static_cast<double>(h_n_constant) / sqrt(static_cast<double>(n));

  MatrixXd variance_inverse;
  variance_inverse.resize(cov_dim, cov_dim);
  variance_inverse.setZero();
  for (int i = 0; i < n; ++i) {
    const double& toggle_none_i = pl_toggle_none(i);
    VectorXd ps_i;
    ps_i.resize(cov_dim);
    for (int p = 0; p < cov_dim; ++p) {
      ps_i(p) = (pl_toggle_one_dim(i, p) - toggle_none_i) / h_n; 
    }
    variance_inverse += ps_i * ps_i.transpose();
  }

  // Sanity-check varinace inverse is invertible.
  FullPivLU<MatrixXd> lu = variance_inverse.fullPivLu();
  if (!lu.isInvertible()) {
    cout << "ERROR: variance_inverse is not invertible:" << endl
         << variance_inverse << endl;
    return false;
  }
  
  *variance = variance_inverse.inverse();
 
  return true;
}

}  // namespace regression
