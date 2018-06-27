// Date: April 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "cox_regression.h"

#include "FileReaderUtils/read_file_structures.h"
#include "FileReaderUtils/read_table_with_header.h"
#include "MathUtils/data_structures.h"
#include "MathUtils/gamma_fns.h"
#include "MathUtils/number_comparison.h"
#include "MathUtils/statistics_utils.h"
#include "Regression/linear_regression.h"

#include <cmath>
#include <cstdlib>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;
using Eigen::Dynamic;
using Eigen::FullPivLU;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace file_reader_utils;
using namespace math_utils;

namespace regression {

bool CoxRegression::GetSortLegendAndTransitions(
    const vector<CensoringData>& dep_var,
    vector<int>* sort_legend, vector<int>* transition_indices) {
  // Get minimum of Survival Time and Censoring Time.
  vector<double> times;
  for (const CensoringData& data : dep_var) {
    times.push_back(min(data.survival_time_, data.censoring_time_));
  }

  // Sort times.
  sort(times.begin(), times.end());

  // Now go through sorted times, finding transition indices, and mapping
  // back to the original index that had that time.
  set<int> used_indices;
  double previous_time = -1.0;
  for (int i = 0; i < times.size(); ++i) {
    const double& time = times[i];
    if (time != previous_time) {
      transition_indices->push_back(i);
      previous_time = time;
    }
    bool found_match = false;
    for (int j = 0; j < dep_var.size(); ++j) {
      if (used_indices.find(j) != used_indices.end()) continue;
      const CensoringData& data = dep_var[j];
      if (FloatEq(time, min(data.survival_time_, data.censoring_time_))) {
        used_indices.insert(j);
        sort_legend->push_back(j);
        found_match = true;
        break;
      }
    }
    if (!found_match) {
      cout << "ERROR in GetSortLegendAndTransitions: Unable to find time "
           << time << ". Aborting.\n";
      return false;
    }
  }
  return true;
}

bool CoxRegression::SortInputByTime(
    const vector<CensoringData>& dep_var, const MatrixXd& indep_vars,
    vector<CensoringData>* sorted_dep_var, MatrixXd* sorted_indep_vars,
    vector<int>* sort_legend, vector<int>* transition_indices) {
  if (!GetSortLegendAndTransitions(dep_var, sort_legend, transition_indices)) {
    return false;
  }
  sorted_indep_vars->resize(indep_vars.rows(), indep_vars.cols());
  for (int i = 0; i < sort_legend->size(); ++i) {
    const int from_index = (*sort_legend)[i];
    sorted_dep_var->push_back(dep_var[from_index]);
    sorted_indep_vars->row(i) = indep_vars.row(from_index);
  }
  return true;
}

bool CoxRegression::ComputePartialSums(
    const vector<int>& transition_indices,
    const VectorXd& exp_logistic_eq,
    VectorXd* partial_sums) {
  double running_total = 0.0;
  int backwards_itr = exp_logistic_eq.size() - 1;  // n - 1
  partial_sums->resize(transition_indices.size());
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

bool CoxRegression::ComputeExponentialOfLogisticEquation(
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

bool CoxRegression::ComputeLogLikelihood(
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

bool CoxRegression::ComputeScoreFunction(
    const VectorXd& exp_logistic_eq,
    const VectorXd& partial_sums,
    const vector<int>& transition_indices,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    VectorXd* score_function) {
  // Sanity-Check input.
  if (score_function == nullptr ||
      indep_vars.rows() != dep_var.size() ||
      indep_vars.rows() != exp_logistic_eq.size()) {
    cout << "\nComputeScoreFunction Failure. indep_vars.size(): "
         << indep_vars.rows() << ", dep_var.size(): "
         << dep_var.size() << ", exp_logistic_eq.size(): "
         << exp_logistic_eq.size() << endl;
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
    *score_function +=
        static_cast<VectorXd>(indep_vars.row(i)) -
        (numerator / partial_sums(transition_itr));
  }
  return true;
}

bool CoxRegression::ComputeInformationMatrix(
    const VectorXd& exp_logistic_eq,
    const VectorXd& partial_sums,
    const vector<int>& transition_indices,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    MatrixXd* info_matrix) {
  // Sanity-Check input.
  if (info_matrix == nullptr || indep_vars.cols() == 0 ||
      indep_vars.rows() != exp_logistic_eq.size()) {
    cout << "\nComputeScoreFunction Failure. indep_vars.size(): "
         << indep_vars.rows() << ", exp_logistic_eq.size(): "
         << exp_logistic_eq.size() << endl;
    return false;
  }
  if (partial_sums.size() != transition_indices.size()) {
    cout << "\nComputeScoreFunction Failure. partial_sums size: "
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
        first_term_numerator / partial_sums(transition_itr) -
        (second_term_numerator * second_term_numerator.transpose() /
         (partial_sums(transition_itr) * partial_sums(transition_itr)));
  }
  return true;
}

bool CoxRegression::ComputeNewBetaHat(
    const vector<int>& transition_indices,
    const VectorXd& beta_hat,
    const double& log_likelihood,
    const VectorXd& score_function,
    const MatrixXd& info_matrix,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    VectorXd* new_beta_hat,
    VectorXd* new_logistic_eq,
    VectorXd* new_exp_logistic_eq,
    VectorXd* new_partial_sums,
    double* new_log_likelihood) {
  // Sanity-check dimensions.
  if (beta_hat.size() != score_function.size() ||
      info_matrix.rows() != info_matrix.cols() ||
      beta_hat.size() != info_matrix.rows()) {
    cout << "ERROR: Bad dimensions on matrices input to ComputeNewBetaHat. "
         << "beta_hat: " << beta_hat.size() << ", score_function: "
         << score_function.size() << ", info_matrix: ("
         << info_matrix.rows() << ", " << info_matrix.cols() << ")\n";
    return false;
  }

  // Check that info_matrix is invertible.
  FullPivLU<MatrixXd> lu = info_matrix.fullPivLu();
  if (!lu.isInvertible()) {
    cout << "ERROR: info_matrix is not invertible:\n" << info_matrix << "\n";
    return false;
  }

  int num_halving_attempts = 0;
  bool found_better_likelihood = false;
  while (!found_better_likelihood &&
         num_halving_attempts < MAX_HALVING_ATTEMPTS) {
    if (num_halving_attempts == 0) {
      // First time through, no need to perform halving yet, just use
      // ordinary iterative formulat for new beta hat.
      *new_beta_hat = beta_hat + info_matrix.inverse() * score_function;
    } else {
      // The first time(s) through failed. Go half the distance as the
      // previous attempt.
      *new_beta_hat = (beta_hat + *new_beta_hat) / 2.0;
    }

    // Get \beta^T * X_i and exp(\beta^T * X_i) for new_beta_hat.
    if (!ComputeExponentialOfLogisticEquation(
            *new_beta_hat, indep_vars, new_logistic_eq, new_exp_logistic_eq)) {
      cout << "ERROR: Unable to compute the exponential of the RHS of the "
           << "logistic equation for beta:\n"
           << *new_beta_hat << "\nAborting.\n";
      return false;
    }

    // Get partial sums S_i(new_beta_hat) for new_beta_hat:
    //   := \sum_j I(T_j >= T_i) * new_exp_logistic_eq_j
    if (!ComputePartialSums(
            transition_indices, *new_exp_logistic_eq, new_partial_sums)) {
      cout << "ERROR: Unable to compute partial sums for beta:\n"
           << *new_beta_hat << "\nAborting.\n";
      return false;
    }

    // Compute Log Likelihood of new \hat{\beta}.
    if (!ComputeLogLikelihood(
            *new_logistic_eq, *new_exp_logistic_eq, *new_partial_sums,
            transition_indices, dep_var, new_log_likelihood)) {
      cout << "ERROR: Unable to compute new log-likelihood for beta:\n"
           << *new_beta_hat << "\nwith new_exp_logistic_eq:\n"
           << (*new_exp_logistic_eq)(0) << "\nAborting.\n";
      return false;
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

bool CoxRegression::RunNewtonRaphson(
    const ConvergenceCriteria& convergence_criteria,
    const vector<int>& transition_indices,
    const VectorXd& beta_hat,
    const VectorXd& logistic_eq,
    const VectorXd& exp_logistic_eq,
    const VectorXd& partial_sums,
    const double& log_likelihood,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    VectorXd* regression_coefficients, int* iterations) {
  if (*iterations > MAX_ITERATIONS) {
    cout << "ERROR: Newton-Raphson failed to converge after "
         << MAX_ITERATIONS << " iterations. Aborting.\n";
    return false;
  }

  (*iterations)++;

  // Compute Score function U(\hat{\beta}).
  VectorXd score_function;
  score_function.resize(beta_hat.size());
  if (!ComputeScoreFunction(
          exp_logistic_eq, partial_sums, transition_indices, indep_vars,
          dep_var, &score_function)) {
    cout << "ERROR: Unable to compute the score function of beta:\n"
         << beta_hat << "\nwith exp_logistic_eq:\n" << exp_logistic_eq(0)
         << "\nAborting.\n";
    return false;
  }

  // Compute Information Matrix V(\hat{\beta}).
  MatrixXd info_matrix;
  if (!ComputeInformationMatrix(
          exp_logistic_eq, partial_sums, transition_indices,
          indep_vars, dep_var, &info_matrix)) {
    cout << "ERROR: Unable to compute the Information Matrix of beta:\n"
         << beta_hat << "\nwith exp_logistic_eq:\n" << exp_logistic_eq(0)
         << "\nAborting.\n";
    return false;
  }

  // Compute new \beta, \beta^T * X_i, exp(\beta^T * X_i), and Log-Likelihood.
  VectorXd new_beta_hat, new_logistic_eq, new_exp_logistic_eq, new_partial_sums;
  double new_log_likelihood;
  if (!ComputeNewBetaHat(
          transition_indices, beta_hat, log_likelihood, score_function,
          info_matrix, indep_vars, dep_var,
          &new_beta_hat, &new_logistic_eq, &new_exp_logistic_eq,
          &new_partial_sums, &new_log_likelihood)) {
    cout << "ERROR: Unable to compute new beta for old beta:\n"
         << beta_hat << "\nScore function:\n" << score_function
         << "\nInformation Matrix:\n" << info_matrix << "\nAborting.\n";
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
     cout << "ERROR: Unrecognized convergence_criteria: "
          << convergence_criteria.to_compare_ << ". Aborting.\n";
     return false;
  }
  if (!convergence_attained) {
    return RunNewtonRaphson(
        convergence_criteria, transition_indices,
        new_beta_hat, new_logistic_eq, new_exp_logistic_eq, new_partial_sums,
        new_log_likelihood, indep_vars, dep_var, regression_coefficients,
        iterations);
  } else {
    *regression_coefficients = new_beta_hat;
    return true;
  }  
}

bool CoxRegression::ComputeRegressionCoefficients(
    const ConvergenceCriteria& convergence_criteria,
    const vector<int>& transition_indices,
    const MatrixXd& indep_vars,
    const vector<CensoringData>& dep_var,
    int* num_iterations,
    MatrixXd* info_matrix_inverse,
    VectorXd* regression_coefficients) {
  int n, p;
  if (!GetDimensions(indep_vars, &n, &p)) {
    cout << "ERROR: Unable to extract n, p from indep_vars. Aborting.\n";
    return false;
  }

  // First guess for \hat{\beta} is a vector of all zeros.
  VectorXd beta_hat;
  beta_hat.resize(p);
  beta_hat.setZero();

  // Perform Newton-Raphson method to find y-intercept of Score function
  // (yields local maximum/minimum of log-likelihood).
  int iterations = 0;
  // For \beta = Zero vector:
  //   - initial logistic_eq (\beta^T * X) is 0.0 for every sample.
  //   - initial exp_logistic_eq is 1.0 for every sample.
  //   - initial log-likelihood is -n * log(2)
  double log_likelihood = static_cast<double>(n) * log(2.0);
  VectorXd initial_logistic_eq, initial_exp_logistic_eq, initial_partial_sums;
  initial_logistic_eq.resize(n);
  initial_exp_logistic_eq.resize(n);
  for (int i = 0; i < n; ++i) {
    initial_logistic_eq(i) = 0.0;
    initial_exp_logistic_eq(i) = 1.0;
  }
  initial_partial_sums.resize(transition_indices.size());
  for (int i = 0; i < transition_indices.size(); ++i) {
    initial_partial_sums(i) = n - transition_indices[i];
  }

  if (!RunNewtonRaphson(
          convergence_criteria, transition_indices, beta_hat,
          initial_logistic_eq, initial_exp_logistic_eq, initial_partial_sums,
          log_likelihood, indep_vars, dep_var, regression_coefficients,
          num_iterations == nullptr ? &iterations : num_iterations)) {
    return false;
  }

  // Compute the final value for exp(\beta^T * X_i).
  VectorXd logistic_eq, exp_logistic_eq;
  if (!ComputeExponentialOfLogisticEquation(
          *regression_coefficients, indep_vars,
          &logistic_eq, &exp_logistic_eq)) {
    cout << "ERROR: Unable to compute the exponential of the RHS of the "
         << "logistic equation for beta:\n"
         << *regression_coefficients << "\nAborting.\n";
    return false;
  }

  // Compute the final partial sums S_i(beta_hat) for beta_hat:
  //   := \sum_j I(T_j >= T_i) * exp_logistic_eq_j
  VectorXd partial_sums;
  if (!ComputePartialSums(
          transition_indices, exp_logistic_eq, &partial_sums)) {
    cout << "ERROR: Unable to compute partial sums for beta:\n"
         << *regression_coefficients << "\nAborting.\n";
    return false;
  }

  // Compute final Log Likelihood of \hat{\beta}.
  double final_log_likelihood;
  if (!ComputeLogLikelihood(
          logistic_eq, exp_logistic_eq, partial_sums, transition_indices,
          dep_var, &final_log_likelihood)) {
    cout << "ERROR: Unable to compute log-likelihood for beta:\n"
         << *regression_coefficients << "\nwith exp_logistic_eq:\n"
         << exp_logistic_eq(0) << "\nAborting.\n";
    return false;
  }

  // Compute final Score function U(\hat{\beta}).
  VectorXd score_function;
  score_function.resize(p);
  if (!ComputeScoreFunction(
          exp_logistic_eq, partial_sums, transition_indices, indep_vars,
          dep_var, &score_function)) {
    cout << "ERROR: Unable to compute final score function of beta:\n"
         << *regression_coefficients << "\nwith exp_logistic_eq:\n"
         << exp_logistic_eq(0) << "\n. Aborting.\n";
    return false;
  }

  // Compute final Information Matrix V(\hat{\beta}).
  MatrixXd info_matrix;
  if (!ComputeInformationMatrix(
        exp_logistic_eq, partial_sums, transition_indices,
        indep_vars, dep_var, &info_matrix)) {
    cout << "ERROR: Unable to compute final information matrix of beta:\n"
         << *regression_coefficients << "\nwith exp_logistic_eq:\n"
         << exp_logistic_eq(0) << "\n. Aborting.\n";
    return false;
  }

  // Check that info_matrix is invertible.
  FullPivLU<MatrixXd> lu = info_matrix.fullPivLu();
  if (!lu.isInvertible()) {
    cout << "ERROR: Final info_matrix is not invertible:\n"
         << info_matrix << "\n";
    return false;
  }

  // Store info_matrix_inverse.
  *info_matrix_inverse = info_matrix.inverse();
  return true;
}

bool CoxRegression::RunCoxRegression(
    const string& output_filename,
    const ModelAndDataParams& params,
    SummaryStatistics* stats) {
  // Sanity check input.
  if (output_filename.empty() && stats == nullptr) return false;
  if (params.model_type_ != ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL) {
    return false;
  }
  const MatrixXd& indep_vars = params.linear_term_values_;
  int n, p;
  if (!GetDimensions(indep_vars, &n, &p)) {
    cout << "ERROR: Unable to extract n, p from indep_vars. Aborting.\n";
    return false;
  }
  const vector<CensoringData>& dep_var = params.dep_vars_.dep_vars_cox_;
  if (dep_var.size() != n) {
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

  // Sort indep_vars based on Survival/Censoring time. Computations are easier
  // based on this sorting; we keep track of the original order in
  // 'sort_legend', and then return the final regression_coefficients with
  // respect to the original input order by undoing the sort at the end.
  vector<CensoringData> sorted_dep_var;
  MatrixXd sorted_indep_vars;
  vector<int> sort_legend;
  vector<int> transition_indices;
  if (!SortInputByTime(
          dep_var, indep_vars,
          &sorted_dep_var, &sorted_indep_vars,
          &sort_legend, &transition_indices)) {
    return false;
  }

  // Perform matrix computations to find Regression Coefficients \hat{\Beta}.
  MatrixXd info_matrix_inverse;
  VectorXd sorted_regression_coefficients;
  int iterations = 0;
  int* iteration_ptr =
      stats == nullptr ? &iterations : &stats->num_iterations_;
  if (!ComputeRegressionCoefficients(
          params.convergence_criteria_, transition_indices,
          sorted_indep_vars, sorted_dep_var, iteration_ptr,
          &info_matrix_inverse, &sorted_regression_coefficients)) {
    return false;
  }

  // Unstandardize final values, if appropriate.
  VectorXd* unstandardized_beta = &sorted_regression_coefficients;
  MatrixXd* unstandardized_covariance = &info_matrix_inverse;
  MatrixXd unstandardized_covariance_matrix;
  VectorXd unstandardized_beta_values;
  bool unstandardization_required = false;
  if (!LinearRegression::UnstandardizeResults(
          params.standardize_vars_, params.linear_terms_mean_and_std_dev_,
          sorted_regression_coefficients, info_matrix_inverse,
          &unstandardization_required,
          &unstandardized_beta_values, &unstandardized_covariance_matrix)) {
    return false;
  }
  if (unstandardization_required) {
    unstandardized_beta = &unstandardized_beta_values;
    unstandardized_covariance = &unstandardized_covariance_matrix;
  }

  // Compute Estimate, (Estimated) Variance, SEE, T-Statistic, and p-value; and
  // print to file.
  string error_msg = "";
  if (!ComputeFinalValuesAndPrint(
          *iteration_ptr, output_filename, params.legend_,
          *unstandardized_covariance, *unstandardized_beta,
          stats, &error_msg)) {
    cout << "ERROR: Invalid Values computed:\n " << error_msg << endl;;
    return false;
  }

  return true;
}


bool CoxRegression::PrintValues(
    const int num_iterations,
    const vector<string>& titles, const vector<double>& estimates,
    const vector<double>& variances, const vector<double>& p_values,
    const vector<double>& standard_estimates_of_error,
    const vector<double>& t_statistics, const string& outfile) {
  // Sanity check all vectors have the same length.
  const int length = titles.size();
  if (length != estimates.size() || length != variances.size() ||
      length != p_values.size() || length != t_statistics.size() ||
      length != standard_estimates_of_error.size()) {
    cout << "ERROR: not all input vectors have same length: "
         << titles.size() << ", " << estimates.size() << ", "
         << variances.size() << ", " << p_values.size() << ", "
         << standard_estimates_of_error.size() << ", "
         << t_statistics.size();
    return false;
  }

  // Write title line.
  ofstream out_file;
  out_file.open(outfile.c_str());
  // Write Num iterations.
  out_file << "Cox Regression completed in " << num_iterations
           << " iterations.\n\n";
  // Write HEADER line.
  out_file << "Variable_Name\t\tEstimate\tVariance\tS.E.E.  "
           << "\tT-Statistic\tp-value\n";

  // Loop over vectors containing the statistics, outputing one value
  // from each on each line of the outfile.
  for (int i = 0; i < length; ++i) {
    out_file << "B_" << i << " (" << titles[i] << ")\t\t";
    char out_line[512] = "";
    sprintf(out_line,
        "%0.06f\t%0.06f\t%0.06f\t%0.06f\t%0.06f\n",
        estimates[i], variances[i], standard_estimates_of_error[i],
        t_statistics[i], p_values[i]);
    out_file << out_line;
  }
  out_file.close();
  return true;
}

bool CoxRegression::ComputeFinalValuesAndPrint(
    const int num_iterations,
    const string& outfile,
    const vector<string>& titles,
    const MatrixXd& info_matrix_inverse,
    const VectorXd& regression_coefficients,
    SummaryStatistics* stats, string* error_msg) {
  if (outfile.empty() && stats == nullptr) return false;

  ofstream out_file;
  if (!outfile.empty()) {
    out_file.open(outfile.c_str());
    // Write Num iterations.
    out_file << "Cox Regression completed in " << num_iterations
             << " iterations.\n\n";
    // Write HEADER line.
    out_file << "Variable_Name\t\tEstimate\tVariance\tS.E.E.  "
             << "\tT-Statistic\tp-value\n";
  }

  const int p = regression_coefficients.size();
  if (stats != nullptr) {
    stats->estimates_.resize(p);
    stats->variances_.resize(p);
    stats->standard_error_.resize(p);
    stats->p_values_.resize(p);
    stats->t_statistics_.resize(p);
  }

  for (int i = 0; i < p; ++i) {
    const string s = "X_" + Itoa(i);
    const string var_title = titles.empty() ? s : titles[i];
    const string var_name = "B_" + Itoa(i) + " (" + var_title + ")";
    double est_val = regression_coefficients(i);
    double est_var = info_matrix_inverse(i, i);
    double est_se = -1.0;
    if (est_var >= 0.0) {
      est_se = sqrt(est_var);
    } else {
      if (error_msg != nullptr) {
        *error_msg += "Negative estimated variance (should never happen): " +
                      Itoa(est_var) + "\n";
      }
      return false;
    }
    double test_stat = -1.0;
    if (est_se > 0.0) {
      test_stat = est_val / est_se;
    } else {
      if (error_msg != nullptr) {
        *error_msg += "Negative S.E.E. (should never happen). est_var: " +
                      Itoa(est_var) + ", S.E.E.: " + Itoa(est_se) + "\n";
      }
      return false;
    }
    double p_value = -1.0;
    p_value =
        RegularizedReverseIncompleteGammaFunction(0.5, (0.5 * test_stat * test_stat));

    // Print to file, if appropriate.
    if (!outfile.empty()) {
      out_file << var_name << "\t\t" << setprecision(8)
               << est_val << "\t" << est_var << "\t"
               << est_se << "\t" << test_stat << "\t" << p_value << "\n";
    }

    // Populate stats, if appropriate.
    if (stats != nullptr) {
      stats->estimates_[i] = est_val;
      stats->variances_[i] = est_var;
      stats->standard_error_[i] = est_se;
      stats->p_values_[i] = p_value;
      stats->t_statistics_[i] = test_stat;
    }
  }

  if (!outfile.empty()) {
    out_file.close();
  }
  return true;
}

}  // namespace regression
