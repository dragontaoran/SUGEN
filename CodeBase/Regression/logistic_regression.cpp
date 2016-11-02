// Date: March 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "logistic_regression.h"

#include "FileReaderUtils/read_file_structures.h"
#include "FileReaderUtils/read_table_with_header.h"
#include "MathUtils/gamma_fns.h"
#include "MathUtils/number_comparison.h"
#include "MathUtils/statistics_utils.h"
#include "Regression/linear_regression.h"

#include <cmath>
#include <cstdlib>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>      // std::setprecision
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using Eigen::Dynamic;
using Eigen::FullPivLU;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace file_reader_utils;
using namespace math_utils;
using namespace std;

namespace regression {

bool LogisticRegression::ComputeExponentialOfLogisticEquation(
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

bool LogisticRegression::ComputeLogLikelihood(
    const VectorXd& logistic_eq,
    const VectorXd& exp_logistic_eq,
    const VectorXd& dep_var,
    double* log_likelihood) {
  *log_likelihood = 0;
  for (int i = 0; i < logistic_eq.size(); ++i) {
    double first_term = FloatEq(0.0, dep_var[i]) ? 0.0 : logistic_eq(i);
    *log_likelihood += first_term - log(1 + exp_logistic_eq(i));
  }
  return true;
}

bool LogisticRegression::ComputeScoreFunction(
    const VectorXd& exp_logistic_eq,
    const MatrixXd& indep_vars,
    const VectorXd& dep_var,
    VectorXd* score_function) {
  // Sanity-Check input.
  if (score_function == nullptr ||
      indep_vars.rows() != dep_var.size() ||
      indep_vars.rows() != exp_logistic_eq.size()) {
    cout << "\nComputeScoreFunction Failure. indep_vars.size(): "
         << indep_vars.rows() << ", dep_var.size(): "
         << dep_var.size() << ", exp_logistic_eq.size(): "
         << exp_logistic_eq.size();
    return false;
  }

  VectorXd exp_regression_rhs;
  exp_regression_rhs.resize(indep_vars.rows());
  for (int i = 0; i < indep_vars.rows(); ++i) {
    exp_regression_rhs(i) = exp_logistic_eq(i) / (1.0 + exp_logistic_eq(i));
  }

  score_function->resize(indep_vars.cols());
  score_function->setZero();
  *score_function =
      indep_vars.transpose() * (dep_var - exp_regression_rhs);

  return true;
}

bool LogisticRegression::ComputeInformationMatrix(
      const VectorXd& exp_logistic_eq,
      const MatrixXd& indep_vars,
      MatrixXd* info_matrix) {
  const int p = indep_vars.cols();
  const int n =  indep_vars.rows();
  if (exp_logistic_eq.size() != n) return false;

  // Construct diagonal (n, n) matrix "A", with i^th entry equal to:
  //   exp(\beta^T * X_i) / (1 + exp(\beta^T * X_i))
  MatrixXd diagonal_coefficient_matrix;
  diagonal_coefficient_matrix.resize(n, n);
  diagonal_coefficient_matrix.setZero();
  for (int i = 0; i < n; ++i) {
    diagonal_coefficient_matrix(i, i) =
      exp_logistic_eq(i) / pow(1.0 + exp_logistic_eq(i), 2);
  }

  // Set info_matrix as:
  //   X^T * A * X
  *info_matrix =
      indep_vars.transpose() * diagonal_coefficient_matrix * indep_vars;

  return true;
}

bool LogisticRegression::ComputeNewBetaHat(
    const VectorXd& beta_hat,
    const double& log_likelihood,
    const VectorXd& score_function,
    const MatrixXd& info_matrix,
    const MatrixXd& indep_vars,
    const VectorXd& dep_var,
    VectorXd* new_beta_hat,
    VectorXd* new_logistic_eq,
    VectorXd* new_exp_logistic_eq,
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

    // Compute Log Likelihood of new \hat{\beta}.
    if (!ComputeLogLikelihood(*new_logistic_eq, *new_exp_logistic_eq, dep_var,
                              new_log_likelihood)) {
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

bool LogisticRegression::RunNewtonRaphson(
    const ConvergenceCriteria& convergence_criteria,
    const VectorXd& beta_hat,
    const VectorXd& logistic_eq,
    const VectorXd& exp_logistic_eq,
    const double& log_likelihood,
    const MatrixXd& indep_vars,
    const VectorXd& dep_var,
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
          exp_logistic_eq, indep_vars, dep_var, &score_function)) {
    cout << "ERROR: Unable to compute the score function of beta:\n"
         << beta_hat << "\nwith exp_logistic_eq:\n" << exp_logistic_eq(0)
         << "\nAborting.\n";
    return false;
  }

  // Compute Information Matrix V(\hat{\beta}).
  MatrixXd info_matrix;
  if (!ComputeInformationMatrix(exp_logistic_eq, indep_vars, &info_matrix)) {
    cout << "ERROR: Unable to compute the Information Matrix of beta:\n"
         << beta_hat << "\nwith exp_logistic_eq:\n" << exp_logistic_eq(0)
         << "\nAborting.\n";
    return false;
  }

  // Compute new \beta, \beta^T * X_i, exp(\beta^T * X_i), and Log-Likelihood.
  VectorXd new_beta_hat, new_logistic_eq, new_exp_logistic_eq;
  double new_log_likelihood;
  if (!ComputeNewBetaHat(
          beta_hat, log_likelihood, score_function, info_matrix, indep_vars,
          dep_var, &new_beta_hat, &new_logistic_eq, &new_exp_logistic_eq,
          &new_log_likelihood)) {
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
        convergence_criteria, new_beta_hat,
        new_logistic_eq, new_exp_logistic_eq, new_log_likelihood,
        indep_vars, dep_var, regression_coefficients, iterations);
  } else {
    *regression_coefficients = new_beta_hat;
    return true;
  }  
}

bool LogisticRegression::ComputeRegressionCoefficients(
    const ConvergenceCriteria& convergence_criteria,
    const MatrixXd& indep_vars,
    const VectorXd& dep_var,
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
  double log_likelihood =
      static_cast<double>(n) * log(2.0);
  VectorXd initial_logistic_eq, initial_exp_logistic_eq;
  initial_logistic_eq.resize(n);
  initial_exp_logistic_eq.resize(n);
  for (int i = 0; i < n; ++i) {
    initial_logistic_eq(i) = 0.0;
    initial_exp_logistic_eq(i) = 1.0;
  }
  if (!RunNewtonRaphson(
          convergence_criteria, beta_hat, initial_logistic_eq,
          initial_exp_logistic_eq, log_likelihood, indep_vars, dep_var,
          regression_coefficients,
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

  // Compute final Log Likelihood of \hat{\beta}.
  double final_log_likelihood;
  if (!ComputeLogLikelihood(logistic_eq, exp_logistic_eq, dep_var,
                            &final_log_likelihood)) {
    cout << "ERROR: Unable to compute log-likelihood for beta:\n"
         << *regression_coefficients << "\nwith exp_logistic_eq:\n"
         << exp_logistic_eq(0) << "\nAborting.\n";
    return false;
  }

  // Compute final Score function U(\hat{\beta}).
  VectorXd score_function;
  score_function.resize(p);
  if (!ComputeScoreFunction(
          exp_logistic_eq, indep_vars, dep_var, &score_function)) {
    cout << "ERROR: Unable to compute final score function of beta:\n"
         << *regression_coefficients << "\nwith exp_logistic_eq:\n"
         << exp_logistic_eq(0) << "\n. Aborting.\n";
    return false;
  }

  // Compute final Information Matrix V(\hat{\beta}).
  MatrixXd info_matrix;
  if (!ComputeInformationMatrix(exp_logistic_eq, indep_vars, &info_matrix)) {
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

bool LogisticRegression::RunLogisticRegression(
    const string& output_filename,
    const ModelAndDataParams& params,
    SummaryStatistics* stats) {
  // Sanity check input.
  if (output_filename.empty() && stats == nullptr) return false;
  if (params.model_type_ != ModelType::MODEL_TYPE_LOGISTIC) return false;
  const MatrixXd& indep_vars = params.linear_term_values_;
  int n, p;
  if (!GetDimensions(indep_vars, &n, &p)) {
    cout << "ERROR: Unable to extract n, p from indep_vars. Aborting.\n";
    return false;
  }
  const vector<bool>& dep_vars = params.dep_vars_.dep_vars_logistic_;
  if (dep_vars.size() != n) {
    cout << "ERROR: Invalid variables. Aborting.\n";
    return false;
  }

  // Put dep vars in a format that makes computations easier (VectorXd).
  VectorXd dep_var;
  dep_var.resize(n);
  for (int i = 0; i < n; ++i) {
    dep_var(i) = dep_vars[i] ? 1.0 : 0.0;
  }


  // The number of independent variables must be less than the number
  // of sample rows (data values), or the system will be under-determined.
  if (p >= n) {
    cout << "ERROR: Invalid input file: Number of Cases (" << n
         << ") must exceed the number of independent variables ("
         << p << ").\n";
    return false;
  }

  // Perform matrix computations to find Regression Coefficients \hat{\Beta}.
  MatrixXd info_matrix_inverse;
  VectorXd regression_coefficients;
  int iterations = 0;
  int* iteration_ptr =
      stats == nullptr ? &iterations : &stats->num_iterations_;
  if (!ComputeRegressionCoefficients(
          params.convergence_criteria_, indep_vars, dep_var, iteration_ptr,
          &info_matrix_inverse, &regression_coefficients)) {
    return false;
  }

  // Unstandardize final values, if appropriate.
  VectorXd* unstandardized_beta = &regression_coefficients;
  MatrixXd* unstandardized_covariance = &info_matrix_inverse;
  MatrixXd unstandardized_covariance_matrix;
  VectorXd unstandardized_beta_values;
  bool unstandardization_required = false;
  if (!LinearRegression::UnstandardizeResults(
          params.standardize_vars_, params.linear_terms_mean_and_std_dev_,
          regression_coefficients, info_matrix_inverse,
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
    cout << "ERROR: Invalid Values computed:\n " << error_msg << endl;
    return false;
  }

  return true;
}

bool LogisticRegression::PrintValues(
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
  out_file << "Logistic Regression completed in " << num_iterations
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

bool LogisticRegression::ComputeFinalValuesAndPrint(
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
    out_file << "Logistic Regression completed in " << num_iterations
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
