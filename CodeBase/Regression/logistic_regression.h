// Date: March 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description: Tools for running logistic regression.

#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "FileReaderUtils/read_file_structures.h"
#include "MathUtils/number_comparison.h"
#include "MathUtils/statistics_utils.h"

#include <cstdlib>
#include <Eigen/Dense>
#include <iostream>
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

const int MAX_ITERATIONS = 20;
const int MAX_HALVING_ATTEMPTS = 10;

class LogisticRegression {
 public:
  // Prints all statistics to a file with the given name.
  static bool PrintValues(
      const int num_iterations,
      const vector<string>& titles, const vector<double>& estimates,
      const vector<double>& variances, const vector<double>& p_values,
      const vector<double>& standard_estimates_of_error,
      const vector<double>& t_statistics, const string& outfile);

  // Given a linear model described by the values in dep_var and indep_vars
  // (and whose linear terms are the contents of 'legend'), performs linear
  // regression to compute an Estimate of the constants factors for each
  // linear term, as well as the Variance, SEE, t-statistic, and p-value for
  // each of these Estimates. Prints these values to 'output_filename'.
  static bool RunLogisticRegression(
      const string& output_filename,
      const ModelAndDataParams& params,
      SummaryStatistics* stats);
  // Same as above, but doesn't print output to file (just populates stats).
  static bool RunLogisticRegression(
      const ModelAndDataParams& params,
      SummaryStatistics* stats) {
    return RunLogisticRegression("", params, stats);
  }
  // Same as above, but only prints to file (no stats to populate).
  static bool RunLogisticRegression(
      const string& output_filename,
      const ModelAndDataParams& params) {
    return RunLogisticRegression(output_filename, params, nullptr);
  }

 private:
  // Reads the input dimensions (number samples, number of terms on RHS of
  // logistic equation).
  static bool GetDimensions(const MatrixXd& indep_vars,
                            int* n, int* p) {
    if (indep_vars.size() == 0) return false;
    *p = indep_vars.cols();
    *n = indep_vars.rows();
    return true;
  }

  // The term exp(\beta^T * X_i) appears in many variables. We compute it once
  // here, storing the result in exp_logistic_eq.
  static bool ComputeExponentialOfLogisticEquation(
      const VectorXd& beta_hat,
      const MatrixXd& indep_vars,
      VectorXd* logistic_eq,
      VectorXd* exp_logistic_eq);

  // Compute the log-likelihood l(\beta).
  static bool ComputeLogLikelihood(
      const VectorXd& logistic_eq,
      const VectorXd& exp_logistic_eq,
      const VectorXd& dep_var,
      double* log_likelihood);

  // Computes the Score Function U(\beta).
  static bool ComputeScoreFunction(
      const VectorXd& exp_logistic_eq,
      const MatrixXd& indep_vars,
      const VectorXd& dep_var,
      VectorXd* score_function);

  // Computes the Information Matrix V(\beta).
  static bool ComputeInformationMatrix(
      const VectorXd& exp_logistic_eq,
      const MatrixXd& indep_vars,
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
      const VectorXd& score_function,
      const MatrixXd& info_matrix,
      const MatrixXd& indep_vars,
      const VectorXd& dep_var,
      VectorXd* new_beta_hat,
      VectorXd* new_logistic_eq,
      VectorXd* new_exp_logistic_eq,
      double* new_log_likelihood);

  // Run the NewtonRaphson Method.
  static bool RunNewtonRaphson(
      const ConvergenceCriteria& convergence_criteria,
      const VectorXd& beta_hat,
      const VectorXd& logistic_eq,
      const VectorXd& exp_logistic_eq,
      const double& log_likelihood,
      const MatrixXd& indep_vars,
      const VectorXd& dep_var,
      VectorXd* regression_coefficients, int* iterations);

  // Given input indep_vars and dep_var, computes the Vector of Regression
  // Coefficients (Beta Hat) using the standard method of Maximum Likelihood
  // Estimation (MLE), with stopping criterion:
  //   |log(L(B^new)) - log(L(B^old))| / log(L(B^old)) < 10^{-6}
  // This function uses Armadillo (a C++ wrapper for LAPACK and BLAS) to do
  // matrix computations.
  static bool ComputeRegressionCoefficients(
      const ConvergenceCriteria& convergence_criteria,
      const MatrixXd& indep_vars,
      const VectorXd& dep_var,
      int* num_iterations,
      MatrixXd* info_matrix_inverse,
      VectorXd* regression_coefficients);

  // Prints Variable titles together with the estimated regression coefficients
  // and covariance; output is print to file indicated by 'outfile', and/or
  // to the provided SummaryStatistics.
  static bool ComputeFinalValuesAndPrint(
      const int num_iterations,
      const string& outfile,
      const vector<string>& titles,
      const MatrixXd& info_matrix_inverse,
      const VectorXd& regression_coefficients,
      SummaryStatistics* stats, string* error_msg);
};

}  // namespace regression
#endif
