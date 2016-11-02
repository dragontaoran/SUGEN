// Date: Feb 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "FileReaderUtils/read_file_structures.h"
#include "MathUtils/statistics_utils.h"

#include <cstdlib>
#include <Eigen/Dense>
#include <string>
#include <vector>

using namespace file_reader_utils;
using namespace math_utils;
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

namespace regression {

class LinearRegression {
 public:
  // Given input indep_vars and dep_var, computes the Vector of Regression
  // Coefficients (Beta Hat) via the standard method of using Ordinary Least
  // Squares to minimize the sum of squared residuals.
  // This function uses Armadillo (a C++ wrapper for LAPACK and BLAS) to do
  // matrix computations.
  static bool ComputeRegressionCoefficients(
      const MatrixXd& indep_vars,
      const VectorXd& dep_var,
      MatrixXd* inverse_of_indep_vars,
      VectorXd* regression_coefficients);

  // Performs unstandardization of Regression coefficients and Covariance Matrix,
  // if appropriate.
  static bool UnstandardizeResults(
      const VariableNormalization& std_type,
      const vector<tuple<bool, double, double>>& linear_terms_mean_and_std_dev,
      const VectorXd& regression_coefficients,
      const MatrixXd& inverse_of_indep_vars,
      bool* unstandardization_required,
      VectorXd* unstandardized_beta, MatrixXd* unstandardized_covariance);

  // Given input indep_vars, dep_var, and regression_coefficients, computes
  // the variance.
  static double ComputeVariance(
      const MatrixXd& indep_vars,
      const VectorXd& dep_var,
      const VectorXd& regression_coefficients);

  // Prints Variable titles together with the estimated regression coefficients
  // and covariance; output is print to file indicated by 'outfile'.
  static bool ComputeFinalValuesAndPrint(
      const string& outfile,
      const vector<string>& titles,
      const double& variance,
      const MatrixXd& inverse_of_indep_vars,
      const VectorXd& regression_coefficients,
      SummaryStatistics* stats, string* error_msg);

  // Given a linear model described by the values in dep_var and indep_vars
  // (and whose linear terms are the contents of 'legend'), performs linear
  // regression to compute an Estimate of the constants factors for each
  // linear term, as well as the Variance, SEE, t-statistic, and p-value for
  // each of these Estimates. If output_filename is non-empty, prints these
  // values to 'output_filename'; if stats is non-null, puts these values in it.
  //   3) Use standardize_vars_ and linear_terms_mean_and_std_dev_ fields of
  //      ModelAndDataParams to unstandardize results
  static bool RunLinearRegression(
      const string& output_filename, const ModelAndDataParams& params,
      SummaryStatistics* stats);
  // Same as above, with alternate API (prints to file only).
  static bool RunLinearRegression(
      const string& output_filename, const ModelAndDataParams& params) {
    return RunLinearRegression(output_filename, params, nullptr);
  }
  // Same as above, with alternate API (prints to stats only).
  static bool RunLinearRegression(
      const ModelAndDataParams& params, SummaryStatistics* stats) {
    return RunLinearRegression("", params, stats);
  }

  // Prints all statistics to a file with the given name.
  static bool PrintValues(
      const vector<string>& titles, const vector<double>& estimates,
      const vector<double>& variances, const vector<double>& p_values,
      const vector<double>& standard_estimates_of_error,
      const vector<double>& t_statistics, const string& outfile);

 private:
};

}  // namespace regression
#endif
