#include "linear_regression.h"

#include "FileReaderUtils/read_table_with_header.h"
#include "MathUtils/gamma_fns.h"

#include <cmath>
#include <cstdlib>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>      // std::setprecision
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using Eigen::FullPivLU;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using file_reader_utils::ReadTableWithHeader;
using namespace math_utils;
using namespace std;

namespace regression {

bool LinearRegression::ComputeRegressionCoefficients(
    const MatrixXd& indep_vars,
    const VectorXd& dep_var,
    MatrixXd* inverse_of_indep_vars,
    VectorXd* regression_coefficients) {
  // The following is safe (no SegFault): Checked indep_vars was non-empty
  // before calling this function.
  const int p = indep_vars.cols();
  // The following matrix will represent: \sum (X_i * X_i^T)
  MatrixXd sum_of_cov_matrices(p, p);
  sum_of_cov_matrices = indep_vars.transpose() * indep_vars;
  // The following (column) vector will represent: \sum (Y_i * X_i)
  VectorXd scaled_indep_vars;
  scaled_indep_vars.resize(p);
  scaled_indep_vars = indep_vars.transpose() * dep_var;

  // Check that sum_of_cov_matrices is invertible.
  FullPivLU<MatrixXd> lu = sum_of_cov_matrices.fullPivLu();
  if (!lu.isInvertible()) {
    return false;
  }
  *inverse_of_indep_vars = sum_of_cov_matrices.inverse();

  // Compute regression coefficients.
  *regression_coefficients = *inverse_of_indep_vars * scaled_indep_vars;
  return true;
}

bool LinearRegression::UnstandardizeResults(
    const VariableNormalization& std_type,
    const vector<tuple<bool, double, double>>& coordinate_mean_and_std_dev,
    const VectorXd& regression_coefficients,
    const MatrixXd& inverse_of_indep_vars,
    bool* unstandardization_required,
    VectorXd* unstandardized_beta, MatrixXd* unstandardized_covariance) {
  if (unstandardized_beta == nullptr || unstandardized_covariance == nullptr) {
    return false;
  }

  // Return if nothing to do.
  if (std_type == VAR_NORM_NONE || coordinate_mean_and_std_dev.empty()) {
    if (unstandardization_required != nullptr) {
      *unstandardization_required = false;
    }
    return true;
  }
  if (std_type == VAR_NORM_STD_NON_BINARY ||
      std_type == VAR_NORM_STD_W_N_MINUS_ONE_NON_BINARY) {
    bool at_least_one_non_binary_var = false;
    for (const tuple<bool, double, double>& stats_itr :
         coordinate_mean_and_std_dev) {
      if (get<0>(stats_itr)) {
        at_least_one_non_binary_var = true;
        break;
      }
    }
    if (!at_least_one_non_binary_var) {
      if (unstandardization_required != nullptr) {
        *unstandardization_required = false;
      }
      return true;
    }
  }

  if (unstandardization_required != nullptr) *unstandardization_required = true;

  // Construct Translation Matrix A, satisfying:
  //   \beta_std = A * \beta_orig
  // Note: A has form:
  //   - Except for top row, the rest of A is a diagonal matrix, with
  //     the i^th column's standard deviation on the diagonal of row i
  //   - For the top row of A, the diagonal element is a '1', while the
  //     remaining columns have value mean_i / std_dev_i
  const int p = regression_coefficients.size();
  MatrixXd A;
  A.resize(p, p);
  A.setZero();
  // Compute \beta * \mu, where \mu is the vector of column means.
  for (int i = 0; i < p; ++i) {
    const tuple<bool, double, double>& stats_itr = coordinate_mean_and_std_dev[i];
    if (i == 0) {
      A(i, i) = 1.0;
      continue;
    }
    if (!get<0>(stats_itr)) {
      A(i, i) = 1.0;
    } else {
      A(i, i) = get<2>(stats_itr);
      A(0, i) = get<1>(stats_itr);
    }
  }

  // Check that A is invertible.
  FullPivLU<MatrixXd> lu = A.fullPivLu();
  if (!lu.isInvertible()) {
    cout << "ERROR: Unable to unstandardize results: non-invertible matrix A:\n"
         << A << endl;
    return false;
  }

  const MatrixXd A_inverse = A.inverse();
  *unstandardized_beta = A_inverse * regression_coefficients;
  *unstandardized_covariance =
      A_inverse * inverse_of_indep_vars * A_inverse.transpose();

  return true;
}

double LinearRegression::ComputeVariance(
    const MatrixXd& indep_vars,
    const VectorXd& dep_var,
    const VectorXd& regression_coefficients) {
  VectorXd var_term(dep_var.size());
  var_term = dep_var - (indep_vars * regression_coefficients);
  MatrixXd variance = var_term.transpose() * var_term;
  return (variance(0,0) / (dep_var.size() - regression_coefficients.size()));
}

bool LinearRegression::PrintValues(
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

bool LinearRegression::ComputeFinalValuesAndPrint(
    const string& outfile,
    const vector<string>& titles,
    const double& variance,
    const MatrixXd& inverse_of_indep_vars,
    const VectorXd& regression_coefficients,
    SummaryStatistics* stats, string* error_msg) {
  if (outfile.empty() && stats == nullptr) return false;

  ofstream out_file;
  if (!outfile.empty()) {
    out_file.open(outfile.c_str());
    // Write title line.
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
    double est_var = variance * inverse_of_indep_vars(i, i);
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

bool LinearRegression::RunLinearRegression(
    const string& output_filename, const ModelAndDataParams& params,
    SummaryStatistics* stats) {
  const vector<double>& dep_var = params.dep_vars_.dep_vars_linear_;
  const MatrixXd& indep_vars = params.linear_term_values_;
  const vector<string>& legend = params.legend_;

  // Sanity check input.
  const int n = dep_var.size();
  if (params.model_type_ != ModelType::MODEL_TYPE_LINEAR ||
      n == 0 || indep_vars.rows() != n) {
    cout << "ERROR: Invalid input data. dep_var.size(): "
         << dep_var.size() << ", indep_vars.size(): "
         << indep_vars.size() << ". Aborting.\n";
    return false;
  }

  // Put dep_var in a more convenient format.
  VectorXd dep_vars;
  dep_vars.resize(n);
  for (int i = 0; i < n; ++i) {
    dep_vars(i) = dep_var[i];
  }

  // The number of independent variables must be less than the number
  // of sample rows (data values), or the system will be under-determined.
  if (indep_vars.cols() >= indep_vars.rows()) {
    cout << "ERROR: Invalid input file: Number of Cases (" << indep_vars.rows()
         << ") must exceed the number of independent variables ("
         << indep_vars.cols() << ").\n"
         << ReadTableWithHeader::PrintInputFormat();
    return false;
  }

  // Perform matrix computations to find Regression Coefficients \hat{\Beta}.
  MatrixXd inverse_of_indep_vars;
  VectorXd regression_coefficients;
  if (!ComputeRegressionCoefficients(
          indep_vars, dep_vars,
          &inverse_of_indep_vars, &regression_coefficients)) {
    cout << "ERROR: Ordinary Least Squares cannot be used for the given input "
         << "values: X * X^T is not invertible. Please check input values "
         << "and try again.\n";
    return false;
  }

  // Perform matrix computations to find (estimated) variance (\hat{\sigma}^2).
  double variance =
      ComputeVariance(indep_vars, dep_vars, regression_coefficients);

  // Unstandardize final values, if appropriate.
  VectorXd* unstandardized_beta = &regression_coefficients;
  MatrixXd* unstandardized_covariance = &inverse_of_indep_vars;
  MatrixXd unstandardized_covariance_matrix;
  VectorXd unstandardized_beta_values;
  bool unstandardization_required = false;
  if (!UnstandardizeResults(
          params.standardize_vars_, params.linear_terms_mean_and_std_dev_,
          regression_coefficients, inverse_of_indep_vars,
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
          output_filename, params.legend_,
          variance, *unstandardized_covariance, *unstandardized_beta,
          stats, &error_msg)) {
    cout << "ERROR: Invalid Values computed:\n " << error_msg;
    return false;
  }

  return true;
}

}  // namespace regression
