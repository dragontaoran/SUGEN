#include "quantile_fns.h"

#include "eq_solver.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>

namespace math_utils {

bool StandardNormalICDF(const double& percentile, double* icdf) {
  // Sanity-Check input.
  if (icdf == nullptr) return false;
  if (percentile < 0.0 || percentile > 1.0) return false;

  // Construct the Expression that represents the equation to solve:
  //   0 = percentile - Phi(x),
  // where Phi(x) is the Standard Normal CDF.
  const string eq_str = Itoa(percentile) + " - Phi(x)";

  // Parse as Expression.
  Expression exp;
  if (!ParseExpression(eq_str, &exp)) {
    cout << "ERROR: Unable to form equation from equation '"
         << eq_str << "' while attempting to compute ICDF for Standard Normal"
         << endl;
    return false;
  }

  // Attempt to solve equation for x.
  RootFinder solution;
  solution.guess_ = 0.0;
  solution.max_iterations_ = 1000;
  solution.relative_distance_from_prev_guess_ = 0.00000001; // 10^-8.
  solution.use_absolute_distance_to_zero_ = false;
  solution.use_absolute_distance_from_prev_guess_ = false;
  solution.use_relative_distance_from_prev_guess_ = true;
  if (!SolveUsingMidpoint(exp, nullptr, &solution)) {
    cout << "Unable to find Standard Normal ICDF; process aborted with Error:"
         << endl << solution.error_msg_ << endl;
    return false;
  }

  *icdf = solution.guess_;
  return true;
}

bool NormalICDF(
    const double& percentile, const double& mean, const double& variance,
    double* icdf) {
  // Sanity-Check input.
  if (icdf == nullptr) return false;
  if (percentile < 0.0 || percentile > 1.0 || variance < 0.0) return false;

  double std_normal_icdf;
  if (!StandardNormalICDF(percentile, &std_normal_icdf)) return false;
  *icdf = sqrt(variance) * std_normal_icdf + mean;
  return true;
}

bool ChiSquaredICDF(
    const double& percentile, const double& degrees_of_freedom,
    double* icdf) {
  // Sanity-Check input.
  if (icdf == nullptr) return false;
  if (percentile < 0.0 || percentile > 1.0) return false;

  // Construct the Expression that represents the equation to solve:
  //   0 = percentile - F_X(x),
  // where F_X(x) = F_X(x | degrees_of_freedom) is the CDF for the Chi-Squared
  // distribution with indicated degrees_of_freedom:
  //   F_X(x | df) = P(df / 2, x / 2),
  // where P(z, x) is the Regularized Incomplete Gamma Function:
  //   P(z, x) = \gamma(0.5 * df, 0.5 * x) / Gamma(0.5 * df)
  const string eq_str =
      "RegIncGamma(0.5 * " + Itoa(degrees_of_freedom) + ", 0.5 * x) - " +
      Itoa(percentile);

  // Parse as Expression.
  Expression exp;
  if (!ParseExpression(eq_str, &exp)) {
    cout << "ERROR: Unable to form equation from equation '"
         << eq_str << "' while attempting to compute ICDF for Chi-Squared"
         << endl;
    return false;
  }

  // Attempt to solve equation for x.
  RootFinder solution;
  solution.guess_ = 0.0;
  solution.max_iterations_ = 1000;
  solution.relative_distance_from_prev_guess_ = 0.00000001; // 10^-8.
  solution.use_absolute_distance_to_zero_ = false;
  solution.use_absolute_distance_from_prev_guess_ = false;
  solution.use_relative_distance_from_prev_guess_ = true;
  if (!SolveUsingMidpoint(exp, nullptr, &solution)) {
    cout << "Unable to find Chi-Squared ICDF; process aborted with Error:"
         << endl << solution.error_msg_ << endl;
    return false;
  }

  *icdf = solution.guess_;
  return true;
}

}  // namespace math_utils
