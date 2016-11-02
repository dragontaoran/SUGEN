// Date: Mar 2016
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Discussion:
//   "Quantile" functions, also known as the Inverse Cumulative Distribution
//   Function (ICDF), is the value at which the probability of a random
//   variable being less than or equal to this value is equal to the given
//   probability. In particular, suppose a R.V. X has CDF F_X(x). Then
//     ICDF(y) = x, where x is the unique value such that F_X(x) = y; i.e.
//     ICDF(y) = (F_X)^{-1}(y)
//   For a given percentile y, we compute ICDF(y) by solving the following
//   equation for x:
//     0 = y - F_X(x)
//   We use eq_solver.cpp, and use the Midpoint Method to solve the above equation.

#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#ifndef QUANTILE_FNS_H
#define QUANTILE_FNS_H

namespace math_utils {

// A.k.a. Probit function.
extern bool StandardNormalICDF(const double& percentile, double* icdf);
extern bool NormalICDF(
    const double& percentile, const double& mean, const double& variance,
    double* icdf);

// Populates icdf with the ICDF (Inverse Cumulative Distribution Function) for
// Chi-Squared R.V.'s. Returns true if computation was successful, false O.W.
extern bool ChiSquaredICDF(
    const double& percentile, const double& degrees_of_freedom,
    double* icdf);

// TODO(PHB): Implement ICDF's for the other supported distributions, e.g.
// Normal, Binomial, Uniform, etc.

}  // namespace math_utils

#endif
