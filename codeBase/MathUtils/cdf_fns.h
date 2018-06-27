// Date: Mar 2016
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Discussion:
//   Cumulative Distribution Functions (CDF) for many of the commond distributions.

#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#ifndef CDF_FNS_H
#define CDF_FNS_H

namespace math_utils {

extern double StandardNormalCDF(const double& x);
extern bool NormalCDF(
    const double& x, const double& mean, const double& variance, double* cdf);

extern bool ChiSquaredCDF(
    const double& x, const double& df, double* cdf);

// TODO(PHB): Implement ICDF's for the other supported distributions, e.g.
// Gamma, Exponential, Poisson, Binomial, Uniform, etc.

}  // namespace math_utils

#endif
