#include "cdf_fns.h"

#include "MathUtils/gamma_fns.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

namespace math_utils {

double StandardNormalCDF(const double&x) {
  return 0.5 * (1.0 + erf(x / sqrt(2)));
}

bool NormalCDF(
    const double& x, const double& mean, const double& variance, double* cdf) {
  if (cdf == nullptr || variance < 0.0) return false;
  if (variance == 0.0) {
    // Delta function, height 1 centered at 'mean'. Integral evaluates to:
    // 0.0, if x < mean, or 1.0, if x >= mean.
    *cdf = x < mean ? 0.0 : 1.0;
  }

  *cdf = StandardNormalCDF((x - mean)/sqrt(variance));
  return true;
}

bool ChiSquaredCDF(
    const double& x, const double& df, double* cdf) {
  // Sanity-test input.
  if (x < 0.0 || df < 0.0 || cdf == nullptr) return false;

  // Note CDF for Chi-Squared distribution is:
  //   F_ChiSq(x | df) = P(df / 2, x / 2),
  // where P(z, x) is the Regularized Incomplete Gamma Function. Since we don't
  // have this Operator defined, we use the equality:
  //   P(df, x) = \gamma(0.5 * df, 0.5 * x) / Gamma(0.5 * df),
  // where Gamma is the Gamma function (Operator GAMMA_FN), and \gamma is
  // the (lower) incomplete gamma function (Operator INC_GAMMA_FN).
  
  *cdf = RegularizedIncompleteGammaFunction(0.5 * df, 0.5 * x);
  return true;
}

}  // namespace math_utils
