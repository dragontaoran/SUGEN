#include "gamma_fns.h"

#include <cmath>
#include <stdio.h>
#include <stdlib.h>

namespace math_utils {

// 'Floating-Point Min', near smallest representable floating point number.
#define FPMIN 1.00208e-292 
// Max number of iterations allowed.
#define ITMAX 500
// Allowed error in estimation ('EPSilon').
#define EPS 0.00000000000000022204460492503 

double Gamma(const double& z) {
  return exp(LogGamma(z));
}

double LogGamma(const double& z) {
  if (z <= 0.0) {
    printf("ERROR: unable to use approximation for Gamma Function on "
           "non-positive values (z = %0.04f. Aborting.\n", z);
    exit(1);
  }
  // The constant 'N' appearing in formula (*) in header file.
  const int N = 6;

  // The coefficients c_0, c_1, ..., c_6 appearing in formula (*) in header file.
  static const double coefficients[N + 1] = {
       1.000000000190015,
      76.18009172947146,
     -86.50532032941677,
      24.01409824083091,
      -1.231739572450155,
       0.1208650973866179e-2,
      -0.5395239384953e-5
  };

  // The constant 'k_0' appearing in formula (*) in header file.
  const int k_0 = 5;

  // Compute \tilde{z}.
  double tilde_z = (z + k_0 + 0.5);

  // Compute \hat{z}.
  double hat_z = coefficients[0];
  for (int i = 1; i <= 6; ++i) {
    hat_z += coefficients[i] / (z + i);
  }

  // Approximate \sqrt{2 * \pi}.
  double sqrt_two_pi = 2.5066282746310005;

  // Return the formula for ln(\Gamma(z)) (see formula (*) at top of header file).
  return -tilde_z + (z + 0.5) * log(tilde_z) + log(sqrt_two_pi * hat_z / z);
}

void GammaContinuedFraction(
    const double& z, const double& x,
    double* gamma_cont_frac, double* ln_gamma) {

  *ln_gamma = LogGamma(z);

  double a_n, delta;
  double b = x + 1.0 - z;
  double c = 1.0 / FPMIN;
  double d = 1.0 / b;
  double h = d;
  int i = 1;
  // Iterate through continued fractions until the difference between
  // terms is small enough (less than EPS), or until ITMAX iterations. 
  for (i = 1; i <= ITMAX; i++) {
    a_n = -i * (i - z);
    b += 2.0;
    d = a_n * d + b;
    if (fabs(d) < FPMIN) d = FPMIN;
    c = b + a_n / c;
    if (fabs(c) < FPMIN) c = FPMIN;
    d = 1.0 / d;
    delta = d * c;
    h *= delta;
    if (fabs(delta - 1.0) <= EPS) break;
  }
  if (i > ITMAX) {
    printf("z (%0.04f) too large; must increase ITMAX (%d) to get convergence "
           "using Continued Fractions.\n", z, ITMAX);
    exit(1);
  }

  *gamma_cont_frac = h * exp(-x + z * log(x) - (*ln_gamma));
}

void GammaSeries(
    const double& z, const double& x,
    double* gamma_series, double* ln_gamma) {
  *ln_gamma = LogGamma(z);

  // Sanity check input: x > 0.
  if (x <= 0.0) {
    if (x < 0.0) {
      printf("x less than 0 in routine GammaSeries\n");
      exit(1);
    }
    *gamma_series = 0.0;
    return;
  }

  double a_n = z;
  double sum = 1.0 / z;
  double delta = sum;
  for (int n = 0; n < ITMAX; n++) {
    a_n += 1.0;
    delta *= x / a_n;
    sum += delta;
    if (fabs(delta) < fabs(sum) * EPS) {
      *gamma_series = sum * exp(-x + z * log(x) - (*ln_gamma));
      return;
    }
  }
  printf("z (%0.04f) too large; must increase ITMAX (%d) to get convergence "
         "in Taylor Series..\n", z, ITMAX);
  exit(1);
}

double RegularizedIncompleteGammaFunction(const double& z, const double& x) {
  // Sanity check x >= 0 and z > 0.
  if (x < 0.0 || z <= 0.0) {
    printf("Invalid arguments in routine RegularizedIncompleteGammaFunction\n");
    exit(1);
  }

  // Use Taylor Series representation of Inc Gamma Fn if x < (z + 1.0).
  double ln_gamma;
  if (x < (z + 1.0)) {
    double gamma_series;
    GammaSeries(z, x, &gamma_series, &ln_gamma);
    return gamma_series;
  // Otherwise, use Continued Fraction approximation.
  } else {
    double gamma_cont_frac;
    GammaContinuedFraction(z, x, &gamma_cont_frac, &ln_gamma);
    return 1.0 - gamma_cont_frac;
  }
}

double RegularizedReverseIncompleteGammaFunction(const double& z, const double& x) {
  // Sanity check x >= 0 and z > 0.
  if (x < 0.0 || z <= 0.0) {
    printf("Invalid arguments in routine RegularizedReverseIncompleteGammaFunction\n");
    exit(1);
  }

  // Use Taylor Series representation of Inc Gamma Fn if x < (z + 1.0).
  double ln_gamma;
  if (x < (z + 1.0)) {
    double gamma_series;
    GammaSeries(z, x, &gamma_series, &ln_gamma);
    return 1.0 - gamma_series;
  // Otherwise, use Continued Fraction approximation.
  } else {
    double gamma_cont_frac;
    GammaContinuedFraction(z, x, &gamma_cont_frac, &ln_gamma);
    return gamma_cont_frac;
  }
}

double LowerIncompleteGammaFunction(const double& z, const double& x) {
  return RegularizedIncompleteGammaFunction(z, x) * Gamma(z);
}

double UpperIncompleteGammaFunction(const double& z, const double& x) {
  return RegularizedReverseIncompleteGammaFunction(z, x) * Gamma(z);
}

}  // namespace math_utils
