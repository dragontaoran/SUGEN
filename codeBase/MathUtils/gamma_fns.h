// Date: Mar 2016
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Discussion:
// ============================== Gamma Function ===============================
// The Gamma Function \Gamma(z) is defined by:
//    \int_0^{\inf} t^{z - 1} * e^{-t} dt
// Notice that for real, integer values:
//    \Gamma(n) = (n - 1)!
// This file computes variants of the Gamma Function, using
// the approximation (valid for Re(z) > 0):
//    (*) \Gamma(z) \approx 
//             e^{- \tilde{z}} * \tilde{z}^{z + 0.5} * \sqrt{2 * \pi} * \hat{z} / z
// Where:
//    \tilde{z} := (z + k_0 + 0.5) for appropriate choice of constant k_0
//    \hat{z} := c_0 + c_1 / (z + 1) + \dots + c_N / (z + N) + \epsilon
//               for appropriate choice of constants N, c_i and error \epsilon
// The above formula (*) is from Lanczos, and has \epsilon < 2e-10 FOR ALL z
// With (*) in hand, we can estimate \Gamma(z) programmatically.
//
// Note that since \Gamma values tend to be quite large (overflow computer
// representation), and computations often involve quotients (so that the
// final value is representatable by computer), it is common to instead
// compute log(\Gamma), and only in the end (if at all) convert back.
// ============================ END: Gamma Function ============================
//
// ========================= Incomplete Gamma Function =========================
// We generalize the definition of Gamma function by taking a finite integral:
//   (1) (Lower) Incomplete Gamma Function:
//         \gamma(z, x) := \int_0^x       t^{z - 1} * e^{-t} dt
//   (2) (Upper) Incomplete Gamma Function:
//         \Gamma(z, x) := \int_x^\infty  t^{z - 1} * e^{-t} dt = 1 - \gamma(z, x) 
// It is immediate then that:
//   (3) \Gamma(z) = lim_{x -> \inf} \gamma(z, x)
//                 = lim_{x -> 0^+} \Gamma(z, x)
//                 = \gamma(z, x) + \Gamma(z, x)
// We define 'Regularized Incomplete Gamma Function', denoted P(z, x) as the
// ratio of the Lower Incomplete Gamma Function and the Gamma Function:
//   (4) P(z, x) := \gamma(z, x) / \Gamma(z)
// We also define 'Regularized Reverse Incomplete Gamma Function', denoted Q(z, x)
// as the ratio of the Upper Incomplete Gamma Function and the Gamma Function:
//   (5) Q(z, x) := \Gamma(z, x) / \Gamma(z) = 1 - P(z, x)
// Notice that domain of the Regularized Incomplete Gamma Fn is [0, \inf),
// and at the limits:
//   P(z, 0) = 0  and P(z, \inf) = 1
// Thus, Incomplete Gamma Function is monotone increasing function from 0 to 1,
// and it has the characteristic that it looks flat at the two extremes (close
// to zero on the left and close to one on the right), and raises quickly
// from 0 to 1 near the value 'x'. Q(z, x) has similar properties, in reverse
// (e.g. it is monotone DECREASING).
//
// There are two approximations that can be used to compute P(z, x); the first
// (which uses Continued Fractions) converges rapidly (and thus can be computed
// quickly) for x < z + 1, and the other (which uses the Series Representation
// for Inc. Gamma Fn) converges rapidly for x > z + 1.
// ====================== END: Incomplete Gamma Function =======================

#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#ifndef GAMMA_FNS_H
#define GAMMA_FNS_H

namespace math_utils {

// Returns \Gamma(z).
extern double Gamma(const double& z);

// For z > 0, returns: ln(\Gamma(z)).
// Using formula (*) above for Gamma, we have:
//   ln(\Gamma(z)) =
//      (z + 0.5) * ln(\tilde{z}) - \tilde{z} + ln(\sqrt{2 * \pi} * \hat{z} / z)
extern double LogGamma(const double& z);

// Returns the Regularized (lower) Incomplete Gamma Function P(z, x),
// as computed either via the Continued Fraction or the Taylor Series
// approximation for P(z, x), (depending on whether x is greater than (z + 1.0)).
extern double RegularizedIncompleteGammaFunction(const double& z, const double& x);
// Returns the Regularized (upper) Incomplete Gamma Function Q(z, x) = 1 - P(z, x)
// as computed either via the Continued Fraction or the Taylor Series
// approximation for P(z, x), (depending on whether x is greater than (z + 1.0)).
extern double RegularizedReverseIncompleteGammaFunction(const double& z, const double& x);
// Returns the (lower) Incomplete Gamma Function \gamma(z, x).
extern double LowerIncompleteGammaFunction(const double& z, const double& x);
// Returns the (upper) Incomplete Gamma Function \Gamma(z, x) = \Gamma(z) - \gamma(z, x).
extern double UpperIncompleteGammaFunction(const double& z, const double& x);

// Helper function for the functions above.
// Populates gamma_series with the Incomplete Gamma Function Q(z, x),
// as evaluated by its Taylor Series representation.
// In case this is an intermediate value (see overflow concerns in comments at top),
// for convenience this also populates ln_gamma with LogGamma(z).
extern void GammaSeries(
    const double& z, const double& x,
    double* gamma_series, double* ln_gamma);

// Helper function for the functions above.
// Populates gamma_cont_frac with the Incomplete Gamma Function Q(z, x),
// as evaluated by its continued fraction representation.
// In case this is an intermediate value (see overflow concerns in comments at top),
// for convenience this also populates ln_gamma with LogGamma(z).
extern void GammaContinuedFraction(
    const double& z, const double& x,
    double* gamma_cont_frac, double* ln_gamma);

}  // namespace math_utils

#endif
