//  Date: 23 February 2010
//  Author: John Burkardt
//  Licensing: This code is distributed under the GNU LGPL license. 
//
//  Description:
//    Computes the Abscissas and Weights for the Gauss-Laguerre
//    quadrature formula:
//      \int_a^\inf g(x) * |x-a|^\alpha * exp(-b(x-a)) dx
//
//  Input:
//    - (int) order:    Number of points for the quadrature
//    - (double) alpha: Exponent for x term (must be greater than -1.0)
//    - (double) a:     Left endpoint of the integral; typically a = 0
//    - (double) b:     Scale factor for the exponent (typically b = 1)
//  Output:
//    - vector<GaussianQuadratureTuple> of the {(weight_i, abscissa_i)}

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>

#ifndef GAUSSIAN_QUADRATURE_H
#define GAUSSIAN_QUADRATURE_H

using namespace std;

namespace math_utils {

extern const int TIMESTAMP_TIME_SIZE;

// The quadrature to perform:
//   1: Legendre,             (a,b)              1.0
//   2: Chebyshev Type 1,     (a,b)              ((b-x)*(x-a))^-0.5)
//   3: Gegenbauer,           (a,b)              ((b-x)*(x-a))^alpha
//   4: Jacobi,               (a,b)              (b-x)^alpha*(x-a)^beta
//   5: Generalized Laguerre, (a,\infty)         (x-a)^alpha*exp(-b*(x-a))
//   6: Generalized Hermite,  (-\infty,\infty)   |x-a|^alpha*exp(-b*(x-a)^2)
//   7: Exponential,          (a,b)              |x-(a+b)/2.0|^alpha
//   8: Rational,             (a,\infty)         (x-a)^alpha*(x+b)^beta
//   9: Chebyshev Type 2,     (a,b)              ((b-x)*(x-a))^(+0.5)
// where:
//   - alpha: Exponent, if relevant (See different Quadratures above)
//   - beta:  Second exponent, if relevant (See different Quadratures above)
//   - a:     Left-endpoint of integral (typically a = 0)
//   - b:     Scale factor for the exponent (typically b = 1), or right-endpoint
//            of integral, as relevant (See different Quadratures above)
enum QuadratureType {
  QT_UNKNOWN,
  QT_LEGENDRE,
  QT_CHEBYSHEV_ONE,
  QT_GEGENBAUER,
  QT_JACOBI,
  QT_GENERALIZED_LAGUERRE,
  QT_GENERALIZED_HERMITE,
  QT_EXPONENTIAL,
  QT_RATIONAL,
  QT_CHEBYSHEV_TWO,
};

struct GaussianQuadratureTuple {
  double weight_;
  double abscissa_;  // A.K.A. 'points'.

  GaussianQuadratureTuple() {
    weight_ = 0.0;
    abscissa_ = -1.0;
  }
};

// Prints current YMDHMS (e.g. 31 May 2001 09:45:54 AM) date as a time stamp.
extern void Timestamp();

// Returns the "R8 roundoff unit": A number X which is a power of 2 and with the
// property that, to the precision of the computer's arithmetic,
//   1 < 1 + X but 1 = 1 + (X / 2)
extern double r8_epsilon();
// Returns -1.0 if the input value is negative, otherwise (including for x = 0.0)
// returns 1.0.
extern double r8_sign(const double& x);

// Performs Gaussian-Laguerre Quadrature on interval (a, infty) of integrand:
//   (x-a)^alpha*exp(-b*(x-a))
extern bool ComputeGaussLaguerreQuadrature(
    const int order,
    const double& alpha,
    const double& a, const double& b,
    vector<GaussianQuadratureTuple>* output);

// Performs Gaussian-Laguerre Quadrature on interval (a, infty) of integrand:
//   (x-a)^alpha*exp(-b*(x-a))
extern bool ComputeGaussHermiteQuadrature(
    const int order,
    const double& alpha,
    const double& a, const double& b,
    vector<GaussianQuadratureTuple>* output);

// Computes Gaussian Quadrature for type 'quadrature_type'.
// Input 'order' specifies the number of quadrature points.
extern bool ComputeGaussianQuadrature(
    const int order, const QuadratureType quadrature_type,
    const double& alpha, const double& beta, const double& a, const double& b,
    vector<GaussianQuadratureTuple>* output);

// Same as above, but uses default values for a (0.0) and b (1.0).
extern bool ComputeDefaultGaussianQuadrature(
    const int order, const QuadratureType quadrature_type,
    const double& alpha, const double& beta,
    vector<GaussianQuadratureTuple>* output);

// Checks parameters alpha and beta for the classical weight functions.
// 'highest_moment' specifies the order of the highest moment to be calculated,
// and is only used if quadrature_type == 8.
extern bool CheckParameters(
    const QuadratureType quadrature_type, const int highest_moment,
    const double& alpha, const double& beta);

// Computes the Jacobi Matrix for the specified Quadrature. More specifically,
// this routine computes the diagonal a_j and sub-diagonal b_j elements of the
// order-m ('matrix_order) tridiagonal symmetric Jacobi matrix associated with
// the polynomials orthogonal with respect to the weight function specified by
// 'quadrature_type'.
// For weight functions 1-7, 'matrix_order' elements are defined in b_j even
// though only (matrix_order - 1) are needed.  For weight function 8,
// b_j(matrix_order) is set to zero.
// Also, zeroth_moment is populated with the 0-th moment of the weight function.
extern bool ComputeJacobiMatrix(
    const QuadratureType quadrature_type, const int matrix_order,
    const double& alpha, const double& beta,
    vector<double>* jacobi_diagonal, vector<double>* jacobi_sub_diagonal,
    double* zeroth_moment);

// This routine computes all the knots (abscissas) and weights of a Gauss
// Quadrature formula with simple knots from the Jacobi matrix and the zero-th
// moment of the weight function, using the Golub-Welsch technique.
// 'order' specifies the number of abscissas, 'zeroth_moment' is the zeroth-
// moment of the weight function, and 'jacobi_[sub_]diagonal' are the relevant
// parts of the Jacobi matrix.
extern bool ComputeAbscissaAndWeights(
    const int order,
    const double& zeroth_moment,
    const vector<double>& jacobi_diagonal,
    const vector<double>& jacobi_sub_diagnoal,
    vector<GaussianQuadratureTuple>* output);

// Diagonalizes a symmetric tridiagonal (entries on main-diagonal and the two
// subdiagonals above and below it, which are identical) matrix. More
// specifically, this routine is a slightly modified version of the EISPACK
// routine to perform the implicit QL algorithm on a symmetric tridiagonal
// matrix. It has been modified to produce the product Q' * Z, where Z is an
// input vector and Q is the orthogonal matrix diagonalizing the input matrix.
// The changes consist (essentialy) of applying the orthogonal transformations
// directly to Z as they are generated.
// PHB NOTE: the main-diagonal of the input matrix is stored in output->abscissa_.
extern bool DiagonalizeTriDiagonalSymmetricMatrix(
    const vector<double>& sub_diagonal,
    vector<GaussianQuadratureTuple>* output);

// Scales a quadrature formula to a nonstandard interval. Note that
// The arrays WTS and SWTS may coincide, and the arrays T and ST may coincide.
// 'multiplicity' is the multiplicity of the knots (abscissa); 'weight_index'
// is used to index the vector of weights unscaled_output.weights_ (for more details see the comments in CAWIQ); 
extern bool ScaleQuadratureForNonStandardInterval(
    const int order,
    const QuadratureType quadrature_type,
    const vector<int>& multiplicity,
    const vector<int>& weight_index,
    const double& alpha, const double& beta,
    const double& a, const double& b,
    const vector<GaussianQuadratureTuple>& unscaled_output,
    vector<GaussianQuadratureTuple>* output);

}  // namespace math_utils

#endif
