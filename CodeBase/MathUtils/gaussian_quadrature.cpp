//  Date: 23 February 2010
//  Author: John Burkardt
//  Licensing: This code is distributed under the GNU LGPL license. 
//  From: http://people.sc.fsu.edu/~jburkardt/cpp_src/gen_laguerre_rule/gen_laguerre_rule.html
//    and http://people.sc.fsu.edu/~jburkardt/cpp_src/gen_laguerre_rule/gen_laguerre_rule.cpp

#include "gaussian_quadrature.h"

#include "MathUtils/constants.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

namespace math_utils {

const int TIMESTAMP_TIME_SIZE = 40;

bool ComputeGaussLaguerreQuadrature(
    const int order,
    const double& alpha,
    const double& a, const double& b,
    vector<GaussianQuadratureTuple>* output) {
  const QuadratureType quadrature_type = QT_GENERALIZED_LAGUERRE;
  const double beta = 0.0;
  return ComputeGaussianQuadrature(
      order, quadrature_type, alpha, beta, a, b, output);
}

bool ComputeGaussHermiteQuadrature(
    const int order,
    const double& alpha,
    const double& a, const double& b,
    vector<GaussianQuadratureTuple>* output) {
  const QuadratureType quadrature_type = QT_GENERALIZED_HERMITE;
  const double beta = 0.0;
  return ComputeGaussianQuadrature(
      order, quadrature_type, alpha, beta, a, b, output);
}

bool ComputeDefaultGaussianQuadrature(
    const int order, const QuadratureType quadrature_type,
    const double& alpha, const double& beta,
    vector<GaussianQuadratureTuple>* output) {
  if (!CheckParameters(quadrature_type, 2 * order, alpha, beta)) return false;

  //  Get the Jacobi matrix and zero-th moment.
  vector<double> jacobi_diagonal(order);
  // Even though the sub-diagonal should have (order - 1) elements, the code
  // assumes it has 'order' elements; so I keep it to have size 'order' to
  // minimize the amount of changes I have to make.
  vector<double> jacobi_sub_diagonal(order);
  double zeroth_moment;
  if (!ComputeJacobiMatrix(
          quadrature_type, order, alpha, beta,
          &jacobi_diagonal, &jacobi_sub_diagonal, &zeroth_moment)) {
    return false;
  }

  //  Compute the knots and weights.
  return ComputeAbscissaAndWeights(
      order, zeroth_moment, jacobi_diagonal, jacobi_sub_diagonal, output);
}

bool ComputeGaussianQuadrature(
    const int order, const QuadratureType quadrature_type,
    const double& alpha, const double& beta,
    const double& a, const double& b, 
    vector<GaussianQuadratureTuple>* output) {
  // Sanity-Check input.
  if (output == nullptr) {
    cout << "ERROR in ComputeGaussLaguerreQuadrature: Null input." << endl;
    return false;
  }
  if (order <= 0) {
    cout << "ERROR in ComputeGaussLaguerreQuadrature: order ("
         << order << ") must be at least 1." << endl;
    return false;
  }

  // NOTE(PHB): Let calling function keep track of time if desired.
  //cout << "\nStarting ComputeGaussianQuadrature." << endl;
  //Timestamp();

  //  First compute the Gauss quadrature formula for default values of a and b.
  vector<GaussianQuadratureTuple> unscaled_output(order);
  if (!ComputeDefaultGaussianQuadrature(
          order, quadrature_type, alpha, beta, &unscaled_output)) {
    return false;
  }

  // Now scale the result for the indicated (a, b).
  vector<int> multiplicity(order);
  for (int i = 0; i < order; i++) {
    multiplicity[i] = 1;
  }
  vector<int> weight_index(order);
  for (int i = 0; i < order; i++) {
    weight_index[i] = i + 1;
  }
  output->resize(order);
  if (!ScaleQuadratureForNonStandardInterval(
          order, quadrature_type, multiplicity, weight_index,
          alpha, beta, a, b, unscaled_output, output)) {
    return false;
  }

  // NOTE(PHB): Let calling function keep track of time if desired.
  //cout << "\nFinished ComputeGaussianQuadrature." << endl;
  //Timestamp();
  return true;
}

bool ComputeJacobiMatrix(
    const QuadratureType quadrature_type, const int matrix_order,
    const double& alpha, const double& beta,
    vector<double>* jacobi_diagonal,
    vector<double>* jacobi_sub_diagonal,
    double* zeroth_moment) {
  if (!CheckParameters(quadrature_type, 2 * matrix_order - 1, alpha, beta)) {
    return false;
  }

  if (500.0 * r8_epsilon() < fabs(pow(tgamma(0.5), 2.0) - PI)) {
    cout << "ERROR in ComputeJacobiMatrix: Gamma function does not match "
         << "machine parameters.\n";
    return false;
  }

  double ab_i;
  double ab_j;

  // Compute Jacobi Matrix for Legendre Quadrature.
  if (quadrature_type == QT_LEGENDRE) {
    *zeroth_moment = 2.0;
    jacobi_diagonal->resize(matrix_order, 0.0);

    const double ab = 0.0;
    for (int i = 1; i <= matrix_order; i++) {
      ab_i = i + ab * (i % 2);
      ab_j = 2 * i + ab;
      (*jacobi_sub_diagonal)[i - 1] = sqrt(ab_i * ab_i / (ab_j * ab_j - 1.0));
    }

  // Compute Jacobi Matrix for Chebyshev (Type 1) Quadrature.
  } else if (quadrature_type == QT_CHEBYSHEV_ONE) {
    *zeroth_moment = PI;
    jacobi_diagonal->resize(matrix_order, 0.0);
    jacobi_sub_diagonal->resize(matrix_order, 0.5);
    (*jacobi_sub_diagonal)[0] = sqrt(0.5);
  
  // Compute Jacobi Matrix for Gegenbauer Quadrature.
  } else if (quadrature_type == QT_GEGENBAUER) {
    const double ab = alpha * 2.0;
    *zeroth_moment =
        pow(2.0, ab + 1.0) * pow(tgamma(alpha + 1.0), 2) / tgamma(ab + 2.0);
    jacobi_diagonal->resize(matrix_order, 0.0);

    (*jacobi_sub_diagonal)[0] = sqrt(1.0 / (2.0 * alpha + 3.0));
    for (int i = 2; i <= matrix_order; i++) {
      (*jacobi_sub_diagonal)[i - 1] =
          sqrt(i * (i + ab) / (4.0 * pow(i + alpha, 2) - 1.0));
    }

  // Compute Jacobi Matrix for Jacobi Quadrature.
  } else if (quadrature_type == QT_JACOBI) {
    const double ab = alpha + beta;
    ab_i = 2.0 + ab;
    *zeroth_moment = pow ( 2.0, ab + 1.0 ) * tgamma ( alpha + 1.0 ) 
      * tgamma ( beta + 1.0 ) / tgamma ( ab_i );
    (*jacobi_diagonal)[0] = (beta - alpha) / ab_i;
    (*jacobi_sub_diagonal)[0] =
        sqrt(4.0 * (1.0 + alpha) * (1.0 + beta) / ((ab_i + 1.0) * ab_i * ab_i));
    double a2b2 = beta * beta - alpha * alpha;

    for (int i = 2; i <= matrix_order; i++) {
      ab_i = 2.0 * i + ab;
      (*jacobi_diagonal)[i - 1] = a2b2 / ((ab_i - 2.0) * ab_i);
      ab_i = ab_i * ab_i;
      (*jacobi_sub_diagonal)[i - 1] =
          sqrt(4.0 * i * (i + alpha) * (i + beta) * (i + ab) /
               ((ab_i - 1.0) * ab_i));
    }

  // Compute Jacobi Matrix for Generalized Laguerre Quadrature.
  } else if (quadrature_type == QT_GENERALIZED_LAGUERRE) {
    *zeroth_moment = tgamma(alpha + 1.0);

    for (int i = 1; i <= matrix_order; i++) {
      (*jacobi_diagonal)[i - 1] = 2.0 * i - 1.0 + alpha;
      (*jacobi_sub_diagonal)[i - 1] = sqrt(i * (i + alpha));
    }

  // Compute Jacobi Matrix for Generalized Hermite Quadrature.
  } else if (quadrature_type == QT_GENERALIZED_HERMITE) {
    *zeroth_moment = tgamma((alpha + 1.0 ) / 2.0);
    jacobi_diagonal->resize(matrix_order, 0.0);

    for (int i = 1; i <= matrix_order; i++) {
      (*jacobi_sub_diagonal)[i - 1] = sqrt((i + alpha * (i % 2)) / 2.0);
    }

  // Compute Jacobi Matrix for Exponential Quadrature.
  } else if (quadrature_type == QT_EXPONENTIAL) {
    const double ab = alpha;
    *zeroth_moment = 2.0 / (ab + 1.0);
    jacobi_diagonal->resize(matrix_order, 0.0);

    for (int i = 1; i <= matrix_order; i++) {
      ab_i = i + ab * (i % 2);
      ab_j = 2 * i + ab;
      (*jacobi_sub_diagonal)[i - 1] = sqrt(ab_i * ab_i / (ab_j * ab_j - 1.0));
    }

  // Compute Jacobi Matrix for Rational Quadrature.
  } else if (quadrature_type == QT_RATIONAL) {
    const double ab = alpha + beta;
    *zeroth_moment =
        tgamma(alpha + 1.0) * tgamma (-1.0 * (ab + 1.0)) /
        tgamma(-1.0 * beta);
    double apone = alpha + 1.0;
    double aba = ab * apone;
    (*jacobi_diagonal)[0] = -1.0 * apone / (ab + 2.0);
    (*jacobi_sub_diagonal)[0] =
        -1.0 * (*jacobi_diagonal)[0] * (beta + 1.0) / (ab + 2.0) / (ab + 3.0);
    double abti;
    for (int i = 2; i <= matrix_order; i++) {
      abti = ab + 2.0 * i;
      (*jacobi_diagonal)[i - 1] = aba + 2.0 * (ab + i) * (i - 1);
      (*jacobi_diagonal)[i - 1] =
          -1.0 * (*jacobi_diagonal)[i - 1] / abti / (abti - 2.0);
    }

    for (int i = 2; i <= matrix_order - 1; i++) {
      abti = ab + 2.0 * i;
      (*jacobi_sub_diagonal)[i - 1] =
        i * (alpha + i) / (abti - 1.0) * (beta + i) /
        (abti * abti) * (ab + i) / (abti + 1.0);
    }
    (*jacobi_sub_diagonal)[matrix_order - 1] = 0.0;
    for (int i = 0; i < matrix_order; i++) {
      (*jacobi_sub_diagonal)[i] = sqrt((*jacobi_sub_diagonal)[i]);
    }
  
  } else {
    cout << "ERROR in ComputeJacobiMatrix: QuadratureType " << quadrature_type
         << " is not supported at this time." << endl;
    return false;
  }

  return true;
}

bool DiagonalizeTriDiagonalSymmetricMatrix(
    const vector<double>& sub_diagonal,
    vector<GaussianQuadratureTuple>* output) {
  // Sanity-check input.
  if (output == nullptr) {
    cout << "ERROR in DiagonalizeTriDiagonalSymmetricMatrix: Null input.";
    return false;
  }
  const int n = output->size();
  if (sub_diagonal.size() != n) {
    cout << "ERROR in DiagonalizeTriDiagonalSymmetricMatrix: Mismatching "
         << "dimensions: Main-diagonal: " << n << ", sub-diagonal: "
         << sub_diagonal.size() << endl;
    return false;
  }

  // Nothing to do for 1-dimensional matrix.
  if (n == 1) return true;

  // Even though sub-diagonal has only n - 1 terms, we size it to 'n',
  // to be consistent with original code (less change of making a mistake
  // in trying to toggle each index by 1), which used spots 1 through n - 1.
  vector<double> new_sub_diagonal = sub_diagonal;
  const int max_iterations = 30;
  int m;
  double c;
  double f;
  double g;
  double p;
  double r;
  double s;
  const double prec = r8_epsilon();

  new_sub_diagonal[n - 1] = 0.0;

  for (int l = 1; l <= n; l++) {
    int j = 0;
    for ( ; ; ) {
      for (m = l; m <= n; m++) {
        if (m == n) break;

        if (fabs(new_sub_diagonal[m - 1]) <=
            prec * (fabs((*output)[m - 1].abscissa_) +
                    fabs((*output)[m].abscissa_))) {
          break;
        }
      }
      p = (*output)[l - 1].abscissa_;
      if (m == l) break;
      if (max_iterations <= j) {
        cout << "ERROR in DiagonalizeTriDiagonalSymmetricMatrix: "
             << "Exceeded iteration limit (" << max_iterations << ")." << endl;
        return false;
      }
      j = j + 1;
      g = ((*output)[l].abscissa_ - p) / (2.0 * new_sub_diagonal[l - 1]);
      r =  sqrt(g * g + 1.0);
      g = (*output)[m - 1].abscissa_ - p + new_sub_diagonal[l - 1] /
          (g + fabs(r) * r8_sign(g));
      s = 1.0;
      c = 1.0;
      p = 0.0;
      const int mml = m - l;

      for (int ii = 1; ii <= mml; ii++) {
        const int i = m - ii;
        f = s * new_sub_diagonal[i - 1];
        const double b = c * new_sub_diagonal[i - 1];

        if (fabs (g) <= fabs (f)) {
          c = g / f;
          r =  sqrt(c * c + 1.0);
          new_sub_diagonal[i] = f * r;
          s = 1.0 / r;
          c = c * s;
        }
        else {
          s = f / g;
          r =  sqrt(s * s + 1.0);
          new_sub_diagonal[i] = g * r;
          c = 1.0 / r;
          s = s * c;
        }
        g = (*output)[i].abscissa_ - p;
        r = ((*output)[i - 1].abscissa_ - g) * s + 2.0 * c * b;
        p = s * r;
        (*output)[i].abscissa_ = g + p;
        g = c * r - b;
        f = (*output)[i].weight_;
        (*output)[i].weight_ = s * (*output)[i - 1].weight_ + c * f;
        (*output)[i - 1].weight_ = c * (*output)[i - 1].weight_ - s * f;
      }
      (*output)[l - 1].abscissa_ = (*output)[l - 1].abscissa_ - p;
      new_sub_diagonal[l - 1] = g;
      new_sub_diagonal[m - 1] = 0.0;
    }
  }

  //  Sort.
  for (int ii = 2; ii <= m; ii++) {
    const int i = ii - 1;
    int k = i;
    p = (*output)[i - 1].abscissa_;

    for (int j = ii; j <= n; j++) {
      if ((*output)[j - 1].abscissa_ < p) {
         k = j;
         p = (*output)[j - 1].abscissa_;
      }
    }

    if (k != i) {
      (*output)[k - 1].abscissa_ = (*output)[i - 1].abscissa_;
      (*output)[i - 1].abscissa_ = p;
      p = (*output)[i - 1].weight_;
      (*output)[i - 1].weight_ = (*output)[k - 1].weight_;
      (*output)[k - 1].weight_ = p;
    }
  }
  return true;
}

bool CheckParameters(
    const QuadratureType quadrature_type, const int highest_moment,
    const double& alpha, const double& beta) {
  // Check that highest_moment is at least zero.
  if (highest_moment < 1) {
    cout << "ERROR in CheckParameters: non-positive 'highest_moment': "
         << highest_moment;
    return false;
  }

  // Check that one of the accepted uses was specfied.
  if (quadrature_type != QT_LEGENDRE && quadrature_type != QT_CHEBYSHEV_ONE &&
      quadrature_type != QT_GEGENBAUER && quadrature_type != QT_JACOBI &&
      quadrature_type != QT_GENERALIZED_LAGUERRE &&
      quadrature_type != QT_GENERALIZED_HERMITE &&
      quadrature_type != QT_EXPONENTIAL && quadrature_type != QT_RATIONAL &&
      quadrature_type != QT_CHEBYSHEV_TWO) {
    cout << "ERROR in CheckParameters: Unexpected value for quadrature_type:"
         << quadrature_type << endl;
    return false;
  }

  //  Check alpha for Gegenbauer, Jacobi, Laguerre, Hermite, Exponential.
  if (quadrature_type == QT_GEGENBAUER || quadrature_type == QT_JACOBI ||
      quadrature_type == QT_GENERALIZED_LAGUERRE ||
      quadrature_type == QT_GENERALIZED_HERMITE ||
      // Not sure about Rational; original code is not clear.
      // quadrature_type == QT_RATIONAL ||
      quadrature_type == QT_EXPONENTIAL) {
    if (alpha <= -1.0) {
      cout << "ERROR in CheckParameters: alpha (" << alpha
           << ") must be greater than -1.0 for quadrature_type "
           << quadrature_type << endl;
      return false;
    }
  }

  //  Check beta for Jacobi.
  if (quadrature_type == QT_JACOBI && beta <= -1.0 ) {
    cout << "ERROR in CheckParameters: alpha (" << beta
         << ") must be greater than -1.0 for Jacobi Quadrature." << endl;
    return false;
  }

  //  Check alpha and beta for Rational.
  if (quadrature_type == QT_RATIONAL) {
    const double tmp = alpha + beta + highest_moment + 1.0;
    if (0.0 <= tmp || tmp <= beta) {
      cout << "ERROR in CheckParameters: alpha (" << alpha << "), beta ("
           << beta << "), and highest moment (" << highest_moment
           << ") must obey certain properties for Rational Quadrature, "
           << "but they fail to do so: Value '" << tmp
           << "' should be between 0.0 and " << beta << endl;
      return false;
    }
  }

  return true;
}

double r8_epsilon() {
  return 2.220446049250313E-016;
}

double r8_sign(const double& x) {
  if (x < 0.0) return -1.0;
  return 1.0;
}

bool ScaleQuadratureForNonStandardInterval(
    const int order,
    const QuadratureType quadrature_type,
    const vector<int>& multiplicity,
    const vector<int>& weight_index,
    const double& alpha, const double& beta,
    const double& a, const double& b,
    const vector<GaussianQuadratureTuple>& unscaled_output,
    vector<GaussianQuadratureTuple>* output) {
  if (!CheckParameters(quadrature_type, 1, alpha, beta)) return false;

  // Sanity check a, b are consistent for the given QuadratureType.
  const double epsilon = r8_epsilon();
  if (quadrature_type == QT_LEGENDRE || quadrature_type == QT_CHEBYSHEV_ONE ||
      quadrature_type == QT_GEGENBAUER || quadrature_type == QT_JACOBI ||
      quadrature_type == QT_EXPONENTIAL || quadrature_type == QT_CHEBYSHEV_TWO) {
    if (fabs(b - a) <= epsilon) {
      cout << "ERROR in ScaleQuadratureForNonStandardInterval: |B - A| "
           << "is too small; B: " << b << ", A: " << a << ", epsilon: "
           << epsilon << "." << endl;
      return false;
    }
  } else if (quadrature_type == QT_GENERALIZED_LAGUERRE ||
             quadrature_type == QT_GENERALIZED_HERMITE) {
    if (b <= 0.0) {
      cout << "ERROR in ScaleQuadratureForNonStandardInterval: b is zero."
           << endl;
      return false;
    }
  } else if (quadrature_type == QT_RATIONAL) {
    if (a + b <= 0.0) {
      cout << "ERROR in ScaleQuadratureForNonStandardInterval: a ("
           << a << ") + b (" << b << ") is non-positive."
           << endl;
      return false;
    }
  } else {
    cout << "ERROR in ScaleQuadratureForNonStandardInterval: unrecognized "
         << "QuadratureType: " << quadrature_type << endl;
    return false;
  }

  // Set al, be, shift, and slp, based on quadrature_type.
  double al;
  double be;
  double shift;
  double slp;
  if (quadrature_type == QT_LEGENDRE) {
    al = 0.0;
    be = 0.0;
    shift = (a + b) / 2.0;
    slp = (b - a) / 2.0;
  } else if (quadrature_type == QT_CHEBYSHEV_ONE) {
    al = -0.5;
    be = -0.5;
    shift = (a + b) / 2.0;
    slp = (b - a) / 2.0;
  } else if (quadrature_type == QT_GEGENBAUER) {
    al = alpha;
    be = alpha;
    shift = (a + b) / 2.0;
    slp = (b - a) / 2.0;
  } else if (quadrature_type == QT_JACOBI) {
    al = alpha;
    be = beta;
    shift = (a + b) / 2.0;
    slp = (b - a) / 2.0;
  } else if (quadrature_type == QT_GENERALIZED_LAGUERRE) {
    shift = a;
    slp = 1.0 / b;
    al = alpha;
    be = 0.0;
  } else if (quadrature_type == QT_GENERALIZED_HERMITE) {
    shift = a;
    slp = 1.0 / sqrt(b);
    al = alpha;
    be = 0.0;
  } else if (quadrature_type == QT_EXPONENTIAL) {
    al = alpha;
    be = 0.0;
    shift = (a + b) / 2.0;
    slp = (b - a) / 2.0;
  } else if (quadrature_type == QT_RATIONAL) {
    shift = a;
    slp = a + b;
    al = alpha;
    be = beta;
  } else if (quadrature_type == QT_CHEBYSHEV_TWO) {
    al = 0.5;
    be = 0.5;
    shift = (a + b) / 2.0;
    slp = (b - a) / 2.0;
  } else {
    return false;
  }

  // Scale the abscissa and weights.
  const double p = pow(slp, al + be + 1.0);
  for (int k = 0; k < order; k++) {
    (*output)[k].abscissa_ = shift + slp * unscaled_output[k].abscissa_;
    int l = abs(weight_index[k]);

    if (l != 0) {
      double tmp = p;
      for (int i = l - 1; i <= l - 1 + multiplicity[k] - 1; i++) {
        (*output)[i].weight_ = unscaled_output[i].weight_ * tmp;
        tmp = tmp * slp;
      }
    }
  }

  return true;
}

bool ComputeAbscissaAndWeights(
    const int order,
    const double& zeroth_moment,
    const vector<double>& jacobi_diagonal,
    const vector<double>& jacobi_sub_diagonal,
    vector<GaussianQuadratureTuple>* output) {
  // Sanity-check input.
  if (output == nullptr) {
    cout << "ERROR in ComputeAbscissaAndWeights: Null input.";
    return false;
  }

  //  Exit if the zero-th moment or order is not positive.
  if (zeroth_moment <= 0.0 || order <= 0) {
    cout << "ERROR in ComputeAbscissaAndWeights: non-positive zeroth-moment: "
         << zeroth_moment << " or order: " << order;
    return false;
  }

  output->resize(order);

  //  Set up vectors for IMTQLX.
  for (int i = 0; i < order; i++) {
    (*output)[i].abscissa_ = jacobi_diagonal[i];
  }
  (*output)[0].weight_ = sqrt(zeroth_moment);
  for (int i = 1; i < order; i++) {
    (*output)[i].weight_ = 0.0;
  }

  //  Diagonalize the Jacobi matrix.
  if (!DiagonalizeTriDiagonalSymmetricMatrix(jacobi_sub_diagonal, output)) {
    return false;
  }

  for (int i = 0; i < order; i++) {
    (*output)[i].weight_ = (*output)[i].weight_ * (*output)[i].weight_;
  }

  return true;
}

void Timestamp() {
  time_t now = time(NULL);
  const struct tm* tm_ptr = localtime(&now);
  static char time_buffer[TIMESTAMP_TIME_SIZE];
  strftime(time_buffer, TIMESTAMP_TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr);

  cout << time_buffer << "\n";

  return;
}

}  // namespace math_utils

