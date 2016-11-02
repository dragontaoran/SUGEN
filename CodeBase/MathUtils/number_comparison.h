// Date: March 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description: Tools for comparing numbers.

#ifndef NUMBER_COMPARISON_H
#define NUMBER_COMPARISON_H

#include "constants.h"

#include <Eigen/Dense>
#include <cstdlib>
#include <vector>

using namespace std;
using Eigen::VectorXd;

namespace math_utils {

extern const double ABSOLUTE_CONVERGENCE_THRESHOLD;

enum ItemsToCompare {
  LIKELIHOOD,
  COORDINATES,
};

struct ConvergenceCriteria {
 public:
  ItemsToCompare to_compare_;
  double delta_;
  double threshold_;

  ConvergenceCriteria() {
    to_compare_ = ItemsToCompare::LIKELIHOOD;
    delta_ = 0.01;
    threshold_ = 0.000001;
  }
};

// ============================ NUMBER COMPARISON ==============================
// Returns true iff AbsoluteConvergenceSafe(one, two) is true. See:
// http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
// for a good discussion of floating point comparison.
extern bool FloatEq(const float& one, const float& two,
                    const double& delta, const double& threshold);
inline bool FloatEq(const float& one, const float& two) {
  return FloatEq(one, two, DELTA, ABSOLUTE_CONVERGENCE_THRESHOLD);
}
extern bool FloatEq(const double& one, const double& two,
                    const double& delta, const double& threshold);
inline bool FloatEq(const double& one, const double& two) {
  return FloatEq(one, two, DELTA, ABSOLUTE_CONVERGENCE_THRESHOLD);
}
// =========================== END NUMBER COMPARISON ===========================

// ============================ CONVERGENCE CRITERIA ===========================
// Discussion: There are 3 kinds of convergence criteria below:
//   1) Absolute Convergence. Returns true if every coordinate i satisfies:
//        threshold_i >= |prev_i - next_i|
//   2) Absolute Convergence Safe. Returns true if every coordinate i satisfies:
//        threshold_i >= |prev_i - next_i|    IF  min(|prev_i|, |next_i|) <= delta_i
//        threshold_i >= |prev_i - next_i| / min(|prev_i|, |next_i|)       OTHERWISE
//   3) Absolute Convergence Safe2. Returns true if every coordinate i satisfies:
//        threshold_i >= |prev_i - next_i| / (|max(prev_i, next_i)| + delta_i)
//   4) Gradient Convergence: See below.
// (1)-(3) can be applied to numeric values or vectors of numeric values.
// Danyu prefers to use (2): Absolute Convergence Safe.
// Additionally, if you want to return the maximum difference (which is what
// is compared to the threshold), use the ABSOLUTE DIFFERENCE functions below.

// ============================ Absolulte Convergence ===========================
// For numeric values (i.e. vectors of size 1).
extern bool AbsoluteConvergence(
    const double& prev, const double& next, const double& threshold);
// Same as above, using ABSOLUTE_CONVERGENCE_THRESHOLD.
inline bool AbsoluteConvergence(const double& prev, const double& next) {
  return AbsoluteConvergence(prev, next, ABSOLUTE_CONVERGENCE_THRESHOLD);
}

// For Vectors.
extern bool VectorAbsoluteConvergence(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& threshold);
// Same as above, using same threshold for all coordinates.
extern bool VectorAbsoluteConvergence(
    const vector<double>& prev, const vector<double>& next,
    const double& threshold);
// Same as above, using ABSOLUTE_CONVERGENCE_THRESHOLD for all coordinates.
inline bool VectorAbsoluteConvergence(
    const vector<double>& prev, const vector<double>& next) {
  return
    VectorAbsoluteConvergence(prev, next, ABSOLUTE_CONVERGENCE_THRESHOLD);
}

// ========================= AbsoluteConvergenceSafe ===========================
// For numeric values (i.e. vectors of size 1).
extern bool AbsoluteConvergenceSafe(
    const double& prev, const double& next,
    const double& delta, const double& threshold);
// Same as above, using DELTA for delta and ABSOLUTE_CONVERGENCE_THRESHOLD
// for threshold.
inline bool AbsoluteConvergenceSafe(const double& prev, const double& next) {
  return AbsoluteConvergenceSafe(
      prev, next, DELTA, ABSOLUTE_CONVERGENCE_THRESHOLD);
}

// For Vectors.
extern bool VectorAbsoluteConvergenceSafe(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& delta, const vector<double>& threshold);
// Same as above, for different API (VectorXd vs. vector<double>).
extern bool VectorAbsoluteConvergenceSafe(
    const VectorXd& prev, const VectorXd& next,
    const VectorXd& delta, const VectorXd& threshold);
// Same as above, using same threshold and delta for all coordinates.
extern bool VectorAbsoluteConvergenceSafe(
    const vector<double>& prev, const vector<double>& next,
    const double& delta, const double& threshold);
// Same as above, for different API (VectorXd vs. vector<double>).
extern bool VectorAbsoluteConvergenceSafe(
    const VectorXd& prev, const VectorXd& next,
    const double& delta, const double& threshold);
// Same as above, using DELTA for delta and ABSOLUTE_CONVERGENCE_THRESHOLD
// for threshold (for all coordinates).
inline bool VectorAbsoluteConvergenceSafe(
    const vector<double>& prev, const vector<double>& next) {
  return VectorAbsoluteConvergenceSafe(
      prev, next, DELTA, ABSOLUTE_CONVERGENCE_THRESHOLD);
}
// Same as above, for different API (VectorXd vs. vector<double>).
inline bool VectorAbsoluteConvergenceSafe(
    const VectorXd& prev, const VectorXd& next) {
  return VectorAbsoluteConvergenceSafe(
      prev, next, DELTA, ABSOLUTE_CONVERGENCE_THRESHOLD);
}
// ======================= END AbsoluteConvergenceSafe =========================

// ======================= AbsoluteConvergenceSafeTwo ==========================
// For numeric values (i.e. vectors of size 1).
extern bool AbsoluteConvergenceSafeTwo(
    const double& prev, const double& next,
    const double& epsilon, const double& threshold);
// Same as above, using EPSILON for epsilon and ABSOLUTE_CONVERGENCE_THRESHOLD
// for threshold.
inline bool AbsoluteConvergenceSafeTwo(
    const double& prev, const double& next) {
  return AbsoluteConvergenceSafeTwo(
      prev, next, EPSILON, ABSOLUTE_CONVERGENCE_THRESHOLD);
}

// For vectors.
extern bool VectorAbsoluteConvergenceSafeTwo(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& epsilon, const vector<double>& threshold);
// Same as above, using same threshold and epsilon for all coordinates.
extern bool VectorAbsoluteConvergenceSafeTwo(
    const vector<double>& prev, const vector<double>& next,
    const double& epsilon, const double& threshold);
// Same as above, using EPSILON for epsilon and ABSOLUTE_CONVERGENCE_THRESHOLD
// for threshold (for all coordinates).
inline bool VectorAbsoluteConvergenceSafeTwo(
    const vector<double>& prev, const vector<double>& next) {
  return VectorAbsoluteConvergenceSafe(
      prev, next, EPSILON, ABSOLUTE_CONVERGENCE_THRESHOLD);
}
// ===================== END AbsoluteConvergenceSafeTwo ========================

// =========================== Gradient Convergence ============================
// Returns true if:
//   (g'_i * I_i * g_i) / (|l_i| + EPSILON) < threshold
// where:
//   i: Denotes iteration index
//   g: Gradient Vector
//   I: Negative (expected) Hessian Matrix
//   l: log-likelihood
// TODO(PHB): Implement this.
//extern bool GradientConvergence();
// ========================= END Gradient Convergence ==========================

// ========================== END CONVERGENCE CRITERIA =========================



// ============================ ABSOLUTE DIFFERENCE ============================
// The following functions all return the maximum "difference" between all
// coordinates of a vector, where we have the same three notions of difference
// as in CONVERGENCE CRITERIA functions above:
//   1) Absolute Difference. Returns:
//        Max_i(|prev_i - next_i|)
//   2) Absolute Convergence Safe. Returns:
//        For coordinates where min(|prev_i|, |next_i|) <= delta_i:
//          Let A := Max_i(|prev_i - next_i|)
//        For all other coordinates:
//          Let B := Max_i(|prev_i - next_i| / min(|prev_i|, |next_i|))
//        Return: Max(A, B).
//   3) Absolute Convergence Safe2. Returns:
//        Max_i(|prev_i - next_i| / (|max(prev_i, next_i)| + delta_i))
//
// ============================ Absolulte Difference ===========================
// For numeric values (i.e. vectors of size 1).
extern double AbsoluteDifference(const double& prev, const double& next);

// For Vectors.
extern double VectorAbsoluteDifference(
    const vector<double>& prev, const vector<double>& next);
// ========================= AbsoluteDifferenceSafe ===========================
// For numeric values (i.e. vectors of size 1).
extern double AbsoluteDifferenceSafe(
    const double& prev, const double& next, const double& delta);

// For Vectors.
extern double VectorAbsoluteDifferenceSafe(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& delta);
// Same as above, for different API (VectorXd vs. vector<double>).
extern double VectorAbsoluteDifferenceSafe(
    const VectorXd& prev, const VectorXd& next, const VectorXd& delta);
// Same as above, using same threshold and delta for all coordinates.
extern double VectorAbsoluteDifferenceSafe(
    const vector<double>& prev, const vector<double>& next, const double& delta);
// Same as above, for different API (VectorXd vs. vector<double>).
extern double VectorAbsoluteDifferenceSafe(
    const VectorXd& prev, const VectorXd& next, const double& delta);
// ======================= END AbsoluteDifferenceSafe =========================

// ======================= AbsoluteDifferenceSafeTwo ==========================
// For numeric values (i.e. vectors of size 1).
extern double AbsoluteDifferenceSafeTwo(
    const double& prev, const double& next, const double& epsilon);

// For vectors.
extern double VectorAbsoluteDifferenceSafeTwo(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& epsilon);
// Same as above, using same threshold and epsilon for all coordinates.
extern double VectorAbsoluteDifferenceSafeTwo(
    const vector<double>& prev, const vector<double>& next,
    const double& epsilon);
// Same as above, for different API (VectorXd instead of vector<double>)
extern double VectorAbsoluteDifferenceSafeTwo(
    const VectorXd& prev, const VectorXd& next,
    const VectorXd& epsilon);
// Same as above, using same threshold and epsilon for all coordinates.
extern double VectorAbsoluteDifferenceSafeTwo(
    const VectorXd& prev, const VectorXd& next,
    const double& epsilon);
// ===================== END AbsoluteDifferenceSafeTwo ========================

// ========================== END ABSOLUTE DIFFERENCE ==========================

}  // namespace math_utils
#endif
