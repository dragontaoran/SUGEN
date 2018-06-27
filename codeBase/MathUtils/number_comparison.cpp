// Date: March 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "number_comparison.h"

#include "constants.h"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

using namespace std;
using Eigen::VectorXd;

namespace math_utils {

const double ABSOLUTE_CONVERGENCE_THRESHOLD = 0.000001;  // 10^-6

bool FloatEq(const float& one, const float& two,
             const double& delta, const double& threshold) {
  return AbsoluteConvergenceSafe(
      static_cast<double>(one), static_cast<double>(two), delta, threshold);
}

bool FloatEq(const double& one, const double& two,
             const double& delta, const double& threshold) {
  return AbsoluteConvergenceSafe(one, two, delta, threshold);
}

bool AbsoluteConvergence(
    const double& prev, const double& next, const double& threshold) {
  return abs(prev - next) <= threshold;
}

bool AbsoluteConvergenceSafe(
    const double& prev, const double& next,
    const double& delta, const double& threshold) {
  if (abs(prev) <= delta || abs(next) <= delta ||
      FloatEq(min(abs(prev), abs(next)), 0.0)) {
    return abs(prev - next) <= threshold;
  }
  return abs(prev - next) / (min(abs(prev), abs(next))) <= threshold;
}

bool AbsoluteConvergenceSafeTwo(
    const double& prev, const double& next,
    const double& epsilon, const double& threshold) {
  return abs(prev - next) / (min(abs(prev), abs(next)) + epsilon) <= threshold;
}

bool VectorAbsoluteConvergence(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& threshold) {
  if (prev.size() != next.size() || prev.size() != threshold.size()) {
    return false;
  }

  for (int i = 0; i < prev.size(); ++i) {
    if (!AbsoluteConvergence(prev[i], next[i], threshold[i])) return false;
  }
  return true;
}

bool VectorAbsoluteConvergence(
    const vector<double>& prev, const vector<double>& next,
    const double& threshold) {
  if (prev.size() != next.size()) {
    return false;
  }

  for (int i = 0; i < prev.size(); ++i) {
    if (!AbsoluteConvergence(prev[i], next[i], threshold)) return false;
  }
  return true;
}

bool VectorAbsoluteConvergenceSafe(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& delta, const vector<double>& threshold) {
  if (prev.size() != next.size() || prev.size() != threshold.size() ||
      prev.size() != delta.size()) {
    return false;
  }

  for (int i = 0; i < prev.size(); ++i) {
    if (!AbsoluteConvergenceSafe(prev[i], next[i], delta[i], threshold[i])) {
      return false;
    }
  }
  return true;
}

bool VectorAbsoluteConvergenceSafe(
    const VectorXd& prev, const VectorXd& next,
    const VectorXd& delta, const VectorXd& threshold) {
  if (prev.size() != next.size() || prev.size() != threshold.size() ||
      prev.size() != delta.size()) {
    return false;
  }

  for (int i = 0; i < prev.size(); ++i) {
    if (!AbsoluteConvergenceSafe(prev[i], next[i], delta[i], threshold[i])) {
      return false;
    }
  }
  return true;
}

bool VectorAbsoluteConvergenceSafe(
    const vector<double>& prev, const vector<double>& next,
    const double& delta, const double& threshold) {
  if (prev.size() != next.size()) {
    return false;
  }

  for (int i = 0; i < prev.size(); ++i) {
    if (!AbsoluteConvergenceSafe(prev[i], next[i], delta, threshold)) {
      return false;
    }
  }
  return true;
}

bool VectorAbsoluteConvergenceSafe(
    const VectorXd& prev, const VectorXd& next,
    const double& delta, const double& threshold) {
  if (prev.size() != next.size()) {
    return false;
  }

  for (int i = 0; i < prev.size(); ++i) {
    if (!AbsoluteConvergenceSafe(prev[i], next[i], delta, threshold)) {
      return false;
    }
  }
  return true;
}

bool VectorAbsoluteConvergenceSafeTwo(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& epsilon, const vector<double>& threshold) {
  if (prev.size() != next.size() || prev.size() != threshold.size() ||
      prev.size() != epsilon.size()) {
    return false;
  }

  for (int i = 0; i < prev.size(); ++i) {
    if (!AbsoluteConvergenceSafeTwo(
            prev[i], next[i], epsilon[i], threshold[i])) {
      return false;
    }
  }
  return true;
}

bool VectorAbsoluteConvergenceSafeTwo(
    const vector<double>& prev, const vector<double>& next,
    const double& epsilon, const double& threshold) {
  if (prev.size() != next.size()) {
    return false;
  }

  for (int i = 0; i < prev.size(); ++i) {
    if (!AbsoluteConvergenceSafeTwo(prev[i], next[i], epsilon, threshold)) {
      return false;
    }
  }
  return true;
}

// ====================== ABSOULTE DIFFERENCE ==================================

double AbsoluteDifference(const double& prev, const double& next) {
  return abs(prev - next);
}

double VectorAbsoluteDifference(
    const vector<double>& prev, const vector<double>& next) {
  if (prev.size() != next.size()) {
    return -1.0;
  }

  double max_difference = 0.0;
  for (int i = 0; i < prev.size(); ++i) {
    const double current_difference = AbsoluteDifference(prev[i], next[i]);
    if (current_difference > max_difference) max_difference = current_difference;
  }
  return max_difference;
}

double AbsoluteDifferenceSafe(
    const double& prev, const double& next, const double& delta) {
  if (abs(prev) <= delta || abs(next) <= delta ||
      FloatEq(min(abs(prev), abs(next)), 0.0)) {
    return abs(prev - next);
  }
  return abs(prev - next) / (min(abs(prev), abs(next)));
}

double VectorAbsoluteDifferenceSafe(
    const VectorXd& prev, const VectorXd& next, const VectorXd& delta) {
  if (prev.size() != next.size()) {
    return -1.0;
  }

  double max_difference = 0.0;
  for (int i = 0; i < prev.size(); ++i) {
    double current_difference =
        AbsoluteDifferenceSafe(prev(i), next(i), delta(i));
    if (current_difference > max_difference) max_difference = current_difference;
  }
  return max_difference;
}

double VectorAbsoluteDifferenceSafe(
    const VectorXd& prev, const VectorXd& next, const double& delta) {
  if (prev.size() != next.size()) {
    return -1.0;
  }

  double max_difference = 0.0;
  for (int i = 0; i < prev.size(); ++i) {
    double current_difference =
        AbsoluteDifferenceSafe(prev(i), next(i), delta);
    if (current_difference > max_difference) max_difference = current_difference;
  }
  return max_difference;
}

double VectorAbsoluteDifferenceSafe(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& delta) {
  if (prev.size() != next.size()) {
    return -1.0;
  }

  double max_difference = 0.0;
  for (int i = 0; i < prev.size(); ++i) {
    double current_difference =
        AbsoluteDifferenceSafe(prev[i], next[i], delta[i]);
    if (current_difference > max_difference) max_difference = current_difference;
  }
  return max_difference;
}

double VectorAbsoluteDifferenceSafe(
    const vector<double>& prev, const vector<double>& next,
    const double& delta) {
  if (prev.size() != next.size()) {
    return -1.0;
  }

  double max_difference = 0.0;
  for (int i = 0; i < prev.size(); ++i) {
    double current_difference =
        AbsoluteDifferenceSafe(prev[i], next[i], delta);
    if (current_difference > max_difference) max_difference = current_difference;
  }
  return max_difference;
}
// PHB

double AbsoluteDifferenceSafeTwo(
    const double& prev, const double& next, const double& epsilon) {
  return abs(prev - next) / (min(abs(prev), abs(next)) + epsilon);
}

double VectorAbsoluteDifferenceSafeTwo(
    const VectorXd& prev, const VectorXd& next, const VectorXd& epsilon) {
  if (prev.size() != next.size() || prev.size() != epsilon.size()) {
    return -1.0;
  }

  double max_difference = 0.0;
  for (int i = 0; i < prev.size(); ++i) {
    double current_difference =
        AbsoluteDifferenceSafeTwo(prev(i), next(i), epsilon(i));
    if (current_difference > max_difference) max_difference = current_difference;
  }
  return max_difference;
}

double VectorAbsoluteDifferenceSafeTwo(
    const VectorXd& prev, const VectorXd& next, const double& epsilon) {
  if (prev.size() != next.size()) {
    return -1.0;
  }

  double max_difference = 0.0;
  for (int i = 0; i < prev.size(); ++i) {
    double current_difference =
        AbsoluteDifferenceSafeTwo(prev(i), next(i), epsilon);
    if (current_difference > max_difference) max_difference = current_difference;
  }
  return max_difference;
}

double VectorAbsoluteDifferenceSafeTwo(
    const vector<double>& prev, const vector<double>& next,
    const vector<double>& epsilon) {
  if (prev.size() != next.size() || prev.size() != epsilon.size()) {
    return -1.0;
  }

  double max_difference = 0.0;
  for (int i = 0; i < prev.size(); ++i) {
    double current_difference =
        AbsoluteDifferenceSafeTwo(prev[i], next[i], epsilon[i]);
    if (current_difference > max_difference) max_difference = current_difference;
  }
  return max_difference;
}

double VectorAbsoluteDifferenceSafeTwo(
    const vector<double>& prev, const vector<double>& next,
    const double& epsilon) {
  if (prev.size() != next.size()) {
    return -1.0;
  }

  double max_difference = 0.0;
  for (int i = 0; i < prev.size(); ++i) {
    double current_difference =
        AbsoluteDifferenceSafeTwo(prev[i], next[i], epsilon);
    if (current_difference > max_difference) max_difference = current_difference;
  }
  return max_difference;
}

}  // namespace math_utils
