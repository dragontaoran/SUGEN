// Date: July 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "eq_solver.h"

#include "cdf_fns.h"
#include "data_structures.h"
#include "gamma_fns.h"
#include "number_comparison.h"

#include <map>
#include <set>

using namespace string_utils;
using namespace std;

namespace math_utils {

bool EvaluateExpression(
    const Expression& expression, const map<string, double>& var_values,
    double* value, set<string>* vars_seen, string* error_msg) {
  string junk_err_msg = "";
  string* error_msg_ptr = (error_msg == nullptr) ? &junk_err_msg : error_msg;

  // Evaluate the expression based on the operation.
  //   - Self-Operation.
  if (expression.op_ == IDENTITY) {
    if (expression.is_constant_) {
      *value = expression.value_;
      return true;
    } else {
      map<string, double>::const_iterator itr = var_values.find(expression.var_name_);
      if (itr == var_values.end()) {
        *error_msg_ptr +=
            "ERROR: Failed to evaluate expression '" +
            GetExpressionString(expression) + "': Could not find var name '" +
            expression.var_name_ + "' in var_values.\n";
        return false;
      }
      *value = itr->second;
      if (vars_seen != nullptr) vars_seen->insert(itr->first);
      return true;
    }
  //   - 2-Term Operations.
  } else if (expression.op_ == ADD || expression.op_ == SUB ||
             expression.op_ == MULT || expression.op_ == DIV ||
             expression.op_ == POW ||
             expression.op_ == EQ || expression.op_ == FLOAT_EQ ||
             expression.op_ == GT || expression.op_ == GTE ||
             expression.op_ == LT || expression.op_ == LTE ||
             expression.op_ == INC_GAMMA_FN || expression.op_ == REG_INC_GAMMA_FN) {
    if (expression.subterm_one_ == nullptr || expression.subterm_two_ == nullptr) {
      *error_msg_ptr +=
          "Unable to perform add/mult/pow: one of the subterms is null.";
      return false;
    }
    double value_one, value_two;
    if (!EvaluateExpression(
            *expression.subterm_one_, var_values, &value_one, error_msg_ptr)) {
      return false;
    }
    if (!EvaluateExpression(
            *expression.subterm_two_, var_values, &value_two, error_msg_ptr)) {
      return false;
    }
    if (expression.op_ == ADD) {
      *value = value_one + value_two;
    } else if (expression.op_ == SUB) {
      *value = value_one - value_two;
    } else if (expression.op_ == MULT) {
      *value = value_one * value_two;
    } else if (expression.op_ == DIV) {
      *value = value_one / value_two;
    } else if (expression.op_ == POW) {
      if (value_one < 0.0 && ceil(value_two) != value_two) {
        *error_msg_ptr +=
            "Unable to raise a negative base (" + Itoa(value_one) +
            ") to a non-integer exponent (" + Itoa(value_two) + ")";
        return false;
      }
      if (FloatEq(value_one, 0.0) && value_two <= 0.0) {
        *error_msg_ptr += "Unable to evaluate 0^0 or 0^negative";
        return false;
      }
      *value = pow(value_one, value_two);
    } else if (expression.op_ == INC_GAMMA_FN) {
      *value = LowerIncompleteGammaFunction(value_one, value_two);
    } else if (expression.op_ == REG_INC_GAMMA_FN) {
      *value = RegularizedIncompleteGammaFunction(value_one, value_two);
    } else if (expression.op_ == EQ) {
      *value = value_one == value_two ? 1.0 : 0.0;
    } else if (expression.op_ == FLOAT_EQ) {
      *value = FloatEq(value_one,value_two) ? 1.0 : 0.0;
    } else if (expression.op_ == GT) {
      *value = value_one > value_two ? 1.0 : 0.0;
    } else if (expression.op_ == GTE) {
      *value = value_one >= value_two ? 1.0 : 0.0;
    } else if (expression.op_ == LT) {
      *value = value_one < value_two ? 1.0 : 0.0;
    } else if (expression.op_ == LTE) {
      *value = value_one <= value_two ? 1.0 : 0.0;
    }
    return true;
  //   - 1-Term Operations.
  } else if (expression.op_ == ABS || expression.op_ == EXP ||
             expression.op_ == FACTORIAL || expression.op_ == LOG ||
             expression.op_ == SQRT || expression.op_ == SIN ||
             expression.op_ == COS || expression.op_ == TAN ||
             expression.op_ == INDICATOR ||
             expression.op_ == GAMMA_FN || expression.op_ == PHI_FN) {
    if (expression.subterm_one_ == nullptr) {
      *error_msg_ptr += "Unable to perform evaluation: subterm_one_ is null";
      return false;
    }
    if (expression.subterm_two_ != nullptr) {
      *error_msg_ptr += "Unable to perform evaluation: subterm_two_ is non-null";
      return false;
    }
    double value_one;
    if (!EvaluateExpression(
            *expression.subterm_one_, var_values, &value_one, error_msg_ptr)) {
      return false;
    }
    if (expression.op_ == TAN) {
      *value = tan(value_one);
    } else if (expression.op_ == SIN) {
      *value = sin(value_one);
    } else if (expression.op_ == COS) {
      *value = cos(value_one);
    } else if (expression.op_ == SQRT) {
      *value = sqrt(value_one);
    } else if (expression.op_ == GAMMA_FN) {
      *value = Gamma(value_one);
    } else if (expression.op_ == PHI_FN) {
      *value = StandardNormalCDF(value_one);
    } else if (expression.op_ == INDICATOR) {
      *value = value_one;  // Subterm will already be 0 or 1, as appropriate
    } else if (expression.op_ == ABS) {
      *value = abs(value_one);
    } else if (expression.op_ == SQRT) {
      if (value_one < 0.0) {
        *error_msg_ptr +=
            "Unable to take square root of negative value: " +
            Itoa(value_one);
        return false;
      }
      *value = sqrt(value_one);
    } else if (expression.op_ == LOG) {
      if (value_one <= 0.0) {
        *error_msg_ptr +=
            "Unable to take log of negative value: " +
            Itoa(value_one);
        return false;
      }
      *value = log(value_one);
    } else if (expression.op_ == EXP) {
      *value = exp(value_one);
    } else if (expression.op_ == FACTORIAL) {
      if (value_one < 0.0 || ceil(value_one) != value_one) {
        *error_msg_ptr +=
            "Unable to do factorial of negative value or non-integer: " +
            Itoa(value_one);
        return false;
      }
      int int_value = (int) value_one;
      *value = 1.0;
      for (int i = 2; i <= int_value; ++i) {
        *value *= i;
      }
    }
    return true;
  }
  *error_msg_ptr +=
      "Unsupported operation: " + Itoa(static_cast<int>(expression.op_));
  return false;
}

bool Solve(const Expression& function, const Expression& derivative,
           RootFinder* solution) {
  if (solution == nullptr) return false;
  if (solution->iteration_index_ > solution->max_iterations_) {
    solution->error_msg_ +=
        "Aborting Newton's Method after " +
        Itoa(solution->iteration_index_ + 1) + " iterations. Current guess: " +
        Itoa(solution->guess_);
    return false;
  }
  
  // Evaluate f(x) and f'(x).
  const double current_guess = solution->guess_;
  map<string, double> var_values;
  var_values.insert(make_pair("x", current_guess));
  double value, derivative_value;
  if (!EvaluateExpression(function, var_values, &value, &solution->error_msg_)) {
    solution->error_msg_ +=
        "Unable to compute function at current guess: " +
        Itoa(current_guess) + ".\n";
    return false;
  }
  if (value == 0.0) {
    solution->summary_info_ += "Newton's Method completed after " +
                    Itoa(solution->iteration_index_ + 1) + " iterations.";
    return true;
  }
  if (!EvaluateExpression(
          derivative, var_values, &derivative_value, &solution->error_msg_)) {
    solution->error_msg_ =
        "Unable to compute derivative at current guess: " +
        Itoa(current_guess) + ".\n";
    return false;
  }
  if (FloatEq(derivative_value, 0.0)) {
    solution->error_msg_ +=
        "Encountered derivative equals zero at iteration " +
        Itoa(solution->iteration_index_ + 1) + ", where current guess was: " +
        Itoa(current_guess) + " and value at that guess: " +
        Itoa(value);
    // Reset iteration index, so SolveUsingMidpoint has a full chance to succeed.
    solution->iteration_index_ = 0;
    return SolveUsingMidpoint(function, &derivative, solution);
  }

  // Run iteration to compute next guess.
  solution->guess_ = current_guess - (value / derivative_value);

  // Check stopping criteria:
  //   - value is sufficiently close to zero
  //   - current guess is sufficiently close to previous guess
  if ((!solution->use_absolute_distance_to_zero_ ||
       AbsoluteConvergence(value, 0.0, solution->absolute_distance_to_zero_)) &&
      (!solution->use_relative_distance_from_prev_guess_ ||
       AbsoluteConvergenceSafeTwo(current_guess, solution->guess_, EPSILON,
                                  solution->relative_distance_from_prev_guess_)) &&
      (!solution->use_absolute_distance_from_prev_guess_ ||
       AbsoluteConvergence(current_guess, solution->guess_,
                           solution->absolute_distance_from_prev_guess_))) {
    solution->summary_info_ += "Newton's Method completed after " +
                    Itoa(solution->iteration_index_ + 1) + " iterations.";
    return true;
  }

  // Update solution's closest_neg_ and/or closest_pos_ (not needed for
  // Newton's Method, but useful if Newton's method fails and we shift
  // to SolveUsingMidpoint).
  if (value > 0.0) {
    if (solution->closest_pos_ == DBL_MAX) {
      // Haven't seen a positive value yet. Set closest_pos_ to current guess.
      solution->closest_pos_ = current_guess;
    } else if (solution->closest_neg_ == DBL_MIN) {
      // Haven't seen a negative value yet, so we don't know if the current
      // guess is closer or farther from the root. We could just leave
      // closest_pos_ to its present value, but we arbitarily decide to
      // take the smaller (in absolute value) value (since closest_pos_
      // started at DBL_MAX, perhaps moving closer to zero will bring us
      // closer to the root?).
      if (abs(current_guess) < abs(solution->closest_pos_)) {
          solution->closest_pos_ = current_guess;
      }
    } else {
      // Update closest_pos_ to be current_guess if the latter is closer to
      // closest_neg_.
      if (abs(solution->closest_neg_ - current_guess) <
          abs(solution->closest_neg_ - solution->closest_pos_)) {
        solution->closest_pos_ = current_guess;
      }
    }
  } else {
    if (solution->closest_neg_ == DBL_MIN) {
      // Haven't seen a negative value yet. Set closest_neg_ to current guess.
      solution->closest_neg_ = current_guess;
    } else if (solution->closest_pos_ == DBL_MAX) {
      // Haven't seen a positive value yet, so we don't know if the current
      // guess is closer or farther from the root. We could just leave
      // closest_neg_ to its present value, but we arbitarily decide to
      // take the smaller (in absolute value) value (since closest_neg_
      // started at DBL_MIN, perhaps moving closer to zero will bring us
      // closer to the root?).
      if (abs(current_guess) < abs(solution->closest_neg_)) {
          solution->closest_neg_ = current_guess;
      }
    } else {
      // Update closest_neg_ to be current_guess if the latter is closer to
      // closest_pos_.
      if (abs(solution->closest_pos_ - current_guess) <
          abs(solution->closest_pos_ - solution->closest_neg_)) {
        solution->closest_neg_ = current_guess;
      }
    }
  }

  solution->iteration_index_++;
  return Solve(function, derivative, solution);
}

bool SolveUsingMidpoint(const Expression& function, const Expression* derivative,
                        RootFinder* solution) {
  // Sanity-check input.
  if (solution == nullptr) return false;

  // Check if we've exceeded solution->max_iterations_.
  if (solution->iteration_index_ > solution->max_iterations_) {
    solution->error_msg_ +=
        "ERROR in SolveUsingMidpoint: Exceeded max number of iterations "
        "for Midpoint method (" +
        Itoa(solution->max_iterations_) + ").";
    return false;
  }
 
  // Sanity-check input.
  if (solution->guess_ == DBL_MIN || solution->guess_ == DBL_MAX) {
    solution->error_msg_ +=
        "ERROR in SolveUsingMidpoint: either no value was provided for "
        "initial guess, or MidpointMethod failed to find an interval "
        "within [DBL_MIN, DBL_MAX] that contained a root (in which case, "
        "try again using a different initial guess).";
    return false;
  }
 
  // Compute f(x) for x = solution->guess_.
  const double current_guess = solution->guess_;
  double value;
  map<string, double> var_values;
  var_values.insert(make_pair("x", current_guess));
  if (!EvaluateExpression(function, var_values, &value, &solution->error_msg_)) {
    solution->error_msg_ =
        "Unable to compute function at current guess: " +
        Itoa(current_guess) + ".\n";
    return false;
  }

  // Return if solution->guess_ is a root.
  const bool convergence_criteria_met =
      (!solution->use_absolute_distance_to_zero_ ||
       AbsoluteConvergence(value, 0.0, solution->absolute_distance_to_zero_)) &&
      (!solution->use_relative_distance_from_prev_guess_ ||
       AbsoluteConvergenceSafeTwo(solution->prev_guess_, solution->guess_, EPSILON,
                                  solution->relative_distance_from_prev_guess_)) &&
      (!solution->use_absolute_distance_from_prev_guess_ ||
       AbsoluteConvergence(solution->prev_guess_, solution->guess_,
                           solution->absolute_distance_from_prev_guess_));
  const bool valid_interval =
      !solution->demand_pos_and_neg_values_found_ ||
      (solution->closest_neg_ != DBL_MIN && solution->closest_pos_ != DBL_MAX);
  if (value == 0.0 || (valid_interval && convergence_criteria_met)) {
    solution->summary_info_ += "MidpointMethod completed after " +
                    Itoa(solution->iteration_index_ + 1) + " iterations.";
    return true;
  }

  // Update message to reflect current iteration's value.
  solution->debug_info_ +=
      "For iteration " + Itoa(solution->iteration_index_ + 1) +
      ", guess " + Itoa(current_guess) + " evaluates to " +
      Itoa(value) + "\n";

  // Record current guess' evaluation value (can be used for next guess to 
  // determine which way to go).
  const double prev_value = solution->value_at_prev_guess_;
  solution->value_at_prev_guess_ = value;

  // Determine what to do based on whether f(current_guess) is
  // positive or negative.
  // TODO(PHB): The below main if/else blocks are mirror-opposites of each
  // other. Rather than duplicating code, set values once at the start that
  // hold solution->closest_[pos | neg]_ and DBL_[MIN | MAX] based on
  // whether value is positive or negative, and then we don't need the duplicate
  // blocks.
  if (value < 0.0) {
    // If this is the first negative value found, record the guess and iteration.
    if (solution->closest_neg_ == DBL_MIN) {
      solution->debug_info_ +=
          "Found first negative value (" + Itoa(value) + ") at iteration " +
          Itoa(solution->iteration_index_ + 1) + " for guess " + Itoa(solution->guess_) +
          "\n";
    }

    // If solution->closest_pos_ is already set, then we have an interval
    // for a solution ([current_guess_, closest_pos_]), and we know exactly
    // which direction to move for our next guess. Otherwise, we'll have
    // to guess which direction to go.
    if (solution->closest_pos_ != DBL_MAX) {
      solution->closest_neg_ = current_guess;
      // We have an interval to work with:
      //   [current_guess, solution->closest_pos_]
      // Bisect and iterate.
      solution->prev_guess_ = solution->guess_;
      solution->guess_ = (solution->closest_pos_ + current_guess) / 2.0;
      solution->iteration_index_++;
      return SolveUsingMidpoint(function, nullptr, solution);
    } else if (solution->closest_neg_ == DBL_MIN) {
      // This is the first guess. Set solution->closest_neg_ to current guess.
      solution->closest_neg_ = current_guess;
      // Use derivative as a guess as to which direction to go to find a root.
      // (not guaranteed to work, e.g. if there is a local max/min before the
      // root, but at least serves as a guess absent of other information).
      double derivative_value;
      if (derivative != nullptr) {
        if (!EvaluateExpression(
                *derivative, var_values, &derivative_value, &solution->error_msg_)) {
          solution->error_msg_ =
              "Unable to compute derivative at current guess: " +
              Itoa(current_guess) + ".\n";
          return false;
        }
        // We move left/right by 2 * abs(current_guess) since
        // current_guess (= user's initial guess) is likely to be of
        // the proper magnitude (the factor of 2 is to avoid trivially
        // checking 0.0 next, which would be okay, but may be more likely
        // to be an asymptote).
        if (derivative_value < 0.0) {
          solution->prev_guess_ = solution->guess_;
          // Derivative is negative and f(current_guess) < 0, so move left.
          solution->guess_ = FloatEq(current_guess, 0.0) ?
              -1.0 : current_guess - 2.0 * abs(current_guess);
          // Derivative no longer needed, so just pass nullptr.
          solution->iteration_index_++;
          return SolveUsingMidpoint(function, nullptr, solution);
        } else {
          solution->prev_guess_ = solution->guess_;
          // Derivative is positive and f(current_guess) < 0, so move right.
          solution->guess_ = FloatEq(current_guess, 0.0) ?
              1.0 : current_guess + 2.0 * abs(current_guess);
          // Derivative no longer needed, so just pass nullptr.
          solution->iteration_index_++;
          return SolveUsingMidpoint(function, nullptr, solution);
        }
      } else {
        solution->prev_guess_ = solution->guess_;
        // No derivative to give us a guess of which direction to move.
        // Arbitrarily choose to move in same direction as sign of
        // current_guess.
        solution->guess_ = (1.0 + EPSILON) * (current_guess + EPSILON);
        solution->iteration_index_++;
        return SolveUsingMidpoint(function, nullptr, solution);
      }
    } else {
      solution->prev_guess_ = solution->guess_;
      // All previous guesses evaluated to a negative value. Use the
      // previous guesses' evaluation value to determine which direction
      // to go (if available); otherwise, progressively
      // move further away from zero (in both directions) to try to find
      // an interval that contains a positive value. Move in the opposite
      // direction as we did for the previous iteration.
      if (prev_value == DBL_MIN) {
          solution->error_msg_ +=
              "Should not reach here: solution->value_at_prev_guess_ should "
              "have been set in previous iteration. Current iteration: " +
              Itoa(solution->iteration_index_ + 1) + ", current guess: " +
              Itoa(current_guess) + ".\n";
          return false;
      } else if (abs(prev_value) < abs(value)) {
        solution->guess_ =
            solution->closest_neg_ -
            (1.0 + EPSILON) * (current_guess - solution->closest_neg_);
      } else {
        solution->guess_ =
            current_guess + 2.0 * (current_guess - solution->closest_neg_);
        solution->closest_neg_ = current_guess;
      }
      solution->iteration_index_++;
      return SolveUsingMidpoint(function, nullptr, solution);
    }
  } else {
    if (solution->closest_pos_ == DBL_MAX) {
      solution->debug_info_ +=
          "Found first positive value (" + Itoa(value) + ") at iteration " +
          Itoa(solution->iteration_index_ + 1) + " for guess " + Itoa(solution->guess_) +
          "\n";
    }
    // If solution->closest_neg_ is already set, then we have an interval
    // for a solution ([current_guess_, closest_neg_]), and we know exactly
    // which direction to move for our next guess. Otherwise, we'll have
    // to guess which direction to go.
    if (solution->closest_neg_ != DBL_MIN) {
      solution->prev_guess_ = solution->guess_;
      solution->closest_pos_ = current_guess;
      // We have an interval to work with:
      //   [current_guess, solution->closest_neg_]
      // Bisect and iterate.
      solution->guess_ = (solution->closest_neg_ + current_guess) / 2.0;
      solution->iteration_index_++;
      return SolveUsingMidpoint(function, nullptr, solution);
    } else if (solution->closest_pos_ == DBL_MAX) {
      // This is the first guess. Set solution->closest_pos_ to current guess.
      solution->closest_pos_ = current_guess;
      // Use derivative as a guess as to which direction to go to find a root.
      // (not guaranteed to work, e.g. if there is a local max/min before the
      // root, but at least serves as a guess absent of other information).
      double derivative_value;
      if (derivative != nullptr) {
        if (!EvaluateExpression(
                *derivative, var_values, &derivative_value, &solution->error_msg_)) {
          solution->error_msg_ =
              "Unable to compute derivative at current guess: " +
              Itoa(current_guess) + ".\n";
          return false;
        }
        // We move left/right by 2 * abs(current_guess) since
        // current_guess (= user's initial guess) is likely to be of
        // the proper magnitude (the factor of 2 is to avoid trivially
        // checking 0.0 next, which would be okay, but may be more likely
        // to be an asymptote).
        if (derivative_value > 0.0) {
          solution->prev_guess_ = solution->guess_;
          // Derivative is positive and f(current_guess) > 0, so move left.
          solution->guess_ = FloatEq(current_guess, 0.0) ?
              -1.0 : current_guess - 2.0 * abs(current_guess);
          // Derivative no longer needed, so just pass nullptr.
          solution->iteration_index_++;
          return SolveUsingMidpoint(function, nullptr, solution);
        } else {
          solution->prev_guess_ = solution->guess_;
          // Derivative is negative and f(current_guess) > 0, so move right.
          solution->guess_ = FloatEq(current_guess, 0.0) ?
              1.0 : current_guess + 2.0 * abs(current_guess);
          // Derivative no longer needed, so just pass nullptr.
          solution->iteration_index_++;
          return SolveUsingMidpoint(function, nullptr, solution);
        }
      } else {
        solution->prev_guess_ = solution->guess_;
        // No derivative to give us a guess of which direction to move.
        // Arbitrarily choose to move in same direction as sign of
        // current_guess.
        solution->guess_ = (1.0 + EPSILON) * (current_guess + EPSILON);
        solution->iteration_index_++;
        return SolveUsingMidpoint(function, nullptr, solution);
      }
    } else {
      solution->prev_guess_ = solution->guess_;
      // All previous guesses evaluated to a negative value. Use the
      // previous guesses' evaluation value to determine which direction
      // to go (if available); otherwise, progressively
      // move further away from zero (in both directions) to try to find
      // an interval that contains a negative value. Move in the opposite
      // direction as we did for the previous iteration.
      if (prev_value == DBL_MIN) {
          solution->error_msg_ +=
              "Should not reach here: solution->value_at_prev_guess_ should "
              "have been set in previous iteration. Current iteration: " +
              Itoa(solution->iteration_index_ + 1) + ", current guess: " +
              Itoa(current_guess) + ".\n";
          return false;
      } else if (abs(prev_value) < abs(value)) {
        solution->guess_ =
            solution->closest_pos_ -
            (1.0 + EPSILON) * (current_guess - solution->closest_pos_);
      } else {
        solution->guess_ =
            current_guess + 2.0 * (current_guess - solution->closest_pos_);
        solution->closest_pos_ = current_guess;
      }
      solution->iteration_index_++;
      return SolveUsingMidpoint(function, nullptr, solution);
    }
  }

  // Shouldn't reach here.
  return false;
}

}  // namespace math_utils
