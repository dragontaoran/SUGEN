// Date: July 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description: Uses Newton (Rhapson) Method for solving an equation.

#ifndef EQ_SOLVER_H
#define EQ_SOLVER_H

#include "MathUtils/data_structures.h"

#include <cstdlib>
#include <map>
#include <set>

using namespace math_utils;
using namespace std;

namespace math_utils {

extern double MAX_ITERATIONS_FOR_NEWTONS_METHOD;
extern double MAX_ITERATIONS_FOR_MIDPOINT_METHOD;

// Evaluates the given expression (by substituting each instance of a variable
// string with its corresponding value, as determined by 'var_values'. Populates
// 'value' with the answer. Also, if vars_seen is not null, keeps track of
// which Keys of 'var_values' were seen/used.
extern bool EvaluateExpression(
    const Expression& expression, const map<string, double>& var_values,
    double* value, set<string>* vars_seen, string* error_msg);
// Same as above, but doesn't keep track of which variables it saw.
inline bool EvaluateExpression(const Expression& expression,
                               const map<string, double>& var_values,
                               double* value, string* error_msg) {
  return EvaluateExpression(expression, var_values, value, nullptr, error_msg);
}
// Same as above, but for just one variable.
inline bool EvaluateExpression(
    const Expression& expression,
    const string& var_string, const double& var_value,
    double* value, string* error_msg) {
  map<string, double> vars;
  vars.insert(make_pair(var_string, var_value));
  return EvaluateExpression(expression, vars, value, error_msg);
}
// Same as above, but for a constant expression (no variables).
inline bool EvaluateConstantExpression(
    const Expression& expression, double* value, string* error_msg) {
  map<string, double> empty_vars;
  return EvaluateExpression(expression, empty_vars, value, error_msg);
}

// Finds a root of the function described by 'function' that is close to
// 'guess' by using N-R method. Returns true if successful (based on
// convergence critera), or false if convergence criteria was not reached
// within solution->max_iterations_.
// TODO(PHB): Generalize this to solve equations of more than one variable.
extern bool Solve(const Expression& function, const Expression& derivative,
                  RootFinder* solution);

// Uses midpoint strategy to find root. If non-null, uses derivative as an
// initial guess as to which direction to go from intitial guess.
// TODO(PHB): Generalize this to solve equations of more than one variable.
extern bool SolveUsingMidpoint(const Expression& function, const Expression* derivative,
                               RootFinder* solution);

// Given an expression, finds the derivative.
// TODO(PHB): Implement this.
extern bool Derivative(const Expression& function,
                       Expression* derivative, string* error_msg);

}  // namespace math_utils

#endif
