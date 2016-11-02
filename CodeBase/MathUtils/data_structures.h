// Author: paulbunn@email.unc.edu (Paul Bunn)
// Last Updated: January 2015
//
// Description: Defines objects that will be useful data storage holders
// for solving systems and running simulations.

#include "constants.h"
#include "StringUtils/string_utils.h"

#include <cfloat>   // For DBL_MIN, DBL_MAX
#include <errno.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <set>
#include <string.h>
#include <vector>

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

using namespace std;
using namespace string_utils;

namespace math_utils {

/* ================================== Enums ================================= */

// An enum representing various operations to apply to a single
// (double) value, or to combine two (double) values. Each
// function that takes in this enum will be responsible for actually
// determining how the operation is performed (e.g. via a 'switch' statement).
enum Operation {
  // Self-Operation.
  IDENTITY,

  // 1-Term Operations (i.e. these operations take a single argument)
  ABS,
  EXP,
  FACTORIAL,
  LOG,
  SQRT,
  SIN,
  COS,
  TAN,
  INDICATOR,
  GAMMA_FN,  // The Gamma function \Gamma(z) = \int_0^{\inf} t^{z - 1} * e^{-t} dt
  PHI_FN,    // The \Phi function (Standard Normal CDF).

  // 2-Term Operations (i.e. these operations take in two arguments).
  ADD,
  SUB,
  MULT,
  DIV,
  POW,
  EQ,   // Equals (==)
  FLOAT_EQ,  // FloatEq()
  GT,   // Greater Than (>)
  GTE,  // Greater Than or Equals (>=)
  LT,   // Less Than (<)
  LTE,  // Less Than or Equals (<=)
  INC_GAMMA_FN,  // (Lower) Incomplete Gamma fn \gamma(df, x) = \int_0^x t^{df - 1} * e^{-t} dt
  REG_INC_GAMMA_FN,  // Regularized Inc. Gamma fn P(df, x) = \gamma(df, x) / \Gamma(df)
};

// Various Distributions that are supported for sampling.
enum Distribution {
  BERNOULLI,          // "bern"
  BINOMIAL,           // "bin"
  CAUCHY,             // "cauchy"
  CHI_SQUARED,        // "chi_sq"
  CONSTANT,           // "C", "const", "constant"
  EXPONENTIAL,        // "exp"
  GAMMA,              // "gamma"
  GEOMETRIC,          // "geo"
  LOG_NORMAL,         // "log_N", "log_n"
  LOG_UNIFORM,        // "log_U", "log_u"
  NEGATIVE_BINOMIAL,  // "neg_bin"
  NORMAL,             // "N", "normal", "norm"
  POISSON,            // "P", "poisson"
  STUDENT_T,          // "t"
  UNIFORM,            // "U", "uniform", "unif"
};

// Describes the nature of a field: string, numeric, etc.
enum DataType {
  DATA_TYPE_UNKNOWN,
  DATA_TYPE_STRING,
  DATA_TYPE_NUMERIC,
  DATA_TYPE_NUMERIC_RANGE,
};

enum KaplanMeierEstimatorType {
  NONE,              // KME = 1.0 for all samples
  LEFT_CONTINUOUS,   // Used when computing statistics (e.g. log-rank, cox)
  RIGHT_CONTINUOUS,  // Used when just printing KME values
};

/* ================================ END Enums =============================== */





/* =============================== Structures =============================== */

// Parameters that specify how a given variable should be sampled.
struct SamplingParameters {
 public:
  // Name of the variable, as it appears in the model.
  string variable_name_;
  // User may want to specify one distribution type for some samples, and
  // a different one for other samples. The next two fields allow the user
  // to specify the range of samples to apply this distribution type to.
  int first_sample_;
  int last_sample_;

  // Type of distribution.
  Distribution type_;
  // Range of distribution.
  double range_start_;
  double range_end_;
  // The following can be the mean (e.g. for Normal distribution), or represent
  // a different paramter to the distribution (e.g. for Poisson, 'lambda' would
  // be stored in 'mean_' field).
  double mean_;
  // The second parameter of a distribution, typically (always?) actually does
  // represent standard deviation.
  double std_dev_;
  // A constant multiplier for the distribution.
  double constant_;

  SamplingParameters() {
    variable_name_ = "";
    first_sample_ = -1;
    last_sample_ = -1;
    range_start_ = 0.0;
    range_end_ = 0.0;
    mean_ = 0.0;
    std_dev_ = 0.0;
    constant_ = 1.0;
  }
};

// Each element of input data will have this type. It is general enough to
// encapsulate either double (e.g. ratio) or string (e.g. nominal or ordinal)
// values.
// For any particular DataHolder item, either value_ should equal 0.0, or
// name_ should equal "". If both are true, we use the double value_ (i.e.
// we assume the item is a double).
struct DataHolder {
 public:
  double value_;
  string name_;
  DataType type_;

  DataHolder() {
    value_ = 0.0;
    name_ = "";
    type_ = DataType::DATA_TYPE_UNKNOWN;
  }
};

// Holds the info for the "dependent variable" for the Cox Proportional Hazards
// Model.
struct CensoringData {
 public:
  double survival_time_;
  double censoring_time_;
  double left_truncation_time_;
  bool is_alive_;

  CensoringData() {
    survival_time_ = DBL_MIN;
    censoring_time_ = DBL_MIN;
    left_truncation_time_ = 0.0;
  }
};    

// Holds the info for the "dependent variable" for the Interval-Censored Surival
// Model.
struct IntervalCensoredData {
 public:
  double left_time_;
  double right_time_;

  IntervalCensoredData() {
    left_time_ = 0.0;
    right_time_ = numeric_limits<double>::infinity();
  }
};    

// Holds all possible information for a given row (Sample, Patient, etc.) of
// input data.
struct DataRow {
  string id_;
  double weight_;
  vector<string> families_;
  double dep_var_value_;
  CensoringData cox_dep_var_value_;
  // Note: This vector is meaningful only if there is a corresponding
  // vector of variable names indicating which column each index corresponds to.
  vector<double> indep_var_values_;

  DataRow() {
    id_ = "";
    weight_ = -1.0;
    dep_var_value_ = DBL_MIN;
  }
};

// Holds information necessary to find a root (solution) of an equation.
struct RootFinder {
  // Fields used in the algorithm to find solution.
  double value_at_prev_guess_;
  double closest_neg_;
  double closest_pos_;
  double guess_;
  double prev_guess_;

  // Fields to determine convergence to the root. At least one of the following
  // three booleans (not counting the last one) must be true.
  bool use_absolute_distance_to_zero_;
  double absolute_distance_to_zero_;
  bool use_absolute_distance_from_prev_guess_;
  double absolute_distance_from_prev_guess_;
  bool use_relative_distance_from_prev_guess_;
  double relative_distance_from_prev_guess_;
  // Whether midpoint method is required to have found a valid interval [a, b],
  // such that f(a) and f(b) have opposite signs, before returning.
  bool demand_pos_and_neg_values_found_;

  // Fields tracking number of iterations.
  int iteration_index_;
  int max_iterations_;

  // Fields that store information (to print) about how the algorithm did.
  string summary_info_;
  string debug_info_;
  string error_msg_;

  RootFinder() {
    // Note that there is no reason why closest_neg_ should be set to neg
    // infinity (DBL_MIN) and closest_pos_ to pos infinity (DBL_MAX), i.e.
    // a priori we don't know what the function approaches at both ends.
    // However, it is convenient to set these to something (so that we
    // can determine if they have been set for real yet), and these choices
    // have the advantage of being different from each other, as well as
    // matching intuition of the basic function y = x. Note that when using
    // these, it should NOT be assumed that closest_neg_ < closest_pos_.
    closest_neg_ = DBL_MIN;
    closest_pos_ = DBL_MAX;
    value_at_prev_guess_ = DBL_MIN;
    guess_ = 0.0;
    prev_guess_ = DBL_MIN;

    use_absolute_distance_to_zero_ = true;
    absolute_distance_to_zero_ = EPSILON;
    use_absolute_distance_from_prev_guess_ = false;
    absolute_distance_from_prev_guess_ = EPSILON;
    use_relative_distance_from_prev_guess_ = true;
    relative_distance_from_prev_guess_ = EPSILON;
    demand_pos_and_neg_values_found_ = true;

    iteration_index_ = 0;
    max_iterations_ = 100;

    summary_info_ = "";
    debug_info_ = "";
    error_msg_ = "";
  }
};

// A generic structure that can represent a constant value, a variable,
// a term in an expression, or a full expression (function). Structure:
//   - If op_ = IDENTITY, then exactly one of {var_name_, value_} should be set
//     (depending on whether this is a variable or a constant), and
//     subterm_one_ and subterm_two_ should both be NULL
//   - For any other operation op_, var_name_ should be empty and value_ should
//     be NULL; either one or both of {subterm_one_, subterm_two_} should be set:
//   - If op_ is EXP, LOG, SQRT, SIN, COS, or TAN, then subterm_two_ should be NULL
//   - If op_ is ADD, SUB, MULT, DIV, or POW, then subterm_one_ and subterm_two_
//     should both be non-null.
// Example:
//   log(2x^2 + 3x - sin(x))
// Then:
//   Outermost Expression log(2x^2 + 3x - sin(x)):
//     op_ = LOG, subterm_one_ = &A
//   Expression A (2x^2 + 3x - sin(x)):
//     op_ = ADD, subterm_one_ = &B, subterm_two_ = &C
//   Expression B (2x^2 + 3x):
//     op_ = ADD, subterm_one_ = &D, subterm_two_ = &E
//   Expression C (-sin(x)):
//     op_ = MULT, subterm_one_ = &F, subterm_two_ = &G
//   Expression D (2x^2):
//     op_ = MULT, subterm_one_ = &H, subterm_two_ = &I
//   Expression E (3x):
//     op_ = MULT, subterm_one_ = &J, subterm_two_ = &K
//   Expression F (-1):
//     op_ = IDENTITY, value_ = -1.0
//   Expression G (sin(x)):
//     op_ = SIN, subterm_one_ = &L
//   Expression H (2):
//     op_ = IDENTITY, value_ = 2.0
//   Expression I (x^2):
//     op_ = POW, subterm_one_ = &M, subterm_two_ = &N
//   Expression J (3):
//     op_ = IDENTITY, value_ = 3.0
//   Expression K (x):
//     op_ = IDENTITY, var_name_ = "x"
//   Expression L (x):
//     op_ = IDENTITY, var_name_ = "x"
//   Expression M (x):
//     op_ = IDENTITY, var_name_ = "x"
//   Expression N (^2):
//     op_ = IDENTITY, value_ = 2.0
//
// Expressions can also be used to encapsulate Indicator functions and boolean
// expressions. For example:
//   I(X \in [a, b])
// Then:
//   Outermost Expression: I(X \in [a, b]) = I(X >= a) * I(X <= b)
//     op_ = MULT, subterm_one = &A, subterm_two = &B
//   Expression A: I(X >= a):
//     op_ = INDICATOR, subterm_one = &C
//   Expression B: I(X <= b):
//     op_ = INDICATOR, subterm_one = &D
//   Expression C: X >= a:
//     op_ = GTE, subterm_one = &E, subterm_two = &F
//   Expression D: X <= b:
//     op_ = LTE, subterm_one = &G, subterm_two = &H
//   Expression E: X
//     op_ = IDENTITY, name_ = X
//   Expreesion F: a:
//     op_ = IDENTITY, value_ = a
//   Expression G: X
//     op_ = IDENTITY, name_ = X
//   Expreesion H: b:
//     op_ = IDENTITY, value_ = b
// Note: To do X | Y (for binary variables X and Y): X | Y = X + Y - X * Y
struct Expression {
  // The operation to apply to the value (e.g. 'IDENTITY' or 'LOG').
  Operation op_;

  // Indicates the title of the variable name, or empty if this expression is
  // a numeric value (i.e. a constant).
  string var_name_;

  // Indicates the value of this variable.
  double value_;
  // Indicates if this expression is a constant (i.e. ignore var_name_ and use value_).
  bool is_constant_;

  // More complicated equations can be handled by recursively combining Expressions.
  // NOTE: (Pointers to) Expressions coming from subterms should *always* be created
  // on the heap: This is becuase sometimes they need to be (in case we want to
  // generate an expression locally in a function, and use it outside, which would
  // be impossible if pointing to an Expression on the stack); and since sometimes
  // they need to be on the heap, we require that they are always on the heap, so
  // that we're not left with the bad scenario where user doesn't know if they
  // need to call 'delete' on the subterms (since there is no way in C++ to
  // determine if a pointer is heap or stack, see:
  // stackoverflow.com/questions/3230420/how-to-know-if-a-pointer-points-to-the-heap-or-the-stack
  Expression* subterm_one_;
  Expression* subterm_two_;

  Expression() {
    op_ = Operation::IDENTITY;
    var_name_ = "";
    value_ = 0.0;
    is_constant_ = false;
    subterm_one_ = nullptr;
    subterm_two_ = nullptr;
  }
};

// Holds a sub-term of one of the additive terms in the linear regression
// formula. For example, for formula:
//   Y = c_0 + c_1 * Log(X_1) * X_2 + c_2 * X_2
// this struct will represent e.g. Log(X_1), or X_2.
// This struct represents the 'title' of the variable term (just describes the
// structure of the term) and the (self) operation applied to it. For example,
// for term Log(X_1), the term_title_ is "X_1" and op_ is LOG, while for term
// X_2, the term_title_ is "X_2" and op_ is IDENTITY.
struct VariableTerm {
  // Indicates the title of the variable name.
  string term_title_;

  // The operation to apply to the value (e.g. 'IDENTITY' or 'LOG').
  Operation op_;

  // In the case op_ indicates POW or EXP, this holds the power (exponent) to apply.
  double exponent_;

  // In case this variable should be simulated (as opposed to read in from
  // a file), this specifies the parameters for how it should be generated.
  SamplingParameters sampling_params_;
};

// Holds one of the additive terms in the linear regression formula.
// For example, for formula:
//   Y = c_0 + c_1 * Log(X_1) * X_2 + c_2 * X_2
// this struct will represent the first linear term: c_1 * Log(X_1) * X_2,
// or the second linear term c_2 * X_2.
struct LinearTerm {
  // The sub-terms comprising this linear term.
  // For example, if this LinearTerm is: c_1 * Log(X_1) * X_2, then terms_
  // will be of size two, with one variable term for Log(X_1), and one
  // for X_2. Or if the LinearTerm is c_2 * X_2, then terms_ is of size one
  // representing X_2. Or if the LinearTerm is c_0, then terms_ will have
  // size zero.
  vector<VariableTerm> terms_;

  // The operation to combine the sub-terms; only used if terms_.size() > 1.
  // Typically, this should be MULT.
  Operation op_;

  // The constant multiplier for this term.
  double constant_;
};
/* ============================= END Structures ============================= */






/* =============================== Functions ================================ */

// For 2-Term operations, returns the string representation of 'op';
// returns empty string if this is not a 2-Term operation.
extern string GetOpString(const Operation op);
// For 1-Term operations, returns the string representation of the
// provided string surrounded by 'op';
// returns empty string if this is not a 1-Term operation.
extern string GetOpString(const Operation op, const string& argument);

// Returns the string representation of 'term' (considers term.op_
// and term.term_title_).
extern string GetExpressionString(const Expression& expression);

// Returns the string representation of the DataHolder. If the DataHolder's
// type_ is not NUMERIC or STRING, returns empty string. If type_ is
// NUMERIC, will either return name_ if non-empty and use_name is true;
// otherwise it will do Itoa() at the desired precision.
extern string GetDataHolderString(
    const DataHolder& data, const bool use_name, const int precision);
// Same as above, but uses default use_name = false.
extern string GetDataHolderString(const DataHolder& data, const int precision);
// Same as above, using default precision (6 digits) for double values.
extern string GetDataHolderString(const DataHolder& data);

// Rewrites all coefficients to have a multiplication sign, e.g. 2x -> 2 * x.
// Also rewrites xx as x * x.
// DEPRECATED. This used to be only used in ParseExpression(), which no
// longer uses it.
extern void RewriteVariables(const string& var_name, string* term);

inline string StripExtraneousParentheses(const string& input) {
  if (!HasPrefixString(input, "(") ||
      !HasSuffixString(input, ")")) return input;
  return StripExtraneousParentheses(input.substr(1, input.length() - 2));
}

// Given a string that starts with a parentheses, sets closing_pos to
// be the position of the closing parentheses. Returns false if string
// doesn't start with '(', or if closing (matching) parentheses is not found.
extern bool GetClosingParentheses(const string& input, size_t* closing_pos);

extern bool GetLinearTerms(
    const string& input, bool is_first_try, const string& current_term,
    vector<pair<string, bool>>* terms);

extern bool GetMultiplicativeTerms(
    const string& input, const string& current_term,
    vector<pair<string, bool>>* terms);

extern bool GetExponentTerms(
    const string& input, string* first_term, string* second_term);

extern bool GetTermsAroundComma(
    const string& input, string* first_term, string* second_term);

// Returns true if the expression is empty (i.e. was initialized via the
// default constructor, but no fields were set).
extern bool IsEmptyExpression(const Expression& exp);

// Parses a string representation of an expression to an Expression.
// NOTE: This function may create 'new' objects on the heap (the
// subterm_[one | two]_ fields of Expression); the caller is responsible
// for deleting these (e.g. by calling DeleteExpression() below on
// 'expression').
extern bool ParseExpression(
    const bool clean_input,
    const string& term_str,
    const bool enforce_var_names, const set<string>& var_names,
    Expression* expression);
// Same as above, with vector instead of set.
inline bool ParseExpression(
    const bool clean_input,
    const string& term_str,
    const bool enforce_var_names, const vector<string>& var_names,
    Expression* expression) {
  set<string> names;
  for (const string& var_name : var_names) names.insert(var_name);
  return ParseExpression(
             clean_input, term_str, enforce_var_names, names, expression);
}
// Same as above, using "x" as the default variable.
inline bool ParseExpression(
    const string& term_str, const bool enforce_var_names, Expression* expression) {
  set<string> var_names;
  if (enforce_var_names) {
    var_names.insert("x");
  }
  return ParseExpression(
             true, term_str, enforce_var_names, var_names, expression);
}
// Same as above, using "x" as the default variable.
inline bool ParseExpression(const string& term_str, Expression* expression) {
  set<string> var_names;
  var_names.insert("x");
  return ParseExpression(true, term_str, true, var_names, expression);
}

// Returns a copy of the input expression.
// NOTE: This function may create 'new' objects on the heap (the
// subterm_[one | two]_ fields of Expression); the caller is responsible
// for deleting these (e.g. by calling DeleteExpression() below on
// 'expression').
extern Expression CopyExpression(const Expression& expression);
// Same as above, with different API.
extern void CopyExpression(
    const Expression& expression, Expression* new_expression);

// (Iteratively) Deletes the subterms of the passed in expression. The
// expression itself is not deleted, since it may have been created on
// the stack; user is responsible for deleting the outermost expression,
// if they created it using "new".
extern void DeleteExpression(Expression& expression);

// Returns the string representation of 'term' (considers term.op_
// and term.term_title_).
extern string GetTermString(const VariableTerm& term);

// Returns the string representation of the input SamplingParameters. For
// example, if params.type_ = NORMAL and params.mean_ = 0 and params.std_dev_
// is 1.0; then would return: N(0.0, 1.0).
extern string GetSamplingParamsString(const SamplingParameters& params);

// Returns:
//   op(value)
// where op is one of:
//   IDENTITY, EXP, SQRT, POW, or LOG
// In the case of 'POW', uses term.exponent_ as the exponent.
// If an error is encountered, it is printed (via cout), and 0.0 is returned.
extern double ComputeSelfOperation(
    const VariableTerm& term, const double& value);

// Computes:
//   v1 op v2
// and stores the value in output.
// Currently, only supported values for 'op' are MULT and ADD.
extern bool ComputeGroupOperation(
    const Operation op, const double& v1, const double& v2, double* output);

/* ============================= END Functions ============================== */

}  // namespace math_utils

#endif
