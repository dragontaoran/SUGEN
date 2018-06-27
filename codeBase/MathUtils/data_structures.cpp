// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "data_structures.h"

#include "StringUtils/string_utils.h"

#include <cfloat>   // For DBL_MIN, DBL_MAX
#include <errno.h>
#include <iostream>
#include <math.h>
#include <set>
#include <string.h>
#include <vector>

using namespace std;
using namespace string_utils;

namespace math_utils {

string GetOpString(const Operation op) {
  switch (op) {
    case Operation::ADD:
      return " + ";
    case Operation::SUB:
      return " - ";
    case Operation::MULT:
      return (" * ");
    case Operation::DIV:
      return " / ";
    case Operation::POW:
      return "^";
    default:
      cout << "Unknown operation, or unexpected self-operation: " << op << "\n";
  }
  return "";
}

string GetOpString(const Operation op, const string& argument) {
  if (op == Operation::FACTORIAL) return "(" + argument + ")!";
  else if (op == Operation::ABS) return "|" + argument + "|";
  else if (op == Operation::EXP) return "exp(" + argument + ")";
  else if (op == Operation::LOG) return "log(" + argument + ")";
  else if (op == Operation::SQRT) return "sqrt(" + argument + ")";
  else if (op == Operation::SIN) return "sin(" + argument + ")";
  else if (op == Operation::COS) return "cos(" + argument + ")";
  else if (op == Operation::TAN) return "tan(" + argument + ")";
  else return "";
}

string GetExpressionString(const Expression& expression) {
  // Self-Operations.
  if (expression.op_ == Operation::IDENTITY) {
    if (expression.is_constant_) {
      return Itoa(expression.value_);
    } else {
      return expression.var_name_;
    }
  // 2-term Operations.
  } else if (expression.op_ == ADD || expression.op_ == SUB ||
             expression.op_ == MULT || expression.op_ == DIV ||
             expression.op_ == POW ||
             expression.op_ == INC_GAMMA_FN || expression.op_ == REG_INC_GAMMA_FN ||
             expression.op_ == EQ || expression.op_ == FLOAT_EQ ||
             expression.op_ == GT || expression.op_ == GTE ||
             expression.op_ == LT || expression.op_ == LTE) {
    if (expression.subterm_one_ == nullptr || expression.subterm_two_ == nullptr) {
      return "Unable to perform add/mult/pow: one of the subterms is null.";
    }
    if (expression.op_ == Operation::ADD) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + " + " +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::INC_GAMMA_FN) {
      return
          "IncGamma(" + GetExpressionString(*expression.subterm_one_) + ", " +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::REG_INC_GAMMA_FN) {
      return
          "RegIncGamma(" + GetExpressionString(*expression.subterm_one_) + ", " +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::SUB) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + " - " +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::MULT) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + " * " +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::DIV) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + " / " +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::POW) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + ")^(" +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::EQ) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + ")==(" +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::FLOAT_EQ) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + ")~=(" +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::GT) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + ")>(" +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::GTE) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + ")>=(" +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::LT) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + ")<(" +
          GetExpressionString(*expression.subterm_two_) + ")";
    } else if (expression.op_ == Operation::LTE) {
      return
          "(" + GetExpressionString(*expression.subterm_one_) + ")<=(" +
          GetExpressionString(*expression.subterm_two_) + ")";
    }
  // 1-term Operations.
  } else if (expression.op_ == TAN || expression.op_ == SIN || expression.op_ == COS ||
             expression.op_ == LOG || expression.op_ == EXP || expression.op_ == FACTORIAL ||
             expression.op_ == GAMMA_FN || expression.op_ == PHI_FN ||
             expression.op_ == ABS || expression.op_ == INDICATOR) {
    if (expression.subterm_one_ == nullptr) {
      return "Unable to perform evaluation: subterm_one_ is null";
    }
    if (expression.subterm_two_ != nullptr) {
      return "Unable to perform evaluation: subterm_two_ is non-null";
    }
    if (expression.op_ == Operation::ABS) {
      return "|" + GetExpressionString(*expression.subterm_one_) + "|";
    } else if (expression.op_ == Operation::GAMMA_FN) {
      return "Gamma(" + GetExpressionString(*expression.subterm_one_) + ")";
    } else if (expression.op_ == Operation::PHI_FN) {
      return "Phi(" + GetExpressionString(*expression.subterm_one_) + ")";
    } else if (expression.op_ == Operation::EXP) {
      return "exp(" + GetExpressionString(*expression.subterm_one_) + ")";
    } else if (expression.op_ == Operation::FACTORIAL) {
      return "(" + GetExpressionString(*expression.subterm_one_) + ")!";
    } else if (expression.op_ == Operation::LOG) {
      return "log(" + GetExpressionString(*expression.subterm_one_) + ")";
    } else if (expression.op_ == Operation::SQRT) {
      return "sqrt(" + GetExpressionString(*expression.subterm_one_) + ")";
    } else if (expression.op_ == Operation::SIN) {
      return "sin(" + GetExpressionString(*expression.subterm_one_) + ")";
    } else if (expression.op_ == Operation::COS) {
      return "cos(" + GetExpressionString(*expression.subterm_one_) + ")";
    } else if (expression.op_ == Operation::TAN) {
      return "tan(" + GetExpressionString(*expression.subterm_one_) + ")";
    } else if (expression.op_ == Operation::INDICATOR) {
      return "I(" + GetExpressionString(*expression.subterm_one_) + ")";
    }
  }
  return
      "ERROR: Unsupported operation: " +
      Itoa(static_cast<int>(expression.op_));
}

string GetDataHolderString(const DataHolder& data) {
  if (data.type_ != DataType::DATA_TYPE_STRING &&
      data.type_ != DataType::DATA_TYPE_NUMERIC) {
    return "";
  }

  if (data.type_ == DataType::DATA_TYPE_STRING) {
    return data.name_;
  }

  return Itoa(data.value_);
}

string GetDataHolderString(
    const DataHolder& data, const bool use_name, const int precision) {
  if (data.type_ != DataType::DATA_TYPE_STRING &&
      data.type_ != DataType::DATA_TYPE_NUMERIC) {
    return "";
  }

  if (data.type_ == DataType::DATA_TYPE_STRING ||
      (use_name && !data.name_.empty())) {
    return data.name_;
  }


  return Itoa(data.value_, precision);
}

string GetDataHolderString(const DataHolder& data, const int precision) {
  return GetDataHolderString(data, false, precision);
}

// DEPRECATED.
void RewriteVariables(const string& var_name, string* term) {
  if (term == nullptr) return;
  size_t var_pos = term->find(var_name);
  size_t chars_processed = 0;
  while (var_pos != string::npos) {
    chars_processed += var_pos + 1;
    int numeric_tester;
    bool updated_formula = false;
    if (chars_processed - 1 > var_name.length()) {
      if (term->substr(chars_processed - 1 - var_name.length(),
                       var_name.length()) == var_name) {
        *term = term->substr(0, chars_processed - 1) + "*" +
                term->substr(chars_processed - 1);
        chars_processed += 1;
        updated_formula = true;
      }
    }
    if (!updated_formula && chars_processed > 1) {
      if (Stoi(term->substr(chars_processed - 2, 1),
                            &numeric_tester)) {
        *term = term->substr(0, chars_processed - 1) + "*" +
                term->substr(chars_processed - 1);
        chars_processed += 1;
      }
    }
    if (chars_processed >= term->length()) break;
    var_pos = term->substr(chars_processed).find(var_name);
  }
}

bool GetClosingParentheses(const string& input, size_t* closing_pos) {
  if (input.empty() || input.substr(0, 1) != "(") return false;
  int current_count = 1;
  for (int i = 1; i < input.length(); ++i) {
    if (input.substr(i, 1) == "(") current_count++;
    if (input.substr(i, 1) == ")") current_count--;
    if (current_count == 0) {
      *closing_pos = i;
      return true;
    }
  }
  return false;
}

bool GetLinearTerms(
    const string& input, bool is_first_try, const string& current_term,
    vector<pair<string, bool>>* terms) {
  if (input.empty()) {
    if (!current_term.empty()) {
      terms->push_back(make_pair(current_term, false));
    }
    return true;
  }
  // Don't try to split around a leading minus sign.
  if (is_first_try && input.substr(0, 1) == "-") {
    return GetLinearTerms(input.substr(1), false, "-", terms);
  }
  size_t add_pos = input.find("+");
  size_t neg_pos = input.find("-");
  if (neg_pos < add_pos) add_pos = neg_pos;
  if (add_pos == string::npos) {
    terms->push_back(make_pair(input, false));
    return true;
  }

  // If addition sign lies inside of parentheses, it isn't the addition
  // sign we're looking for.
  size_t parentheses_pos = input.find("(");
  if (parentheses_pos != string::npos && parentheses_pos < add_pos) {
    const string after_parentheses = input.substr(parentheses_pos);
    size_t closing_pos;
    if (!GetClosingParentheses(after_parentheses, &closing_pos)) {
      cout << "ERROR: Unable to GetClosingParentheses for: '"
           << after_parentheses << "'" << endl;
      return false;
    }
    add_pos = after_parentheses.find("+");
    neg_pos = after_parentheses.find("-");
    if (neg_pos < add_pos) add_pos = neg_pos;
    return GetLinearTerms(
        input.substr(parentheses_pos + closing_pos + 1), false,
        (current_term + input.substr(0, parentheses_pos) +
         after_parentheses.substr(0, closing_pos + 1)),
        terms);
  }
  terms->push_back(make_pair(
      current_term + input.substr(0, add_pos), add_pos != neg_pos));
  return GetLinearTerms(input.substr(add_pos + 1), false, "", terms);
}

bool GetMultiplicativeTerms(
    const string& input, const string& current_term,
    vector<pair<string, bool>>* terms) {
  if (input.empty()) {
    if (!current_term.empty()) {
      terms->push_back(make_pair(current_term, false));
    }
    return true;
  }
  size_t mult_pos = input.find("*");
  size_t div_pos = input.find("/");
  if (div_pos < mult_pos) mult_pos = div_pos;
  if (mult_pos == string::npos) {
    terms->push_back(make_pair(input, false));
    return true;
  }

  // If multiplication sign lies inside of parentheses, it isn't the
  // multiplication sign we're looking for.
  size_t parentheses_pos = input.find("(");
  if (parentheses_pos != string::npos && parentheses_pos < mult_pos) {
    const string after_parentheses = input.substr(parentheses_pos);
    size_t closing_pos;
    if (!GetClosingParentheses(after_parentheses, &closing_pos)) {
      cout << "ERROR: Unable to GetClosingParentheses for: '"
           << after_parentheses << "'" << endl;
      return false;
    }
    mult_pos = after_parentheses.find("*");
    div_pos = after_parentheses.find("/");
    if (div_pos < mult_pos) mult_pos = div_pos;
    return GetMultiplicativeTerms(
        input.substr(parentheses_pos + closing_pos + 1),
        (current_term + input.substr(0, parentheses_pos) +
         after_parentheses.substr(0, closing_pos + 1)),
        terms);
  }
  terms->push_back(make_pair(
      current_term + input.substr(0, mult_pos), mult_pos != div_pos));
  return GetMultiplicativeTerms(input.substr(mult_pos + 1), "", terms);
}

bool GetExponentTerms(
    const string& input, string* first_term, string* second_term) {
  size_t exp_pos = input.find("^");
  if (exp_pos == string::npos) {
    *first_term += input;
    return true;
  }

  // If power sign lies inside of parentheses, it may not be the
  // power sign we're looking for.
  size_t parentheses_pos = input.find("(");
  if (parentheses_pos != string::npos && parentheses_pos < exp_pos) {
    const string after_parentheses = input.substr(parentheses_pos);
    exp_pos = after_parentheses.find("^");
    size_t closing_pos;
    if (!GetClosingParentheses(after_parentheses, &closing_pos)) {
      cout << "ERROR: Unable to GetClosingParentheses for: '"
           << after_parentheses << "'" << endl;
      return false;
    }
    *first_term +=
        input.substr(0, parentheses_pos) +
        after_parentheses.substr(0, closing_pos + 1);
    return GetExponentTerms(
        input.substr(parentheses_pos + closing_pos + 1),
        first_term, second_term);
  }
  *first_term += input.substr(0, exp_pos);
  *second_term = input.substr(exp_pos + 1);
  return true;
}

bool GetBooleanTerms(
    const string& input, const string& boolean_sep,
    string* first_term, string* second_term) {
  size_t exp_pos = input.find(boolean_sep);
  if (exp_pos == string::npos) {
    *first_term += input;
    return true;
  }

  // If the boolean separator lies inside of parentheses, it may not be the
  // power sign we're looking for.
  size_t parentheses_pos = input.find("(");
  if (parentheses_pos != string::npos && parentheses_pos < exp_pos) {
    const string after_parentheses = input.substr(parentheses_pos);
    exp_pos = after_parentheses.find(boolean_sep);
    size_t closing_pos;
    if (!GetClosingParentheses(after_parentheses, &closing_pos)) {
      cout << "ERROR: Unable to GetClosingParentheses for: '"
           << after_parentheses << "'" << endl;
      return false;
    }
    *first_term +=
        input.substr(0, parentheses_pos) +
        after_parentheses.substr(0, closing_pos + 1);
    return GetBooleanTerms(
        input.substr(parentheses_pos + closing_pos + 1), boolean_sep,
        first_term, second_term);
  }
  *first_term += input.substr(0, exp_pos);
  *second_term = input.substr(exp_pos + 1);
  return true;
}

bool GetTermsAroundComma(
    const string& input, string* first_term, string* second_term) {
  size_t comma_pos = input.find(",");
  if (comma_pos == string::npos) {
    *first_term += input;
    return true;
  }

  // If comma lies inside of parentheses, it may not be the comma
  // we're looking for.
  size_t parentheses_pos = input.find("(");
  if (parentheses_pos != string::npos && parentheses_pos < comma_pos) {
    const string after_parentheses = input.substr(parentheses_pos);
    comma_pos = after_parentheses.find(",");
    size_t closing_pos;
    if (!GetClosingParentheses(after_parentheses, &closing_pos)) {
      cout << "ERROR: Unable to GetClosingParentheses for: '"
           << after_parentheses << "'" << endl;
      return false;
    }
    *first_term +=
        input.substr(0, parentheses_pos) +
        after_parentheses.substr(0, closing_pos + 1);
    return GetTermsAroundComma(
        input.substr(parentheses_pos + closing_pos + 1),
        first_term, second_term);
  }
  *first_term += input.substr(0, comma_pos);
  *second_term = input.substr(comma_pos + 1);
  return true;
}

bool IsEmptyExpression(const Expression& exp) {
  if (exp.op_ != Operation::IDENTITY) return false;
  return exp.var_name_.empty() && !exp.is_constant_;
}

bool ParseExpression(
    const bool clean_input,
    const string& term_str,
    const bool enforce_var_names, const set<string>& var_names,
    Expression* expression) {
  if (clean_input) {
    // Remove all whitespace.
    string term_str_cleaned = RemoveAllWhitespace(term_str);

    // Replace all scientific notation with actual numbers, e.g.
    //   5.1e-2 -> 5.1 * 0.01
    term_str_cleaned = RemoveScientificNotation(term_str_cleaned);
    return ParseExpression(
               false, term_str_cleaned, enforce_var_names, var_names, expression);
  }
  
  // Try to split formula into two LINEAR terms, if possible. For example:
  //   (2x + 1)(x - 1) - x(2 + x)(x - 1) + ((x + 1)(x - 1)) * sqrt(2x)
  // Would be split as:
  //   (2x + 1)(x - 1), 
  //   -x(2 + x)(x - 1) + ((x + 1)(x - 1)) * sqrt(2x)
  string first_term = "";
  string second_term = "";
  string input = term_str;
  vector<pair<string, bool>> terms;
  if (!GetLinearTerms(input, true, "", &terms)) return false;
  if (terms.size() > 1) {
    Expression* current_expression = expression;
    bool prev_term_was_subtraction = false;
    for (int i = 1; i < terms.size(); ++i) {
      if (terms[i - 1].second) {
        current_expression->op_ =
            prev_term_was_subtraction ? Operation::SUB : Operation::ADD;
      } else {
        current_expression->op_ =
            prev_term_was_subtraction ? Operation::ADD : Operation::SUB;
        prev_term_was_subtraction = !prev_term_was_subtraction;
      }
      current_expression->subterm_one_ = new Expression();
      if (!ParseExpression(
                     false, terms[i - 1].first, enforce_var_names, var_names,
                     current_expression->subterm_one_)) {
        return false;
      }
      current_expression->subterm_two_ = new Expression();
      if (i == terms.size() - 1) {
        // Reached the last term; make it the second term of "current_expression", as
        // opposed to creating a new expression like the 'else' part of this
        // conditional block.
        return ParseExpression(
                   false, terms[i].first, enforce_var_names, var_names,
                   current_expression->subterm_two_);
      } else {
        // Not at the last term; so we'll need to create multiple terms to
        // hold the remaining terms.
        current_expression = current_expression->subterm_two_;
      }
    }
  }

  // Primary formula is not the sum of two (or more) terms. Try splitting it
  // into MULTIPLICATIVE terms.
  terms.clear();
  if (!GetMultiplicativeTerms(input, "", &terms)) return false;
  if (terms.size() > 1) {
    Expression* current_expression = expression;
    bool last_term_was_division = false;
    for (int i = 1; i < terms.size(); ++i) {
      if (terms[i - 1].second) {
        current_expression->op_ =
            last_term_was_division ? Operation::DIV : Operation::MULT;
      } else {
        current_expression->op_ =
            last_term_was_division ? Operation::MULT : Operation::DIV;
        last_term_was_division = !last_term_was_division;
      }
      current_expression->subterm_one_ = new Expression();
      if (!ParseExpression(
                     false, terms[i - 1].first, enforce_var_names, var_names,
                     current_expression->subterm_one_)) {
        return false;
      }
      current_expression->subterm_two_ = new Expression();
      if (i == terms.size() - 1) {
        // Reached the last term; make it the second term of "current_expression", as
        // opposed to creating a new term like the 'else' part of this
        // conditional block.
        return ParseExpression(
                   false, terms[i].first, enforce_var_names, var_names,
                   current_expression->subterm_two_);
      } else {
        // Not at the last term; so we'll need to create multiple terms to
        // hold the remaining terms.
        current_expression = current_expression->subterm_two_;
      }
    }
  }

  // Primary formula is not the product of two (or more) terms. Try splitting it
  // around '^' into base, exponent.
  first_term = "";
  second_term = "";
  if (!GetExponentTerms(input, &first_term, &second_term)) return false;
  if (!second_term.empty()) {
    expression->op_ = Operation::POW;
    expression->subterm_one_ = new Expression();
    expression->subterm_two_ = new Expression();
    return (ParseExpression(
              false, first_term, enforce_var_names, var_names, expression->subterm_one_) &&
            ParseExpression(
              false, second_term, enforce_var_names, var_names, expression->subterm_two_));
  }

  // Primary formula is not a power (base, exponent). Try parsing it as a boolean
  // expression: split around '==', '~=', '>', '>=', '<', '<=".
  vector<pair<string, Operation>> boolean_operators;
  boolean_operators.push_back(make_pair("==", Operation::EQ));
  boolean_operators.push_back(make_pair("~=", Operation::FLOAT_EQ));
  boolean_operators.push_back(make_pair(">", Operation::GT));
  boolean_operators.push_back(make_pair(">=", Operation::GTE));
  boolean_operators.push_back(make_pair("<", Operation::LT));
  boolean_operators.push_back(make_pair("<=", Operation::LTE));
  for (int i = 0; i < boolean_operators.size(); ++i) {
    first_term = "";
    second_term = "";
    const string& boolean_sep = boolean_operators[i].first;
    if (!GetBooleanTerms(input, boolean_sep, &first_term, &second_term)) return false;
    if (!second_term.empty()) {
      expression->op_ = boolean_operators[i].second;
      expression->subterm_one_ = new Expression();
      expression->subterm_two_ = new Expression();
      return (ParseExpression(
                false, first_term, enforce_var_names, var_names, expression->subterm_one_) &&
              ParseExpression(
                false, second_term, enforce_var_names, var_names, expression->subterm_two_));
    }
  }

  // Formula has a single primary term. Check if it is a special function.
  if (HasPrefixString(term_str, "LOG(") ||
      HasPrefixString(term_str, "Log(") ||
      HasPrefixString(term_str, "log(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::LOG;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(4, term_str.length() - 5),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "LN(") ||
      HasPrefixString(term_str, "Ln(") ||
      HasPrefixString(term_str, "ln(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::LOG;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(3, term_str.length() - 4),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "EXP(") ||
             HasPrefixString(term_str, "Exp(") ||
             HasPrefixString(term_str, "exp(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::EXP;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(4, term_str.length() - 5),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "ABS(") ||
             HasPrefixString(term_str, "Abs(") ||
             HasPrefixString(term_str, "abs(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::ABS;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(4, term_str.length() - 5),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "SQRT(") ||
             HasPrefixString(term_str, "Sqrt(") ||
             HasPrefixString(term_str, "sqrt(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::SQRT;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(5, term_str.length() - 6),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "SIN(") ||
             HasPrefixString(term_str, "Sin(") ||
             HasPrefixString(term_str, "sin(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::SIN;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(4, term_str.length() - 5),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "COS(") ||
             HasPrefixString(term_str, "Cos(") ||
             HasPrefixString(term_str, "cos(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::COS;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(4, term_str.length() - 5),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "TAN(") ||
             HasPrefixString(term_str, "Tan(") ||
             HasPrefixString(term_str, "tan(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::TAN;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(4, term_str.length() - 5),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "I(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::INDICATOR;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(2, term_str.length() - 3),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "Phi(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::PHI_FN;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(4, term_str.length() - 5),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "Gamma(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::GAMMA_FN;
    expression->subterm_one_ = new Expression();
    return ParseExpression(
               false, term_str.substr(6, term_str.length() - 7),
               enforce_var_names, var_names, expression->subterm_one_);
  } else if (HasPrefixString(term_str, "IncGamma(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::INC_GAMMA_FN;
    expression->subterm_one_ = new Expression();
    expression->subterm_two_ = new Expression();
    string first_term, second_term;
    string input =
        StripExtraneousParentheses(term_str.substr(9, term_str.length() - 10));
    if (!GetTermsAroundComma(input, &first_term, &second_term)) {
      return false;
    }
    return (ParseExpression(
              false, first_term, enforce_var_names, var_names, expression->subterm_one_) &&
            ParseExpression(
              false, second_term, enforce_var_names, var_names, expression->subterm_two_));
  } else if (HasPrefixString(term_str, "RegIncGamma(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::REG_INC_GAMMA_FN;
    expression->subterm_one_ = new Expression();
    expression->subterm_two_ = new Expression();
    string first_term, second_term;
    string input =
        StripExtraneousParentheses(term_str.substr(12, term_str.length() - 13));
    if (!GetTermsAroundComma(input, &first_term, &second_term)) {
      return false;
    }
    return (ParseExpression(
              false, first_term, enforce_var_names, var_names, expression->subterm_one_) &&
            ParseExpression(
              false, second_term, enforce_var_names, var_names, expression->subterm_two_));
  } else if (HasPrefixString(term_str, "POW(") ||
             HasPrefixString(term_str, "Pow(") ||
             HasPrefixString(term_str, "pow(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    expression->op_ = Operation::POW;
    expression->subterm_one_ = new Expression();
    expression->subterm_two_ = new Expression();
    string first_term, second_term;
    string input =
        StripExtraneousParentheses(term_str.substr(4, term_str.length() - 5));
    if (!GetTermsAroundComma(input, &first_term, &second_term)) {
      return false;
    }
    return (ParseExpression(
              false, first_term, enforce_var_names, var_names, expression->subterm_one_) &&
            ParseExpression(
              false, second_term, enforce_var_names, var_names, expression->subterm_two_));
  }

  // Primary formula has only one term. Try to parse.

  // Check factorial first, since it is the unique time a formula may
  // start with '(' but not end in ')', as in e.g. (n - 1)!.
  if (HasSuffixString(term_str, "!")) {
    if (term_str.length() < 2) return false;
    expression->op_ = Operation::FACTORIAL;
    expression->subterm_one_ = new Expression();
    // Check if character before '!' is a closing parenthesis.
    if (term_str.substr(term_str.length() - 2, 1) == ")") {
      if (!HasPrefixString(term_str, "(")) return false;
      return ParseExpression(
                 false, term_str.substr(1, term_str.length() - 3),
                 enforce_var_names, var_names, expression->subterm_one_);
    }
    return ParseExpression(
               false, term_str.substr(0, term_str.length() - 1),
               enforce_var_names, var_names, expression->subterm_one_);
  }

  // Check for extraneous extra enclosing parentheses.
  if (HasPrefixString(term_str, "(")) {
    if (!HasSuffixString(term_str, ")")) return false;
    return ParseExpression(\
               false, term_str.substr(1, term_str.length() - 2),
               enforce_var_names, var_names, expression);
  }

  // Check for leading negative sign.
  if (HasPrefixString(term_str, "-")) {
    expression->op_ = Operation::MULT;
    expression->subterm_one_ = new Expression();
    expression->subterm_one_->op_ = Operation::IDENTITY;
    expression->subterm_one_->value_ = -1.0;
    expression->subterm_one_->is_constant_ = true;
    expression->subterm_two_ = new Expression();
    return ParseExpression(
               false, term_str.substr(1), enforce_var_names, var_names, expression->subterm_two_);
  }

  // If we've made it here, there is no more reduction that can be done.
  // Check to see if it is a variable name.
  if (var_names.find(term_str) != var_names.end()) {
    expression->op_ = Operation::IDENTITY;
    expression->var_name_ = term_str;
    return true;
  }

  // Check if this term is a coefficient followed by a variable name, e.g. "2x".
  for (const string& var_name : var_names) {
    size_t var_pos = term_str.find(var_name);
    if (var_pos != string::npos) {
      const string non_var = term_str.substr(0, var_pos);
      const string var_str = term_str.substr(var_pos);
      double value;
      if (!Stod(non_var, &value)) return false;
      expression->op_ = Operation::MULT;
      expression->subterm_one_ = new Expression();
      expression->subterm_one_->op_ = Operation::IDENTITY;
      expression->subterm_one_->value_ = value;
      expression->subterm_one_->is_constant_ = true;
      expression->subterm_two_ = new Expression();
      return ParseExpression(
                 false, var_str, enforce_var_names, var_names, expression->subterm_two_);
    }
  }

  // Not a variable name. Try to parse as a numeric value; return false if not.
  double value;
  if (Stod(term_str, &value)) {
    expression->op_ = Operation::IDENTITY;
    expression->value_ = value;
    expression->is_constant_ = true;
    return true;
  }

  // Failed to parse this term. If enforce_var_names is true, return false.
  // Otherwise, just treat this term as a variable name.
  if (enforce_var_names) return false;
  expression->op_ = Operation::IDENTITY;
  expression->var_name_ = term_str;
  return true;
}

void CopyExpression(const Expression& expression, Expression* new_expression) {
  new_expression->op_ = expression.op_;
  new_expression->var_name_ = expression.var_name_;
  new_expression->value_ = expression.value_;
  new_expression->is_constant_ = expression.is_constant_;

  if (expression.subterm_one_ != nullptr) {
    new_expression->subterm_one_ = new Expression();
    CopyExpression(*expression.subterm_one_, new_expression->subterm_one_);
  }
  if (expression.subterm_two_ != nullptr) {
    new_expression->subterm_two_ = new Expression();
    CopyExpression(*expression.subterm_two_, new_expression->subterm_two_);
  }
}

Expression CopyExpression(const Expression& expression) {
  Expression to_return;
  CopyExpression(expression, &to_return);
  return to_return;
}

void DeleteExpression(Expression& expression) {
  if (expression.subterm_one_ != nullptr) {
    DeleteExpression(*expression.subterm_one_);
    delete expression.subterm_one_;
  }
  if (expression.subterm_two_ != nullptr) {
    DeleteExpression(*expression.subterm_two_);
    delete expression.subterm_two_;
  }
}

string GetTermString(const VariableTerm& term) {
  const Operation op = term.op_;
  const string& title = term.term_title_;
  const string exponent =
    term.op_ == Operation::POW ? Itoa(term.exponent_) : "";

  switch (op) {
    case Operation::IDENTITY:
      return title;
    case Operation::EXP:
      return ("exp(" + title + ")");
    case Operation::SQRT:
      return ("sqrt(" + title + ")");
    case Operation::POW:
      return title + "^" + exponent;
    case Operation::LOG:
      return ("Log(" + title + ")");
    default:
      cout << "Unknown operation or unexpected group operation: " << op << "\n";
  }
  return "";
}

string GetSamplingParamsString(const SamplingParameters& params) {
  string range = "";
  if (params.first_sample_ >= 0) {
    if (params.last_sample_ >= 0) {
      range = " on Samples [" + Itoa(params.first_sample_) +
              ", " + Itoa(params.last_sample_) + "]";
    } else {
      range = " on Samples [" + Itoa(params.first_sample_) +
              ", N]";
    }
  } else if (params.last_sample_ >= 0) {
      range =
          " on Samples [1, " + Itoa(params.last_sample_) + "]";
  }

  const string constant = params.constant_ == 1.0 ?
      "" : Itoa(params.constant_) + " * ";
  switch (params.type_) {
    // TODO(PHB): Implement remaining types.
    case Distribution::BERNOULLI: {
      return "BERNOULLI_STRING";
    }
    case Distribution::BINOMIAL: {
      return "BINOMIAL_STRING";
    }
    case Distribution::CAUCHY: {
      return "CAUCHY_STRING";
    }
    case Distribution::CHI_SQUARED: {
      return "CHI_SQUARED_STRING";
    }
    case Distribution::CONSTANT: {
      return "Constant(" + Itoa(params.constant_) + ")" + range;
    }
    case Distribution::NEGATIVE_BINOMIAL: {
      return "NEGATIVE_BINOMIAL_STRING";
    }
    case Distribution::EXPONENTIAL: {
      return "EXPONENTIAL_STRING";
    }
    case Distribution::GAMMA: {
      return "GAMMA_STRING";
    }
    case Distribution::GEOMETRIC: {
      return "GEOMETRIC_STRING";
    }
    case Distribution::NORMAL: {
      return (constant + "N(" + Itoa(params.mean_) + "," +
              Itoa(params.std_dev_) + ")") + range;
    }
    case Distribution::LOG_NORMAL: {
      return "LOG_NORMAL_STRING";
    }
    case Distribution::POISSON: {
      return "POISSON_STRING";
    }
    case Distribution::STUDENT_T: {
      return "STUDENT_T_STRING";
    }
    case Distribution::UNIFORM: {
      return (constant + "U(" + Itoa(params.range_start_) + "," +
              Itoa(params.range_end_) + ")") + range;
    }
    case Distribution::LOG_UNIFORM: {
      return (constant + "log U(" + Itoa(params.range_start_) +
              "," + Itoa(params.range_end_) + ")") + range;
    }
  }
  return "";
}

double ComputeSelfOperation(
    const VariableTerm& term, const double& value) {
  const Operation& op = term.op_;
  const double& exponent = term.exponent_;

  switch (op) {
    case Operation::IDENTITY:
      return value;
    case Operation::EXP:
      return exp(value);
    case Operation::SQRT: {
      if (value < 0.0) {
        cout << "Unable to compute sqrt of a negative value: "
             << value << ". Using 0.0 instead.\n";
        return 0.0;
      }
      return pow(value, 0.5);
    }
    case Operation::POW: {
      errno = 0;
      double pow_value = pow(value, exponent);
      if (errno == ERANGE || errno == EDOM) {
        cout << "Attempting to compute pow(" << value << ", "
             << exponent << ") resulted in error: " << strerror(errno)
             << ". Using 0.0 instead.\n";
        return 0.0;
      }
      return pow_value;
    }
    case Operation::LOG: {
      if (value <= 0.0) {
        cout << "Unable to compute log of a non-positive value: "
             << value << ". Using log(" << value << ") = 0.0.\n";
        return 0.0;
      }
      return log(value);
    }
    default:
      cout << "Unknown operation or unexpected group operation: " << op << "\n";
  }
  return 0.0;
}

bool ComputeGroupOperation(
    const Operation op, const double& v1, const double& v2, double* output) {
  switch (op) {
    case Operation::ADD:
      *output = v1 + v2;
      return true;
    case Operation::MULT:
      *output = v1 * v2;
      return true;
    default:
      cout << "Unknown operation, or unexpected self-operation: " << op << "\n";
  }
  return false;
}

}  // namespace math_utils

