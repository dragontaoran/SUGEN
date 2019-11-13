// Author: paulbunn@email.unc.edu (Paul Bunn)
// Last Updated: March 2015

#include "string_utils.h"

#include <cstring>     // For memcpy
//#include <iostream>    // For cout
#include <locale>      // For std::locale, std::tolower, and std::toupper.
#include <set>
#include <sstream>     // For Itoa (uses ostringstream).
#include <stdlib.h>    // For atoi, atof.
#include <string>
#include <vector>

using namespace std;

namespace string_utils {

// Returns true if input is a string of length one, representing one
// of the 10 digits (0-9); returns false otherwise.
bool IsStringDigit(const string& input) {
  if (input.length() != 1) return false;
  return (input == "0" || input == "1" || input == "2" || input == "3" ||
          input == "4" || input == "5" || input == "6" || input == "7" ||
          input == "8" || input == "9");
}

// Returns true if input is the exponent part of scientific notation,
// e.g. "E+01"; returns false otherwise.
bool IsScientificNotation(const string& input) {
  if (input.length() < 2) return false;
  if (input.substr(0, 1) != "E" && input.substr(0, 1) != "e") return false;
  string rest = input.substr(1);
  for (unsigned int i = 0; i < rest.length(); ++i) {
    if (i == 0 && (rest.substr(i, 1) == "+" || rest.substr(i, 1) == "-")) {
      continue;
    }
    if (!IsStringDigit(rest.substr(i, 1))) return false;
  }
  return true;
}

// Returns true if str is a string representation of zero (e.g. "0" or "0.0").
bool IsZeroString(const string& input) {
  if (input.empty()) return false;
  const string str = input.substr(0, 1) == "-" ? input.substr(1) : input;
  if (str == "0.0" || str == "0") return true;

  // None of the simple cases above were true, but str may still represent
  // zero, e.g. "0.0000". Instead of testing for every possible length of
  // trailing zeros, we split the string around a decimal point, and test
  // if all characters on either side are '0'.
  vector<string> decimal_parts;
  Split(str, ".", &decimal_parts);
  if (decimal_parts.size() > 2) return false;
  const string& part_one = decimal_parts[0];
  for (unsigned int i = 0; i < part_one.length(); ++i) {
    if (part_one.substr(i, 1) != "0") return false;
  }
  // No decimal, return true.
  if (decimal_parts.size() == 1) return true;

  // Check that all text after decimal are zeros, with one exception that
  // it may be in scientific notation, e.g. 0.00E+00
  const string& part_two = decimal_parts[1];
  for (unsigned int i = 0; i < part_two.length(); ++i) {
    if (part_two.substr(i, 1) == "E" || part_two.substr(i, 1) == "e") {
      return IsScientificNotation(part_two.substr(i));
    }
    if (part_two.substr(i, 1) != "0") return false;
  }

  return true;
}

// Returns true if input is a number, false otherwise. Currently, hex not supported.
bool IsNumeric(const string& input) {
  string str = input;
  
  // Strip enclosing parentheses.
  while (HasPrefixString(str, "(") &&
         HasSuffixString(str, ")")) {
    str = StripPrefixString(str, "(");
    str = StripSuffixString(str, ")");
  }

  // Handle corner-cases
  if (str.empty() || str == "." || str == "+" || str == "-") {
    return false;
  }

  // Move over leading +/- sign.
  if (str.substr(0, 1) == "+" || str.substr(0, 1) == "-") {
    str = str.substr(1);
  }

  // Iterate through all characters of string, returning false if any non-digit
  // is encountered, unless the non-digit is a decimal or scientific notation.
  bool has_decimal = false;
  for (unsigned int i = 0; i < str.length(); ++i) {
    const string current = str.substr(i, 1);

    // Handle potential decimal.
    if (current == ".") {
      if (has_decimal) {
        // A decimal was already found, so this is the second.
        return false;
      }
      has_decimal = true;
      continue;
    }

    // Handle scientific notation.
    if (current == "E" || current == "e") {
      return IsScientificNotation(str.substr(i));
    }

    // Return false for any non-numeric character.
    if (!IsStringDigit(current)) {
      return false;
    }
  }
  return true;
}

bool IsScientificNotation(const string& input, double* value) {
	if (!IsNumeric(input)) return false;

	// First check for "e", e.g. "1.43e-10".
	size_t e_pos = input.find("e");
	if (e_pos != string::npos) {
		if (e_pos == 0 || e_pos == input.length() - 1) {
			return false;
		}
		const string first_half = input.substr(0, e_pos);
		if (IsNumeric(first_half) && IsScientificNotation(input.substr(e_pos))) {
			if (value != nullptr) {
				Stod(input, value);
			}
			return true;
		}
		return false;
	}

	// Also check for "E", e.g. "1.43E10"
	size_t E_pos = input.find("E");
	if (E_pos != string::npos) {
		if (E_pos == 0 || E_pos == input.length() - 1) {
			return false;
		}
		const string first_half = input.substr(0, E_pos);
		const string second_half = input.substr(E_pos + 1);
		if (IsNumeric(first_half) && IsScientificNotation(input.substr(E_pos))) {
			if (value != nullptr) {
				Stod(input, value);
			}
			return true;
		}
		return false;
	}
  
	return false;
}

bool IsScientificNotationWithSign(const string& input, double* value) {
  size_t e_pos = input.find("e");
  size_t E_pos = input.find("E");
  size_t plus_pos = input.find("+");
  size_t neg_pos = input.find("-");
  const bool exactly_one_e =
      (e_pos != string::npos && E_pos == string::npos) ||
      (e_pos == string::npos && E_pos != string::npos);
  const bool exactly_one_sign =
      (plus_pos != string::npos && neg_pos == string::npos) ||
      (plus_pos == string::npos && neg_pos != string::npos);
  if (!exactly_one_e || !exactly_one_sign) return false;
  const size_t exp_pos = e_pos == string::npos ? E_pos : e_pos;
  const size_t sign_pos = plus_pos == string::npos ? neg_pos : plus_pos;
  if (sign_pos != exp_pos + 1) return false;
  return IsScientificNotation(input, value);
}

string RemoveScientificNotation(const string& input) {
  string output = input;
  size_t e_pos = input.find("e");
  size_t E_pos = input.find("E");
  if (e_pos == string::npos && E_pos == string::npos) return output;
  // Find first occurrence of "e" or "E".
  const size_t exp_pos =
      e_pos == string::npos ? E_pos :
      E_pos == string::npos ? e_pos :
      e_pos < E_pos ? e_pos : E_pos;

  // If "e" is the first character of the input string, it cannot represent
  // scientific notation. Look for other potential instances further along
  // in the string.
  if (exp_pos == 0) {
    return input.substr(0, 1) + RemoveScientificNotation(input.substr(1));
  }
  // Ditto for last character, except no more possible matches beyond.
  if (exp_pos == input.length() - 1) return output;

  // Check that character after the "e" is either a digit or "+/-" sign.
  const string& next_char = input.substr(exp_pos + 1, 1);
  if (next_char != "+" && next_char != "-" && !IsStringDigit(next_char)) {
    return (input.substr(0, exp_pos + 1) +
            RemoveScientificNotation(input.substr(exp_pos + 1)));
  }

  // Check that character before the "e" is either a digit or decimal.
  const string& prev_char = input.substr(exp_pos - 1, 1);
  if (prev_char != "." && !IsStringDigit(prev_char)) {
    return (input.substr(0, exp_pos + 1) +
            RemoveScientificNotation(input.substr(exp_pos + 1)));
  }

  // Current "e" indeed is scientific notation (probably, still have potential
  // corner case, e.g. "Hi. E-Mail is good.", which after removing whitespace,
  // has hit ".E-"; so we need to further make sure that there is at least one
  // digit before and after the "e".

  // Find starting position (first digit of scientific notation).
  bool has_digit_before = false;
  bool has_decimal_before = false;
  size_t sci_notation_begin = exp_pos - 1;
  while (true) {
    if (sci_notation_begin < 0) {
      break;
    }
    const string current_char = input.substr(sci_notation_begin, 1);
    if (current_char == ".") {
      // Check if we've already seen a decimal.
      if (has_decimal_before) break;
      has_decimal_before = true;
      sci_notation_begin -= 1;
      continue;
    }
    if (IsStringDigit(current_char)) {
      has_digit_before = true;
      sci_notation_begin -= 1;
      continue;
    } else {
      break;
    }
  }
  // We went one further than the start position, increment back to start.
  sci_notation_begin += 1;

  // Can rule out scientific notation if there wasn't a digit preceding the "e".
  if (has_digit_before == false) {
    return (input.substr(0, exp_pos + 1) +
            RemoveScientificNotation(input.substr(exp_pos + 1)));
  }

  // Now find ending position (last digit of exponent of scientific notation).
  bool has_digit_after = false;
  bool has_pos_sign_after = false;
  size_t sci_notation_end = exp_pos + 1;
  while (true) {
    if (sci_notation_end >= input.length()) {
      break;
    }
    const string current_char = input.substr(sci_notation_end, 1);
    if (current_char == "+" || current_char == "-") {
      // Check that we are at the very next position after the "e".
      if (sci_notation_end != exp_pos + 1) break;
      if (current_char == "+") {
        has_pos_sign_after = true;
      }
      sci_notation_end += 1;
      continue;
    }
    if (IsStringDigit(current_char)) {
      has_digit_after = true;
      sci_notation_end += 1;
      continue;
    } else {
      break;
    }
  }
  // We went one further than the start position, increment back to start.
  sci_notation_end -= 1;

  // Can rule out scientific notation if there wasn't a digit following the "e".
  if (has_digit_after == false) {
    return (input.substr(0, exp_pos + 1) +
            RemoveScientificNotation(input.substr(exp_pos + 1)));
  }

  // This string *does* represent scientific notation. Convert it to a double
  // and back to a string (no longer in scientific notation), then prepend the
  // prefix, and send the suffix for additional checking.
  const string prefix =
      sci_notation_begin == 0 ? "" : input.substr(0, sci_notation_begin);
  const string base =
      input.substr(sci_notation_begin, exp_pos - sci_notation_begin);
  const string exp = has_pos_sign_after ?
      input.substr(exp_pos + 2, sci_notation_end - (exp_pos + 1)) :
      input.substr(exp_pos + 1, sci_notation_end - exp_pos);
  const string suffix = input.substr(sci_notation_end + 1);

  // Sanity-check prefix and suffix of "e" can be parsed as a double.
  double base_value;
  if (!Stod(base, &base_value)) {
    return output;
  }

  return prefix + base + "*10^(" + exp + ")" + RemoveScientificNotation(suffix);
}

// ============================= CONCATENATE =================================
string StrCat(const string& str1, const string& str2) {
  return str1 + str2;
}

string StrCat(
    const string& str1, const string& str2, const string& str3) {
  return str1 + str2 + str3;
}

string StrCat(
    const string& str1, const string& str2, const string& str3,
    const string& str4) {
  return str1 + str2 + str3 + str4;
}

string StrCat(
    const string& str1, const string& str2, const string& str3,
    const string& str4, const string& str5) {
  return str1 + str2 + str3 + str4 + str5;
}

string StrCat(
    const string& str1, const string& str2, const string& str3,
    const string& str4, const string& str5, const string& str6) {
  return str1 + str2 + str3 + str4 + str5 + str6;
}
// =========================== END CONCATENATE =============================== 

// ============================== NUMERIC ====================================
bool Stoi(const string& str, int* i) {
  *i = atoi(str.c_str());
  if (*i == 0 && str != "0") {
    return false;
  }
  return true;
}

bool Stoi(const string& str, int64_t* i) {
  *i = atoi(str.c_str());
  if (*i == 0 && str != "0") {
    return false;
  }
  return true;
}

bool Stoi(const string& str, uint64_t* i) {
  *i = atoi(str.c_str());
  if (*i == 0 && str != "0") {
    return false;
  }
  return true;
}

bool Stoi(const string& str, unsigned int* i) {
  *i = atoi(str.c_str());
  if (*i < 0 || (*i == 0 && str != "0")) {
    return false;
  }
  return true;
}

bool Stod(const string& str, double* d) {
  if (!IsNumeric(str)) {
    return false;
  }

  // Handle fractions, if necessary.
  size_t fraction = str.find("/");
  if (fraction != string::npos && fraction != str.length()) {
    const string numerator = str.substr(0, fraction);
    const string denominator = str.substr(fraction + 1);
    // Only one fraction symbol '/' allowed.
    if (numerator.find("/") != string::npos ||
        denominator.find("/") != string::npos) {
      return false;
    }
    double num, denom;
    if (!Stod(numerator, &num) || !Stod(denominator, &denom)) return false;
    *d = num / denom;
    return true;
  }

  // No fraction symbol.
  *d = atof(str.c_str());
  if (*d == 0.0 && !IsZeroString(str)) {
    return false;
  }
  return true;
}

string Itoa(const int i) {
  std::ostringstream s;
  s << i;
  return s.str();
}

string Itoa(const unsigned int i) {
  std::ostringstream s;
  s << i;
  return s.str();
}

string Itoa(const long& l) {
  std::ostringstream s;
  s << l;
  return s.str();
}

string Itoa(const unsigned long& l) {
  std::ostringstream s;
  s << l;
  return s.str();
}

string Itoa(const float& f) {
  std::ostringstream s;
  s << f;
  return s.str();
}

string Itoa(const float& f, const int precision) {
  std::ostringstream s;
  s.precision(precision);
  s << fixed << f;
  return s.str();
}

string Itoa(const double& d) {
  std::ostringstream s;
  s << d;
  return s.str();
}

string Itoa(const double& d, const int precision) {
  std::ostringstream s;
  s.precision(precision);
  s << fixed << d;
  return s.str();
}

double ParseDouble(const char* input, const int start_byte) {
  double d;
  memcpy(&d, &input[start_byte], sizeof(double));
  return d;
}

float ParseFloat(const char* input, const int start_byte) {
  float f;
  memcpy(&f, &input[start_byte], sizeof(float));
  return f;
}

int ParseInt(const char* input, const int start_byte) {
  int i;
  memcpy(&i, &input[start_byte], sizeof(int));
  return i;
  //return static_cast<int>(input[start_byte]);
}
// ============================ END NUMERIC ================================== 

// =============================== STRIP =====================================
bool HasPrefixString(const string& input, const string& to_match) 
{
	if (to_match.empty()) return true;
	if (input.empty() || input.length() < to_match.length()) return false;
	return input.substr(0, to_match.length()) == to_match;
}

bool StripPrefixString(
    const string& input, const string& to_match, string* output) {
  if (output == NULL) return false;
  if (to_match.empty() || input.empty()) {
    output->assign(input);
    return false;
  }
  if (HasPrefixString(input, to_match)) {
    output->assign(input.substr(to_match.length()));
    return true;
  }
  output->assign(input);
  return false;
}

string StripPrefixString(
    const string& input, const string& to_match) {
  string temp;
  StripPrefixString(input, to_match, &temp);
  return temp;
}

bool HasSuffixString(const string& input, const string& to_match) {
	if (to_match.empty()) return true;
	if (input.empty() || input.length() < to_match.length()) return false;
	return input.substr(input.length() - to_match.length()) == to_match;
}

bool StripSuffixString(
    const string& input, const string& to_match, string* output) {
  if (output == NULL) return false;
  if (to_match.empty() || input.empty()) {
    output->assign(input);
    return false;
  }
  if (HasSuffixString(input, to_match)) {
    output->assign(input.substr(0, input.length() - to_match.length()));
    return true;
  }
  output->assign(input);
  return false;
}

string StripSuffixString(
    const string& input, const string& to_match) {
  string temp;
  StripSuffixString(input, to_match, &temp);
  return temp;
}

void RemoveLeadingWhitespace(const string& input, string* output) {
  if (output == NULL) return;

  int first_non_whitespace_index = -1;
  for (unsigned int i = 0; i < input.length(); ++i) {
    const char c = input[i];
    if (!::isspace(c)) {
      first_non_whitespace_index = i;
      break;
    }
  }

  if (first_non_whitespace_index >= 0) {
    *output = input.substr(first_non_whitespace_index);
  } else {
    *output = "";
  }
}

string RemoveLeadingWhitespace(const string& input) {
  string output;
  RemoveLeadingWhitespace(input, &output);
  return output;
}

void RemoveTrailingWhitespace(const string& input, string* output) {
  if (output == NULL) return;

  int last_non_whitespace_index = -1;
  for (int i = input.length() - 1; i >= 0; --i) {
    const char c = input[i];
    if (!::isspace(c)) {
      last_non_whitespace_index = i + 1;
      break;
    }
  }

  if (last_non_whitespace_index > 0) {
    *output = input.substr(0, last_non_whitespace_index);
  } else {
    *output = "";
  }
}

string RemoveTrailingWhitespace(const string& input) {
  string output;
  RemoveTrailingWhitespace(input, &output);
  return output;
}

void RemoveExtraWhitespace(const string& input, string* output) {
  if (output == NULL) return;
  bool seen_non_whitespace = false;
  bool prev_was_space = false;
  output->clear();
  for (unsigned int i = 0; i < input.length(); ++i) {
    const char c = input[i];
    if (::isspace(c)) {
      // Keep space only if not a leading (prefix) space and
      // not a consecutive space.
      if (seen_non_whitespace && !prev_was_space) {
        (*output) += c;
      }
      prev_was_space = true;
    } else {
      (*output) += c;
      seen_non_whitespace = true;
      prev_was_space = false;
    }
  }
  
  // It's possible there remains one trailing whitespace character.
  // If so, remove it now.
  const int final_pos = output->length() - 1;
  if (::isspace((*output)[final_pos])) {
    output->erase(final_pos);
  }
}

string RemoveExtraWhitespace(const string& input) {
  string output;
  RemoveExtraWhitespace(input, &output);
  return output;
}

void RemoveAllWhitespace(const string& input, string* output)
{
	if (output == NULL) return;
	output->clear();
	for (unsigned int i = 0; i < input.length(); ++i)
	{
		const char c = input[i];
		if (!::isspace(c)) (*output) += c;
	}
}

string RemoveAllWhitespace(const string& input) {
  string output;
  RemoveAllWhitespace(input, &output);
  return output;
}

void Strip(
    const string& input, const string& to_match, string* output) {
  if (output == NULL || to_match.empty() || input.length() < to_match.length()) {
    return;
  }
  output->assign(input);
  size_t itr = input.find(to_match);
  while (itr != string::npos) {
    output->erase(itr, itr + to_match.length());
    itr = output->find(to_match);
  }
}

string StripQuotes(const string& input) {
  if (!HasPrefixString(input, "\"") || !HasSuffixString(input, "\"")) {
    return input;
  }
  return StripPrefixString(StripSuffixString(input, "\""), "\"");
}

string StripSingleQuotes(const string& input) {
  if (!HasPrefixString(input, "'") || !HasSuffixString(input, "'")) {
    return input;
  }
  return StripPrefixString(StripSuffixString(input, "'"), "'");
}

string StripBrackets(const string& input) {
  if (!HasPrefixString(input, "[") || !HasSuffixString(input, "]")) {
    return input;
  }
  return StripPrefixString(StripSuffixString(input, "]"), "[");
}

string StripBraces(const string& input) {
  if (!HasPrefixString(input, "{") || !HasSuffixString(input, "}")) {
    return input;
  }
  return StripPrefixString(StripSuffixString(input, "}"), "{");
}

string StripParentheses(const string& input) {
  if (!HasPrefixString(input, "(") || !HasSuffixString(input, ")")) {
    return input;
  }
  return StripPrefixString(StripSuffixString(input, ")"), "(");
}

string StripAllEnclosingPunctuationAndWhitespace(
    const string& input) {
  return StripBrackets(StripParentheses(StripBraces(
            StripSingleQuotes(StripQuotes(RemoveAllWhitespace(input))))));
}

// ============================= END STRIP ===================================

// ============================ JOIN and SPLIT ===============================
// Vector of Chars.
bool Join(
    const vector<char>& input, const string& delimiter, string* output) {
  if (output == NULL) return false;
  output->clear();
  bool first_element_is_empty_char = false;
  for (const char& itr : input) {
    if (output->empty() && !first_element_is_empty_char) {
      output->assign(&itr, 1);
      if (itr == '\0') first_element_is_empty_char = true;
    } else {
      (*output) += delimiter + itr;
    }
  }
  return true;
}
string Join(const vector<char>& input, const string& delimiter) {
  string to_return;
  Join(input, delimiter, &to_return);
  return to_return;
}
// Vector of Strings.
bool Join(
    const vector<string>& input, const string& delimiter, string* output) {
  if (output == NULL) return false;
  output->clear();
  bool first_element_is_empty_str = false;
  for (const string& itr : input) {
    if (output->empty() && !first_element_is_empty_str) {
      output->assign(itr);
      if (itr == "") first_element_is_empty_str = true;
    } else {
      (*output) += delimiter + itr;
    }
  }
  return true;
}
string Join(const vector<string>& input, const string& delimiter) {
  string to_return;
  Join(input, delimiter, &to_return);
  return to_return;
}
// Vector of Integers.
bool Join(
    const vector<int>& input, const string& delimiter, string* output) {
  if (output == NULL) return false;
  output->clear();
  for (const int& itr : input) {
    if (output->empty()) {
      output->assign(Itoa(itr));
    } else {
      (*output) += delimiter + Itoa(itr);
    }
  }
  return true;
}
string Join(const vector<int>& input, const string& delimiter) {
  string to_return;
  Join(input, delimiter, &to_return);
  return to_return;
}
// Vector of Doubles.
bool Join(
    const vector<double>& input, const string& delimiter, string* output) {
  if (output == NULL) return false;
  output->clear();
  for (const double& itr : input) {
    if (output->empty()) {
      output->assign(Itoa(itr));
    } else {
      (*output) += delimiter + Itoa(itr);
    }
  }
  return true;
}
string Join(const vector<double>& input, const string& delimiter) {
  string to_return;
  Join(input, delimiter, &to_return);
  return to_return;
}
// Vector of Booleans.
bool Join(
    const vector<bool>& input, const string& delimiter, string* output) {
  if (output == NULL) return false;
  output->clear();
  for (const bool& itr : input) {
    if (output->empty()) {
      output->assign(Itoa(itr));
    } else {
      (*output) += delimiter + Itoa(itr);
    }
  }
  return true;
}
string Join(const vector<bool>& input, const string& delimiter) {
  string to_return;
  Join(input, delimiter, &to_return);
  return to_return;
}
// Set of Strings.
bool Join(
    const set<string>& input, const string& delimiter, string* output) {
  if (output == NULL) return false;
  output->clear();
  bool first_element_is_empty_str = false;
  for (const string& itr : input) {
    if (output->empty() && !first_element_is_empty_str) {
      output->assign(itr);
      if (itr == "") first_element_is_empty_str = true;
    } else {
      (*output) += delimiter + itr;
    }
  }
  return true;
}
string Join(const set<string>& input, const string& delimiter) {
  string to_return;
  Join(input, delimiter, &to_return);
  return to_return;
}
// Set of Integers.
bool Join(
    const set<int>& input, const string& delimiter, string* output) {
  if (output == NULL) return false;
  output->clear();
  for (const int& itr : input) {
    if (output->empty()) {
      output->assign(Itoa(itr));
    } else {
      (*output) += delimiter + Itoa(itr);
    }
  }
  return true;
}
string Join(const set<int>& input, const string& delimiter) {
  string to_return;
  Join(input, delimiter, &to_return);
  return to_return;
}
// Set of Doubles.
bool Join(
    const set<double>& input, const string& delimiter, string* output) {
  if (output == NULL) return false;
  output->clear();
  for (const double& itr : input) {
    if (output->empty()) {
      output->assign(Itoa(itr));
    } else {
      (*output) += delimiter + Itoa(itr);
    }
  }
  return true;
}
string Join(const set<double>& input, const string& delimiter) {
  string to_return;
  Join(input, delimiter, &to_return);
  return to_return;
}
// Set of Booleans.
bool Join(
    const set<bool>& input, const string& delimiter, string* output) {
  if (output == NULL) return false;
  output->clear();
  for (const bool& itr : input) {
    if (output->empty()) {
      output->assign(Itoa(itr));
    } else {
      (*output) += delimiter + Itoa(itr);
    }
  }
  return true;
}
string Join(const set<bool>& input, const string& delimiter) {
  string to_return;
  Join(input, delimiter, &to_return);
  return to_return;
}

bool Split(
    const string& input, const string& delimiter, const bool skip_empty,
    vector<string>* output) {
  // Nothing to do if empty input or delimiter.
  if (input.empty() || delimiter.empty()) return true;

  // Sanity check output is not null.
  if (output == nullptr) return false;

  // Iterate through input, splitting at each instance of the delimiter.
  size_t pos = input.find(delimiter);
  string suffix = input;
  while (pos != string::npos) {
    if (pos == 0) {
      if (!skip_empty) output->push_back("");
    } else {
      output->push_back(suffix.substr(0, pos));
    }
    suffix = suffix.substr(pos + delimiter.length());
    pos = suffix.find(delimiter);
  }

  // Store final suffix.
  if (suffix.empty()) {
    if (!skip_empty) output->push_back("");
  } else {
    output->push_back(suffix);
  }

  return true;
}

bool Split(
    const string& input, const set<string>& delimiters, const bool skip_empty,
    vector<string>* output) {
  // Nothing to do if empty input or delimiter.
  if (input.empty() || delimiters.empty()) return true;

  // Sanity check output is not null.
  if (output == nullptr) return false;

  // Iterate through input, splitting at first found delimiter.
  string suffix = input;
  string split_delimiter;
  size_t split_pos, pos;
  while (true) {
    split_delimiter = "";
    split_pos = suffix.length();
    for (const string& delimiter : delimiters) {
      pos = suffix.find(delimiter);
      if (pos != string::npos && pos < split_pos) {
        split_pos = pos;
        split_delimiter = delimiter;
      }
    }
    // No delimiter found. Break.
    if (split_delimiter.empty()) break;
    if (split_pos == 0) {
      if (!skip_empty) output->push_back("");
    } else {
      output->push_back(suffix.substr(0, split_pos));
    }
    suffix = suffix.substr(split_pos + split_delimiter.length());
  }

  // Store final suffix.
  if (suffix.empty()) {
    if (!skip_empty) output->push_back("");
  } else {
    output->push_back(suffix);
  }

  return true;
}

bool Split(
    const string& input, const set<char>& delimiters, const bool skip_empty,
    vector<string>* output) {
  // Nothing to do if empty input or delimiter.
  if (input.empty() || delimiters.empty()) return true;

  // Sanity check output is not null.
  if (output == nullptr) return false;

  // Iterate through input one character at a time, splitting at any delimiter.
  string current_token;
  for (string::const_iterator itr = input.begin();
       itr != input.end(); ++itr) {
    // Check if current character is a delimiter.
    if (delimiters.find(*itr) != delimiters.end()) {
      // Current character is a delimiter.
      // Write current_token to output.
      if (current_token.empty() && !skip_empty) {
        output->push_back("");
      } else if (!current_token.empty()) {
        output->push_back(current_token);
      }
      current_token = "";
    } else {
      // Current character is not a delimiter. Append it to current_token.
      current_token += *itr;
    }
  }

  // Store final token.
  if (!current_token.empty()) {
    output->push_back(current_token);
  }

  return true;
}

bool Split(
    const string& input, const string& delimiter, const bool skip_empty,
    set<string>* output) {
  // Nothing to do if empty input or delimiter.
  if (input.empty() || delimiter.empty()) return true;

  // Sanity check output is not null.
  if (output == nullptr) return false;

  // Iterate through input, splitting at each instance of the delimiter.
  size_t pos = input.find(delimiter);
  string suffix = input;
  while (pos != string::npos) {
    if (pos == 0) {
      if (!skip_empty) output->insert("");
    } else {
      output->insert(suffix.substr(0, pos));
    }
    suffix = suffix.substr(pos + delimiter.length());
    pos = suffix.find(delimiter);
  }

  // Store final suffix.
  if (suffix.empty()) {
    if (!skip_empty) output->insert("");
  } else {
    output->insert(suffix);
  }

  return true;
}

bool Split(
    const string& input, const set<string>& delimiters, const bool skip_empty,
    set<string>* output) {
  // Nothing to do if empty input or delimiter.
  if (input.empty() || delimiters.empty()) return true;

  // Sanity check output is not null.
  if (output == nullptr) return false;

  // Iterate through input, splitting at first found delimiter.
  string suffix = input;
  string split_delimiter;
  size_t split_pos, pos;
  while (true) {
    split_delimiter = "";
    split_pos = suffix.length();
    for (const string& delimiter : delimiters) {
      pos = suffix.find(delimiter);
      if (pos != string::npos && pos < split_pos) {
        split_pos = pos;
        split_delimiter = delimiter;
      }
    }
    // No delimiter found. Break.
    if (split_delimiter.empty()) break;
    if (split_pos == 0) {
      if (!skip_empty) output->insert("");
    } else {
      output->insert(suffix.substr(0, split_pos));
    }
    suffix = suffix.substr(split_pos + split_delimiter.length());
  }

  // Store final suffix.
  if (suffix.empty()) {
    if (!skip_empty) output->insert("");
  } else {
    output->insert(suffix);
  }

  return true;
}

bool Split(
    const string& input, const set<char>& delimiters, const bool skip_empty,
    set<string>* output) {
  // Nothing to do if empty input or delimiter.
  if (input.empty() || delimiters.empty()) return true;

  // Sanity check output is not null.
  if (output == nullptr) return false;

  // Iterate through input one character at a time, splitting at any delimiter.
  string current_token;
  for (string::const_iterator itr = input.begin();
       itr != input.end(); ++itr) {
    // Check if current character is a delimiter.
    if (delimiters.find(*itr) != delimiters.end()) {
      // Current character is a delimiter.
      // Write current_token to output.
      if (current_token.empty() && !skip_empty) {
        output->insert("");
      } else if (!current_token.empty()) {
        output->insert(current_token);
      }
      current_token = "";
    } else {
      // Current character is not a delimiter. Append it to current_token.
      current_token += *itr;
    }
  }

  // Store final token.
  if (!current_token.empty()) {
    output->insert(current_token);
  }

  return true;
}
// ========================== END JOIN and SPLIT =============================

// ============================ MISCELLANEOUS ================================
int CountOccurrences(const string& value, const char target) {
  int count = 0;
  for (unsigned int i = 0; i < value.length(); ++i) {
    if (value[i] == target) ++count;
  }
  return count;
}

int CountOccurrences(const string& value, const string& target) {
  if (target.empty()) return 0;

  int count = 0;
  size_t found_pos = value.find(target);
  string tail = value;
  while(found_pos != string::npos) {
    ++count;
    tail = tail.substr(found_pos + target.length());
    found_pos = tail.find(target);
  }
  return count;
}

string ToLowerCase(const string& input) {
  locale loc;
  string output = "";
  for (string::size_type i = 0; i < input.length(); ++i) {
    output += tolower(input[i], loc);
  }
  return output;
}

string ToUpperCase(const string& input) {
  locale loc;
  string output = "";
  for (string::size_type i = 0; i < input.length(); ++i) {
    output += toupper(input[i], loc);
  }
  return output;
}

bool EqualsIgnoreCase(const string& one, const string& two) {
  return ToLowerCase(one) == ToLowerCase(two);
}

// ========================== END MISCELLANEOUS ==============================

}  // namespace string_utils
