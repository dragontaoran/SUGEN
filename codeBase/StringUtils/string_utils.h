#include <set>
#include <string>
#include <vector>

#ifndef STRING_UTILS_H
#define STRING_UTILS_H

using namespace std;

namespace string_utils {

// ============================= CONCATENATE =================================
extern string StrCat(const string& str1, const string& str2);
extern string StrCat(
    const string& str1, const string& str2, const string& str3);
extern string StrCat(
    const string& str1, const string& str2, const string& str3,
    const string& str4);
extern string StrCat(
    const string& str1, const string& str2, const string& str3,
    const string& str4, const string& str5);
extern string StrCat(
    const string& str1, const string& str2, const string& str3,
    const string& str4, const string& str5, const string& str6);
// =========================== END CONCATENATE ===============================

// ============================ JOIN and SPLIT ===============================
// Join String Containers.
// Joins the elements of 'input' into a single string, seperated by
// 'delimiter'.
extern bool Join(const vector<char>& input, const string& delimiter,
                 string* output);
// Same as above, but returns a string instead of populating an input pointer.
extern string Join(const vector<char>& input, const string& delimiter);
// Same as above with default delimiter space (" ").
inline bool Join(const vector<char>& input, string* output) {
  return Join(input, " ", output);
}
// Same as above with default delimiter space (" ").
inline string Join(const vector<char>& input) {
  return Join(input, " ");
}

// Join String Containers.
// Joins the elements of 'input' into a single string, seperated by
// 'delimiter'.
extern bool Join(const vector<string>& input, const string& delimiter,
                 string* output);
// Same as above, but for set container.
extern bool Join(const set<string>& input, const string& delimiter,
                 string* output);
// Same as above, but returns a string instead of populating an input pointer.
extern string Join(const vector<string>& input, const string& delimiter);
// Same as above, but for set container.
extern string Join(const set<string>& input, const string& delimiter);
// Same as above with default delimiter space (" ").
inline bool Join(const vector<string>& input, string* output) {
  return Join(input, " ", output);
}
// Same as above, but for set container.
inline bool Join(const set<string>& input, string* output) {
  return Join(input, " ", output);
}
// Same as above with default delimiter space (" ").
inline string Join(const vector<string>& input) {
  return Join(input, " ");
}
// Same as above, but for set container.
inline string Join(const set<string>& input) {
  return Join(input, " ");
}

// Join Integer Containers.
extern bool Join(const vector<int>& input, const string& delimiter,
                 string* output);
// Same as above, but for set container.
extern bool Join(const set<int>& input, const string& delimiter,
                 string* output);
// Same as above with default delimiter space (" ").
inline bool Join(const vector<int>& input, string* output) {
  return Join(input, " ", output);
}
// Same as above, but for set container.
inline bool Join(const set<int>& input, string* output) {
  return Join(input, " ", output);
}
// Same as above, but returns string instead of bool.
extern string Join(const vector<int>& input, const string& delimiter);
// Same as above, but for set container.
extern string Join(const set<int>& input, const string& delimiter);
// Same as above with default delimiter space (" ").
inline string Join(const vector<int>& input) {
  return Join(input, " ");
}
// Same as above, but for set container.
inline string Join(const set<int>& input) {
  return Join(input, " ");
}

// Join Double Containers.
extern bool Join(const vector<double>& input, const string& delimiter,
                 string* output);
// Same as above, but for set container.
extern bool Join(const set<double>& input, const string& delimiter,
                 string* output);
// Same as above with default delimiter space (" ").
inline bool Join(const vector<double>& input, string* output) {
  return Join(input, " ", output);
}
// Same as above, but for set container.
inline bool Join(const set<double>& input, string* output) {
  return Join(input, " ", output);
}
// Same as above, but returns string instead of bool.
extern string Join(const vector<double>& input, const string& delimiter);
// Same as above, but for set container.
extern string Join(const set<double>& input, const string& delimiter);
// Same as above with default delimiter space (" ").
inline string Join(const vector<double>& input) {
  return Join(input, " ");
}
// Same as above, but for set container.
inline string Join(const set<double>& input) {
  return Join(input, " ");
}

// Join Boolean Containers.
extern bool Join(const vector<bool>& input, const string& delimiter,
                 string* output);
// Same as above, but for set container.
extern bool Join(const set<bool>& input, const string& delimiter,
                 string* output);
// Same as above with default delimiter space (" ").
inline bool Join(const vector<bool>& input, string* output) {
  return Join(input, " ", output);
}
// Same as above, but for set container.
inline bool Join(const set<bool>& input, string* output) {
  return Join(input, " ", output);
}
// Same as above, but returns string instead of bool.
extern string Join(const vector<bool>& input, const string& delimiter);
// Same as above, but for set container.
extern string Join(const set<bool>& input, const string& delimiter);
// Same as above with default delimiter space (" ").
inline string Join(const vector<bool>& input) {
  return Join(input, " ");
}
// Same as above, but for set container.
inline string Join(const set<bool>& input) {
  return Join(input, " ");
}

// Returns the first index of 'substring' in 'input', returning string::npos
// if not found. This is the same as the standard C++ 'substr' method, except
// that it handles the special input chars: "\t", "\n", and "\s".
extern size_t FindSubstring(const string& input, const string& substring);

// Separates 'input' at each instance of 'delimiter', putting each
// segment into 'output' (optionally omitting empty strings based on
// 'skip_empty'). Note that this respects the following special characters:
//   - "\t": Tab
//   - "\s": Space
extern bool Split(
    const string& input, const string& delimiter, const bool skip_empty,
    vector<string>* output);
// Same as above, with default value for skip_empty = true.
inline bool Split(
    const string& input, const string& delimiter, vector<string>* output) {
  return Split(input, delimiter, true, output);
}
// Same as above, but will split at any delimeter in the set of 'delimiters'.
// NOTE: Unexpected behavior may result if any items in delimiters are a
// substring of another; and even more generally, if there are any overlapping
// characters. For example, if "foo" and "of" are both in delimiters, then
// should split "roofoor" as [ro, oor] or as [roo, r]?
extern bool Split(
    const string& input, const set<string>& delimiters, const bool skip_empty,
    vector<string>* output);
// Same as above, with default value for skip_empty = true.
// NOTE: See Note above about uncertainty in output if delimiters contains
// overlapping strings.
inline bool Split(
    const string& input, const set<string>& delimiters,
    vector<string>* output) {
  return Split(input, delimiters, true, output);
}
// Same as above, but set of delimeters is of type char.
extern bool Split(
    const string& input, const set<char>& delimiters, const bool skip_empty,
    vector<string>* output);
// Same as above, with default value for skip_empty = true.
inline bool Split(
    const string& input, const set<char>& delimiters,
    vector<string>* output) {
  return Split(input, delimiters, true, output);
}
// The below are the same as the above, but they put things in a set
// instead of a vector.
extern bool Split(
    const string& input, const string& delimiter, const bool skip_empty,
    set<string>* output);
inline bool Split(
    const string& input, const string& delimiter, set<string>* output) {
  return Split(input, delimiter, true, output);
}
extern bool Split(
    const string& input, const set<string>& delimiters, const bool skip_empty,
    set<string>* output);
inline bool Split(
    const string& input, const set<string>& delimiters,
    set<string>* output) {
  return Split(input, delimiters, true, output);
}
extern bool Split(
    const string& input, const set<char>& delimiters, const bool skip_empty,
    set<string>* output);
inline bool Split(
    const string& input, const set<char>& delimiters,
    set<string>* output) {
  return Split(input, delimiters, true, output);
}
// ========================== END JOIN and SPLIT =============================

// =============================== STRIP =====================================
// If 'to_match' is a prefix of 'input', then returns true and populates
// output with this prefix removed from input. Otherwise returns false
// and populates output with input.
extern bool StripPrefixString(
    const string& input, const string& to_match, string* output);
// Same above, but different API: returns the string instead of taking in a
// pointer. Returns (copy of) original string if prefix is not found.
extern string StripPrefixString(const string& input, const string& to_match);
// Returns true if 'to_match' is a prefix of 'input'; false otherwise.
extern bool HasPrefixString(const string& input, const string& to_match);
// If 'to_match' is a suffix of 'input', then returns true and populates
// output with this suffix removed from input. Otherwise returns false
// and populates output with input.
extern bool StripSuffixString(
    const string& input, const string& to_match, string* output);
// Same above, but different API: returns the string instead of taking in a
// pointer. Returns (copy of) original string if suffix is not found.
extern string StripSuffixString(const string& input, const string& to_match);
// Returns true if 'to_match' is a suffix of 'input'; false otherwise.
extern bool HasSuffixString(const string& input, const string& to_match);
// Removes Leading whitespace from input and puts resulting string in output.
extern void RemoveLeadingWhitespace(const string& input, string* output);
// Same as above with different API (returns original string if no whitespace)
extern string RemoveLeadingWhitespace(const string& input);
// Removes Trailing whitespace from input and puts resulting string in output.
extern void RemoveTrailingWhitespace(const string& input, string* output);
// Same as above with different API (returns original string if no whitespace)
extern string RemoveTrailingWhitespace(const string& input);
// Removes Leading, Trailing, and Consecutive whitespace from input, and
// puts the resulting string in output.
extern void RemoveExtraWhitespace(const string& input, string* output);
// Same as above with different API (returns original string if no whitespace)
extern string RemoveExtraWhitespace(const string& input);
// Removes all whitespace from input, and puts the resulting string in output.
extern void RemoveAllWhitespace(const string& input, string* output);
// Same as above with different API (returns original string if no whitespace)
extern string RemoveAllWhitespace(const string& input);
// Removes all instances of 'to_match' from input and places result in output.
extern void Strip(
    const string& input, const string& to_match, string* output);
// Removes "" from the start/end of a string (only removes if both are present).
extern string StripQuotes(const string& input);
// Removes '' from the start/end of a string (only removes if both are present).
extern string StripSingleQuotes(const string& input);
// Removes {} from the start/end of a string (only removes if both are present).
extern string StripBraces(const string& input);
// Removes [] from the start/end of a string (only removes if both are present).
extern string StripBrackets(const string& input);
// Removes () from the start/end of a string (only removes if both are present).
extern string StripParentheses(const string& input);
extern string StripAllEnclosingPunctuationAndWhitespace(const string& input);
// ============================= END STRIP ===================================

// ============================== NUMERIC ====================================
// Coverts the input string 'str' to a numeric value (of the appropriate type).
// Returns true if the conversion was successful, false otherwise.
// This function only necessary because C++ has no good way to convert a
// string to an int, and check that the conversion doesn't have errors.
// Note that std::stoi() does this, but is not an option for me, since
// this was introduced in C++11, and this particular function is not
// compatible using MinGW on Windows.
extern bool Stoi(const string& str, int* i);
extern bool Stoi(const string& str, int64_t* i);
extern bool Stoi(const string& str, uint64_t* i);
extern bool Stoi(const string& str, unsigned int* i);

// Coverts the input string 'str' to a double value, storing the result in 'd'.
// Returns true if the conversion was successful, false otherwise.
extern bool Stod(const string& str, double* d);

// Converts an (int, long, double, float) to a string.
extern string Itoa(const int i);
// NOTE: We cannot have the following two, as [u]int64_t are actually not
// concrete types, whose underlying types (long or long long) depend on
// the compilier. Thus, we cannot support both API for input 'long' (or
// 'long long' for that matter) and int64_t, because for a compiler that
// treats int64_t as long, it will have ambiguous overload error.
//PHBextern string Itoa(const int64_t i);
//PHBextern string Itoa(const uint64_t i);
extern string Itoa(const unsigned int i);
extern string Itoa(const long& l);
extern string Itoa(const unsigned long& l);
extern string Itoa(const double& d, const int precision);
// Same as above, with default precision set to 6 digits.
extern string Itoa(const double& d);
extern string Itoa(const float& f, const int precision);
// Same as above, with default precision set to 6 digits.
extern string Itoa(const float& f);

// Reads a char array, and parses various numeric types.
extern double ParseDouble(const char* input, const int start_byte);
extern float ParseFloat(const char* input, const int start_byte);
extern int ParseInt(const char* input, const int start_byte);

// Checks whether the input string represents a numeric value in Scientific
// notation, and if so, returns true and populates value with the value.
extern bool IsScientificNotation(const string& input, double* value);
// Same as above, but demands presence of "+" or "-" after the "e" or "E".
extern bool IsScientificNotationWithSign(const string& input, double* value);
// Replaces any instances of Scientific Notation with the numeric equivalent, e.g.
//   5.1e-05 -> 5.1 * 0.00001
extern string RemoveScientificNotation(const string& input);
// ============================ END NUMERIC ==================================

// ============================ MISCELLANEOUS ================================
// Returns the number of times 'target' appears in 'value'.
extern int CountOccurrences(const string& value, const char target);
// Returns the number of times 'target' appears in 'value'.
extern int CountOccurrences(const string& value, const string& target);

// Returns the string, with all letters converted to lowercase.
extern string ToLowerCase(const string& input);
// Returns the string, with all letters converted to uppercase.
extern string ToUpperCase(const string& input);
// Returns whether the two strings are equal, up to case.
extern bool EqualsIgnoreCase(const string& one, const string& two);
// ========================== END MISCELLANEOUS ==============================

}  // namespace string_utils

#endif
