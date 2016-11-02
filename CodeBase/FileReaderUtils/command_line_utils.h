// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description:
//   Utility functions for processing command-line arguments for commonly
//   used parameters.

#include "FileReaderUtils/read_file_structures.h"
#include "MathUtils/data_structures.h"

#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef COMMAND_LINE_UTILS_H
#define COMMAND_LINE_UTILS_H

using namespace math_utils;
using namespace std;

namespace file_reader_utils {

// Strings used to identify these param types.
extern const char kInterpolationTypeString[];
extern const char kOutsideIntervalParamsString[];

// Combines the char* argv arguments into a single string, which is the
// command the user issued to run the program.
extern string GetCommand(const int argc, char* argv[]);

// Populates info with the provided information.
extern bool ParseFileInfo(
    const string& filename, const string& delimiter, const string& comment_char,
    const string& infinity_char, const string& na_strings_str,
    FileInfo* info, string* error_msg);

// Input has one of the following forms:
//   1) Single String:                        NA_STRING_ONE
//   2) Comma-Separated List of Strings:      NA_STRING_ONE,NA_STRING_TWO
//   3) Same as (2), with enclosing brackets: {NA_STRING_ONE,NA_STRING_TWO}
// Input may optionally be surrounded by enclosing quotes.
extern bool ParseNaStrings(const string& input, set<string>* na_strings);

// The index of any term in 'header' that has a '$' suffix is added to
// nominal_columns.
extern bool GetNominalColumnsFromTitles(
    const vector<string>& header, set<int>* nominal_columns);

// There are two valid input formats:
//   1) VAR_1 = VariableNormalization_1; ...; VAR_n = VariableNormalization_n
//   2) VAR_1; VAR_2; ...; VAR_n
// In the former case, we apply the VariableNormalization_i to the indicated
// variable; in the latter case, we apply VAR_NORM_STD_W_N_MINUS_ONE. We
// allow mixtures of the two formats.
extern bool ParseVariableNormalizationParams(
    const string& input,
    map<string, VariableNormalization>* var_name_to_normalization,
    string* error_msg);

// Parses 'input' into one the InterpolationType enum types.
extern bool ParseInterpolationType(
    const string& input, InterpolationType* type, string* error_msg);

// Parses a string representation of ExtrapolationType. String should have format:
//    EXTRAPOLATION_TYPE
// or
//    EXTRAPOLATION_TYPE(double)
// where the latter format is expected iff EXTRAPOLATION_TYPE is ET_CONSTANT.
// In particular, EXTRAPOLATION_TYPE should be one of the keywords: {constant,
// left (or first), right (or last)}. 'value' will be popluated by this function
// iff EXTRAPOLATION_TYPE is 'constant', in which case it will assume the value
// in parentheses.
extern bool ParseOutsideIntervalParams(
    const string& input,
    ExtrapolationType* type, DataHolder* value, string* error_msg);

// Input looks like:
//   "VAR_1 = inside : INTERPOLATION_TYPE(double),
//            outside_left : EXTRAPOLATION_TYPE(double),
//            outside_right : EXTRAPOLATION_TYPE(double);
//    VAR_2 = INTERPOLATION_TYPE(double);
//    VAR_3 = inside : INTERPOLATION_TYPE(double),
//            outside_left : EXTRAPOLATION_TYPE(double),
//            outside_right : EXTRAPOLATION_TYPE(double);
//    ..."
// Where:
//   - Notice there are two formats (VAR_2 has a different format than VAR_1 and
//     VAR_3): If there is only one type, it is assumed to describe the
//     InterpolationType; otherwise, a keyword (inside, outside_left, outside_right)
//     followed by a colon is required
//   - The ';' separates the parameter specifications for the different variables
//   - 'inside', 'outside_left', and 'outside_right' are keywords that identify
//      the category of the params to follow
//   - The ',' separates the three categories mentioned above (inside,
//     outside_left, outside_right). The appearance of each of these
//     is optional (i.e. can specify one, two, or all three).
//   - The ':' separates the category of params from the param values
//   - The INTERPOLATION_TYPE should be one of the keywords {nearest, left,
//     right, constant (or baseline), or lin_interp}; see the InterpolationType
//     these correspond to in FileReaderUtils/read_file_utils.h. The desired
//     constant value should follow in parentheses iff type is 'constant.'
//   - The EXTRAPOLATION_TYPE should be one of the keywords {left,
//     right, or constant}; see the ExtrapolationType these correspond to in
//     FileReaderUtils/read_file_utils.h. The desired constant value should
//     follow in parentheses iff type is 'constant.'
extern bool ParseTimeDependentParams(
    const string& input,
    map<string, TimeDependentParams>* var_name_to_time_params,
    string* error_msg);

// Input has one of the following forms:
//   VAR_1 = bucket(N)
//   VAR_1 = round(VALUE)
//   VAR_1 = {0, [2.4..inf], 6} -> 1.5; VAR_1 = {foo} -> bar; VAR_2 = {None} -> '0'
// Where:
//   - The top form takes positive integer values N, and partitions the (min, max)
//     range of values for this variable into N buckets, sending each data value
//     to the nearest bucket (partition point).
//   - The middle form takes a double value, and rounds each data value to the
//     closest multiple of that value. Warning: If VALUE doesn't exactly divide
//     1.0, this may not do what is expected. For example, if VALUE is 0.33 and
//     a data value is 3.34, then this rounds to 3.30 rather than 3.33, since
//     the latter is not a multiple of 0.33.
//   - For the last form:
//     - Input may optionally be surrounded by enclosing quotes
//     - Strings can optionally be surrounded by SINGLE quotes
//     - Values will be treated as numeric if possible, unless surrounded by
//       single quotes, in which case they are treated as strings (hence the
//       zero is surrounded in single quotes at the end, emphasizing to collapse
//       "None" to the *string* 0, not the numeric value of 0.
//     - Spaces are optional, and will be ignored
//     - Numeric ranges are possible, and have format: [a..b]. You can use
//       "-inf_char" (resp. "inf_char") for b (resp. a), which will be treated
//       as +/- infinity.
extern bool ParseCollapseParams(
    const string& input, const string& inf_char,
    map<string, vector<VariableCollapseParams>>* var_name_to_collapse_params,
    string* error_msg);

// Takes in a string expression for the subgroup(s) and a header that
// contains the column names/titles (in the correct order), and outputs
// the columns that should be used to determine the subgroups, and the
// various values that define each subgroup.
// Format for subgroup_str:
//   "(COL_NAME_1, COL_NAME_2, ..., COL_NAME_n) = {
//       (VAL_1,1, VAL_1,2, ..., VAL_1,n),
//       (VAL_2,1, VAL_2,2, ..., VAL_2,n),
//       ...
//       (VAL_m,1, VAL_m,2, ..., VAL_m,n)}"
// Note that the '*' character is a special character that indicates ANY
// value for that column applies; e.g. if (GENDER, AGE) is the LHS of the input,
// and one of the elements on the RHS is (*, 37), then both (M, 37) and (F, 37)
// will be put into the same subgroup.
// For the above input, 'subgroup_cols' will have length n (position i will
// indicate the column index of COL_NAME_i), and 'subgroups' will have length
// m, with index j representing the j^th subgroup; so each index j of 'subgroups'
// will be a vector of length n, and represents the values for each column that
// indicate membership in that subgroup.
extern bool ParseSubgroups(
    const string& subgroup_str, const vector<string>& header,
    vector<VariableColumn>* subgroup_cols, vector<vector<string>>* subgroups,
    set<int>* input_cols_used, string* error_msg);

// Parses the --strata string, which is just a comma-separated list of the
// column name(s) that should be used to determine strata. Both the column
// name and index is populated.
extern bool ParseStrata(
    const string& strata_str, const vector<string>& header,
    set<VariableColumn>* strata_cols, set<int>* input_cols_used,
    string* error_msg);

// Parses the --id_col, --weight_col, and --family_cols strings.
extern bool ParseIdWeightAndFamily(
    const string& id_str, const string& weight_str, const string& family_str,
    const vector<string>& header,
    VariableColumn* id_col, VariableColumn* weight_col,
    vector<VariableColumn>* family_cols,
    set<int>* input_cols_used, string* error_msg);

// Parses any additional columns, currently none.
/*
extern bool ParseMiscellaneousColumns(
    const string& left_truncation_str,
    const vector<string>& header,
    VariableColumn* left_truncation_col,
    set<int>* input_cols_used, string* error_msg);
*/
  
}  // namespace file_reader_utils

#endif
