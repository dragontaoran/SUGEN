// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)

#include "command_line_utils.h"

#include "FileReaderUtils/read_file_structures.h"
#include "MapUtils/map_utils.h"
#include "MathUtils/data_structures.h"
#include "StringUtils/string_utils.h"

#include <map>
#include <set>
#include <string>
#include <vector>

using namespace map_utils;
using namespace math_utils;
using namespace string_utils;
using namespace std;

namespace file_reader_utils {

const char kInterpolationTypeString[] = "inside";
const char kExtrapolationLeftString[] = "outside_left";
const char kExtrapolationRightString[] = "outside_right";

string GetCommand(const int argc, char* argv[]) {
  vector<string> terms;
  bool next_param_has_quotes = false;
  for (int i = 0; i < argc; ++i) {
    const string quotes = next_param_has_quotes ? "\"" : "";
    const string current_term = string(argv[i]);
    terms.push_back(quotes + current_term + quotes);
    next_param_has_quotes =
        (current_term == "--params" || current_term == "--model" ||
         current_term == "--model_two" || current_term == "--strata" ||
         current_term == "--strata_two" || current_term == "--subgroup" ||
         current_term == "--subgroup_two");
  }
  return Join(terms, " ");
}

bool GetNominalColumnsFromTitles(
    const vector<string>& header, set<int>* nominal_columns) {
  if (nominal_columns == nullptr) return false;
  for (int i = 0; i < header.size(); ++i) {
    if (HasSuffixString(header[i], "$")) {
      nominal_columns->insert(i);
    }
  }
  return true;
}

bool ParseNaStrings(const string& input, set<string>* na_strings) {
  if (na_strings == nullptr) return false;
  Split(StripAllEnclosingPunctuationAndWhitespace(
          input), ",", na_strings);
  return true;
}

bool ParseFileInfo(
    const string& filename, const string& delimiter, const string& comment_char,
    const string& infinity_char, const string& na_strings_str,
    FileInfo* info, string* error_msg) {
  if (info == nullptr) {
    return false;
  }

  info->name_ = filename;
  info->delimiter_ = delimiter;
  info->comment_char_ = comment_char;
  info->infinity_char_ = infinity_char;
  if (!na_strings_str.empty()) {
    return ParseNaStrings(na_strings_str, &(info->na_strings_));
  }
  return true;
}

bool ParseVariableNormalizationParams(
    const string& input,
    map<string, VariableNormalization>* var_name_to_normalization,
    string* error_msg) {
  if (var_name_to_normalization == nullptr) {
    return false;
  }

  const string expression = StripQuotes(RemoveAllWhitespace(input));

  // Variables can be separated by either ";" or ",". Try ";" first, and
  // if only one term is found, try ",".
  const bool has_semicolon = expression.find(";") != string::npos;
  const string separator = has_semicolon ? ";" : ",";
  vector<string> parts;
  Split(expression, separator, &parts);
  if (parts.empty() || parts[0].empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing normalization parameters: "
                    "No parameter specification found.\n";
    }
    return false;
  }

  for (const string& var_part : parts) {
    vector<string> var_parts;
    Split(var_part, "=", &var_parts);
    if (var_parts.size() > 2) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing normalization parameters: Bad format "
                      "for specification of variable normalization.\n";
      }
      return false;
    }
    // Determine which input format (see comments in .h file) was specified
    // for this variable, based on presence of equal sign.
    if (var_parts.size() == 1) {
      var_name_to_normalization->insert(make_pair(
          var_parts[0], VariableNormalization::VAR_NORM_STD_W_N_MINUS_ONE));
    } else {
      // Attempt to parse the normalization type to one of the accepted
      // VariableNormalization types.
      if (EqualsIgnoreCase(var_parts[1], "none")) {
        var_name_to_normalization->insert(make_pair(
            var_parts[0], VariableNormalization::VAR_NORM_NONE));
      } else if (EqualsIgnoreCase(var_parts[1] , "std") ||
                 EqualsIgnoreCase(var_parts[1] , "population_std") ||
                 EqualsIgnoreCase(var_parts[1] , "std_population") ||
                 EqualsIgnoreCase(var_parts[1] , "population_standard") ||
                 EqualsIgnoreCase(var_parts[1] , "standard_population") ||
                 EqualsIgnoreCase(var_parts[1] , "standard")) {
        var_name_to_normalization->insert(make_pair(
            var_parts[0], VariableNormalization::VAR_NORM_STD));
      } else if (EqualsIgnoreCase(var_parts[1], "sample_std") ||
                 EqualsIgnoreCase(var_parts[1], "std_sample") ||
                 EqualsIgnoreCase(var_parts[1], "standard_sample") ||
                 EqualsIgnoreCase(var_parts[1], "sample_standard")) {
        var_name_to_normalization->insert(make_pair(
            var_parts[0], VariableNormalization::VAR_NORM_STD_W_N_MINUS_ONE));
      } else {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in Parsing normalization parameters: Unrecognized "
                        "Variable Normalization type '" + var_parts[1] + "'.\n";
        }
        return false;
      }
    }
  }
  return true;
}

bool ParseInterpolationType(
    const string& input, InterpolationType* type, string* error_msg) {
  if (type == nullptr) {
    return false;
  }

  if (EqualsIgnoreCase(input, "left") ||
      EqualsIgnoreCase(input, "nearest_left")) {
    *type = InterpolationType::IT_LEFT;
    return true;
  } else if (EqualsIgnoreCase(input, "right") ||
             EqualsIgnoreCase(input, "nearest_right")) {
    *type = InterpolationType::IT_RIGHT;
    return true;
  } else if (EqualsIgnoreCase(input, "first") ||
             EqualsIgnoreCase(input, "baseline") ||
             EqualsIgnoreCase(input, "baseline_const") ||
             EqualsIgnoreCase(input, "baseline_constant")) {
    *type = InterpolationType::IT_BASELINE_CONSTANT;
    return true;
  } else if (EqualsIgnoreCase(input, "nearest")) {
    *type = InterpolationType::IT_NEAREST;
    return true;
  } else if (EqualsIgnoreCase(input, "lin_int") ||
             EqualsIgnoreCase(input, "lin_interp") ||
             EqualsIgnoreCase(input, "linear_int") ||
             EqualsIgnoreCase(input, "linear_interp") ||
             EqualsIgnoreCase(input, "linear_interpolation")) {
    *type = InterpolationType::IT_LINEAR_INTERP;
    return true;
  } else {
    if (error_msg != nullptr) {
      *error_msg += "ERROR: Unable to parse '" +
                    input + "' as an ExtrapolationType.\n";
    }
    return false;
  }
}

bool ParseOutsideIntervalParams(
    const string& input,
    ExtrapolationType* type, DataHolder* value, string* error_msg) {
  if (type == nullptr || value == nullptr) {
    return false;
  }

  const string input_stripped =
      StripQuotes(RemoveAllWhitespace(input));

  // If the type is 'ET_CONSTANT', separate out the constant part in parentheses.
  vector<string> type_and_value_sep;
  Split(input_stripped, "(", &type_and_value_sep);
  if (type_and_value_sep.size() != 1 && type_and_value_sep.size() != 2) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing --outside params: Bad format:\n" +
                    input_stripped + "\n";
    }
    return false;
  }

  if (type_and_value_sep.size() == 1) {
    // Handle the non-ET_CONSTANT case.
    const string type_str = ToLowerCase(type_and_value_sep[0]);
    if (type_str == "default" || type_str == "constant" || type_str == "const") {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing --outside params: Bad format: "
                      "specifying 'constant' for the extrapolation type "
                      "requires including the constant value inside "
                      "parentheses.\n" + input_stripped + "\n";
      }
      return false;
    } else if (type_str == "left" ||
               type_str == "leftmost" ||
               type_str == "first") {
      *type = ExtrapolationType::ET_LEFTMOST;
    } else if (type_str == "right" ||
               type_str == "rightmost" ||
               type_str == "last") {
      *type = ExtrapolationType::ET_RIGHTMOST;
    } else {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing --outside params: Unrecognized "
                      "string used to describe Extrapolation Type: '" +
                      type_and_value_sep[0] + "'.\n";
      }
      return false;
    }
  } else {
    // Handle the ET_CONSTANT case.
    const string type_str = ToLowerCase(type_and_value_sep[0]);
    if (type_str != "default" && type_str != "constant" && type_str != "const") {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing --outside params: Bad format: "
                      "only extrapolation type 'constant' for"
                      "the extrapolation type requires including the "
                      "constant value inside parentheses.\n" +
                      input_stripped + "\n";
      }
      return false;
    }
    *type = ExtrapolationType::ET_RIGHTMOST;
    //PHB*type = ExtrapolationType::ET_CONSTANT;
    string value_str;
    if (!StripSuffixString(type_and_value_sep[1], ")", &value_str)) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing --outside params: Bad format: "
                      "specifying 'constant' "
                      "for the extrapolation type requires including the "
                      "constant value inside parentheses.\n" +
                      input_stripped + "\n";
      }
      return false;
    }
    if (value_str.empty()) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing --outside params: Bad format: "
                      "specifying 'constant' "
                      "for the extrapolation type requires including the "
                      "constant value inside parentheses.\n" +
                      input_stripped + "\n";
      }
      return false;
    }
    const string value_str_stripped = StripSingleQuotes(value_str);
    double default_val;
    if (value_str_stripped != value_str ||
        !Stod(value_str_stripped, &default_val)) {
      value->name_ = value_str_stripped;
      value->type_ = DataType::DATA_TYPE_STRING;
    } else {
      value->value_ = default_val;
      value->type_ = DataType::DATA_TYPE_NUMERIC;
    }
  }

  return true;
}

bool ParseTimeDependentParams(
    const string& input,
    map<string, TimeDependentParams>* var_name_to_time_params,
    string* error_msg) {
  if (var_name_to_time_params == nullptr) {
    return false;
  }

  // Separate param specifications for each variable.
  vector<string> vars;
  Split(StripQuotes(RemoveAllWhitespace(input)), ";", &vars);
  if (vars.empty() || vars[0].empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in parsing 'extrapolation' arguments: Empty arguments.\n";
    }
    return false;
  }

  // Go over params for each variable.
  for (const string& var : vars) {
    // Separate variable name from the params.
    vector<string> var_name_sep;
    Split(var, "=", &var_name_sep);
    if (var_name_sep.size() != 2) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in parsing 'extrapolation' arguments: Bad format for "
                      "variable:\n" + var + "\n";
      }
      return false;
    }
    TimeDependentParams& time_params =
        (var_name_to_time_params->insert(make_pair(
            var_name_sep[0], TimeDependentParams()))).first->second;

    // Separate the various kinds of TimeDependentParams specifications
    // (currently just InterpolationType and OutsideIntervalParams).
    vector<string> time_params_categories;
    Split(var_name_sep[1], ",", &time_params_categories);
    for (const string& category : time_params_categories) {
      // Separate the category name from the category params.
      vector<string> category_name_sep;
      Split(category, ":", &category_name_sep);
      if (category_name_sep.size() > 2) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in parsing 'extrapolation' arguments: Bad format "
                        "for variable '" +
                        var_name_sep[0] + "':\n" + category + "\n";
        }
        return false;
      }

      if (category_name_sep.size() == 1) {
        // Only one Type provided, parse it as the InterpolationType.
        if (!ParseInterpolationType(
                category_name_sep[0], &(time_params.interp_type_), error_msg)) {
          return false;
        }
      } else {
        // Process each kind of Param Category.
        const string& category_name = category_name_sep[0];
        if (category_name == kInterpolationTypeString) {
          if (!ParseInterpolationType(
                  category_name_sep[1], &(time_params.interp_type_), error_msg)) {
            return false;
          }
        } else if (category_name == kExtrapolationLeftString) {
          if (!ParseOutsideIntervalParams(
                  category_name_sep[1],
                  &(time_params.outside_params_.outside_left_type_),
                  &(time_params.outside_params_.default_left_val_),
                  error_msg)) {
            return false;
          }
          time_params.outside_params_.left_explicitly_set_ = true;
        } else if (category_name == kExtrapolationRightString) {
          if (!ParseOutsideIntervalParams(
                  category_name_sep[1],
                  &(time_params.outside_params_.outside_right_type_),
                  &(time_params.outside_params_.default_right_val_),
                  error_msg)) {
            return false;
          }
          time_params.outside_params_.right_explicitly_set_ = true;
        } else {
          if (error_msg != nullptr) {
            *error_msg += "ERROR in parsing 'extrapolation' arguments: Bad format "
                          "for variable '" +
                          var_name_sep[0] + "': Unrecognized Param type '" +
                          category_name + "'\n";
          }
          return false;
        }
      }
    }
  }

  return true;
}

bool ParseCollapseParams(
    const string& input, const string& inf_char,
    map<string, vector<VariableCollapseParams>>* var_name_to_collapse_params,
    string* error_msg) {
  if (var_name_to_collapse_params == nullptr) {
    return false;
  }

  const string expression = StripQuotes(RemoveAllWhitespace(input));
  if (expression.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in parsing 'collapse' argument: Empty argument.\n";
    }
    return false;
  }

  // Split into the collapse expressions for each variable.
  vector<string> var_collapse;
  Split(expression, ";", &var_collapse);
  for (const string& var_collapse_expression : var_collapse) {
    // Parse LHS of var_collapse_expression.
    vector<string> var_name_desc_sep;
    Split(var_collapse_expression, "=", &var_name_desc_sep);
    if (var_name_desc_sep.size() != 2) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in parsing 'collapse' argument: Unable to parse component "
                      "on RHS of collapse expression: '" +
                      var_collapse_expression + "'\n";
      }
      return false;
    }
    // Parse Variable name.
    const string& var_name = var_name_desc_sep[0];
    vector<VariableCollapseParams>& collapse_params =
        var_name_to_collapse_params->insert(make_pair(
            var_name, vector<VariableCollapseParams>())).first->second;

    // Parse Variable collapse to/from components.
    //   - Test for first format (bucket).
    if (HasPrefixString(var_name_desc_sep[1], "bucket")) {
      const string num_buckets_str =
          StripParentheses(StripPrefixString(var_name_desc_sep[1], "bucket"));
      int num_buckets;
      if (!Stoi(num_buckets_str, &num_buckets) || num_buckets <= 0) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in parsing 'collapse' argument: Unable to parse "
                        "the number of buckets as a positive integer for collapse "
                        "argument '" + var_name + "'.\n";
        }
        return false;
      }
      collapse_params.push_back(VariableCollapseParams());
      VariableCollapseParams& current_collapse_params = collapse_params.back();
      current_collapse_params.num_buckets_ = num_buckets;
      continue;
    }
    if (HasPrefixString(var_name_desc_sep[1], "round")) {
      const string round_to_str =
          StripParentheses(StripPrefixString(var_name_desc_sep[1], "round"));
      double round_to;
      if (!Stod(round_to_str, &round_to) || round_to <= 0.0) {
        if (error_msg != nullptr) {
          *error_msg += "ERROR in parsing 'collapse' argument: Unable to parse "
                        "the round to as a positive numeric value for collapse "
                        "argument '" + var_name + "'.\n";
        }
        return false;
      }
      collapse_params.push_back(VariableCollapseParams());
      VariableCollapseParams& current_collapse_params = collapse_params.back();
      current_collapse_params.round_to_nearest_ = round_to;
      continue;
    }
    vector<string> to_from_parts;
    Split(var_name_desc_sep[1], "->", &to_from_parts);
    if (to_from_parts.size() != 2) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in parsing 'collapse' argument: Unable to parse component on "
                      "RHS: '" + var_collapse_expression + "'.\n";
      }
      return false;
    }

    // Attempt to parse 'to' value as a numeric value; otherwise, store as string.
    const string to_stripped = StripSingleQuotes(to_from_parts[1]);
    double to_val;
    string to_str = "";
    if (to_stripped != to_from_parts[1] || !Stod(to_stripped, &to_val)) {
      to_str = to_stripped;
    }

    // Parse Variable 'from' components.
    const string from_components = StripBraces(to_from_parts[0]);
    vector<string> from_parts;
    Split(from_components, ",", &from_parts);
    for (const string& from_part : from_parts) {
      collapse_params.push_back(VariableCollapseParams());
      VariableCollapseParams& current_collapse_params = collapse_params.back();
      if (to_str.empty()) {
        current_collapse_params.to_val_ = to_val;
        current_collapse_params.to_type_ = DataType::DATA_TYPE_NUMERIC;
      } else {
        current_collapse_params.to_str_ = to_str;
        current_collapse_params.to_type_ = DataType::DATA_TYPE_STRING;
      }
      if (HasPrefixString(from_part, "[") &&
          HasSuffixString(from_part, "]")) {
        current_collapse_params.from_type_ = DataType::DATA_TYPE_NUMERIC_RANGE;
        // This from component is a range: [a..b]. Parse range.
        const string range_only = StripBrackets(from_part);
        vector<string> range_parts;
        Split(range_only, "..", false, &range_parts);
        if (range_parts.size() != 2) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR in parsing 'collapse' argument: Unable to parse range: '[" +
                          range_only + "].\n";
          }
          return false;
        }
        double a_value;
        if (range_parts[0] == ("-" + inf_char) || range_parts[0] == "-inf") {
          current_collapse_params.from_range_.first =
              -1.0 * numeric_limits<double>::infinity();
        } else if (!Stod(range_parts[0], &a_value)) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR in parsing 'collapse' argument: Unable to parse range: '[" +
                          range_only + "]: Cannot parse '" + range_parts[0] +
                          "' as a numeric value.\n";
          }
          return false;
        } else {
          current_collapse_params.from_range_.first = a_value;
        }
        double b_value;
        if (range_parts[1] == inf_char || range_parts[1] == "inf") {
          current_collapse_params.from_range_.second =
              numeric_limits<double>::infinity();
        } else if (!Stod(range_parts[1], &b_value)) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR in parsing 'collapse' argument: Unable to parse range: '[" +
                          range_only + "]: Cannot parse '" + range_parts[1] +
                          "' as a numeric value.\n";
          }
          return false;
        } else {
          current_collapse_params.from_range_.second = b_value;
        }
        if (current_collapse_params.from_range_.first >
            current_collapse_params.from_range_.second) {
          if (error_msg != nullptr) {
            *error_msg += "ERROR in parsing 'collapse' argument: Unable to parse range: '[" +
                          range_only + "]: first endpoint is greater than the "
                          "second!\n";
          }
          return false;
        }
      } else {
        const string from_stripped = StripSingleQuotes(from_part);
        double from_val;
        if (from_stripped != from_part || !Stod(from_stripped, &from_val)) {
          // This 'from' part is a string.
          current_collapse_params.from_str_ = from_stripped;
          current_collapse_params.from_type_ = DataType::DATA_TYPE_STRING;
        } else {
          current_collapse_params.from_val_ = from_val;
          current_collapse_params.from_type_ = DataType::DATA_TYPE_NUMERIC;
        }
      }
    }
  }

  return true;
}

bool ParseSubgroups(
    const string& subgroup_str, const vector<string>& header,
    vector<VariableColumn>* subgroup_cols, vector<vector<string>>* subgroups,
    set<int>* input_cols_used, string* error_msg) {
  if (subgroup_cols == nullptr || subgroups == nullptr || header.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "Null input to ParseSubgroups.\n";
    }
    return false;
  }

  // Nothing to do if no subgroup to parse.
  if (subgroup_str.empty()) return true;

  // Remove extraneous whitespace.
  const string expression = StripQuotes(RemoveAllWhitespace(subgroup_str));

  // Split subgroup expression around the '=' sign.
  vector<string> subgroup_terms;
  Split(expression, "=", &subgroup_terms);
  if (subgroup_terms.size() != 2) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in parsing subgroup '" +
                    subgroup_str + "': Expected Column Title(s) on LHS of an "
                    "equals sign, and a list of value-tuples on the RHS.\n";
    }
    return false;
  }

  // Parse LHS of subgroup expression.
  vector<string> titles;
  Split(StripParentheses(subgroup_terms[0]), ",", &titles);
  if (titles.empty() || titles[0].empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in parsing subgroups: Empty LHS of "
                    "--subgroup expression.\n";
    }
    return false;
  }
  for (const string& title : titles) {
    subgroup_cols->push_back(VariableColumn());
    VariableColumn& current_subgroup_col = subgroup_cols->back();
    current_subgroup_col.name_ = title;
    for (int i = 0; i < header.size(); ++i) {
      if (header[i] == title) {
        current_subgroup_col.index_ = i;
        input_cols_used->insert(i);
        break;
      }
    }
    // Make sure a match was found.
    if (current_subgroup_col.index_ == -1) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing Subgroups: Unable to find "
                      "subgroup column '" + title + "' among the titles in "
                      "the Header row of the input file:\n\"" +
                      Join(header, ", ") + "\"\n";
      }
      return false;
    }
  }

  // Parse RHS of subgroup expression.
  const int num_expected_subgroups = subgroup_cols->size();
  const string subgroup_rhs_stripped = StripBraces(subgroup_terms[1]);
  if (subgroup_rhs_stripped.empty()) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing Subgroups: No RHS found for subgroup:\n\t" +
                    subgroup_terms[1] + "\n";
    }
    return false;
  }
  vector<string> rhs_terms;
  Split(subgroup_rhs_stripped, "),(", &rhs_terms);
  for (int i = 0; i < rhs_terms.size(); ++i) {
    string& term = rhs_terms[i];
    if (i == 0) {
      term = StripPrefixString(term, "(");
    } else if (i == rhs_terms.size() - 1) {
      term = StripSuffixString(term, ")");
    }
    subgroups->push_back(vector<string>());
    vector<string>& subgroup = subgroups->back();
    Split(term, ",", &subgroup);
    if (subgroup.size() != num_expected_subgroups) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing Subgroups: subgroup '" + term +
                      "' does not have the expected number of column values (" +
                      Itoa(num_expected_subgroups) + ").\n";
      }
      return false;
    }
  }

  // Sanity check all subgroups are distinct.
  set<string> unique_subgroups;
  const string& kDummySeperator = "PHB_FOO_PHB";
  for (const vector<string>& subgroup_terms : *subgroups) {
    const string subgroup_concat =
        Join(subgroup_terms, kDummySeperator);
    if (unique_subgroups.find(subgroup_concat) == unique_subgroups.end()) {
      unique_subgroups.insert(subgroup_concat);
    } else {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing Subgroups: Detected duplicate subgroup "
                      " categories ('" + Join(subgroup_terms, ",") +
                      "') on RHS of --subgroup_model expression: " +
                      subgroup_rhs_stripped + "\n";
      }
      return false;
    }
  }

  return true;
}

bool ParseStrata(
    const string& strata_str, const vector<string>& header,
    set<VariableColumn>* strata_cols, set<int>* input_cols_used,
    string* error_msg) {
  string strata_str_cleaned = StripQuotes(RemoveAllWhitespace(strata_str));
  if (strata_str_cleaned.empty()) return true;

  vector<string> strata_col_names;
  Split(strata_str_cleaned, ",", &strata_col_names);

  for (const string& strata_col : strata_col_names) {
    VariableColumn col;
    col.name_ = strata_col;
    for (int i = 0; i < header.size(); ++i) {
      if (header[i] == strata_col) {
        input_cols_used->insert(i);
        col.index_ = i;
        strata_cols->insert(col);
        break;
      }
    }
    if (col.index_ == -1) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing Strata: Unable to find "
                      "strata column '" + strata_col + "' among the titles in "
                      "the Header row of the input file:\n\"" +
                      Join(header, ", ") + "\"\n";
      }
      return false;
    }
  }

  return true;
}

bool ParseIdWeightAndFamily(
    const string& id_str, const string& weight_str, const string& family_str,
    const vector<string>& header,
    VariableColumn* id_col, VariableColumn* weight_col,
    vector<VariableColumn>* family_cols, set<int>* input_cols_used,
    string* error_msg) {
  // Parse Id.
  string id_str_cleaned = StripQuotes(RemoveAllWhitespace(id_str));
  if (!id_str_cleaned.empty()) {
    id_col->name_ = id_str_cleaned;
    for (int i = 0; i < header.size(); ++i) {
      if (header[i] == id_str_cleaned) {
        input_cols_used->insert(i);
        id_col->index_ = i;
        break;
      }
    }
    if (id_col->index_ == -1) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing Id: Unable to find "
                      "id column '" + id_str_cleaned + "' among the titles in "
                      "the Header row of the input file:\n\"" +
                      Join(header, ", ") + "\"\n";
      }
      return false;
    }
  }

  // Parse Weight.
  string weight_str_cleaned = StripQuotes(RemoveAllWhitespace(weight_str));
  if (!weight_str_cleaned.empty()) {
    weight_col->name_ = weight_str_cleaned;
    for (int i = 0; i < header.size(); ++i) {
      if (header[i] == weight_str_cleaned) {
        input_cols_used->insert(i);
        weight_col->index_ = i;
        break;
      }
    }
    if (weight_col->index_ == -1) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing Id: Unable to find "
                      "weight column '" + weight_str_cleaned + "' among the titles in "
                      "the Header row of the input file:\n\"" +
                      Join(header, ", ") + "\"\n";
      }
      return false;
    }
  }

  // Parse Family.
  string family_str_cleaned = StripQuotes(RemoveAllWhitespace(family_str));
  if (family_str_cleaned.empty()) return true;
  vector<string> family_col_names;
  Split(family_str_cleaned, ",", &family_col_names);
  for (const string& family_col : family_col_names) {
    family_cols->push_back(VariableColumn());
    VariableColumn& col = family_cols->back();
    col.name_ = family_col;
    for (int i = 0; i < header.size(); ++i) {
      if (header[i] == family_col) {
        input_cols_used->insert(i);
        col.index_ = i;
        break;
      }
    }
    if (col.index_ == -1) {
      if (error_msg != nullptr) {
        *error_msg += "ERROR in Parsing Strata: Unable to find "
                      "family column '" + family_col + "' among the titles in "
                      "the Header row of the input file:\n\"" +
                      Join(header, ", ") + "\"\n";
      }
      return false;
    }
  }

  return true;
}

/*
bool ParseMiscellaneousColumns(
    const string& left_truncation_str,
    const vector<string>& header,
    VariableColumn* left_truncation_col,
    set<int>* input_cols_used, string* error_msg) {
  string left_truncation_str_cleaned =
      StripQuotes(RemoveAllWhitespace(left_truncation_str));
  if (left_truncation_str_cleaned.empty()) return true;

  left_truncation_col->name_ = left_truncation_str_cleaned;
  for (int i = 0; i < header.size(); ++i) {
    if (header[i] == left_truncation_str_cleaned) {
      input_cols_used->insert(i);
      left_truncation_col->index_ = i;
      break;
    }
  }

  if (left_truncation_col->index_ == -1) {
    if (error_msg != nullptr) {
      *error_msg += "ERROR in Parsing Left-Truncation Column: Unable to find "
                    "left_truncation column '" + left_truncation_str_cleaned +
                    "' among the titles in the Header row of the input file:\n\"" +
                    Join(header, ", ") + "\"\n";
    }
    return false;
  }

  return true;
}
*/

}  // namespace file_reader_utils
