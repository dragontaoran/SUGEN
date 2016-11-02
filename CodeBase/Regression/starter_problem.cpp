#include "MathUtils/gamma_fns.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Matrix2d;
using Eigen::FullPivLU;
using namespace math_utils;
using namespace std;

// Coverts the input string 'str' to an int value, storing the result in 'i'.
// Returns true if the conversion was successful, false otherwise.
// This function only necessary because C++ has no good way to convert a
// string to an int, and check that the conversion doesn't have errors.
// Note that std::stoi() does this, but is not an option for me, since
// this was introduced in C++11, and this particular function is not
// compatible using MinGW on Windows.
bool Stoi(const string& str, int* i) {
  *i = atoi(str.c_str());
  if (*i == 0 && str != "0") {
    return false;
  }
  return true;
}

// Coverts the input string 'str' to a double value, storing the result in 'd'.
// Returns true if the conversion was successful, false otherwise.
bool Stod(const string& str, double* d) {
  *d = atof(str.c_str());
  if (*d == 0.0 && str != "0.0" && str != "0") {
    return false;
  }
  return true;
}

// Returns info on the appropriate format for the input file.
string PrintInputFormat() {
  return  "Note that input.txt should have format:\n  "
          "\tCASE\tX_1\tX_2\t...\tX_p\tY\n"
          "\t1\tX_1,1\tX_2,1\t...\tX_p,1\tY_1\n"
          "\t2\tX_1,2\tX_2,2\t...\tX_p,2\tY_2\n"
          "\t...\t...\t...\t...\t...\t...\n"
          "\tn\tX_1,n\tX_2,n\t...\tX_p,n\tY_n\n"
          "Where the first row is the 'TITLE' row, "
          "with the first column representing the subject/case number, "
          "the next p columns the title of your p variables, "
          "and the last column the title of the dependent variable; "
          "you can use simply X_1, ..., X_p and Y for the titles in "
          "the first row, if you don't want to explicitly name them; "
          "but the first row MUST be the 'TITLE' row (i.e. it must not "
          "contain the actual data). The next n rows should be filled "
          "with the appropriate data (the first column should simply "
          "mark the subject number, i.e. the first column should count "
          "from 1 to n).\nThe input/output file names cannot contain "
          "spaces.\n";
}

// Prints a message for appropriate usage of this program.
void PrintUsage() {
  cout << "ERROR: Incorrect usage. Proper Usage:\n  "
       << "linear_regression.exe --in /path/to/input.txt "
       << "--out /path/to/output.txt\n\n"
       << PrintInputFormat();
}

// Reads all data values (from file) for input and output variables;
// stores in indep_vars and dep_vars, respectively. Returns true if
// file is successfully parsed; returns false otherwise.
bool ReadInputData(
    ifstream& file, const int& p,
    vector<VectorXd>* indep_vars, vector<double>* dep_vars, char* error_msg) {
  string line, token;
  int line_num = 2;
  while (getline(file, line)) {
    token = "";
    indep_vars->push_back(VectorXd(p - 1));
    VectorXd& indep_values = indep_vars->back();
    int current_col_index = 0;
    // Iterate through line, stopping at white space.
    for (string::iterator itr = line.begin(); itr != line.end(); ++itr) {
      if (*itr != ' ' && *itr != '\t') {
        token += *itr;
      // Sanity check line doesn't have too many entries.
      } else if (current_col_index > p - 1) {
        sprintf(error_msg, "Unable to parse line %d: either line ends in extra whitespace "
                "(space or tab), or too many data values).\n", line_num);
        return false;
      // Store token in indep_values.
      } else if (!token.empty()) {
        // First token on line should simply be the Subject/Case number, which should
        // match the line_num. Sanity check this for the first token.
        if (current_col_index == 0) {
          int case_num;
          if (!Stoi(token, &case_num)) {
            sprintf(error_msg, "First token of line %d should represent Subject/"
                    "Case number, but instead found '%s'.\n",
                    line_num, token.c_str());
            return false;
          } else if (case_num != line_num - 1) {
            sprintf(error_msg, "First entry on line %d should be Case Number (equal to "
                    "%d); found instead %d\n", line_num, line_num - 1, case_num);
            return false;
          }
          // First element of each independent variable vector is always '1',
          // representing the constant term of the linear model.
          indep_values(current_col_index) = 1.0;
          current_col_index++;
          token = "";
          continue;
        }
        // Insert next data value.
        double value;
        if (!Stod(token, &value)) {
          sprintf(error_msg, "Unable to parse '%s' as a (double) value (from line %d).\n",
                  token.c_str(), line_num);
          return false;
        }
        indep_values(current_col_index) = value;
        current_col_index++;
        token = "";
      }
    }
    // Reached end of line. Make sure line had proper number of entries.
    if (token.empty()) {
      sprintf(error_msg, "Unable to parse line %d:  Line ends in extraneous "
              "whitespace (space or tab).\n", line_num);
      return false;
    }
    if (current_col_index != p - 1) {
      sprintf(error_msg, "Unable to parse line: Wrong number of entries on line"
              " %d: expected %d entries (based on TITLE line), but found %d.\n",
              line_num, p, current_col_index + 1);
      return false;
    }
    // Store final token on the line, which represents the value of the
    // dependent variable.
    double y_value;
    if (!Stod(token, &y_value)) {
      sprintf(error_msg, "Unable to parse '%s' as a (double) value (from line %d).\n",
              token.c_str(), line_num);
      return false;
    }
    dep_vars->push_back(y_value);
    token = "";
    ++line_num;
  }
  return true;
}

// Read first line of input file to get Titles of (In)dependent variables.
// Return true if first line was successfully parsed (and the titles will
// appear in order in 'titles'); otherwise return false.
bool GetTitles(ifstream& file, vector<string>* titles) {
  string title_line;
  if (!getline(file, title_line)) {
    return false;
  }
  string temp_title = "";
  for (string::iterator itr = title_line.begin(); itr != title_line.end(); ++itr) {
    if (*itr != ' ' && *itr != '\t') {
      temp_title += *itr;
    } else if (!temp_title.empty()) {
      titles->push_back(temp_title);
      temp_title = "";
    }
  }
  // Store final word on the first line.
  if (!temp_title.empty()) titles->push_back(temp_title);
  return true;
}

// Given input indep_vars and dep_vars, computes the Matrix of Regression
// Coefficients (Beta Hat) via the standard method of using Ordinary Least
// Squares to minimize the sum of squared residuals.
// This function uses Armadillo (a C++ wrapper for LAPACK and BLAS) to do
// matrix computations.
bool ComputeRegressionCoefficients(
    const vector<VectorXd>& indep_vars,
    const vector<double>& dep_vars,
    MatrixXd* inverse_of_indep_vars,
    MatrixXd* regression_coefficients) {
  // The following is safe (no SegFault): Checked indep_vars was non-empty
  // before calling this function.
  const int p = indep_vars[0].size();
  // The following matrix will represent: \sum (X_i * X_i^T)
  MatrixXd sum_of_cov_matrices(p, p);
  sum_of_cov_matrices.setZero();
  // The following (column) vector will represent: \sum (Y_i * X_i)
  VectorXd scaled_indep_vars;
  scaled_indep_vars.resize(p);
  scaled_indep_vars.setZero();
  for (int i = 0; i < indep_vars.size(); ++i) {
    sum_of_cov_matrices += indep_vars[i] * (indep_vars[i].transpose());
    scaled_indep_vars += dep_vars[i] * indep_vars[i];
  }

  // Check that sum_of_cov_matrices is invertible.
  FullPivLU<MatrixXd> lu = sum_of_cov_matrices.fullPivLu();
  if (!lu.isInvertible()) {
    return false;
  }
  *inverse_of_indep_vars = sum_of_cov_matrices.inverse();

  // Compute regression coefficients.
  *regression_coefficients = *inverse_of_indep_vars * scaled_indep_vars;
  return true;
}

// Given input indep_vars, dep_vars, and regression_coefficients, computes
// the variance.
double ComputeVariance(
    const vector<VectorXd>& indep_vars,
    const vector<double>& dep_vars,
    const MatrixXd& regression_coefficients) {
  double variance = 0.0;
  MatrixXd temp_term;
  for (int i = 0; i < indep_vars.size(); ++i) {
    temp_term = regression_coefficients.transpose() * indep_vars[i];
    variance += (dep_vars[i] - temp_term(0, 0)) * (dep_vars[i] - temp_term(0, 0));
  }
  return (variance / (indep_vars.size() - regression_coefficients.size()));
}

// Prints Variable titles together with the estimated regression coefficients
// and covariance; output is print to file indicated by 'outfile'.
bool Print(
    const string& outfile,
    const vector<string>& titles,
    const double& variance,
    const MatrixXd& inverse_of_indep_vars,
    const MatrixXd& regression_coefficients,
    char* error_msg) {
  ofstream out_file;
  out_file.open(outfile.c_str());
  // Write title line.
  out_file << "Variable_Name\t\tEstimate\tVariance\tS.E.E.  \tT-Statistic\tp-value\n";
  char var_name[512] = "";
  for (int i = 0; i < regression_coefficients.size(); ++i) {
    if (i == 0) {
      sprintf(var_name, "Constant (B_0)");
    } else {
      sprintf(var_name, "%s (B_%d)", titles[i].c_str(), i);
    }
    double est_val = regression_coefficients(i, 0);
    double est_var = variance * inverse_of_indep_vars(i, i);
    double see = -1.0;
    if (est_var >= 0.0) {
      see = sqrt(est_var);
    } else {
      sprintf(error_msg, "Negative estimated variance (should never happen): %.04f\n", est_var);
    }
    double test_stat = -1.0;
    if (see > 0.0) {
      test_stat = est_val / see;
    } else {
      sprintf(error_msg, "Negative S.E.E. (should never happen). est_var: %.04f,"
              "S.E.E.: %.04f.\n", est_var, see);
    }
    double p_value = -1.0;
    p_value = RegularizedReverseIncompleteGammaFunction(0.5, (0.5 * test_stat * test_stat));
    char out_line[512] = "";
    sprintf(out_line, "%s\t\t%0.06f\t%0.06f\t%0.06f\t%0.06f\t%0.06f\n",
            var_name, est_val, est_var, see, test_stat, p_value);
    out_file << out_line;
  }
  out_file.close();
  return true;
}

int main (int argc, char* argv[]) {
  // Sanity check command line arguments. Expected format:
  //  $> linear_regression --in foo_input.txt --out foo_output.txt
  if (argc != 5 || string(argv[1]) != "--in" || string(argv[3]) != "--out") {
    PrintUsage();
    return -1;
  }

  // Open and read input file.
  ifstream input_file(argv[2]);
  if (!input_file.is_open()) {
    cout << "ERROR: Unable to open file: " << argv[2]
         << ". Please check path and try again.\n";
    return -1;
  }
  // Read Title line of input file.
  vector<string> titles;
  if (!GetTitles(input_file, &titles) || titles.size() < 4) {
    cout << "ERROR: Improper format of input file.\n"
         << PrintInputFormat();
  }
  // Read the rest (actual data) of input file.
  vector<VectorXd> indep_vars;
  vector<double> dep_vars;
  char error_msg[512] = "";
  if (!ReadInputData(
          input_file, titles.size(), &indep_vars, &dep_vars, error_msg)) {
    cout << "ERROR: Improper format of input file:\n  " << error_msg
         << "\n" << PrintInputFormat();
    return -1;
  }
  if (indep_vars.empty() || indep_vars.size() != dep_vars.size()) {
    cout << "ERROR: Improper format of input file:\n"
         << PrintInputFormat();
    return -1;
  }
  // The number of independent variables (p - 1) must be less than the number
  // of data points (cases), or the system will be under-determined.
  if (indep_vars[0].size() >= indep_vars.size()) {
    cout << "ERROR: Invalid input file: Number of Cases (" << indep_vars.size()
         << ") must exceed the number of independent variables ("
         << indep_vars[0].size() << ").\n"
         << PrintInputFormat();
    return -1;
  }

  // Perform matrix computations to find Regression Coefficients \hat{\Beta}.
  MatrixXd inverse_of_indep_vars, regression_coefficients;
  if (!ComputeRegressionCoefficients(
        indep_vars, dep_vars, &inverse_of_indep_vars, &regression_coefficients)) {
    cout << "ERROR: Ordinary Least Squares cannot be used for the given input "
         << "values: X * X^T is not invertible. Please check input values "
         << "and try again.\n";
    return -1;
  }

  // Perform matrix computations to find (estimated) variance (\hat{\sigma}^2).
  double variance =
      ComputeVariance(indep_vars, dep_vars, regression_coefficients);
  
  // Compute Estimate, (Estimated) Variance, SEE, T-Statistic, and p-value; Print to file.
  if (!Print(
          string(argv[4]), titles, variance, inverse_of_indep_vars, regression_coefficients, error_msg)) {
    cout << "ERROR: Invalid Values computed:\n " << error_msg;
    return -1;
  }

  return 0;
}
