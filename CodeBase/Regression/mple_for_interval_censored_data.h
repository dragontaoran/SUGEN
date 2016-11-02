// Date: Nov 2015
// Author: paulbunn@email.unc.edu (Paul Bunn)
//
// Description:
//   Performs Maximum-Likelihood Estimation for Transformation Models with
//   Interval Censored Data.
//   More specifically, takes in a TimeDepIntervalCensoredData object
//   (that represents the data, in a special format), and runs the EM
//   algorithm for parameter estimation to simultaneously estimate \hat{\beta}
//   and \hat{\lambda}.

#include "FileReaderUtils/read_time_dep_interval_censored_data.h"

#include "MathUtils/data_structures.h"
#include "MathUtils/gaussian_quadrature.h"
#include "TestUtils/test_utils.h"

#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef MPLE_FOR_INTERVAL_CENSORED_DATA_H
#define MPLE_FOR_INTERVAL_CENSORED_DATA_H

using Eigen::MatrixXd;
using Eigen::VectorXd;
using file_reader_utils::TimeDepIntervalCensoredData;
using file_reader_utils::SubjectInfo;
using test_utils::Timer;
using namespace std;
using namespace math_utils;

namespace regression {

// Forward declare, so can make it a friend class.
class MultivariateMpleForIntervalCensoredData;
class ClusteredMpleForIntervalCensoredData;

enum MpleReturnValue {
  SUCCESS,
  FAILED_BAD_INPUT,
  FAILED_PRELIMINARY_COMPUTATION,
  FAILED_E_STEP,
  FAILED_M_STEP,
  FAILED_MAX_ITR,
  FAILED_VARIANCE,
  FAILED_NEGATIVE_VARIANCE,
};

class MpleForIntervalCensoredData {
 friend class MultivariateMpleForIntervalCensoredData;
 friend class ClusteredMpleForIntervalCensoredData;
 public:
  // Constructor.
  MpleForIntervalCensoredData() {
    logging_on_ = true;
    force_one_right_censored_ = true;
    r_ = -1.0;
    convergence_threshold_ = 0.0001;
    h_n_constant_ = 5;
    max_itr_ = 1000;
    num_gaussian_laguerre_points_ = 40;
    integral_constant_factor_ = 0.0;
  }
  // Constructor that should be used before any non-static call to
  // PerformEmAlgorithmForParameterEstimation();
  MpleForIntervalCensoredData(
      const bool logging_on, const bool force_one_right_censored,
      const double& r, const double& convergence_threshold,
      const int h_n_constant, const int max_itr,
      const int num_gaussian_laguerre_points) {
    logging_on_ = logging_on;
    force_one_right_censored_ = force_one_right_censored;
    r_ = r;
    convergence_threshold_ = convergence_threshold;
    h_n_constant_ = h_n_constant;
    max_itr_ = max_itr;
    num_gaussian_laguerre_points_ = num_gaussian_laguerre_points;
  }

  int GetHn() { return h_n_constant_; }
  void SetHn(const int h_n_constant) { h_n_constant_ = h_n_constant; } 

  // Runs the E-M Algorithm to solve simultaneously for solutions \beta and
  // \lambda to a non-parametric transformation model with interval-censored data.
  // Member fields that have already been set:
  //   r_, convergence_threshold_, h_n_constant_, max_itr_,
  //   distinct_times_, lower_time_bounds_, upper_time_bounds_, x_
  MpleReturnValue PerformEmAlgorithmForParameterEstimation(
      int* num_iterations, double* log_likelihood,
      VectorXd* final_beta, VectorXd* final_lambda, MatrixXd* variance);
  // Same as above, for static use.  
  static MpleReturnValue PerformEmAlgorithmForParameterEstimation(
      const double& r, const double& convergence_threshold,
      const int h_n_constant, const int max_itr,
      const set<double>& distinct_times,               // Size M
      const vector<double>& lower_time_bounds,         // Size n
      const vector<double>& upper_time_bounds,         // Size n
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Size n, VectorXd size p_indep,
                                                       // MatrixXd dim (p_dep, M)
      int* num_iterations, double* log_likelihood,
      VectorXd* final_beta, VectorXd* final_lambda, MatrixXd* variance);

  // Compute Variance.
  MpleReturnValue ComputeVariance(
      const VectorXd& beta, const VectorXd& lambda,
      MatrixXd* variance);
  // Same as above, but static (so it needs to get everything passed-in,
  // rather than using non-static member fields).
  static MpleReturnValue ComputeVariance(
      const Expression& transformation_G,
      const Expression& transformation_G_prime,
      const double& r,
      const double& integral_constant_factor,
      const double& convergence_threshold,
      const int h_n_constant, const int max_itr,
      const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
      const set<double>& distinct_times,               // Size M
      const vector<double>& lower_time_bounds,         // Size n
      const vector<double>& upper_time_bounds,         // Size n
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Size n, VectorXd size p_indep,
                                                       // MatrixXd dim (p_dep, M)
      const vector<vector<bool>>& r_i_star_indicator,  // n x M
      const VectorXd& final_beta,                      // Size p
      const VectorXd& final_lambda,                    // Size M
      MatrixXd* variance);                             // Dim (p, p)

  // Prints output.
  static bool PrintOutput(
      const string& output_file,
      const vector<string>& legend,
      const set<double>& distinct_times,
      const VectorXd& beta, const VectorXd& lambda,
      const MatrixXd& variance);
  // Same as above, but for beta and lambda only (no Covariance matrix).
  static bool PrintOutput(
      const string& output_file,
      const vector<string>& legend,
      const set<double>& distinct_times,
      const VectorXd& beta, const VectorXd& lambda) {
    MatrixXd junk;
    junk.resize(0, 0);
    return PrintOutput(output_file, legend, distinct_times, beta, lambda, junk);
  }

  // Transfers the data from TimeDepIntervalCensoredData to the Mple
  // private fields x_, lower_time_bounds_, upper_time_bounds_,
  // distinct_times_, and time_indep_vars_.
  bool InitializeData(const TimeDepIntervalCensoredData& data);

  // Initialize fields the Multivariate MPLE doesn't have, so that it can call
  // MpleForIntervalCensoredData::ComputeVariance in case K = 1.
  static bool InitializeInput(
      const double& r, 
      double* integral_constant_factor,
      Expression* transformation_G, Expression* transformation_G_prime);

  // Temp functions for time benchmarks.
  static void InitializeTimers();
  static void PrintTimers();
  static void SetConvergenceThreshold(const double& threshold) {
    convergence_threshold_ = threshold;
  }
  static void SetMaxItr(const int max_itr) {
    max_itr_ = max_itr;
  }
  static void SetLoggingOn(const bool logging_on) {
    logging_on_ = logging_on;
  }
  static void SetForceOneRightCensored(bool force_one_right_censored) {
      force_one_right_censored_ = force_one_right_censored;
  }
  static void SetNoUsePositiveDefiniteVariance(bool no_use_pos_def_var) {
    no_use_pos_def_variance_ = no_use_pos_def_var;
  }
  set<double> GetDistinctTimes() {
    return distinct_times_;
  }

 private:
  // Static Member fields.
  static bool logging_on_;
  static bool force_one_right_censored_;
  static bool no_use_pos_def_variance_;
  static double convergence_threshold_;
  static int max_itr_;
  // Non-Static Member fields.
  double r_;
  double integral_constant_factor_;
  int h_n_constant_;
  int num_gaussian_laguerre_points_;
  vector<GaussianQuadratureTuple> gaussian_laguerre_points_;
  Expression transformation_G_;
  Expression transformation_G_prime_;
  set<double> distinct_times_;          
  vector<double> lower_time_bounds_;    
  vector<double> upper_time_bounds_;    
  vector<vector<bool>> time_indep_vars_;  // Outer vector size n, inner vector size p.
  vector<pair<VectorXd, MatrixXd>> x_;  
  vector<vector<MatrixXd>> x_x_transpose_;
  vector<vector<bool>> r_i_star_indicator_;

  // Temp field for time benchmarks.
  static vector<Timer> timers_;

  // Internal method for Performing EM Algorithm. Caller should use
  // the public versions above.
  static MpleReturnValue PerformEmAlgorithmForParameterEstimation(
      const double& r, const double& convergence_threshold,
      const double& integral_constant_factor,
      const int h_n_constant, const int max_itr,
      const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
      const Expression& transformation_G,
      const Expression& transformation_G_prime,
      const set<double>& distinct_times,               // Size M
      const vector<double>& lower_time_bounds,         // Size n
      const vector<double>& upper_time_bounds,         // Size n
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Size n, VectorXd size p_indep,
                                                       // MatrixXd dim (p_dep, M)
      const vector<vector<MatrixXd>>& x_x_transpose,   // n x M x (p, p) (inner vector has size
                                                       // 1 instead of M if all cov time-indep
      const vector<vector<bool>>& r_i_star_indicator,  // n x M
      int* num_iterations, double* log_likelihood,
      VectorXd* final_beta, VectorXd* final_lambda, MatrixXd* variance);

  // Compute constants that will be used throughout the E-M algorithm.
  // Prior to calling, the following member fields should already have been set:
  //   r_ distinct_times_, lower_time_bounds_, upper_time_bounds_, x_
  bool InitializeInput();
  // Same as above but for static use.
  static bool InitializeInput(
      const int num_gauss_laguerre_points,
      const double& r, 
      const set<double>& distinct_times,               // Size M
      const vector<double>& lower_time_bounds,         // Size n
      const vector<double>& upper_time_bounds,         // Size n
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Size n, VectorXd size p_indep,
                                                       // MatrixXd dim (p_dep, M)
      double* integral_constant_factor,
      vector<GaussianQuadratureTuple>* gaussian_laguerre_points,
      Expression* transformation_G, Expression* transformation_G_prime,
      vector<vector<MatrixXd>>* x_x_transpose,
      vector<vector<bool>>* r_i_star_indicator);

  // Finds the weights and abscissa (knots) for the Gaussian-Laguerre
  // Quadrature for n points for \alpha := -1 + 1 / r; a := 0; and b := 1.0.
  static bool ComputeGaussianLaguerrePoints(
      const int n, const double& r,
      vector<GaussianQuadratureTuple>* gaussian_laguerre_points);

  // Finds the factor that appears as a constant (quotient) factor when
  // computing the weights:
  //   \Gamma(1/r) * r^(1/r)
  static bool ComputeIntegralConstantFactor(
      const double& r, double* integral_constant_factor);

  // Constructs G_r(x) and G'_r(x), where G_r(x) is:
  //   If r = 0.0: x
  //   Otherwise:  log(1 + r * x) / r
  static bool ConstructTransformation(
      const double& r,
      Expression* transformation_G,
      Expression* transformation_G_prime);

  // Computes X * X^T.
  static bool ComputeXTimesXTranspose(
      const vector<vector<bool>>& time_indep_vars,  // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,    // Size n, VectorXd size p_indep,
                                                    // MatrixXd dim (p_dep, M)
      vector<vector<MatrixXd>>* x_x_transpose);     // n x M x (p, p) (inner vector has size
                                                    // 1 instead of M if all cov time-indep)

  // Constructs Subject i's covariate values at the m^th distinct timepoint by
  // combining the values of his time-independent and time-dependent values.
  static bool GetXim(
      const int m, const vector<bool>& time_indep_vars_i,
      const pair<VectorXd, MatrixXd>& x_i,
      VectorXd* x_im);

  // Computes \beta^T * X_im for a particluar Subject i \in [1..n] and time m \in [0..M].
  static bool ComputeBetaDotXim(
      const int m, const vector<bool>& time_indep_vars_i,
      const VectorXd& beta, const pair<VectorXd, MatrixXd>& x_i,
      double* dot_product);

  // Computes c * X_im for a given constant c and a particluar Subject
  // i \in [1..n] and time m \in [0..M]; and then adds this to 'input'.
  static bool AddConstantTimesXim(
      const int m, const vector<bool>& time_indep_vars_i,
      const double& constant, const pair<VectorXd, MatrixXd>& x_i,
      VectorXd* input);

  // Computes c * (X_im - v) for a given constant c, vector v, and a particluar
  // Subject i \in [1..n] and time m \in [0..M]; and then adds this to 'input'.
  static bool AddConstantTimesXimMinusVector(
      const int m, const vector<bool>& time_indep_vars_i,
      const double& constant, const VectorXd& v,
      const pair<VectorXd, MatrixXd>& x_i,
      VectorXd* input);


  // Computes the terms:
  //   exp(\beta^T * X_jm)
  // for each i \in [1..n] and m \in [0..M]
  static bool ComputeExpBetaX(
      const VectorXd& beta,                         // Size p
      const vector<vector<bool>>& time_indep_vars,  // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,    // Size n, VectorXd size p_indep,
                                                    // MatrixXd dim (p_dep, M)
      vector<VectorXd>* exp_beta_x);  // Outer vector size n, VectorXd size M (or 1)

  // For each m \in [0..M] and i \in [1..n], computes I(t_m <= R*_i).
  static bool ComputeRiStarIndicator(
      const set<double>& distinct_times,          // Size M
      const vector<double>& lower_time_bounds,    // Size n
      const vector<double>& upper_time_bounds,    // Size n
      vector<vector<bool>>* r_i_star_indicator);  // n x M

  // Whether the convergence criterion has been reached.
  static bool EmAlgorithmHasConverged(
      const double& convergence_threshold,
      const VectorXd& old_beta, const VectorXd& new_beta,
      const VectorXd& old_lambda, const VectorXd& new_lambda,
      double* current_difference);
  // Same as above, but just for the lambda parameter.
  static bool ProfileEmAlgorithmHasConverged(
      const double& convergence_threshold,
      const VectorXd& old_lambda, const VectorXd& new_lambda);

  static bool DoEStep(
      const Expression& transformation_G,
      const Expression& transformation_G_prime,
      const set<double>& distinct_times,        // Size M
      const vector<double>& lower_time_bounds,  // Size n
      const vector<double>& upper_time_bounds,  // Size n
      const VectorXd& beta,                     // Size p
      const VectorXd& lambda,                   // Size M
      const vector<VectorXd>& exp_beta_x,       // Outer vector size n, VectorXd size M (or 1)
      const double& r,
      const double& integral_constant_factor,
      const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
      MatrixXd* weights,                        // Dim (n, M)
      VectorXd* posterior_means);               // Size n

  // Computes S_i1 or S_i2 (depending on whether the input 'time_bounds'
  // correspond to {L_i} or {U_i}):
  //   S_i1 := \sum_{t_m <= L_i} \lambda_m * exp(\beta^T * X_im)
  //   S_i2 := \sum_{t_m <= U_i} \lambda_m * exp(\beta^T * X_im)
  static bool ComputeS(
      const set<double>& distinct_times,        // Size M
      const vector<double>& time_bounds,        // Size n
      const VectorXd& lambda,                   // Size M
      const vector<VectorXd>& exp_beta_x,       // Outer vector size n, VectorXd size M (or 1)
      VectorXd* S);                             // Size n

  // Computes exp(-G(S_L)) or exp(-G(S_U)), depending on if input S is S_L or S_U.
  static bool ComputeExpTransformation(
      const Expression& transformation_G,
      const VectorXd& S,   // Size n
      VectorXd* exp_g_S);  // Size n

  // Computes G'(S_L) or G'(S_U), depending on if input S is S_L or S_U.
  static bool ComputeTransformationDerivative(
      const Expression& transformation_G_prime,
      const VectorXd& S,   // Size n
      VectorXd* g_prime_S);  // Size n

  // Computes the posterior mean v_i for each Subject:
  //   v_i = (exp(-G(S^L_i)) * G'(S^L_i) - exp(-G(S^U_i)) * G'(S^U_i) /
  //         (exp(-G(S^L_i)) - exp(-G(S^U_i)))
  static bool ComputePosteriorMeans(
      const VectorXd& exp_neg_g_S_L,   // Size n
      const VectorXd& exp_neg_g_S_U,   // Size n
      const VectorXd& g_prim_S_L,  // Size n
      const VectorXd& g_prim_S_U,  // Size n
      VectorXd* v);                // Size n

  // Computes w_im, which depends on how t_m relates to (L_i, U_i):
  //   - If t_m <= L_i:
  //       w_im = 0.0
  //   - If L_i < t_m <= U_i < \infty:
  //       w_im = \lambda_m * exp(\beta^T * X_im) * Integral_{\ksi_i} /
  //              (exp(-G(S^L_i)) - exp(-G(S^U_i)))
  //     where Integral_{\ksi_i} is defined below
  //   - If t_m > U_i OR L_i < t_m < U_i = \infty:
  //       w_im = \lambda_m * exp(\beta^T * X_im) * v_i
  //     where v_i is the i^th posterior mean (see equation below).
  static bool ComputeWeights(
      const set<double>& distinct_times,        // Size M
      const vector<double>& lower_time_bounds,  // Size n
      const vector<double>& upper_time_bounds,  // Size n
      const VectorXd& lambda,                   // Size M
      const VectorXd& S_L,                      // Size n
      const VectorXd& S_U,                      // Size n
      const VectorXd& exp_neg_g_S_L,            // Size n
      const VectorXd& exp_neg_g_S_U,            // Size n
      const VectorXd& v,                        // Size n
      const vector<VectorXd>& exp_beta_x,       // Outer vector size n, VectorXd size M (or 1)
      const double& r,
      const double& integral_constant_factor,
      const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
      MatrixXd* weights);                       // Dim (n, M)

  // Computes Integral_{\ksi_i} :=
  //     \int_{\ksi} \ksi_i / (1 - exp(-\ksi_i(S^U_i - S^L_i))) * \phi(\ksi_i) *
  //                 (exp(-\ksi_i * S^L_i) - exp(-\ksi_i * S^U_i)) d\ksi_i
  static bool GetIntegralForWeightForTimeWithinLU(
      const double& S_L_i, const double& S_U_i, const double& r,
      const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
      double* integral_value);

  static bool DoMStep(
      const int itr_index,
      const VectorXd& beta,                // Size p
      const VectorXd& posterior_means,     // Size n
      const MatrixXd& weights,             // Dim (n, M)
      const vector<VectorXd>& exp_beta_x,  // Outer vector size n, VectorXd size M (or 1)
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Size n, VectorXd size p_indep,
                                                       // MatrixXd dim (p_dep, M)
      const vector<vector<MatrixXd>>& x_x_transpose,   // n x M x (p, p) (inner vector has size
                                                       // 1 instead of M if all cov time-indep
      const vector<vector<bool>>& r_i_star_indicator,  // n x M
      VectorXd* new_beta,               // Size p
      VectorXd* new_lambda);            // Size M

  // Computes the terms:
  //   v_j * exp(\beta^T * X_jm)
  // for each i \in [1..n] and m \in [0..M]
  static bool ComputeSummandTerm(
      const vector<VectorXd>& exp_beta_x,  // Outer vector size n, VectorXd size M (or 1)
      const VectorXd& posterior_means,     // Size n
      vector<VectorXd>* v_exp_beta_x);     // Outer vector size n, VectorXd size M (or 1)          

   // Computes S0_m, S1_m, and S2_m, which are defined as:
  //   S0_m := \sum_{i=1}^n v_j * exp(\beta^T X_jm)
  //   S1_m := \sum_{i=1}^n v_j * exp(\beta^T X_jm) * X_jm
  //   S2_m := \sum_{i=1}^n v_j * exp(\beta^T X_jm) * X_jm * X^T_jm
  // for each m \in [0..M].
  static bool ComputeSValues(
      const vector<VectorXd>& v_exp_beta_x,       // Outer vector size n, VectorXd size M (or 1)
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Outer vector size n, VectorXd size
                                                       // p_indep, MatrixXd dim (p_dep, M)
      const vector<vector<MatrixXd>>& x_x_transpose,   // n x M x (p, p) (inner vector has size
                                                       // 1 instead of M if all cov time-indep
      const vector<vector<bool>>& r_i_star_indicator,  // n x M
      vector<double>* S0,            // Size M
      vector<VectorXd>* S1,          // M x p
      vector<MatrixXd>* S2);         // M x (p, p)
 
  // Computes S0_m (similar to above, but with different API).
  static bool ComputeS0(
      const VectorXd& beta,             // Size p
      const VectorXd& posterior_means,  // Size n
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Size n, VectorXd size p_indep,
                                                       // MatrixXd dim (p_dep, M)
      const vector<vector<bool>>& r_i_star_indicator,  // n x M
      vector<double>* S0);              // Size M

  // Computes \Sigma, where:
  //   \Sigma = \sum_i \sum_m w_im * ((S1_m / S0_m) * (S1_m / S0_m)^T - S2_m / S0_m)
  static bool ComputeSigma(
      const MatrixXd& weights,     // Dim (n, M)
      const vector<double>& S0,    // Size M (or size 1, if all covariates time-indep)
      const vector<VectorXd>& S1,  // Outer vector size M (or 1), VectorXd size p
      const vector<MatrixXd>& S2,  // Outer vector size M (or 1), Matrix dim (p, p)
      const vector<vector<bool>>& r_i_star_indicator,  // Outer vector size n, inner size M
      MatrixXd* Sigma);            // Dim (p, p)

  // Computes new \beta value via:
  //   \beta_new = \beta_old - \Sigma^-1 * (\sum_i \sum_m w_im * (X_im - S1_m / S0_m))
  static bool ComputeNewBeta(
      const int itr_index,
      const VectorXd& old_beta,     // Size p
      const MatrixXd& Sigma,        // Dim (p, p)
      const MatrixXd& weights,      // Dim (n, M)
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Size n, VectorXd size p_indep,
                                                       // MatrixXd dim (p_dep, M)
      const vector<double>& S0,     // Size M
      const vector<VectorXd>& S1,   // M x p
      const vector<vector<bool>>& r_i_star_indicator,  // n x M
      VectorXd* new_beta);          // Size p

  // Computes new \lambda value via:
  //   \lambda_m = (\sum_i w_im) / (\sum_i v_i * exp(\beta_new^T * X_im))
  //             = (\sum_i w_im) / S0^new_m
  static bool ComputeNewLambda(
      const MatrixXd& weights,       // Dim (n, M)
      const vector<double>& S0_new,  // Size M
      const vector<vector<bool>>& r_i_star_indicator,  // n x M
      VectorXd* new_lambda);         // Size M

  // Computes the Profile-Likelihood values required for computing variance.
  static bool ComputeProfileLikelihoods(
      const Expression& transformation_G,
      const Expression& transformation_G_prime,
      const double& r,
      const double& integral_constant_factor,
      const double& convergence_threshold,
      const int h_n_constant, const int max_itr,
      const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
      const set<double>& distinct_times,        // Size M
      const vector<double>& lower_time_bounds,  // Size n
      const vector<double>& upper_time_bounds,  // Size n
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Size n, VectorXd size p_indep,
                                                       // MatrixXd dim (p_dep, M)
      const vector<vector<bool>>& r_i_star_indicator,  // n x M
      const VectorXd& final_beta,               // Size p
      const VectorXd& final_lambda,             // Size M
      double* pl_at_beta,
      VectorXd* pl_toggle_one_dim,              // Size depends on doing pos-def variance:
                                                //   No Pos-Def: p;  Pos-Def: n
      MatrixXd* pl_toggle_two_dim);             // Dim depends on doing pos-def variance:
                                                //   No Pos-Def: (p)^2 (only use Upper-Tri);
                                                //   Pos-Def:    (n, p)

  // Computes the Profile-Likelihood at the given \beta value:
  //   pl_n(\beta) = max_{\Lambda} log(L_n(\beta, \Lambda))
  // First, uses final_lambda as the initial guess for the \lambda value
  // that maximizes the RHS of pl_n, then run through the E-M algorithm
  // to find the maximizing \lambda (since we are doing the profile-likelihood,
  // we don't update \beta when running the E-M algorithm for maximizing lambda).
  static bool ComputeProfileLikelihood(
      const Expression& transformation_G,
      const Expression& transformation_G_prime,
      const double& r,
      const double& integral_constant_factor,
      const double& convergence_threshold,
      const int max_itr,
      const vector<GaussianQuadratureTuple>& gaussian_laguerre_points,
      const set<double>& distinct_times,        // Size M
      const vector<double>& lower_time_bounds,  // Size n
      const vector<double>& upper_time_bounds,  // Size n
      const vector<vector<bool>>& time_indep_vars,     // Outer vector size n, inner size p
      const vector<pair<VectorXd, MatrixXd>>& x,       // Size n, VectorXd size p_indep,
                                                       // MatrixXd dim (p_dep, M)
      const vector<vector<bool>>& r_i_star_indicator,  // n x M
      const VectorXd& beta,                     // Size p
      const VectorXd& final_lambda,             // Size M
      double* pl,
      VectorXd* pl_alternate);                  // Size n
  
  // Evaluates the (discretized) likelihood function L_n(\beta, \Lambda):
  //   L_n = \Pi_i exp(-G(S_L_i)) - exp(-G(S_U_i))
  static bool EvaluateLogLikelihoodFunctionAtBetaLambda(
      const Expression& transformation_G,
      const set<double>& distinct_times,        // Size M
      const vector<double>& lower_time_bounds,  // Size n
      const vector<double>& upper_time_bounds,  // Size n
      const VectorXd& lambda,                   // Size M
      const vector<VectorXd>& exp_beta_x,       // Outer vector size n, VectorXd size M (or 1)
      double* likelihood,
      VectorXd* e_i_likelihoods);               // Size n
  // Same as above, but for different API (the above function will first
  // compute the requisite exp_neg_g_S_[L | U] values, then call this one).
  static bool EvaluateLogLikelihoodFunctionAtBetaLambda(
      const VectorXd& exp_neg_g_S_L,   // Size n
      const VectorXd& exp_neg_g_S_U,   // Size n
      double* likelihood);
  // Same as above, for the alternate (pos-def) version of computing variance.
  static bool EvaluateAlternateLogLikelihoodFunctionAtBetaLambda(
      const VectorXd& exp_neg_g_S_L,   // Size n
      const VectorXd& exp_neg_g_S_U,   // Size n
      VectorXd* e_i_likelihoods);      // Size n

  // Applies the formula for estimated variance.
  static bool ComputeVarianceFromProfileLikelihoods(
      const int n, const int h_n_constant, const double& pl_at_beta,
      const VectorXd& pl_toggle_one_dim,      // Size p
      const MatrixXd& pl_toggle_two_dim,      // Dim (p, p)
      MatrixXd* variance);                    // Dim (p, p)
  // Same as above, for the alternate (pos-def) version of computing variance.
  static bool ComputeAlternateVarianceFromProfileLikelihoods(
      const int n, const int h_n_constant,
      const VectorXd& pl_toggle_none,      // Size n
      const MatrixXd& pl_toggle_one_dim,   // Dim (n, p)
      MatrixXd* variance);                 // Dim (p, p)
};

}  // namespace regression

#endif
