// Date: Dec 2015
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
#include "Regression/mple_for_interval_censored_data.h"

#include "MathUtils/data_structures.h"
#include "MathUtils/gaussian_quadrature.h"
#include "TestUtils/test_utils.h"

#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef MULTIVARIATE_MPLE_FOR_INTERVAL_CENSORED_DATA_H
#define MULTIVARIATE_MPLE_FOR_INTERVAL_CENSORED_DATA_H

using Eigen::MatrixXd;
using Eigen::VectorXd;
using file_reader_utils::DependentCovariateEstimates;
using file_reader_utils::TimeDepIntervalCensoredData;
using file_reader_utils::SubjectInfo;
using test_utils::Timer;
using namespace std;
using namespace math_utils;

namespace regression {

// The Multivariate MPLE algorithm essentially reduces to the Univariate MPLE
// algorithm, treating each (dependent) covariate separately. This structure
// holds all the information necessary to run the Univariate MPLE algorithm.
struct DependentCovariateInfo {
  double r_;
  Expression transformation_G_;
  // Let N_k denote the number of Gaussian-Laguerre points used for k^th dep covariate.
  vector<GaussianQuadratureTuple> points_and_weights_;  // Size N_k
  // \sum_{q=1}^{N_k} points_and_weights_[q].abscissa_ * points_and_weights_[q].weight_
  double sum_points_times_weights_;
  set<double> distinct_times_;            // Size M_k
  vector<double> lower_time_bounds_;      // Size n
  vector<double> upper_time_bounds_;      // Size n
  // Specifies, for each subject, which of the p_k covariates are time-independent.
  vector<vector<bool>> time_indep_vars_;  // Outer vector has Size n, inner vector Size p_k
  vector<pair<VectorXd, MatrixXd>> x_;  // Vector has Size n, each VectorXd size is p_k_indep,
                                        // and each Matrix has Dim (p_k_dep, M_k) or (0, 0)
                                        // if all covariates are time-independent.
  vector<vector<MatrixXd>> x_x_transpose_;  // n x M_k x (p_k, p_k); if all covariates are
                                            // time-indep, inner-vector has size 1.
  vector<vector<bool>> r_star_;  // Outer vector size n, inner vector size M_k
};

// There will be a vector (of size K) of this object, which holds
// intermediate values for each of the K dependent covariates. 
struct DependentCovariateIntermediateValues {
  VectorXd beta_;           // Size p_k
  VectorXd lambda_;         // Size M_k
  vector<vector<double>> exp_beta_x_;   // Outer vector size n, Inner vector size M_k, or
                                        // size 1 if all p_k covariates are time-indep
  vector<MatrixXd> exp_beta_x_plus_b_;  // Outer vector size n, Matrix Dim (N_s, M_k), or
                                        // Dim (N_s, 1) if all p_k covariates are time-indep
  // The following represent S^L_ki(\sqrt{2} * \sigma * y_s) (resp. S^U_ki()),
  // where y_s = Gaussian-Hermite Points.
  MatrixXd S_L_;            // Dim (n, N_s), where N_s = number of Gaussian-Hermite points
  MatrixXd S_U_;            // Dim (n, N_s), where N_s = number of Gaussian-Hermite points
  // The following represent exp(-G_k(S^L_ki(\sqrt{2} * \sigma * y_s))).
  MatrixXd exp_neg_g_S_L_;  // Dim (n, N_s), where N_s = number of Gaussian-Hermite points
  MatrixXd exp_neg_g_S_U_;  // Dim (n, N_s), where N_s = number of Gaussian-Hermite points
  MatrixXd a_is_;           // Size (n, N_s), where N_s = number of Gaussian-Hermite points
  MatrixXd c_is_;           // Size (n, N_s), where N_s = number of Gaussian-Hermite points
  MatrixXd d_is_;           // Size (n, N_s), where N_s = number of Gaussian-Hermite points
  VectorXd f_i_;            // Size n
};

class MultivariateMpleForIntervalCensoredData {
 public:
  // Runs the E-M Algorithm to solve simultaneously for solutions \beta and
  // \lambda to a non-parametric transformation model with interval-censored data.
  static MpleReturnValue PerformEmAlgorithmForParameterEstimation(
      const double& convergence_threshold,
      const int h_n_constant, const int max_itr,
      const vector<DependentCovariateInfo>& input,  // Size K
      int* num_iterations, double* log_likelihood, double* b_variance,
      vector<DependentCovariateEstimates>* estimates, MatrixXd* variance);

  // Compute Variance. This public interface just calls the private interface
  // below, after computing a few constants.
  static MpleReturnValue ComputeVariance(
      const int n, const vector<int>& p, const int h_n_constant, const int max_itr,
      const double& convergence_threshold,
      const double& final_sigma,
      const vector<DependentCovariateInfo>& input,
      const vector<DependentCovariateEstimates>& estimates,
      MatrixXd* variance);

  // Initializes input to the main EM algorithm above.
  static bool InitializeInput(
      const vector<double>& r,                // Size K
      const TimeDepIntervalCensoredData& data,
      vector<DependentCovariateInfo>* input);  // Size K

  // Temp functions for time benchmarks.
  static void InitializeTimers();
  static void PrintTimers();
  static void SetLoggingOn(const bool logging_on) {
    logging_on_ = logging_on;
  }
  static void SetForceOneRightCensored(bool force_one_right_censored) {
      force_one_right_censored_ = force_one_right_censored;
  }
  static void SetNoUsePositiveDefiniteVariance(bool no_use_pos_def_var) {
    no_use_pos_def_variance_ = no_use_pos_def_var;
  }
  static void SetNoComputeVariance(bool no_variance) {
    PHB_no_variance_ = no_variance;
  }
  static void SetUseExpLambdaConvergenceCriteria(bool use_exp_lambda_convergence) {
    PHB_use_exp_lambda_convergence_ = use_exp_lambda_convergence;
  }

 private:
  // Member Variables.
  static bool logging_on_;
  static bool force_one_right_censored_;
  static bool no_use_pos_def_variance_;
  static bool PHB_no_variance_;
  static bool PHB_use_exp_lambda_convergence_;
  static int num_failed_variance_computations_;
  // Dimensions (number of linear terms) for each dependent variable (event type).
  static vector<int> p_;
  // Temp structure for time benchmarks.
  static vector<Timer> timers_;

  // Finds the weights and abscissa (knots) for the Gaussian-Laguerre
  // Quadrature for n points for \alpha := -1 + 1 / r; a := 0; and b := 1.0.
  static bool ComputeGaussianLaguerrePoints(
      const int n, const double& r,
      vector<GaussianQuadratureTuple>* gaussian_laguerre_points);

  // Finds the weights and abscissa (knots) for the Gaussian-Hermite
  // Quadrature for n points for \alpha := 0; a := 0; and b := 1.0.
  static bool ComputeGaussianHermitePoints(
      const int n, vector<GaussianQuadratureTuple>* gaussian_hermite_points);
  // Same as above, but also comutes the sum of the weights.
  static bool ComputeGaussianHermitePointsAndSumWeights(
      const int N_s,
      double* sum_hermite_weights,
      vector<GaussianQuadratureTuple>* gaussian_hermite_points);

  // Constructs G_r(x) and G'_r(x), where G_r(x) is:
  //   If r = 0.0: x
  //   Otherwise:  log(1 + r * x) / r
  static bool ConstructTransformation(
      const double& r_k, Expression* transformation_G_k);

  // Computes X * X^T.
  static bool ComputeXTimesXTranspose(
      const vector<vector<bool>>& time_indep_vars_k,  // Outer-vector size n, inner size p_k
      const vector<pair<VectorXd, MatrixXd>>& x_k,  // n x (p_k_indep, (p_k_dep, M_k))
      vector<vector<MatrixXd>>* x_k_x_k_transpose);  // n x K x (p_k, p_k)

  // For a fixed k in [1..K], computes the terms:
  //   exp(\beta_k^T * X_kim + \sqrt{2} * \sigma * y_s)
  // for each i \in [1..n], s \in [1..N_s] and m \in [1..M_k], where
  // N_s is number of points used for Gaussian-Hermite integral approximation
  // and y_s is the s^th abscissa.
  static bool ComputeExpBetaXPlusB(
      const VectorXd& beta_k,       // Size p_k
      const VectorXd& b,            // Size N_s, the contants added to \beta_k^T * X_kim
      const vector<vector<bool>>& time_indep_vars_k,  // Outer vector size n, inner size p_k
      const vector<pair<VectorXd, MatrixXd>>& x_k,  // n x (p_k_indep, (p_k_dep, M_k))
      vector<vector<double>>* exp_beta_x_k,    // Outer vector size n, inner vector size M_k,
                                               // or size 1 if all p_k cov's are time-indep
      vector<MatrixXd>* exp_beta_x_plus_b_k);  // Outer vector size n, Matrix Dim (N_s, M_k),
                                               // or (N_s, 1) if all p_k cov's are time-indep

  // Whether the convergence criterion has been reached.
  static bool EmAlgorithmHasConverged(
      const double& convergence_threshold,
      const double& b_variance_old, const double& b_variance_new,
      const vector<DependentCovariateIntermediateValues>& intermediate_values,  // Size K
      const vector<DependentCovariateEstimates>& estimates,  // Size K
      double* current_difference);
  // Same as above, but just for the lambda parameter.
  static bool ProfileEmAlgorithmHasConverged(
      const double& convergence_threshold, const vector<VectorXd>& old_lambda,
      const vector<DependentCovariateIntermediateValues>& intermediate_values);  // Size K

  // The 'intermediate_values' parameter acts both as input and output. In
  // particular, when DoEStep is called, only the beta_k_, lambda_k_, and
  // exp_beta_x_plus_b_ fields are current (valid). In the process of doing
  // the E-Step, the other fields will be computed/populated.
  static bool DoEStep(
      const double& current_sigma, const double& sum_hermite_weights,
      const VectorXd& b,                            // Size N_s
      const vector<DependentCovariateInfo>& input,  // Size K
      const vector<GaussianQuadratureTuple>& gaussian_hermite_points,     // Size N_s
      vector<DependentCovariateIntermediateValues>* intermediate_values,  // Size K
      vector<MatrixXd>* weights,                    // Vector Size K, matrix Dim (n, M_k)
      vector<VectorXd>* posterior_means,            // Outer vector Size K, inner vector Size n
      VectorXd* phi);                               // Size n

  // Computes S^L_ki or S^U_ki (depending on whether the input 'time_bounds'
  // correspond to {L_i} or {U_i}):
  //   S^U_ki(b_i) := \sum_{t_km <= L_ki} \lambda_km * exp(\beta_k^T * X_kim + b_i)
  //   S^L_ki(b_i) := \sum_{t_km <= U_ki} \lambda_km * exp(\beta_k^T * X_kim + b_i)
  static bool ComputeS(
      const set<double>& t_k,    // Size M_k, the distinct times for k^th covariate
      // The (lower or upper) time values for each Subject
      const vector<double>& time_bounds_k, // Size n
      const VectorXd& lambda_k,     // Size M_k
      const vector<MatrixXd>& exp_beta_x_plus_b_k,  // Vector has Size n, Matrix Dim (N_s, M_k)
      MatrixXd* S_k);               // Dim (n, N_s), where N_s is num Gaussian-Hermite points

  // Computes exp(-G(S_L)) or exp(-G(S_U)), depending on if input S is S_L or S_U.
  static bool ComputeExpTransformation(
      const Expression& transformation_G_k,
      const MatrixXd& S_k,       // Dim (n, N_s)
      MatrixXd* exp_neg_g_S_k);  // Dim (n, N_s)

  // Computes e_is for each s in [1..N_s], where N_s is number of Gaussian-
  // Hermite points:
  //   e_is := \Pi_k [ exp(-G_k(S^L_ki(\sqrt{2} * \sigma * y_s))) -
  //                   exp(-G_k(S^U_ki(\sqrt{2} * \sigma * y_s))) ],
  // where y_s are the Gaussian-Hermite points, and k ranges from [1..K]
  static bool ComputeEis(
      const vector<DependentCovariateIntermediateValues>& intermediate_values,  // Size K
      MatrixXd* e  /* Dim (n, N_s) */);

  // Computes the constants (from "Nonparametric Maximum Likelihood Estimation
  // for Multiple Events With Interval-Censored Data"): a_kis, c_kis, d_kis, f_ki:
  //   a_kis := S^L_ki(\sqrt{2} * \sigma * y_s) + (1 / r_k)
  //   c_kis := S^U_ki(\sqrt{2} * \sigma * y_s) + (1 / r_k)
  //   d_kis := e_is / [ exp(-G_k(S^L_ki(\sqrt{2} * \sigma * y_s))) -
  //                     exp(-G_k(S^U_ki(\sqrt{2} * \sigma * y_s))) ]
  //   f_ki := W_L * \sum_s \mu_s * d_kis * [a_kis^(-1 / r_k) -
  //                                         I_(R_ki < \inf) * c_kis^(-1 / r_k) ]
  // where W_L := Sum of Gaussian-Laguerre weights, and s ranges over [1..N_s],
  // \mu_s is the s^th Gaussian-Hermite weight, and I_(R_ki < \inf) is the
  // indicator function for R_ki < infinity.
  static bool ComputeConstants(
      const double& r_k_inverse, const double& sum_hermite_weights,
      const vector<GaussianQuadratureTuple>& hermite_points_and_weights,     // Size N_s
      const vector<double>& upper_time_bounds_k,    // Size n
      const MatrixXd& e,                            // Dim (n, N_s)
      const MatrixXd& S_L_k,                        // Dim (n, N_s)
      const MatrixXd& S_U_k,                        // Dim (n, N_s)
      const MatrixXd& exp_neg_g_S_L_k,              // Dim (n, N_s)
      const MatrixXd& exp_neg_g_S_U_k,              // Dim (n, N_s)
      MatrixXd* a_k, MatrixXd* c_k, MatrixXd* d_k,  // Dim (n, N_s)
      VectorXd* f_k);                               // Size n

  // Computes the posterior mean v_ki for each Subject:
  // If r_k = 0:
  //   v_ki := \sum_s exp(\sqrt{2} * \sigma * y_s) * \mu_s * e_is /
  //           \sum_s \mu_s * e_is
  // If r_k \neq 0:
  //   v_ki := (1 / f_ki) * PW_L *
  //           \sum_s exp(\sqrt{2} * \sigma * y_s) * \mu_s * d_kis *
  //                  [a_kis^(-1 / r_k) - I_(R_ki < \inf) * c_kis^(-1 / r_k) ]
  // where PW_L := Sum of Laguerre points times weights (\sum_q \nu_kq * x_kq),
  // and other variables are as defined in ComputeConstants() above.
  static bool ComputePosteriorMeans(
      const double& r_k,
      const double& sum_points_times_weights,
      const VectorXd& b,                          // Size N_s
      const vector<GaussianQuadratureTuple>& hermite_points_and_weights,     // Size N_s
      const vector<double>& upper_time_bounds_k,  // Size n
      const MatrixXd& e,                          // Dim (n, N_s)
      const MatrixXd& a_k,                        // Dim (n, N_s)
      const MatrixXd& c_k,                        // Dim (n, N_s)
      const MatrixXd& d_k,                        // Dim (n, N_s)
      const VectorXd& f_k,                        // Size n
      VectorXd* v_k);                             // Size n

  // Computes w_kim, which depends on r_k and how t_km relates to (L_ki, U_ki):
  // - If r_k = 0:
  //   - If t_km <= L_ki:
  //       w_kim = 0.0
  //   - If L_ki < t_km <= U_ki < \infty:
  //       w_kim = (\sum_s \mu_s * e_is)^-1 *
  //               \sum_s \mu_s * e_is *
  //                      [ (\lambda_km * exp(\beta_k^T * X_kim + \sqrt{2} * \sigma * y_s)) /
  //                        (1 - exp(S^U_ki(\sqrt{2} * \sigma * y_s)) -
  //                             exp(S^L_ki(\sqrt{2} * \sigma * y_s))) ]
  //   - If t_km > U_ki OR L_ki < t_km < U_ki = \infty:
  //       w_kim = \lambda_km * exp(\beta_k^T * X_kim) * v_ki
  // - If r_k \neq 0:
  //   - If t_km <= L_ki:
  //       w_kim = 0.0
  //   - If L_ki < t_km <= U_ki < \infty:
  //       w_kim = f_ki^-1 *
  //               \sum \mu_s * d_kis * \lambda_km *
  //                    exp(\beta_k^T * X_kim + \sqrt{2} * \sigma * y_s) *
  //                    (A_kis - I_(R_ki < \inf) * C_kis)
  //     where:
  //       A_kis := a_kis^(-1 - 1 / r_k) *
  //                \sum_q x_kq * \nu_kq / 
  //                       (1 - exp(S^U_ki(\sqrt{2} * \sigma * y_s)) -
  //                            exp(S^L_ki(\sqrt{2} * \sigma * y_s)))
  //       C_kis := A_kis, with the first term (a_kis) replaced with c_kis
  //   - If t_km > U_ki OR L_ki < t_km < U_ki = \infty:
  //       w_kim = \lambda_km * exp(\beta_k^T * X_kim) * v_ki
  static bool ComputeWeights(
      const double& r_k,
      const vector<GaussianQuadratureTuple>& hermite_points_and_weights,     // Size N_s
      const vector<GaussianQuadratureTuple>& laguerre_points_and_weights_k,  // Size N_k
      const set<double>& distinct_times_k,        // Size M_k
      const vector<double>& lower_time_bounds_k,  // Size n
      const vector<double>& upper_time_bounds_k,  // Size n
      const vector<MatrixXd>& exp_beta_x_plus_b_k, // Vector Size n, Matrix Dim (N_s, M_k)
      const VectorXd& lambda_k,                   // Size M_k
      const MatrixXd& e,                          // Dim (n, N_s)
      const MatrixXd& S_L_k,                      // Dim (n, N_s)
      const MatrixXd& S_U_k,                      // Dim (n, N_s)
      const MatrixXd& a_k,                        // Dim (n, N_s)
      const MatrixXd& c_k,                        // Dim (n, N_s)
      const MatrixXd& d_k,                        // Dim (n, N_s)
      const VectorXd& f_k,                        // Size n
      const VectorXd& v_k,                        // Size n
      MatrixXd* weights_k);                       // Dim (n, M_k)

  static bool DoMStep(
      const int itr_index,
      const vector<DependentCovariateInfo>& input,  // Size K
      const vector<DependentCovariateIntermediateValues>& intermediate_values,  // Size K
      const VectorXd& phi,                          // Size n
      const vector<VectorXd>& posterior_means,      // Outer vector size K, inner Vector Size n
      const vector<MatrixXd>& weights,              // Outer vector size K, Matrix Dim (n, M_k)
      double* new_sigma_squared,
      vector<DependentCovariateEstimates>* new_estimates);  // Size K

  // Computes the terms required in the M-Step:
  //   v_ki * exp(\beta_k^T * X_kim)
  // for each i \in [1..n] and m \in [1..M_k]
  static bool ComputeSummandTerm(
      const VectorXd& v_k,          // Size n
      const vector<vector<double>>& exp_beta_x_k, // Outer vector size n, inner vector size M_k,
                                                  // or size 1 if all p_k cov's are time-indep
      vector<VectorXd>* v_exp_beta_x_k);    // Outer vector size n, VectorXd size M or 1

   // Computes S0_km, S1_km, and S2_km, which are defined as:
  //   S0_km := \sum_{i=1}^n v_ki * exp(\beta_k^T X_kim)
  //   S1_km := \sum_{i=1}^n v_ki * exp(\beta_k^T X_kim) * X_kim
  //   S2_km := \sum_{i=1}^n v_ki * exp(\beta_k^T X_kim) * X_kim * X^T_kim
  // for each k \in [1..K] and each m in [1..M_k].
  static bool ComputeSValues(
      const vector<VectorXd>& v_exp_beta_x_k,    // Outer vector size n, VectorXd size M or 1
      const vector<vector<bool>>& time_indep_vars_k,  // Outer-vector size n, inner size p_k
      const vector<pair<VectorXd, MatrixXd>>& x_k,  // n x (p_k_indep, (p_k_dep, M_k))
      const vector<vector<MatrixXd>>& x_k_x_k_transpose,  // n x M_k x (p_k, p_k)
      const vector<vector<bool>>& r_star_k,  // n x M_k
      vector<double>* S0_k,            // Size M_k
      vector<VectorXd>* S1_k,          // Outer Vector Size M_k, Inner Vector Size p_k
      vector<MatrixXd>* S2_k);         // Outer Vector Size M_k, Matrix Dim (p_k, p_k)
 
  // Computes S0_km (similar to above, but with different API).
  static bool ComputeS0(
      const VectorXd& beta_k,             // Size p_k
      const VectorXd& posterior_means_k,  // Size n
      const vector<vector<bool>>& time_indep_vars_k,  // Outer-vector size n, inner size p_k
      const vector<pair<VectorXd, MatrixXd>>& x_k,  // n x (p_k_indep, (p_k_dep, M_k))
      const vector<vector<bool>>& r_star_k,  // n x M_k
      vector<double>* S0_k);              // Size M_k

  // Computes \Sigma, where:
  //   \Sigma = \sum_i \sum_m w_kim * ((S1_km / S0_km) * (S1_km / S0_km)^T - S2_km / S0_km)
  static bool ComputeSigma(
      const MatrixXd& weights_k,     // Dim (n, M_k)
      const vector<double>& S0_k,    // Size M_k
      const vector<VectorXd>& S1_k,  // M_k x p_k
      const vector<MatrixXd>& S2_k,  // M_k x (p_k, p_k)
      const vector<vector<bool>>& r_star_k,  // n x M_k
      MatrixXd* Sigma_k);            // Dim (p_k, p_k)

  // Computes new \beta value via:
  //   \beta_new = \beta_old - \Sigma^-1 * (\sum_i \sum_m w_kim * (X_kim - S1_km / S0_km))
  static bool ComputeNewBeta(
      const int itr_index,
      const VectorXd& old_beta_k,     // Size p_k
      const MatrixXd& Sigma_k,        // Dim (p_k, p_k)
      const MatrixXd& weights_k,      // Dim (n, M_k)
      const vector<vector<bool>>& time_indep_vars_k, // Outer-vector size n, inner size p_k
      const vector<pair<VectorXd, MatrixXd>>& x_k,  // n x (p_k_indep, (p_k_dep, M_k))
      const vector<double>& S0_k,     // Size M_k
      const vector<VectorXd>& S1_k,   // M_k x p_k
      const vector<vector<bool>>& r_star_k,  // n x M_k
      VectorXd* new_beta_k);          // Size p_k

  // Computes new \lambda_k value via:
  //   \lambda_km = (\sum_i w_kim) / (\sum_i v_ki * exp(\beta_new_k^T * X_kim))
  //              = (\sum_i w_kim) / S0^new_km
  static bool ComputeNewLambda(
      const MatrixXd& weights_k,       // Dim (n, M_k)
      const vector<double>& S0_new_k,  // Size M_k
      const vector<vector<bool>>& r_star_k,  // n x M_k
      VectorXd* new_lambda_k);         // Size M_k

  // Computes an estimate for Variance.
  static MpleReturnValue ComputeVariance(
      const int n, const vector<int>& p, const int h_n_constant, const int max_itr,
      const double& convergence_threshold,
      const double& final_sigma, const double& sum_hermite_weights,
      const vector<GaussianQuadratureTuple>& gaussian_hermite_points,  // Size N_s
      const VectorXd& b,                                               // Size N_s
      const vector<DependentCovariateInfo>& input,                     // Size K
      const vector<DependentCovariateEstimates>& estimates,            // Size K
      MatrixXd* variance);                      // Dim (1 + \sum p_k, 1 + \sum p_k)

  // Computes the Profile-Likelihood values required for computing variance.
  static bool ComputeProfileLikelihoods(
      const int n, const vector<int>& p, const int h_n_constant, const int max_itr,
      const double& convergence_threshold,
      const double& final_sigma, const double& sum_hermite_weights,
      const vector<GaussianQuadratureTuple>& gaussian_hermite_points,  // Size N_s
      const VectorXd& b,                                               // Size N_s
      const vector<DependentCovariateInfo>& input,                     // Size K
      const vector<DependentCovariateEstimates>& estimates,            // Size K
      double* pl_at_beta,
      VectorXd* pl_toggle_one_dim,   // Size depends on doing pos-def variance:
                                     //   No Pos-Def: 1 + \sum_k p_k;  Pos-Def: n
      MatrixXd* pl_toggle_two_dim);  // Dim depends on doing pos-def variance:
                                     //   No Pos-Def: (1 + \sum p_k)^2 (only use Upper-Tri);
                                     //   Pos-Def:    (n, \sum p_k)

  // Computes the Profile-Likelihood at the given \theta value:
  //   pl_n(\theta) = max_{\Lambda} log(L_n(\theta, \Lambda))
  // First, uses final_lambda as the initial guess for the \lambda value
  // that maximizes the RHS of pl_n, then run through the E-M algorithm
  // to find the maximizing \lambda (since we are doing the profile-likelihood,
  // we don't update \theta when running the E-M algorithm for maximizing lambda).
  static bool ComputeProfileLikelihood(
      const int max_itr,
      const double& convergence_threshold,
      const double& sum_hermite_weights,
      const vector<GaussianQuadratureTuple>& gaussian_hermite_points,  // Size N_s
      const VectorXd& b,                            // Size N_s
      const vector<DependentCovariateInfo>& input,  // Size K
      const vector<VectorXd>& beta,                 // Size K
      const vector<VectorXd>& lambda,               // Size K
      double* pl,                                   // No Pos-Def: pl; Pos-Def: nullptr
      VectorXd* pl_alternate);                      // No Pos-Def: nullptr; Pos-Def: Size n
  
  // Evaluates the (discretized) likelihood function L_n(\beta, \sigma^2, \Lambda).
  // On input, intermediate_values has just beta_, lambda_, and exp_beta_x_plus_b_
  // fields set; the other fields will be computed/updated within this function.
  // Note that exactly one of {likelihood, e_i_likelihoods} will be set
  // (whichever is non-null), based on use_pos_def_variance_.
  static bool EvaluateLogLikelihoodFunctionAtBetaLambda(
      const vector<GaussianQuadratureTuple>& gaussian_hermite_points,     // Size N_s
      const vector<DependentCovariateInfo>& input,                        // Size K
      vector<DependentCovariateIntermediateValues>* intermediate_values,  // Size K
      double* likelihood,
      VectorXd* e_i_likelihoods);                                         // Size n
  // Same as above, but for different API (the above function will first
  // compute the requisite exp_neg_g_S_[L | U] values, then call this one).
  static bool EvaluateLogLikelihoodFunctionAtBetaLambda(
      const vector<GaussianQuadratureTuple>& gaussian_hermite_points,  // Size N_s
      const vector<DependentCovariateIntermediateValues>& intermediate_values,  // Size K
      double* likelihood);
  // Same as above, but for the Positive-Definite version of the computation
  // (same as original, except doesn't sum over all n, and includes a factor
  // of 1 / h_n).
  static bool EvaluateAlternateLogLikelihoodFunctionAtBetaLambda(
      const vector<GaussianQuadratureTuple>& gaussian_hermite_points,  // Size N_s
      const vector<DependentCovariateIntermediateValues>& intermediate_values,  // Size K
      VectorXd* e_i_likelihoods);                     // Size n

  // Applies the formula for estimated variance.
  static bool ComputeVarianceFromProfileLikelihoods(
      const int n, const int h_n_constant, const double& pl_at_theta,
      const VectorXd& pl_toggle_one_dim,      // Size 1 + \sum p_k
      const MatrixXd& pl_toggle_two_dim,      // Dim (1 + \sum p_k, 1 + \sum p_k)
      MatrixXd* variance);                    // Dim (1 + \sum p_k, 1 + \sum p_k)
  static bool ComputeAlternateVarianceFromProfileLikelihoods(
      const int n, const int h_n_constant,
      const VectorXd& pl_toggle_none,      // Size n
      const MatrixXd& pl_toggle_one_dim,   // Dim (n, \sum p_k)
      MatrixXd* variance);                 // Dim (1 + \sum p_k, 1 + \sum p_k)
};

}  // namespace regression

#endif
