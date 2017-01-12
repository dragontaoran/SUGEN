#ifndef SUGEN_UTILS_H
#define SUGEN_UTILS_H

#include <fstream>
#include <string>
#include <map>
#include <vector>
#include "InputFile.h"
#include "VcfFileReader.h"
#include "VcfRecordGenotype.h"
#include "VcfHeader.h"
#include "MathUtils/data_structures.h"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;
using namespace math_utils;


/**** structures *****************************************************************************************************************************/
struct INPUT_UTILS {
	
	/**** Phenotype *********************************************************************/
	map<string, int> rawPheno_ID_;
	map<string, int> rawCOV_name_;
	vector<string> rawF_;
	vector<string> rawStrata_;
	VectorXd rawY_;
	VectorXd rawWT_;
	MatrixXd rawW_;
	vector<CensoringData> rawY_cox_;
	/**** Phenotype *********************************************************************/
	
	/**** Genotype **********************************************************************/
	vector<string> rawGeno_ID_;
	vector< vector<double> > rawCOND_;
	VectorXi rawCOND_Miss_;
	/**** Genotype **********************************************************************/

	/**** Pairwise Inclusion Probabilities **********************************************/
	vector<string> FN_sps_;
	vector<MatrixXd> rawSPS_;
	vector< vector<string> > rawSPS_ID_;
	/**** Pairwise Inclusion Probabilities **********************************************/
	
	/**** ID Linker *********************************************************************/
	VectorXi raw_geno_pheno_linker_;
	MatrixXi raw_prob_pheno_linker_;
	/**** ID Linker *********************************************************************/
};

struct SVA_UTILS {
	
	/**** all regression models *********************************************************/
	VectorXd theta_;
	MatrixXd vartheta_;	
	VectorXd WtY_;
	MatrixXd WtW_;
	MatrixXd Ahat_; 
	MatrixXd Bhat_;
	vector<VectorXd> resi_; 
	vector<VectorXd> Miss_;
	vector<MatrixXd> Uhat_;
	VectorXd theta_initilization_;
	VectorXd GE_U_;
	MatrixXd GE_V_;
	/**** all regression models *********************************************************/
	
	/**** linear regression *************************************************************/
	double sigma2_;
	vector<VectorXd> het_sigma2_; // --hetero-variance
	vector<VectorXd> het_sigma2_0_; // --hetero-variance
	vector<MatrixXd> het_miss_; // --hetero-variance
	vector<VectorXd> het_miss_inv_sigma2_; // --hetero-variance
	bool flag_het_converge_; // --hetero-variance	
	/**** linear regression *************************************************************/
	
	/**** logistic regression ***********************************************************/
	VectorXd theta0_;	
	vector<VectorXd> Wttheta_;
	vector<VectorXd> e_Wttheta_; 
	vector<VectorXd> logi_Wttheta_;
	vector<VectorXd> logi2_Wttheta_;
	bool flag_logistic_converge_;
	/**** logistic regression ***********************************************************/
	
	/**** cox proportional hazards regression *******************************************/
	double S0_;
	VectorXd S1_;
	MatrixXd S2_;
	vector<VectorXd> riskset_;
	bool flag_coxph_converge_;
	VectorXd S1_G_;
	MatrixXd S2_GW_;
	MatrixXd S2_GG_;
	double newton_raphson_step_size_;
	/**** cox proportional hazards regression *******************************************/
	
	/**** robust variance estimator *****************************************************/
	vector<VectorXd> wtd_;
	/**** robust variance estimator *****************************************************/
	
	/**** SNP info **********************************************************************/
	int n_;
	int mac_;
	int n0_count_;
	int n1_count_;
	int n2_count_; 
	int n_dose_;
	double maf_;
	bool flag_multiallelic_;
	int n_case_;
	double maf_case_;
	/**** SNP output info ***************************************************************/
	
	/**** output ************************************************************************/
	IFILE FO_out_;
	double pvalue_;
	/**** output ************************************************************************/

	/**** score test ********************************************************************/
	IFILE FO_score_mass_;
	IFILE FO_score_snp_;
	IFILE FO_score_group_;
	vector<VectorXd> rawG_;
	vector< vector<VectorXd> > rawGene_;
	vector<MatrixXd> G_;
	vector<MatrixXd> Uhat_G_;
	string gene_ID_;	
	vector<string> SNP_chr_;
	vector<int> SNP_pos_;
	vector<string> SNP_ID_;
	vector<string> SNP_ref_;
	vector<string> SNP_alt_;
	vector<int> SNP_n_;
	vector<int> SNP_mac_;
	vector<int> SNP_n0_count_;
	vector<int> SNP_n1_count_;
	vector<int> SNP_n2_count_; 
	vector<int> SNP_n_dose_;
	vector<double> SNP_maf_;
	vector<int> SNP_n_case_;
	vector<double> SNP_maf_case_;
	int nSNP_;
	MatrixXd Ahat_rescale_;
	MatrixXd Bhat_rescale_;
	VectorXd U_;
	MatrixXd V_;
	MatrixXd V_A_;
	VectorXd beta_;
	VectorXd se_;
	vector<int> A_snplist_;
	vector<int> B_snplist_;
	MatrixXd Bhat_cov_;
	/**** score test ********************************************************************/
	
	/**** functions *********************************************************************/
	SVA_UTILS ();
	/**** functions *********************************************************************/
};

struct SUGEN {
	
	/**** definition ********************************************************************/
	enum METHOD_ {LS, logistic, right_censored};
	enum SNP_Analysis_Mode_ {ST, GE, COND};
	enum SVA_OUTPUT_TYPE_ {sva_header, sva_results_miss, sva_no_miss};
	enum EXTRACT_TYPE_ {EXTRACT_TYPE_NONE, EXTRACT_TYPE_CHR, EXTRACT_TYPE_RANGE, EXTRACT_TYPE_FILE};
	enum TEST_TYPE_ {WALD, SCORE};
	enum RESCALE_TYPE_ {NAIVE, OPTIMAL};
	/**** definition ********************************************************************/
	
	/**** I/O ***************************************************************************/
	string FN_pheno_; // phenotype file name
	string FN_geno_; // genotype file name
	string FN_sps_file_; // list of probabilities files
	string FN_prefix_; // prefix of output files
	
	METHOD_ method_; // 1. linear 2. logistic 3. coxph
	EXTRACT_TYPE_ extract_type_;
	bool flag_dosage_; // analysis dosage data in VCF file
	bool flag_robust_; // robust variance
	bool flag_uw_; // unweighted analysis
	bool flag_out_zip_; // zip output file
	bool flag_subset_;
	bool flag_strata_;
	bool flag_left_truncation_;
	bool flag_pairwise_inclusion_prob_;
	string formula_; // regression formula in the form Y=X1+X2+X3
	string id_col_; // ID column name 
	string family_col_; // family ID column name
	string weight_col_; // weight column name
	string subset_;
	string strata_;
	string left_truncation_col_;
	
	SNP_Analysis_Mode_ snp_analysis_mode_; // 1. standard 2. gene-environment 3. conditional
	string FN_COND_SNPS_;
	
	TEST_TYPE_ test_type_;
	string FN_group_;
	bool flag_group_;
	double group_maf_;
	double group_callrate_;
	RESCALE_TYPE_ rescale_type_;
	/**** I/O ***************************************************************************/
		
	/**** others variables **************************************************************/
	string FN_log_;
	ofstream FO_log_;	
	string FN_out_;
	string FN_score_mass_;
	string FN_score_snp_;
	string FN_score_group_;
	
	int N_Sample_geno_;
	int N_total_;
	int N_study_;
	int p_;
	int nadd_;
	int ncov_;
	int nhead_;
	
	vector<string> ENVI_names_; // environmenal covariates
	VectorXi ENVI_col_;
	
	vector<string> COND_names_; // conditional SNPs
	vector<string> COND_chrs_;
	vector<int32_t> COND_pos_;

	vector<VectorXd> Y_;
	vector< vector<CensoringData> > Y_cox_;
	vector<MatrixXd> W_;
	vector<vector<string>> F_; 
	vector<VectorXi> geno_pheno_linker_;
	vector<VectorXd> wt_; 	
	vector<MatrixXd> wtds_;
	vector<VectorXi> fam_ind_;
	vector<MatrixXd> strata_ind_;
	
	VcfFileReader VCF_reader_;
	VcfHeader VCF_header_;
	VcfRecord VCF_record_;
	string dosage_key_;
	
	string extract_chr_;
	int32_t extract_start_;
	int32_t extract_end_;
	string FN_extract_;
	/**** others variables **************************************************************/
		
	/**** functions *********************************************************************/
	SUGEN ();
	
	void CommandLineArgs_ (const int argc, char *argv[]);
	
	void InputData_LoadPheno_ (INPUT_UTILS& input_item);
	double ReadOneVariant_ (const int idx, VcfRecordGenotype& genoInfo);
	void InputData_CheckGeno_ (INPUT_UTILS& input_item);
	void InputData_LoadProb_ (INPUT_UTILS& input_item);
	void InputData_IDLinker_ (INPUT_UTILS& input_item);
	void InputData_PrepareAnalysis_ (INPUT_UTILS& input_item);
	void InputData_ ();
	
	void SingleVariantAnalysis_Initialization_ (SVA_UTILS& sva_item);
	void SingleVariantAnalysis_OutputSNPCountHeader_ (SVA_UTILS& sva_item, const SVA_OUTPUT_TYPE_ sva_output_type);
	void SingleVariantAnalysis_Output_ (SVA_UTILS& sva_item, const SVA_OUTPUT_TYPE_ sva_output_type);
	void SingleVariantAnalysis_GetSNP_ (SVA_UTILS& sva_item);
	void LinearWald_ (SVA_UTILS& sva_item);
	void LogisticWald_ (SVA_UTILS& sva_item);
	void CoxphWald_ (SVA_UTILS& sva_item);	
	void SingleVariantAnalysis_PerSNPAnalysis_ (SVA_UTILS& sva_item);
	void SingleVariantAnalysis_ ();

	void ScoreTests_GlobalInitialization_ (SVA_UTILS& sva_item);
	void LinearScoreNull_ (SVA_UTILS& sva_item);
	void LogisticScoreNull_ (SVA_UTILS& sva_item);
	void CoxphScoreNull_ (SVA_UTILS& sva_item);
	void ScoreTests_Output_ (SVA_UTILS& sva_item, const SVA_OUTPUT_TYPE_ sva_output_type);
	void ScoreTests_GetSNP_ (SVA_UTILS& sva_item);
	void CalculateUV_ (SVA_UTILS& sva_item); 
	void LinearScore_ (SVA_UTILS& sva_item);
	void LogisticScore_ (SVA_UTILS& sva_item);
	void CoxphScore_ (SVA_UTILS& sva_item);
	void CoxphLogRank_ (SVA_UTILS& sva_item);
	void ScoreTests_GroupAnalysis_ (SVA_UTILS& sva_item);
	void ScoreTests_PerSNPAnalysis_ (SVA_UTILS& sva_item);
	void ScoreTests_SNPAnalysis_ (SVA_UTILS& sva_item);
	void ScoreTests_ ();
	
	void Analysis_ ();
	/**** functions *********************************************************************/
};
/**** structures *****************************************************************************************************************************/

/**** functions ******************************************************************************************************************************/
void stdError (const string reason);
void error (ofstream& FO_log, const string reason);
double gammq (double a, double x);
void gser (double *gamser, double a, double x, double *gln);
void gcf (double *gammcf, double a, double x, double *gln);
double gammln (double xx);
bool CheckSinglularVar(const Ref<const MatrixXd>& vartheta); // Check if the covaraince matrix of the parameter estimates is singular or not

template <typename T1, typename T2> 
void dim (T1& nrow, T2& ncol, const string file, const bool header, ofstream& FO_log) {
	ifstream fin;
	char c, lastchar;

	nrow = 0;
	ncol = 0;
	lastchar = '\n';

	fin.open(file.c_str());

	if (!fin.is_open()) {
		error(FO_log, "Error: Cannot open file "+file);
	} else {
		fin >> noskipws >> c;
		while (true) {
			fin >> noskipws >> c;
			if (fin.eof()) {
				if (lastchar != '\n') nrow++;
				break;
			}
			if (c == '\t') ncol++;
			if (c == '\n') nrow++;
			lastchar = c;
		}
		fin.close();
	}

	ncol /= nrow;
	ncol++;
	if (header == true) nrow--;
} // dim

template <typename T1> 
void nrow (T1& nrow, const string file, const bool header, ofstream& FO_log) {
	unsigned long long ncol;
	dim(nrow, ncol, file, header, FO_log);
} // nrow

template <typename T>
vector<int> sortIndexes (const vector<T> &v) {

  /**** initialize original index locations *******************************************/
  vector<int> idx(v.size());
  for (int i = 0; i != idx.size(); ++i) idx[i] = i;
  /**** initialize original index locations *******************************************/

  /**** sort indexes based on comparing values in v ***********************************/
  sort(idx.begin(), idx.end(), [&v](int i1, int i2){return v[i1] < v[i2];});
  /**** sort indexes based on comparing values in v ***********************************/

  return idx;
} // sortIndexes

template<typename T>
void sortByIndexes (const MatrixBase<T>& m, const vector<int>& indx) {

	typedef typename internal::plain_matrix_type<T>::type MatrixType;
	MatrixType tmp = m;

	for (int i=0; i<m.rows(); i++) {
		const_cast< MatrixBase<T>& >(m).row(i) = tmp.row(indx[i]);
	}
} // sortByIndexes

template<typename T>
void sortByIndexes (vector<T>& m, const vector<int>& indx) {

	vector<T> tmp = m;

	for (int i=0; i<m.size(); i++) {
		m[i] = tmp[indx[i]];
	}
} // sortByIndexes
/**** functions ******************************************************************************************************************************/
 
#endif
