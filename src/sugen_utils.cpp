#include "sugen_utils.h"
#include <iostream>
#include <sstream>
#include <ctime>
#include <cstdio>
#include <cmath>
#include <ctgmath>
#include "FileReaderUtils/read_file_structures.h"
#include "FileReaderUtils/read_file_utils.h"
#include "FileReaderUtils/read_input.h"
#include "FileReaderUtils/read_table_with_header.h"
#include "StringUtils/string_utils.h"

using namespace file_reader_utils;
using namespace string_utils;

#define MISSIND -999.
#define FPMIN 1.00208e-292 // by gammq, gser, gcf, gammln
#define ITMAX 500 // by gammq, gser, gcf, gammln
#define EPS 0.00000000000000022204460492503 // by gammq, gser, gcf, gammln
#define TOL 1e-4 // by SUGEN::LogisticWald_, LinearWald_, CoxphWald_, LogisticScoreNull_, LinearScoreNull_, CoxphScoreNull_
#define MAX_ITER 500 // by SUGEN::LogisticWald_, LinearWald_, CoxphWald_, LogisticScoreNull_, LinearScoreNull_, CoxphScoreNull_
#define ERROR_MARGIN 1e-8 // by SUGEN::LinearWald_, LogisticWald_, CalculateUV_
#define NEWTON_RAPHSON_MAXIMUM_STEP 500 // by SUGEN::CoxphWald_, CoxphScoreNull_
#define STOP_MAXIMUM 5 // by SUGEN::CoxphWald_, CoxphScoreNull_

void stdError (const string reason) 
{	
	cout << reason << endl;
	cout << "Program was stopped due to error(s) listed above." << endl;
	exit(1);
} // stdError

void error (ofstream& FO_log, const string reason) 
{	
	FO_log << reason << endl;
	FO_log << "Program was stopped due to error(s) listed above." << endl;
	exit(1);	
} // error

double gammq (double a, double x) 
{	
    double gammcf, gamser, gln;

    if (x < 0.0 || a <= 0.0) {
        printf("Invalid arguments in routine gammp\n");
        exit(1);
    }

    if (x < (a + 1.0)) {
        gser(&gamser, a, x, &gln);
        return 1.0-gamser;
    } else {
        gcf(&gammcf, a, x, &gln);
        return gammcf;
    }
} // gammq

void gser (double *gamser, double a, double x, double *gln) 
{	
    int n;
    double ap, del, sum;

    *gln = gammln(a);
    if (x <= 0.0) {
        if (x < 0.0) {
            printf("x less than 0 in routine gser\n");
            exit(1);
        }
        *gamser = 0.0;
        return;
    } else {
        ap = a;
        del = sum = 1.0/a;
        for (n = 1; n <= ITMAX; n++) {
            ap = ap+1.0;
            del *= x/ap;
            sum += del;
            if (fabs(del) < fabs(sum)*EPS) {
                *gamser = sum*exp(-x + a*log(x)-(*gln));
                return;
            }
        }
        printf("a too large, ITMAX too small in routine gser\n");
        exit(1);
    }
} // gser

void gcf (double *gammcf, double a, double x, double *gln) 
{	
    /* Local variables */
    int i;
    double an, b, c, d, del, h;

    *gln = gammln(a);
    b = x + 1.0 - a;
    c = 1.0/FPMIN;
    d = 1.0/b;
    h = d;

    for (i = 1; i <= ITMAX; i++) {
        an = -i*(i-a);
        b += 2.0;
        d = an*d+b;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = b+an/c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0/d;
        del = d*c;
        h *= del;
        if (fabs(del-1.0) < EPS) break;
    }
    if (i > ITMAX) {
        printf("A too large, ITMAX too small\n");
        exit(1);
    }

    *gammcf = exp(-x+a*log(x)-(*gln))*h;
} // gcf

double gammln (double xx) 
{
    double x, y, tmp, ser;
    static const double cof[6] = { 76.18009172947146,-86.50532032941677,24.01409824083091,
            -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};

    int j;

    y = x = xx;
    tmp = x+5.5;
    tmp -= (x+0.5)*log(tmp);
    ser = 1.000000000190015;

    for (j=0;j<6;j++) ser += cof[j]/++y;

    return -tmp+log(2.5066282746310005*ser/x);
} // gammln

bool CheckSingularVar (const Ref<const MatrixXd>& vartheta) 
{	
	/**** Check if the covaraince matrix of the parameter *******************************/
	/**** estimates is valid or not *****************************************************/
	
	// check the positivity of diagonal elements
	for (int i=0; i<vartheta.cols(); i++) 
	{
		if (vartheta(i,i) <= 0. || ::isnan(vartheta(i,i))) 
		{
			return true;	
		}
	}
	
	// check if the matrix is symmetric or not
	MatrixXd tmp = vartheta.selfadjointView<Eigen::Upper>();
	if ((vartheta-tmp).sum() > 1.) 
	{
		return true;
	}
	return false;
} // CheckSingularVar

SUGEN::SUGEN () 
{	
	/**** I/O ***************************************************************************/
	FN_pheno_ = "pheno.txt";
	FN_geno_ = "geno.vcf.gz";
	FN_sps_file_ = "probmatrix.txt";
	FN_prefix_ = "results";
	method_ = LS;
	snp_analysis_mode_ = ST;
	extract_type_ = EXTRACT_TYPE_NONE;
	flag_robust_ = false;
	flag_dosage_ = false;
	flag_uw_ = false;
	flag_out_zip_ = false;
	flag_subset_ = false;
	flag_strata_ = false;
	flag_left_truncation_ = false;
	flag_pairwise_inclusion_prob_ = false;
	flag_ge_full_output_ = false;
	FN_COND_SNPS_ = "NULL";
	formula_ = "NULL";
	id_col_ = "IID";
	family_col_ = "FID";
	weight_col_ = "WT";
	subset_ = "NULL";
	strata_ = "NULL";
	test_type_ = WALD;
	FN_group_ = "NULL";
	flag_group_ = false;
	group_maf_ = 0.05;
	group_callrate_ = 0.0;
	rescale_type_ = NAIVE;
	/**** I/O ***************************************************************************/
	
	/**** other variables ***************************************************************/
	dosage_key_ = "DS";
	/**** other variables ***************************************************************/
} // SUGEN::SUGEN

SVA_UTILS::SVA_UTILS()
{
	newton_raphson_step_size_ = NEWTON_RAPHSON_MAXIMUM_STEP;
}

void SUGEN::CommandLineArgs_ (const int argc, char *argv[]) 
{			
	bool flag_condition = false;
	bool flag_GE = false;
	bool flag_formula = false;
	bool flag_extract_chr = false;
	bool flag_extract_range = false;
	bool flag_extract_file = false;
	bool flag_group_maf = false;
	bool flag_group_callrate = false;
	bool flag_score_rescale = false;
	time_t now = time(NULL);
	
	/**** command line input ************************************************************/
	for (int i=1; i<argc; i++) 
	{
		if (strcmp(argv[i], "--pheno") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--pheno'!");
			}
			FN_pheno_ = StripQuotes(string(argv[++i])); 
			continue;
		}
		if (strcmp(argv[i], "--vcf") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--vcf'!");
			}
			FN_geno_ = StripQuotes(string(argv[++i])); 
			continue;
		}
		if (strcmp(argv[i], "--probmatrix") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--probmatrix'!");
			}
			flag_pairwise_inclusion_prob_ = true;
			FN_sps_file_ = StripQuotes(string(argv[++i])); 
			continue;
		}
		if (strcmp(argv[i], "--out-prefix") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--out-prefix'!");
			}
			FN_prefix_ = StripQuotes(string(argv[++i]));
			FN_log_ = FN_prefix_+".log";
			continue;
		}		
		if (strcmp(argv[i], "--out-zip") == 0) 
		{
			flag_out_zip_ = true;
			continue;
		}	
		if (strcmp(argv[i], "--formula") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--formula'!");
			}
			formula_ = StripQuotes(string(argv[++i]));
			flag_formula = true;			
			continue;
		}	
		if (strcmp(argv[i], "--id-col") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--id-col'!");
			}
			id_col_ = StripQuotes(string(argv[++i])); 
			continue;
		}
		if (strcmp(argv[i], "--family-col") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--family-col'!");
			}
			family_col_ = StripQuotes(string(argv[++i])); 
			continue;
		}
		if (strcmp(argv[i], "--weight-col") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--weight-col'!");
			}
			weight_col_ = StripQuotes(string(argv[++i])); 
			continue;
		}
		if (strcmp(argv[i], "--subset") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--subset'!");
			}
			subset_ = StripQuotes(string(argv[++i]));
			flag_subset_ = true;
			continue;
		}
		if (strcmp(argv[i], "--hetero-variance") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--hetero-variance'!");
			}
			strata_ = StripQuotes(string(argv[++i]));
			flag_strata_ = true;
			continue;
		}
		if (strcmp(argv[i], "--ge") == 0) {			
			if (i == argc - 1) 
			{
				stdError("Error: Expected environmental variables after '--ge', multiple variables should be separated by comma!");
			}
			string ENVI = StripQuotes(string(argv[++i]));
			if (!Split(ENVI, ",", &ENVI_names_)) 
			{
				stdError("Error: Cannot parse --ge argument "+ENVI+"!");
			}
			flag_GE = true;
			continue;
		}
		if (strcmp(argv[i], "--ge-output-detail") == 0)
		{
			flag_ge_full_output_ = true;
			continue;
		}		
		if (strcmp(argv[i], "--dosage") == 0)
		{
			flag_dosage_ = true;
			continue;
		}		
		if (strcmp(argv[i], "--unweighted") == 0) 
		{
			flag_uw_ = true;
			continue;
		}		
		if (strcmp(argv[i], "--model") == 0)
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--model'!");
			}
			i++;
			if (strcmp(argv[i], "linear") == 0) 
			{
				method_ = LS;
			} 
			else if (strcmp(argv[i], "logistic") == 0) 
			{
				method_ = logistic;		
			}
			else if (strcmp(argv[i], "coxph") == 0)
			{
				method_ = right_censored;
			}
			else 
			{
				stdError("Error: Unknown parameter for option --model "+string(argv[i])+"!");
			}
			continue;
		}
		if (strcmp(argv[i], "--left-truncation") == 0)
		{
			if (i == argc-1)
			{
				stdError("Error: Expected argument after '--left-truncation'!");
			}
			left_truncation_col_ = argv[++i];
			flag_left_truncation_ = true;
			continue;
		}
		if (strcmp(argv[i], "--robust-variance") == 0) 
		{
			flag_robust_ = true;
			continue;
		}	
		if (strcmp(argv[i], "--cond") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--cond'!");
			}
			FN_COND_SNPS_ = StripQuotes(string(argv[++i]));
			flag_condition = true;
			continue;
		}	
		if (strcmp(argv[i], "--extract-chr") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--extract-chr'!");
			}
			extract_chr_ = argv[++i];
			flag_extract_chr = true;
			continue;
		}
		if (strcmp(argv[i], "--extract-range") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--extract-range'!");
			}
			string range = StripQuotes(string(argv[++i]));
			vector<string> range_vec;
			if (!Split(range, "-", &range_vec)) 
			{
				stdError("Error: Cannot parse --extract-range argument "+range+"!");
			}			
			extract_start_ = atoi(range_vec[0].c_str());
			extract_end_ = atoi(range_vec[1].c_str());
			flag_extract_range = true;
			continue;
		}		
		if (strcmp(argv[i], "--extract-file") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--extract-file'!");
			}
			FN_extract_ = StripQuotes(string(argv[++i]));
			flag_extract_file = true;
			continue;
		}		
		if (strcmp(argv[i], "--score") == 0) 
		{
			test_type_ = SCORE;
			continue;
		}
		if (strcmp(argv[i], "--score-rescale") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--score-rescale'!");
			}
			i++;
			if (strcmp(argv[i], "naive") == 0) 
			{
				rescale_type_ = NAIVE;
			} 
			else if (strcmp(argv[i], "optimal") == 0) 
			{
				rescale_type_ = OPTIMAL;
			} 
			else 
			{
				stdError("Error: Unknown parameter for option --score-rescale "+string(argv[i])+"!");
			}
			flag_score_rescale = true;
			continue;
		}		
		if (strcmp(argv[i], "--group") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--group'!");
			}
			FN_group_ = StripQuotes(string(argv[++i]));
			flag_group_ = true;
			continue;
		}
		if (strcmp(argv[i], "--group-maf") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--group-maf'!");
			}
			group_maf_ = atof(argv[++i]);
			flag_group_maf = true;
			continue;
		}
		if (strcmp(argv[i], "--group-callrate") == 0) 
		{
			if (i == argc - 1) 
			{
				stdError("Error: Expected argument after '--group-callrate'!");
			}
			group_callrate_ = atof(argv[++i]);
			flag_group_callrate = true;
			continue;
		}
		stdError("Error: Unknown command line option "+string(argv[i])+"!");
	}
	
	if (test_type_ == WALD) 
	{
		if (flag_GE && flag_condition) 
		{
			stdError("Error: Can only specify one of '--cond' and '--ge'!");
		} 
		else if (flag_GE && !flag_condition) 
		{
			snp_analysis_mode_ = GE;
		} 
		else if (flag_condition && !flag_GE) 
		{
			snp_analysis_mode_ = COND;
		}
		if (flag_group_ || flag_group_callrate || flag_group_maf)
		{
			stdError("Error: Can only specify '--group', '--group-maf', or '--group-callrate' when '--score' is specified!");
		}
		if (flag_score_rescale)
		{
			stdError("Error: Can only specify '--score-rescale' when '--score' is specified!");
		}
	} 
	else if (test_type_ == SCORE)
	{
		if (flag_GE || flag_condition) 
		{
			stdError("Error: Can only specify '--ge' or '--cond' when '--score' is not specified!");
		}
	}
	
	if (!flag_formula) 
	{
		stdError("Error: --formula must be specified!");
	}
	if ((method_ == logistic || method_ == right_censored) && flag_strata_)
	{
		stdError("Error: --hetero-variance is not needed for logistic regression or Cox proportional hazards regression!");
	}
	if ((method_ == LS || method_ == logistic) && flag_left_truncation_)
	{
		stdError("Error: --left-truncation is an option for Cox proportional hazards regression only!");
	}
	if (flag_group_) 
	{
		if (flag_extract_chr || flag_extract_range || flag_extract_file)
		{
			stdError("Error: --extract-chr, --extract-range, or --extract-file cannot be specified with --group together!");
		}			
	}
	else
	{
		if (flag_group_maf || flag_group_callrate)
		{
			stdError("Error: --group-maf or --group-callrate cannot be specified when --group is not specified!");
		}		
		if (flag_extract_chr && !flag_extract_range && !flag_extract_file) 
		{
			extract_type_ = EXTRACT_TYPE_CHR;
		} 
		else if (flag_extract_chr && flag_extract_range && !flag_extract_file) 
		{
			extract_type_ = EXTRACT_TYPE_RANGE;
		} 
		else if (!flag_extract_chr && !flag_extract_range && flag_extract_file) 
		{
			extract_type_ = EXTRACT_TYPE_FILE;
		} 
		else if (!flag_extract_chr && flag_extract_range && !flag_extract_file) 
		{
			stdError("Error: --extract-chr must also be specified!");
		} 
		else if (!flag_extract_chr && !flag_extract_range && !flag_extract_file) 
		{
			extract_type_ = EXTRACT_TYPE_NONE;
		} 
		else 
		{
			stdError("Error: --extract-chr or --extract-range cannot be used with --extract-file together!");
		}
	}
	/**** command line input ************************************************************/
	
	/**** command line input summary ****************************************************/
	FO_log_.open(FN_log_.c_str());
	if (!FO_log_.is_open()) 
	{
		stdError("Can not open log file " + FN_log_ + " !");
	}		
	FO_log_ << "SUGEN v8.5 (03/21/2017) starts at " << asctime(localtime(&now)) << endl;
	FO_log_ << "Author: Ran Tao" << endl;
	FO_log_ << "Email: r.tao@vanderbilt.edu" << endl;
	FO_log_ << "Documentation & citation: https://github.com/dragontaoran/SUGEN" << endl << endl;

	FO_log_ << "The phenotype file is " << FN_pheno_ << "." << endl;
	FO_log_ << "The formula is " << formula_ << "." << endl;
	FO_log_ << "The suject ID column in the phenotype file is " << id_col_ << "." << endl;
	FO_log_ << "The family ID column in the phenotype file is " << family_col_ << "." << endl;
	if (flag_subset_)
	{
		FO_log_ << "The subsetting rule is: " << subset_ << "." << endl;
	}
	FO_log_ << endl;
	
	FO_log_ << "The VCF file is " << FN_geno_ << "." << endl;
	if (flag_dosage_)
	{
		FO_log_ << "Analyze dosage data in the VCF file." << endl;
	}
	FO_log_ << endl;
	
	if (flag_uw_) 
	{
		FO_log_ << "Unweighted analysis will be performed." << endl;
	} 
	else 
	{
		FO_log_ << "Weighted analysis will be performed." << endl;
		FO_log_ << "The weight column in the phenotype file is " << weight_col_ << "." << endl;
		if (flag_pairwise_inclusion_prob_)
		{
			FO_log_ << "The pairwise inclusion probabilities file is " << FN_sps_file_ << "." << endl;
		}
		else
		{
			FO_log_ << "No pairwise inclusion probabilities are provided." << endl;
		}
	}
	FO_log_ << endl;
	
	FO_log_ << "The log file is " << FN_log_ << "." << endl << endl;
	
	if (method_ == LS) 
	{
		FO_log_ << "Perform linear regression." << endl;
		if (flag_strata_)
		{
			FO_log_ << "Allow residual variance in different groups (defined by " << strata_ << ") to be different." << endl;
		}
	} 
	else if (method_ == logistic)
	{
		FO_log_ << "Perform logistic regression." << endl;
	}
	else if (method_ == right_censored)
	{
		FO_log_ << "Perform Cox proportional harzards regression." << endl;
		if (flag_left_truncation_)
		{
			FO_log_ << "The left-truncation variable is " << left_truncation_col_ << "." << endl;
		}
	}
	
	if (flag_robust_) 
	{
		FO_log_ << "Use robust variance estimators." << endl;
	} 
	else 
	{
		FO_log_ << "Use model-based variance estimators." << endl << endl;
	}
	
	if (!flag_group_) 
	{
		if (extract_type_ == EXTRACT_TYPE_CHR) 
		{
			FO_log_ << "Restrict analysis to variants in chromosome " << extract_chr_ << "." << endl;
		} 
		else if (extract_type_ == EXTRACT_TYPE_RANGE) 
		{
			FO_log_ << "Restrict analysis to variants in chromosome " << extract_chr_ << ", from position " << extract_start_ << " to position " << extract_end_ << "." << endl;
		} 
		else if (extract_type_ == EXTRACT_TYPE_FILE) 
		{
			FO_log_ << "Restrict analysis to variants in " << FN_extract_ << "." << endl;
		}
	}
	
	if (test_type_ == WALD) 
	{
		FO_log_ << "Perform Wald tests." << endl;				
		if (snp_analysis_mode_ == GE) 
		{
			FO_log_ << "Perform gene by environment interaction analysis." << endl;
			FO_log_ << "The environment variables are:";
			for (int i=0; i<ENVI_names_.size(); i++) 
			{
				FO_log_ << ' ' << ENVI_names_[i];
			}
			FO_log_ << "." << endl;
			if (flag_ge_full_output_)
			{
				FO_log_ << "Output the covariances of the regression coefficient estimators of the SNP, environment variables, and SNP by environment interaction variables.";
				FO_log_ << endl;
			}
		} 
		else if (snp_analysis_mode_ == COND) 
		{
			FO_log_ << "Perform conditional analysis conditioning on the SNPS in " << FN_COND_SNPS_ << "." << endl; 
		} 
		else 
		{
			FO_log_ << "Perform standard association analysis." << endl;
		}
		
		if (flag_out_zip_) 
		{
			FN_out_ = FN_prefix_+".wald.out.gz";
		} 
		else 
		{
			FN_out_ = FN_prefix_+".wald.out";
		}
		FO_log_ << "The results file is " << FN_out_ << "." << endl;
	}
	else if (test_type_ == SCORE)
	{
		FO_log_ << "Perform score tests." << endl;
		if (flag_group_)
		{
			FO_log_ << "Perform group-based analysis." << endl;
			FO_log_ << "The group file is " << FN_group_ << "." << endl;
			FO_log_ << "variants with call rates greater than or equal to " << group_callrate_ << " are included." << endl;
			FO_log_ << "variants with minor allele frequencies smaller than or equal to " << group_maf_ << " are included." << endl;
		} 
		else
		{
			FO_log_ << "Perform single-variant analysis." << endl;
		}
		
		if (flag_score_rescale) 
		{
			if (rescale_type_ == NAIVE) 
			{
				FO_log_ << "The score statistics and its corresponding covariances are rescaled according to the naive rule." << endl;
			}
			else
			{
				FO_log_ << "The score statistics and its corresponding covariances are rescaled according to the optimal rule." << endl;
			}
		}

		if (flag_out_zip_) 
		{
			FN_score_mass_ = FN_prefix_+".mass.out.gz";
			FN_score_snp_ = FN_prefix_+".score.snp.out.gz";
		} 
		else 
		{
			FN_score_mass_ = FN_prefix_+".mass.out";
			FN_score_snp_ = FN_prefix_+".score.snp.out";
		}
		FO_log_ << "The mass output file is " << FN_score_mass_ << "." << endl;		
		if (flag_group_)
		{
			FO_log_ << "The SNP output file is " << FN_score_snp_ << "." << endl;
		}
		else 
		{
			FO_log_ << "The SNP output file is " << FN_score_snp_ << "." << endl;
		}
	}
	/**** command line input summary ****************************************************/
	
	FO_log_ << endl;
	FO_log_.close();
} // SUGEN::CommandLineArgs_

void SUGEN::InputData_LoadPheno_ (INPUT_UTILS& input_item) 
{	
	FO_log_ << "Loading phenotype data..." << endl;
			
	/**** link the ModelAndDataParams obeject params ************************************/
	ModelAndDataParams params;
	if (method_ == LS || method_ == logistic)
	{
		params.model_type_ = ModelType::MODEL_TYPE_LINEAR;
	}
	else if (method_ == right_censored)
	{
		params.model_type_ = ModelType::MODEL_TYPE_RIGHT_CENSORED_SURVIVAL;
		if (flag_left_truncation_)
		{
			params.left_truncation_str_ = left_truncation_col_;
		}
	}
	params.file_.na_strings_.insert("NA");
	params.file_.name_ = FN_pheno_;
	params.model_str_ = formula_;
	params.id_str_ = id_col_;
	if (flag_strata_)
	{
		params.family_str_ = family_col_+","+strata_;
	}
	else
	{
		params.family_str_ = family_col_;
	}
	params.standardize_vars_ = VariableNormalization::VAR_NORM_NONE;
	if (!flag_uw_) 
	{
		params.weight_str_ = weight_col_;
	}
	if (flag_subset_)
	{
		params.subgroup_str_ = subset_;
	}
	if (!ReadInput::FillModelAndDataParams(&params)) 
	{
		error(FO_log_, params.error_msg_);
	}
	/**** link the ModelAndDataParams obeject params ************************************/
		
	/**** read data *********************************************************************/
	int N = params.ids_.size();
	if (N == 0) 
	{
		error(FO_log_, "Error: No Samples found. Check phenotype file.");
	}
	if (method_ == LS || method_ == logistic)
	{
		ncov_ = params.linear_term_values_.cols()-1;
	}
	else if (method_ == right_censored)
	{
		ncov_ = params.linear_term_values_.cols();
	}
	
	if (ncov_ > 0) 
	{
		if (snp_analysis_mode_ == GE) 
		{
			ENVI_col_.resize(ENVI_names_.size());
			ENVI_col_.setConstant(-1);
			input_item.rawCOV_name_.clear();
			for (int i=1; i<=ncov_; i++) 
			{
				if (method_ == LS || method_ == logistic)
				{
					input_item.rawCOV_name_[params.legend_[i]] = i-1;
				}
				else if (method_ == right_censored)
				{
					input_item.rawCOV_name_[params.legend_[i-1]] = i-1;
				}
			}
			for (int i=0; i<ENVI_col_.size(); i++) 
			{
				if (input_item.rawCOV_name_.count(ENVI_names_[i])) 
				{
					ENVI_col_(i) = input_item.rawCOV_name_[ENVI_names_[i]];
				} 
				else 
				{
					error(FO_log_, "Error: The environment variable "+ENVI_names_[i]+" is not in the regression formula!");
				}
			}
		}
		input_item.rawW_ = params.linear_term_values_.rightCols(ncov_);
	}
	
	input_item.rawPheno_ID_.clear();
	
	if (method_ == LS || method_ == logistic)
	{
		input_item.rawY_.resize(N);
	}
	else if (method_ == right_censored)
	{
		input_item.rawY_cox_.resize(N);
	}
	
	input_item.rawF_.resize(N);
	
	if (!flag_uw_) 
	{
		input_item.rawWT_.resize(N);
	}
	
	if (flag_strata_)
	{
		input_item.rawStrata_.resize(N);
	}
	
	for (int i=0; i<N; i++) 
	{
		input_item.rawPheno_ID_[params.ids_[i]] = i;
		
		if (method_ == LS || method_ == logistic)
		{
			input_item.rawY_(i) = params.dep_vars_.dep_vars_linear_[i];
		}
		else if (method_ == right_censored)
		{
			input_item.rawY_cox_[i] = params.dep_vars_.dep_vars_cox_[i];
		}
		
		if (method_ == logistic && fabs(input_item.rawY_(i)-0.0) > 1e-8 && fabs(input_item.rawY_(i)-1.) > 1e-8) 
		{
			error(FO_log_, "Error: In logistic regression, the trait variable is not coded as 0/1 in the phenotype file!");
		}
		
		input_item.rawF_[i] = params.families_[i][0];
		if (flag_strata_)
		{
			input_item.rawStrata_[i] = params.families_[i][1];
		}
		
		if (!flag_uw_) 
		{
			input_item.rawWT_(i) = params.weights_[i];
		}
	}
	/**** read data *********************************************************************/
		
	FO_log_ << "The phenotype file contains " << N << " individual(s) with complete records." << endl;
	FO_log_ << "There are " << ncov_ << " covariate(s) included in the analysis." << endl;
	FO_log_ << "Done!" << endl << endl;
} // SUGEN::InputData_LoadPheno_

double SUGEN::ReadOneVariant_ (const int idx, VcfRecordGenotype& genoInfo) 
{
	int numGTs;
	double genotype;
	
	if (flag_dosage_)
	{
		const string *dosage_tmp = genoInfo.getString(dosage_key_.c_str(), idx);
		if(strcmp((*dosage_tmp).c_str(), ".") > 0)
		{
			genotype = atof((*dosage_tmp).c_str());
		}
		else
		{
			genotype = MISSIND;
		}
	}
	else
	{
		numGTs = VCF_record_.getNumGTs(idx);
		genotype = 0;
		for (int j=0; j<numGTs; j++) 
		{
			if(VCF_record_.getGT(idx,j) == VcfGenotypeSample::MISSING_GT) 
			{
				genotype = MISSIND;
				break;
			}
			genotype += VCF_record_.getGT(idx,j);
		}
	}
	return genotype;
} // SUGEN::ReadOneVariant_

void SUGEN::InputData_CheckGeno_ (INPUT_UTILS& input_item) 
{	
	FO_log_ << "Processing the VCF file..." << endl;
	
	/**** check dimension ***************************************************************/
	VCF_reader_.open(String(FN_geno_.c_str()), VCF_header_);		
	N_Sample_geno_ = VCF_header_.getNumSamples();
	FO_log_ << "The VCF file contains " << N_Sample_geno_ << " individual(s)." << endl;
	VCF_reader_.close();
	/**** check dimension ***************************************************************/
	
	/**** read subject ID ***************************************************************/		
	input_item.rawGeno_ID_.resize(N_Sample_geno_);
	for(int i=0; i<N_Sample_geno_; i++) 
	{
	  input_item.rawGeno_ID_[i] = string(VCF_header_.getSampleName(i));
	}
	/**** read subject ID ***************************************************************/
	
	/**** read conditional SNPs *********************************************************/
	if (snp_analysis_mode_ == COND) 
	{	
		int NSNP;
		int32_t pos;
		string SNP_name;
		vector<string> chr_pos;
		ifstream FI;
		vector<double> SNPGeno(N_Sample_geno_, 0.);
		
		input_item.rawCOND_.clear();
		input_item.rawCOND_Miss_.setOnes(N_Sample_geno_);
		COND_names_.clear();
		COND_chrs_.clear();
		COND_pos_.clear();
		
		nrow(NSNP, FN_COND_SNPS_, false, FO_log_); 
		
		FI.open(FN_COND_SNPS_);
		VCF_reader_.open(String(FN_geno_.c_str()), VCF_header_);
		VCF_reader_.readVcfIndex();
				
		for (int i=0, nsnp=0; i<NSNP; i++) 
		{		
			FI >> SNP_name;
			chr_pos.clear();
			if (!Split(SNP_name, ":", &chr_pos)) 
			{
				stdError("Error: Cannot parse SNP "+SNP_name+" in "+FN_COND_SNPS_+"!\n");
			}
			pos = atoi(chr_pos[1].c_str());
			VCF_reader_.set1BasedReadSection(chr_pos[0].c_str(), pos, pos+1);
			if (VCF_reader_.readRecord(VCF_record_)) 
			{	
				if (VCF_record_.getNumAlts() > 1) 
				{
					error(FO_log_, "Error: In conditional analysis, SNP "+SNP_name+" is multiallelic!");
				}
				VcfRecordGenotype & genoInfo = VCF_record_.getGenotypeInfo();
				
				COND_names_.push_back(SNP_name);
				COND_chrs_.push_back(chr_pos[0]);
				COND_pos_.push_back(pos);
				input_item.rawCOND_.push_back(SNPGeno);
				
				for (int i=0; i<N_Sample_geno_; i++) 
				{
					input_item.rawCOND_[nsnp][i] = ReadOneVariant_(i, genoInfo);
					if(input_item.rawCOND_[nsnp][i] == MISSIND) 
					{
						input_item.rawCOND_Miss_(i) = 0;
					}		
				}
				nsnp ++;
			}
		}
				
		nadd_ = COND_names_.size();
		if (nadd_ > 0) 
		{
			FO_log_ << "variants ";
			for (int i=0; i<nadd_; i++) 
			{
				FO_log_ << COND_names_[i] << " ";
			}
			FO_log_ << "in "+FN_COND_SNPS_+" are present in the VCF file and thus are included in the conditional analysis." << endl;
		} 
		else 
		{
			FO_log_ << "Warning: There is no SNP in "+FN_COND_SNPS_+" that is also in the VCF file!" << endl;
			FO_log_ << "Warning: No conditional analysis will be performed!" << endl;
			exit(0);
		}
		
		FI.close();
		VCF_reader_.close();						
	}
	/**** read conditional SNPs *********************************************************/
	
	FO_log_ << "Done!" << endl << endl;
} // SUGEN::InputData_CheckGeno_

void SUGEN::InputData_LoadProb_ (INPUT_UTILS& input_item) 
{
	if (!flag_uw_ && flag_pairwise_inclusion_prob_) {
		FO_log_ << "Loading pairwise probability data..." << endl;
		
		/**** initialization ************************************************************/
		ifstream FI_sps_file, FI_sps;
		int NR_sps_file, NR_sps, NC_sps;
		/**** initialization ************************************************************/
		
		/**** check the number of pairwise inclusion probability matrices ***************/
		nrow(NR_sps_file, FN_sps_file_, false, FO_log_);
		if (NR_sps_file == 0) {
			error(FO_log_, "Error: There is no pairwise inclusion matrix specified!");
		} else if (NR_sps_file == 1) {
			FO_log_ << "The pairwise inclusion probabilities file contains " << NR_sps_file << " pairwise inclusion probability matrix." << endl;
		} else if (NR_sps_file >=1) {
			FO_log_ << "The pairwise inclusion probabilities file contains " << NR_sps_file << " pairwise inclusion probability matrices." << endl;
		}
		/**** check the number of pairwise inclusion probability matrices ***************/
		
		/**** read pairwise inclusion probability matrices ******************************/
		FI_sps_file.open(FN_sps_file_.c_str());
		input_item.rawSPS_.resize(NR_sps_file);
		input_item.rawSPS_ID_.resize(NR_sps_file);
		input_item.FN_sps_.resize(NR_sps_file);
		for (int nstudy=0; nstudy<NR_sps_file; nstudy++) {
			FI_sps_file >> input_item.FN_sps_[nstudy];
			dim(NR_sps, NC_sps, input_item.FN_sps_[nstudy], true, FO_log_);
			if (NR_sps != NC_sps) {
				error(FO_log_, "Error: Pairwise inclusion probability in file "+input_item.FN_sps_[nstudy]+" is not a square matrix!");
			}
			FO_log_ << "The pairwise inclusion probability matrix " << input_item.FN_sps_[nstudy] << " contains " << NR_sps << " individual(s)." << endl;
			input_item.rawSPS_[nstudy].resize(NR_sps, NC_sps);
			input_item.rawSPS_ID_[nstudy].resize(NC_sps);
			FI_sps.open(input_item.FN_sps_[nstudy].c_str());
			for (int i=0; i<NC_sps; i++) {
				FI_sps >> input_item.rawSPS_ID_[nstudy][i];
			}		
			for (int i=0; i<NR_sps; i++) {
				for (int j=0; j<NC_sps; j++) {
					FI_sps >> input_item.rawSPS_[nstudy](i,j);
				}
			}
			FI_sps.close();
		}
		FI_sps_file.close();
		/**** read pairwise inclusion probability matrices ******************************/			
		FO_log_ << "Done!" << endl << endl;
	}
} // SUGEN::InputData_LoadProb_

void SUGEN::InputData_IDLinker_ (INPUT_UTILS& input_item) 
{	
	int N;
	if (method_ == LS || method_ == logistic)
	{
		N = input_item.rawY_.size();
	}
	else if (method_ == right_censored)
	{
		N = input_item.rawY_cox_.size();
	}
	
	/**** link genotype and phenotype IDs ***********************************************/
	input_item.raw_geno_pheno_linker_.resize(N);
	input_item.raw_geno_pheno_linker_.setConstant(-1);	
	for (int i=0; i<input_item.rawGeno_ID_.size(); i++) 
	{
		if (input_item.rawPheno_ID_.count(input_item.rawGeno_ID_[i])) 
		{
			if (test_type_ == WALD && snp_analysis_mode_ == COND) 
			{
				if (input_item.rawCOND_Miss_(i) == 1)
				{
					input_item.raw_geno_pheno_linker_(input_item.rawPheno_ID_[input_item.rawGeno_ID_[i]]) = i;
				}
			}
			else
			{
				input_item.raw_geno_pheno_linker_(input_item.rawPheno_ID_[input_item.rawGeno_ID_[i]]) = i;
			}
		}
	}
	if ((input_item.raw_geno_pheno_linker_.array() > -1).count() <= 0) {
		error(FO_log_, "Error: There are problems in linking the phenotype and VCF files!");
	}
	/**** link genotype and phenotype IDs ***********************************************/
	
	/**** link probability and phenotype IDs ********************************************/
	if (!flag_uw_ && flag_pairwise_inclusion_prob_) {
		const int N_study = input_item.rawSPS_.size();
		input_item.raw_prob_pheno_linker_.resize(N, N_study);
		input_item.raw_prob_pheno_linker_.setConstant(-1);
		for (int nstudy=0; nstudy<N_study; nstudy++) {
			for (int i=0; i<input_item.rawSPS_ID_[nstudy].size(); i++) {
				if (input_item.rawPheno_ID_.count(input_item.rawSPS_ID_[nstudy][i])) {
					input_item.raw_prob_pheno_linker_(input_item.rawPheno_ID_[input_item.rawSPS_ID_[nstudy][i]], nstudy) = i;
				}
			}
		}
		if ((input_item.raw_prob_pheno_linker_.array() > -1).count() <= 0) {
			error(FO_log_, "Error: There are problems in linking the phenotype file and pairwise inclusion probability matrices!");
		}
	}
	/**** link probability and phenotype IDs ********************************************/
} // SUGEN::InputData_IDLinker_

void SUGEN::InputData_PrepareAnalysis_ (INPUT_UTILS& input_item) 
{	
	FO_log_ << "Preparing analysis..." << endl;

	/**** initialization ****************************************************************/
	int length_rawY, nfam, rawN1, N_study_pseudo;
	vector<int> idindx;
	bool flag_uw_decision;
	VectorXi prob_pheno_linker;
	MatrixXd wts, wtps;
	vector<string> strata_tmp; // hetero-variance
	int numStrata; // hetero-variance
	map<string, vector<int>> strata_map; // hetero-variance
	vector<string> strata_unique; // hetero-variance
	
	if (method_ == LS || method_ == logistic)
	{
		length_rawY = input_item.rawY_.size();
	}
	else if (method_ == right_censored)
	{
		length_rawY = input_item.rawY_cox_.size();
	}
	/**** initialization ****************************************************************/
		
	/**** count the number of valid studies *********************************************/
	if (flag_uw_) 
	{
		N_study_ = 1;
	} 
	else if (flag_pairwise_inclusion_prob_)
	{
		N_study_ = 0;
		for (int nstudy=0; nstudy<input_item.FN_sps_.size(); nstudy++) 
		{
			if ((input_item.raw_prob_pheno_linker_.col(nstudy).array() > -1).count() > 0) 
			{
				N_study_++;
			}
		}
	}
	else
	{
		N_study_ = 1;
	}
	/**** count the number of valid studies *********************************************/
	
	/**** data initialization ***********************************************************/
	if (method_ == LS || method_ == logistic)
	{
		Y_.resize(N_study_);
	}
	else if (method_ == right_censored)
	{
		Y_cox_.resize(N_study_);
	}
	if (!(method_ == right_censored && test_type_ == SCORE && ncov_ == 0))
	{
		W_.resize(N_study_);
	}
	geno_pheno_linker_.resize(N_study_);
	F_.resize(N_study_);
	fam_ind_.resize(N_study_);
	if (!flag_uw_) 
	{
		wt_.resize(N_study_);
		if (flag_pairwise_inclusion_prob_)
		{
			wtds_.resize(N_study_);
		}
	}
	if (flag_strata_)
	{
		strata_ind_.resize(N_study_);
	}
	/**** data initialization ***********************************************************/
	
	/**** count the number of covariates in the regression model ************************/
	if (test_type_ == WALD) 
	{
		if (snp_analysis_mode_ == ST) 
		{
			nadd_ = 0;
		} 
		else if (snp_analysis_mode_ == GE) 
		{
			nadd_ = ENVI_col_.size();
		} 
		else if (snp_analysis_mode_ == COND) 
		{
			nadd_ = input_item.rawCOND_.size();
		}
		
		if (method_ == LS || method_ == logistic)
		{
			nhead_ = 2;		
		}
		else if (method_ == right_censored)
		{
			nhead_ = 1;
		}
		p_ = nhead_+nadd_+ncov_;
	}
	else if (test_type_ == SCORE)
	{
		if (method_ == LS || method_ == logistic)
		{
			nhead_ = 1;	
		}
		else if (method_ == right_censored)
		{
			nhead_ = 0;
		}
		p_ = nhead_+ncov_;
	}
	/**** count the number of covariates in the regression model ************************/
		
	N_total_ = 0;
	if (flag_uw_)
	{
		N_study_pseudo = 1;
	}
	else if (flag_pairwise_inclusion_prob_)
	{
		N_study_pseudo = input_item.FN_sps_.size();
	}
	else
	{
		N_study_pseudo = 1;
	}

	for (int nstudy=0, nstudy1=0; nstudy<N_study_pseudo; nstudy++) 
	{	
		if (flag_uw_) 
		{
			N_total_ = rawN1 = (input_item.raw_geno_pheno_linker_.array() > -1).count(); 
		} 
		else if (flag_pairwise_inclusion_prob_)
		{
			rawN1 = 0;
			for (int i=0; i<length_rawY; i++) 
			{
				if (input_item.raw_prob_pheno_linker_(i,nstudy) > -1 && input_item.raw_geno_pheno_linker_(i) > -1) 
				{
					rawN1++;
				}
			}
			N_total_ += rawN1;
		}
		else
		{
			N_total_ = rawN1 = (input_item.raw_geno_pheno_linker_.array() > -1).count();
		}

		if (rawN1 > 0) 
		{
			if (method_ == LS || method_ == logistic)
			{
				Y_[nstudy1].resize(rawN1);
			}
			else if (method_ == right_censored)
			{
				Y_cox_[nstudy1].resize(rawN1);
			}
			if (p_ > 0)
			{
				W_[nstudy1].resize(rawN1, p_);
			}
			F_[nstudy1].resize(rawN1);
			geno_pheno_linker_[nstudy1].resize(rawN1);
			idindx.resize(rawN1);
			if (!flag_uw_) 
			{
				if (flag_pairwise_inclusion_prob_)
				{
					prob_pheno_linker.resize(rawN1);
				}
				wt_[nstudy1].resize(rawN1);
			}
			if (flag_strata_)
			{
				strata_tmp.clear();
				strata_tmp.resize(rawN1);
			}
			
			for (int i=0, j = 0; i<length_rawY; i++) 
			{
				if (flag_uw_)
				{
					flag_uw_decision = (input_item.raw_geno_pheno_linker_(i) != -1);
				}
				else if (flag_pairwise_inclusion_prob_)
				{
					flag_uw_decision = (input_item.raw_geno_pheno_linker_(i) != -1 && input_item.raw_prob_pheno_linker_(i,nstudy) != -1);
				}
				else
				{
					flag_uw_decision = (input_item.raw_geno_pheno_linker_(i) != -1);
				}
				
				if (flag_uw_decision) 
				{
					F_[nstudy1][j] = input_item.rawF_[i];
					
					if (method_ == LS || method_ == logistic)
					{	
						Y_[nstudy1](j) = input_item.rawY_(i);
					}
					else if (method_ == right_censored)
					{
						Y_cox_[nstudy1][j] = input_item.rawY_cox_[i];
					}
					
					if (test_type_ == WALD) 
					{
						if (ncov_ > 0) 
						{
							W_[nstudy1].block(j,nhead_+nadd_,1,ncov_) = input_item.rawW_.row(i);
						}
						if (method_ == LS || method_ == logistic)
						{
							W_[nstudy1](j,1) = 1.;
						}
						W_[nstudy1](j,0) = MISSIND;
						if (snp_analysis_mode_ == GE) 
						{
							W_[nstudy1].block(j,nhead_,1,nadd_).setConstant(MISSIND);
						} 
						else if (snp_analysis_mode_ == COND) 
						{
							for (int nsnp=0; nsnp<nadd_; nsnp++) 
							{
								W_[nstudy1](j,nhead_+nsnp) = input_item.rawCOND_[nsnp][input_item.raw_geno_pheno_linker_(i)];
							}
						}
					}
					else if (test_type_ == SCORE) 
					{
						if (method_ == LS || method_ == logistic)
						{
							if (ncov_ > 0) 
							{
								W_[nstudy1].block(j,nhead_,1,ncov_) = input_item.rawW_.row(i);
							}
							W_[nstudy1](j,0) = 1.;
						}
						else if (method_ == right_censored)
						{
							if (ncov_ > 0)
							{
								W_[nstudy1].row(i) = input_item.rawW_.row(i);
							}
							else
							{
								// error(FO_log_, "Error: In Cox proportional hazards regression, there must be at leasy one covariate!");
							}
						}
					}
					
					geno_pheno_linker_[nstudy1](j) = input_item.raw_geno_pheno_linker_(i);
					if (!flag_uw_) 
					{
						wt_[nstudy1](j) = input_item.rawWT_(i);
						if (flag_pairwise_inclusion_prob_)
						{
							prob_pheno_linker(j) = input_item.raw_prob_pheno_linker_(i,nstudy);
						}
					}
					if (flag_strata_)
					{
						strata_tmp[j] = input_item.rawStrata_[i];
					}
					j++;
				}
			}
			
			idindx = sortIndexes(F_[nstudy1]);
			sortByIndexes(F_[nstudy1], idindx);
			if (method_ == LS || method_ == logistic)
			{
				sortByIndexes(Y_[nstudy1], idindx);
			}
			else if (method_ == right_censored)
			{
				sortByIndexes(Y_cox_[nstudy1], idindx);
			}
			if (p_ > 0)
			{
				sortByIndexes(W_[nstudy1], idindx);
			}
			sortByIndexes(geno_pheno_linker_[nstudy1], idindx);
			if (!flag_uw_) 
			{
				if (flag_pairwise_inclusion_prob_)
				{
					sortByIndexes(prob_pheno_linker, idindx);
				}
				sortByIndexes(wt_[nstudy1], idindx);
			}
			if (flag_strata_)
			{
				sortByIndexes(strata_tmp, idindx);
			}
			idindx.resize(0);

			nfam = 1;
			for (int i=1; i<rawN1; i++) 
			{
				if (F_[nstudy1][i] != F_[nstudy1][i-1]) 
				{
					nfam++;
				}
			}
			fam_ind_[nstudy1].resize(nfam+1);
			fam_ind_[nstudy1](0) = -1;
			fam_ind_[nstudy1](nfam) = rawN1-1;
			for (int i=1, j=1; i<rawN1; i++) 
			{
				if (F_[nstudy1][i] != F_[nstudy1][i-1]) 
				{
					fam_ind_[nstudy1](j++) = i-1;
				}
			}
			
			if (!flag_uw_ && flag_pairwise_inclusion_prob_) 
			{
				wts.resize(rawN1,rawN1);
				for (int i=0; i<rawN1; i++)
				{
					for (int j=0; j<rawN1; j++) 
					{
						wts(i,j) = input_item.rawSPS_[nstudy](prob_pheno_linker(i),prob_pheno_linker(j));
					}
				}
				wtds_[nstudy1] = wt_[nstudy1]*wt_[nstudy1].transpose();
				
				wtps.resize(rawN1,rawN1);			
				for (int i=0; i<rawN1; i++) 
				{
					for (int j=0; j<rawN1; j++) 
					{
						wtps(i,j) = (wts(i,j)-wts(i,i)*wts(j,j))*wtds_[nstudy1](i,j)/wts(i,j);
					}
				}
							
				for (int i=0; i<nfam-1; i++) 
				{
					for (int j=i+1; j<nfam; j++) 
					{
						wtds_[nstudy1].block(fam_ind_[nstudy1](i)+1,fam_ind_[nstudy1](j)+1,fam_ind_[nstudy1](i+1)-fam_ind_[nstudy1](i),fam_ind_[nstudy1](j+1)-fam_ind_[nstudy1](j)) =
							wtps.block(fam_ind_[nstudy1](i)+1,fam_ind_[nstudy1](j)+1,fam_ind_[nstudy1](i+1)-fam_ind_[nstudy1](i),fam_ind_[nstudy1](j+1)-fam_ind_[nstudy1](j));
						wtds_[nstudy1].block(fam_ind_[nstudy1](j)+1,fam_ind_[nstudy1](i)+1,fam_ind_[nstudy1](j+1)-fam_ind_[nstudy1](j),fam_ind_[nstudy1](i+1)-fam_ind_[nstudy1](i)) =
							wtps.block(fam_ind_[nstudy1](j)+1,fam_ind_[nstudy1](i)+1,fam_ind_[nstudy1](j+1)-fam_ind_[nstudy](j),fam_ind_[nstudy1](i+1)-fam_ind_[nstudy1](i));
					}
				}			
				wts.resize(0,0);
				wtps.resize(0,0);
				prob_pheno_linker.resize(0);
			}
			
			if (flag_strata_)
			{
				strata_map.clear();
				strata_unique.clear();
				for (int i=0; i<rawN1; i++)
				{
					if (strata_map.count(strata_tmp[i]) == 0)
					{
						strata_unique.push_back(strata_tmp[i]);
					}
					strata_map[strata_tmp[i]].push_back(i);
				}
				numStrata = strata_map.size();
				strata_ind_[nstudy1].resize(rawN1, numStrata);
				strata_ind_[nstudy1].setZero();
				for (int i=0; i<numStrata; i++)
				{
					for (int j=0; j<strata_map[strata_unique[i]].size(); j++)
					{
						strata_ind_[nstudy1](strata_map[strata_unique[i]][j], i) = 1.;
					}
				}
			}
			nstudy1++;	
		} 
		else 
		{
			FO_log_ << "\tThere is no individual eligible for association analysis, skipped." << endl;
		}
	}
	FO_log_ << "In summary, " << N_total_ << " individual(s) from " << N_study_ << " study (studies) are included in the analysis." << endl;
	FO_log_ << "Done!" << endl << endl;	
} // SUGEN::InputData_PrepareAnalysis

void SUGEN::InputData_ () 
{
	INPUT_UTILS input_item;

	InputData_LoadPheno_(input_item);
	InputData_CheckGeno_(input_item);
	InputData_LoadProb_(input_item);
	InputData_IDLinker_(input_item);
	InputData_PrepareAnalysis_(input_item);
} // SUGEN::InputData_

void SUGEN::SingleVariantAnalysis_Initialization_ (SVA_UTILS& sva_item) 
{
	int N1;
	
	sva_item.theta_.resize(p_);
	sva_item.vartheta_.resize(p_, p_);
	sva_item.WtW_.resize(p_, p_);
	sva_item.WtY_.resize(p_);
	sva_item.Ahat_.resize(p_, p_);
	sva_item.Bhat_.resize(p_, p_);
	if (snp_analysis_mode_ == GE)
	{
		sva_item.GE_U_.resize(nadd_+1);
		sva_item.GE_V_.resize(nadd_+1,nadd_+1);
	}
			
	sva_item.resi_.resize(N_study_);
	sva_item.Uhat_.resize(N_study_);
	if (!flag_robust_ && !flag_uw_ && flag_pairwise_inclusion_prob_) 
	{
		sva_item.wtd_.resize(N_study_);
	}
	if (flag_strata_)
	{
		sva_item.het_sigma2_.resize(N_study_);
		sva_item.het_sigma2_0_.resize(N_study_);
		sva_item.het_miss_.resize(N_study_);
		sva_item.theta0_.resize(p_);
		sva_item.theta_initilization_.setZero(p_);
		sva_item.het_miss_inv_sigma2_.resize(N_study_);
	}
	sva_item.Miss_.resize(N_study_);		
	if (method_ == logistic) 
	{
		sva_item.Wttheta_.resize(N_study_);
		sva_item.e_Wttheta_.resize(N_study_);
		sva_item.logi_Wttheta_.resize(N_study_);
		sva_item.logi2_Wttheta_.resize(N_study_);
		sva_item.theta0_.resize(p_);
		sva_item.theta_initilization_.setZero(p_);
	}
	else if (method_ == right_censored)
	{
		sva_item.Wttheta_.resize(N_study_);
		sva_item.e_Wttheta_.resize(N_study_);
		sva_item.theta0_.resize(p_);
		sva_item.S1_.resize(p_);
		sva_item.S2_.resize(p_, p_);
		sva_item.riskset_.resize(N_study_);
		sva_item.theta_initilization_.setZero(p_);
	}
			
	N_total_ = 0;
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		N1 = F_[nstudy].size();
		N_total_ += N1;
		sva_item.resi_[nstudy].resize(N1);
		if (flag_strata_)
		{
			sva_item.het_sigma2_[nstudy].resize(strata_ind_[nstudy].cols());
			sva_item.het_sigma2_0_[nstudy].resize(strata_ind_[nstudy].cols());
			sva_item.het_miss_[nstudy].resize(N1, strata_ind_[nstudy].cols());
			sva_item.het_miss_inv_sigma2_[nstudy].resize(N1);
		}
		sva_item.Miss_[nstudy].resize(N1);
		sva_item.Uhat_[nstudy].resize(N1, p_);
		if (!flag_robust_ && !flag_uw_ && flag_pairwise_inclusion_prob_) 
		{
			sva_item.wtd_[nstudy].resize(N1);
			for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
			{
				for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
				{
					if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
					{
						sva_item.wtd_[nstudy](j) = wtds_[nstudy](j,j);
						wtds_[nstudy](j,j) = 0.;						
					}
					else
					{
						sva_item.wtd_[nstudy](j) = 0.;
					}
				}
			}
		}
		if (method_ == logistic) 
		{
			sva_item.Wttheta_[nstudy].resize(N1);
			sva_item.e_Wttheta_[nstudy].resize(N1);
			sva_item.logi_Wttheta_[nstudy].resize(N1);
			sva_item.logi2_Wttheta_[nstudy].resize(N1);
		}
		else if (method_ == right_censored)
		{
			sva_item.Wttheta_[nstudy].resize(N1);
			sva_item.e_Wttheta_[nstudy].resize(N1);
			sva_item.riskset_[nstudy].resize(N1);
		}
	}	
} // SUGEN::SingleVariantAnalysis_Initialization_

void SUGEN::SingleVariantAnalysis_OutputSNPCountHeader_ (SVA_UTILS& sva_item, const SVA_OUTPUT_TYPE_ sva_output_type) 
{	
	if (sva_output_type == sva_header) 
	{
		*sva_item.FO_out_ << "CHROM\tPOS\tVCF_ID\tREF\tALT\tALT_AF\tALT_AC\tN_INFORMATIVE\tN_REF\tN_HET\tN_ALT\tN_DOSE";
		if (method_ == logistic)
		{
			*sva_item.FO_out_ << "\tALT_AF_CASE\tN_CASE";
		}
		else if (method_ == right_censored)
		{
			*sva_item.FO_out_ << "\tALT_AF_EVENT\tN_EVENT";
		}
	} 
	else 
	{
		*sva_item.FO_out_ << VCF_record_.getChromStr() << '\t' << VCF_record_.get1BasedPosition() << '\t' << VCF_record_.getIDStr() << '\t'
			<< VCF_record_.getRefStr() << '\t' << VCF_record_.getAltStr();
		*sva_item.FO_out_ << '\t' << sva_item.maf_<< '\t' << sva_item.mac_ << '\t' << sva_item.n_ << '\t' << sva_item.n0_count_ 
			<< '\t'  << sva_item.n1_count_ << '\t' << sva_item.n2_count_ << '\t' << sva_item.n_dose_;
		if (method_ == logistic || method_ == right_censored)
		{
			*sva_item.FO_out_ << "\t" << sva_item.maf_case_ << "\t" << sva_item.n_case_;
		}
	}		
} // SUGEN::SingleVariantAnalysis_OutputSNPCountHeader_

void SUGEN::SingleVariantAnalysis_Output_ (SVA_UTILS& sva_item, const SVA_OUTPUT_TYPE_ sva_output_type) 
{		
	SingleVariantAnalysis_OutputSNPCountHeader_(sva_item, sva_output_type);
		
	if (snp_analysis_mode_ == ST) 
	{
		/**** standard association analysis *********************************************/		
		if (sva_output_type == sva_header) 
		{
			*sva_item.FO_out_ << "\tBETA\tSE\tPVALUE";
		} 
		else if (sva_output_type == sva_results_miss) 
		{
			*sva_item.FO_out_ << "\tNA\tNA\tNA";			
		} 
		else if (sva_output_type == sva_no_miss) 
		{
			sva_item.pvalue_ = gammq(0.5, sva_item.theta_(0)*sva_item.theta_(0)/(2.*sva_item.vartheta_(0,0)));
			*sva_item.FO_out_ << '\t' << sva_item.theta_(0) << '\t' << sqrt(sva_item.vartheta_(0,0)) << '\t' << sva_item.pvalue_;
		}
		/**** standard association analysis *********************************************/
	} 
	else if (snp_analysis_mode_ == COND) 
	{
		/**** conditional analysis ******************************************************/
		if (sva_output_type == sva_header) 
		{
			*sva_item.FO_out_ << "\tBETA\tSE\tPVALUE";
			for (int i=0; i<nadd_; i++) 
			{
				*sva_item.FO_out_ << "\tBETA_" << COND_names_[i] << "\tSE_" << COND_names_[i] << "\tPVALUE_" << COND_names_[i];
			}
		} 
		else if (sva_output_type == sva_results_miss) 
		{
			for (int i=0; i<nadd_+1; i++) 
			{
				*sva_item.FO_out_ << "\tNA\tNA\tNA";
			}
		} 
		else if (sva_output_type == sva_no_miss) 
		{
			sva_item.pvalue_ = gammq(0.5, sva_item.theta_(0)*sva_item.theta_(0)/(2.*sva_item.vartheta_(0,0)));	
			*sva_item.FO_out_ << '\t' << sva_item.theta_(0) << '\t' << sqrt(sva_item.vartheta_(0,0)) << '\t' << sva_item.pvalue_;

			for (int i=0; i<nadd_; i++) 
			{
				sva_item.pvalue_ = gammq(0.5, sva_item.theta_(nhead_+i)*sva_item.theta_(nhead_+i)/(2.*sva_item.vartheta_(nhead_+i,nhead_+i)));
				*sva_item.FO_out_ << "\t" << sva_item.theta_(nhead_+i) << "\t" << sqrt(sva_item.vartheta_(nhead_+i,nhead_+i)) << "\t" << sva_item.pvalue_;
			}
		}
		/**** conditional analysis ******************************************************/		
	} 
	else if (snp_analysis_mode_ == GE) 
	{
		/**** gene-environment interaction analysis *************************************/
		if (sva_output_type == sva_header) 
		{
			*sva_item.FO_out_ << "\tPVALUE_G\tPVALUE_INTER\tPVALUE_BOTH";
			*sva_item.FO_out_ << "\tBETA_G";
			for (int i=0; i<nadd_; i++) 
			{
				*sva_item.FO_out_ << "\tBETA_"+ENVI_names_[i];
			}
			for (int i=0; i<nadd_; i++) 
			{
				*sva_item.FO_out_ << "\tBETA_G:"+ENVI_names_[i];
			}
			
			*sva_item.FO_out_ << "\tCOV_G_G";
			if (flag_ge_full_output_)
			{
				for (int i=0; i<nadd_; i++) 
				{
					*sva_item.FO_out_ << "\tCOV_G_"+ENVI_names_[i];
				}
				for (int i=0; i<nadd_; i++) 
				{
					*sva_item.FO_out_ << "\tCOV_G_G:"+ENVI_names_[i];
				}
			}
			
			for (int i=0; i<nadd_; i++) 
			{
				if (flag_ge_full_output_)
				{
					for (int j=i; j<nadd_; j++) 
					{
						*sva_item.FO_out_ << "\tCOV_"+ENVI_names_[i] << "_" << ENVI_names_[j];
					}
					for (int j=0; j<nadd_; j++) 
					{
						*sva_item.FO_out_ << "\tCOV_"+ENVI_names_[i] << "_G:" << ENVI_names_[j];
					}
				}
				else
				{
					*sva_item.FO_out_ << "\tCOV_"+ENVI_names_[i] << "_" << ENVI_names_[i];
				}
			}
			
			for (int i=0; i<nadd_; i++) 
			{
				if (flag_ge_full_output_)
				{
					for (int j=i; j<nadd_; j++) 
					{
						*sva_item.FO_out_ << "\tCOV_G:"+ENVI_names_[i] << "_G:" << ENVI_names_[j];
					}
				}
				else
				{
					*sva_item.FO_out_ << "\tCOV_G:"+ENVI_names_[i] << "_G:" << ENVI_names_[i];
				}
			}						
		} 
		else if (sva_output_type == sva_results_miss) 
		{
			*sva_item.FO_out_ << "\tNA\tNA\tNA";
			if (flag_ge_full_output_)
			{
				for (int i=0; i<(1+2*nadd_)*(2+nadd_); i++) 
				{
					*sva_item.FO_out_ << "\tNA";
				}
			}
			else
			{
				for (int i=0; i<(1+2*nadd_)*2; i++) 
				{
					*sva_item.FO_out_ << "\tNA";
				}				
			}
		} 
		else if (sva_output_type == sva_no_miss) 
		{
			sva_item.GE_U_(0) = sva_item.theta_(0);
			sva_item.GE_U_.tail(nadd_) = sva_item.theta_.segment(nhead_,nadd_);
			sva_item.GE_V_(0,0) = sva_item.vartheta_(0,0);
			sva_item.GE_V_.topRightCorner(1,nadd_) = sva_item.vartheta_.block(0,nhead_,1,nadd_);
			sva_item.GE_V_.bottomLeftCorner(nadd_,1) = sva_item.GE_V_.topRightCorner(1,nadd_).transpose();
			sva_item.GE_V_.bottomRightCorner(nadd_,nadd_) = sva_item.vartheta_.block(nhead_,nhead_,nadd_,nadd_);
			
			double test_statistic;
			bool flag_singular = false;
			test_statistic = sva_item.GE_U_(0)*sva_item.GE_U_(0)/sva_item.GE_V_(0,0);
			if (test_statistic <= 0. || ::isnan(test_statistic) || ::isinf(test_statistic))
			{
				*sva_item.FO_out_ << "\tNA";
				flag_singular = true;
			}
			else
			{
				sva_item.pvalue_ = gammq(0.5, test_statistic/2.);
				*sva_item.FO_out_ << "\t" << sva_item.pvalue_;
			}
			test_statistic = (sva_item.GE_U_.tail(nadd_).transpose()*sva_item.GE_V_.bottomRightCorner(nadd_, nadd_).inverse()*sva_item.GE_U_.tail(nadd_))(0,0);
			if (test_statistic <= 0. || ::isnan(test_statistic) || ::isinf(test_statistic))
			{
				*sva_item.FO_out_ << "\tNA";
				flag_singular = true;
			}
			else
			{
				sva_item.pvalue_ = gammq(0.5*nadd_, test_statistic/2.);
				*sva_item.FO_out_ << "\t" << sva_item.pvalue_;
			}
			test_statistic = sva_item.GE_U_.transpose()*sva_item.GE_V_.inverse()*sva_item.GE_U_;
			if (test_statistic <= 0. || ::isnan(test_statistic) || ::isinf(test_statistic))
			{
				*sva_item.FO_out_ << "\tNA";
				flag_singular = true;
			}
			else
			{
				sva_item.pvalue_ = gammq(0.5*(nadd_+1), test_statistic/2.);
				*sva_item.FO_out_ << "\t" << sva_item.pvalue_;
			}
			
			if (flag_singular)
			{
				if (flag_ge_full_output_)
				{
					for (int i=0; i<(1+2*nadd_)*(2+nadd_); i++) 
					{
						*sva_item.FO_out_ << "\tNA";
					}
				}
				else
				{
					for (int i=0; i<(1+2*nadd_)*2; i++) 
					{
						*sva_item.FO_out_ << "\tNA";
					}				
				}
			}
			else
			{
				*sva_item.FO_out_ << "\t" << sva_item.theta_(0);
				for (int i=0; i<nadd_; i++) 
				{
					*sva_item.FO_out_ << "\t" << sva_item.theta_(nhead_+nadd_+ENVI_col_(i));
				}
				for (int i=0; i<nadd_; i++) 
				{
					*sva_item.FO_out_ << "\t" << sva_item.theta_(nhead_+i);
				}
			
				*sva_item.FO_out_ << "\t" << sva_item.vartheta_(0,0);
				if (flag_ge_full_output_)
				{
					for (int i=0; i<nadd_; i++) 
					{
						*sva_item.FO_out_ << "\t" << sva_item.vartheta_(0,nhead_+nadd_+ENVI_col_(i));
					}
					for (int i=0; i<nadd_; i++) 
					{
						*sva_item.FO_out_ << "\t" << sva_item.vartheta_(0,nhead_+i);
					}
				}
				
				for (int i=0; i<nadd_; i++) 
				{
					if (flag_ge_full_output_)
					{
						for (int j=i; j<nadd_; j++) 
						{
							*sva_item.FO_out_ << "\t" << sva_item.vartheta_(nhead_+nadd_+ENVI_col_(i),nhead_+nadd_+ENVI_col_(j));
						}
						for (int j=0; j<nadd_; j++) 
						{
							*sva_item.FO_out_ << "\t" << sva_item.vartheta_(nhead_+nadd_+ENVI_col_(i),nhead_+j);
						}
					}
					else
					{
						*sva_item.FO_out_ << "\t" << sva_item.vartheta_(nhead_+nadd_+ENVI_col_(i),nhead_+nadd_+ENVI_col_(i));				
					}
				}
				
				for (int i=0; i<nadd_; i++) 
				{
					if (flag_ge_full_output_)
					{
						for (int j=i; j<nadd_; j++) 
						{
							*sva_item.FO_out_ << "\t" << sva_item.vartheta_(nhead_+i,nhead_+j);
						}
					}
					else
					{
						*sva_item.FO_out_ << "\t" << sva_item.vartheta_(nhead_+i,nhead_+i);
					}
				}
			}
		}
		/**** gene-environment interaction analysis *************************************/
	}
	*sva_item.FO_out_ << '\n';	
} // SUGEN::SingleVariantAnalysis_Output_

void SUGEN::SingleVariantAnalysis_GetSNP_ (SVA_UTILS& sva_item) 
{	
	if (VCF_record_.getNumAlts() > 1) 
	{
		sva_item.flag_multiallelic_ = true;
		return;
	} 
	else 
	{
		sva_item.flag_multiallelic_ = false;
	}
	VcfRecordGenotype & genoInfo = VCF_record_.getGenotypeInfo();
	
	sva_item.mac_ = 0;
	sva_item.maf_ = 0.;
	sva_item.n0_count_ = 0;
	sva_item.n1_count_ = 0;
	sva_item.n2_count_ = 0;
	sva_item.n_dose_ = 0;
	sva_item.n_ = 0;
	if (method_ == logistic || method_ == right_censored)
	{
		sva_item.maf_case_ = 0.;
		sva_item.n_case_ = 0.;
	}
		
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		for (int i=0; i<F_[nstudy].size(); i++) 
		{
			W_[nstudy](i,0) = ReadOneVariant_(geno_pheno_linker_[nstudy](i), genoInfo);
			if (W_[nstudy](i,0) == MISSIND)
			{
				sva_item.Miss_[nstudy](i) = 0.;
				if (snp_analysis_mode_ == GE) 
				{
					W_[nstudy].block(i,nhead_,1,nadd_).setConstant(MISSIND);
				}
			}
			else 
			{
				sva_item.Miss_[nstudy](i) = 1.;
				sva_item.maf_ += W_[nstudy](i,0);
				if (flag_dosage_)
				{
					sva_item.n_dose_++;
				}
				else
				{
					if (fabs(W_[nstudy](i,0)-0.0) < ERROR_MARGIN) 
					{
						sva_item.n0_count_++;
					} 
					else if (fabs(W_[nstudy](i,0)-1.0) < ERROR_MARGIN) 
					{
						sva_item.n1_count_++;
					} 
					else if (fabs(W_[nstudy](i,0)-2.0) < ERROR_MARGIN) 
					{
						sva_item.n2_count_++;
					}
				}
				if (snp_analysis_mode_ == GE) 
				{
					for (int j=0; j<nadd_; j++) 
					{
						W_[nstudy](i,nhead_+j) = W_[nstudy](i,0)*W_[nstudy](i,nhead_+nadd_+ENVI_col_(j));
					}
				}
				if (method_ == logistic && Y_[nstudy](i) == 1)
				{
					sva_item.maf_case_ += W_[nstudy](i,0);
					sva_item.n_case_ += 1;
				}
				if (method_ == right_censored && !Y_cox_[nstudy][i].is_alive_)
				{
					sva_item.maf_case_ += W_[nstudy](i,0);
					sva_item.n_case_ += 1;					
				}
			}
		}
	}
						
	sva_item.n_ = sva_item.n0_count_+sva_item.n1_count_+sva_item.n2_count_+sva_item.n_dose_;
	if (sva_item.n_ > 0) 
	{
		sva_item.maf_ /= 2.*sva_item.n_;
		sva_item.mac_ = sva_item.n1_count_+2*sva_item.n2_count_;
	}
	if (method_ == logistic || method_ == right_censored)
	{
		if (sva_item.n_case_ > 0)
		{
			sva_item.maf_case_ /= 2.*sva_item.n_case_;
		}
	}
} // SUGEN::SingleVariantAnalysis_GetSNP_

void SUGEN::LinearWald_ (SVA_UTILS& sva_item) 
{	
	/**** GEE for linear models *********************************************************/ 
	/**** independent working correlation matrix ****************************************/
	/**** Wald statistics ***************************************************************/
	double numerator, denominator;
	
	/**** effect estimation *************************************************************/
	sva_item.WtW_.setZero();
	sva_item.WtY_.setZero();
	if (flag_uw_) 
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.Miss_[nstudy].asDiagonal()*W_[nstudy];
			sva_item.WtY_.noalias() += W_[nstudy].transpose()*sva_item.Miss_[nstudy].asDiagonal()*Y_[nstudy];
		}		
	} 
	else 
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.WtW_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.Miss_[nstudy].asDiagonal()*W_[nstudy];
			sva_item.WtY_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.Miss_[nstudy].asDiagonal()*Y_[nstudy];
		}
	}
	sva_item.theta_ = sva_item.WtW_.selfadjointView<Eigen::Upper>().ldlt().solve(sva_item.WtY_);
	/**** effect estimation *************************************************************/

	/**** calculate sigma2 **************************************************************/
	numerator = 0.0;
	denominator = 0.0;
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		sva_item.resi_[nstudy] = (Y_[nstudy]-W_[nstudy]*sva_item.theta_).array()*sva_item.Miss_[nstudy].array();
		if (flag_uw_) 
		{
			numerator += sva_item.resi_[nstudy].squaredNorm();
			denominator += sva_item.Miss_[nstudy].sum();
		}
		else
		{
			numerator += (sva_item.resi_[nstudy].array()*wt_[nstudy].array()*sva_item.resi_[nstudy].array()).sum();
			denominator += (sva_item.Miss_[nstudy].array()*wt_[nstudy].array()).sum();		
	
		}
	}
	sva_item.sigma2_ = numerator/denominator;
	/**** calculate sigma2 **************************************************************/	
		
	if (flag_strata_)
	{
		double tol;
		int iter;
		
		sva_item.theta0_ = sva_item.theta_;
		for (int nstudy=0; nstudy<N_study_; nstudy++)
		{
			sva_item.het_miss_[nstudy] = sva_item.Miss_[nstudy].asDiagonal()*strata_ind_[nstudy];
			for (int i=0; i<strata_ind_[nstudy].cols(); i++)
			{	
				sva_item.het_sigma2_[nstudy](i) = sva_item.sigma2_;
				sva_item.het_sigma2_0_[nstudy](i) = sva_item.sigma2_;
			}
		}
		sva_item.flag_het_converge_ = false;

		for (iter=1; iter<=MAX_ITER; iter++) 
		{	
			/**** effect estimation *****************************************************/
			sva_item.WtW_.setZero();
			sva_item.WtY_.setZero();
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.het_miss_inv_sigma2_[nstudy].setZero();
				for (int i=0; i<strata_ind_[nstudy].cols(); i++)
				{
					sva_item.het_miss_inv_sigma2_[nstudy].noalias() += sva_item.het_miss_[nstudy].col(i)/sva_item.het_sigma2_[nstudy](i);
				}
				if (flag_uw_)
				{
					sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
					sva_item.WtY_.noalias() += W_[nstudy].transpose()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*Y_[nstudy];
				}
				else
				{
					sva_item.WtW_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
					sva_item.WtY_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*Y_[nstudy];						
				}
			}		
			sva_item.theta_ = sva_item.WtW_.selfadjointView<Eigen::Upper>().ldlt().solve(sva_item.WtY_);
			/**** effect estimation *****************************************************/
			
			/**** calculate residual variance *******************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{
				sva_item.resi_[nstudy] = Y_[nstudy]-W_[nstudy]*sva_item.theta_;
				for (int i=0; i<strata_ind_[nstudy].cols(); i++)
				{	
					if (flag_uw_) 
					{
						numerator = (sva_item.resi_[nstudy].array()*sva_item.het_miss_[nstudy].col(i).array()).matrix().squaredNorm();
						denominator = sva_item.het_miss_[nstudy].col(i).sum();
					}
					else
					{
						numerator = (sva_item.resi_[nstudy].array()*wt_[nstudy].array()*sva_item.het_miss_[nstudy].col(i).array()
							*sva_item.resi_[nstudy].array()).sum();
						denominator = (sva_item.het_miss_[nstudy].col(i).array()*wt_[nstudy].array()).sum();					
					}
					sva_item.het_sigma2_[nstudy](i) = numerator/denominator;
				}
			}		
			/**** calculate residual variance *******************************************/
			
			tol = (sva_item.theta_-sva_item.theta0_).array().abs().sum();
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{
				tol += (sva_item.het_sigma2_[nstudy]-sva_item.het_sigma2_0_[nstudy]).array().abs().sum();
			}
			if (tol < TOL) 
			{
				sva_item.flag_het_converge_ = true;
				break;
			} 
			else 
			{
				sva_item.theta0_ = sva_item.theta_;
				for (int nstudy=0; nstudy<N_study_; nstudy++)
				{
					sva_item.het_sigma2_0_[nstudy] = sva_item.het_sigma2_[nstudy];
				}				
			}
		}
		
		/**** calculate WtW_, U_hat *****************************************************/
		sva_item.WtW_.setZero();
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.het_miss_inv_sigma2_[nstudy].setZero();
			for (int i=0; i<strata_ind_[nstudy].cols(); i++)
			{
				sva_item.het_miss_inv_sigma2_[nstudy].noalias() += sva_item.het_miss_[nstudy].col(i)/sva_item.het_sigma2_[nstudy](i);
			}
			if (flag_uw_)
			{
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
			}
			else
			{
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];					
			}
			sva_item.resi_[nstudy] = Y_[nstudy]-W_[nstudy]*sva_item.theta_;
			sva_item.Uhat_[nstudy] = (sva_item.resi_[nstudy].array()*sva_item.het_miss_inv_sigma2_[nstudy].array()).matrix().asDiagonal()*W_[nstudy];
		}		
		/**** calculate WtW_, U_hat *****************************************************/
	}
	else
	{
		/**** calculate Uhat_ ***********************************************************/
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Uhat_[nstudy] = sva_item.resi_[nstudy].asDiagonal()*W_[nstudy]/sva_item.sigma2_;
		}
		sva_item.WtW_ /= sva_item.sigma2_;
		/**** calculate Uhat_ ***********************************************************/		
	}
		
	/**** variance estimation ***********************************************************/
	sva_item.Bhat_.setZero();
	if (flag_uw_) 
	{
		/**** unweighted analysis *******************************************************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
						}
					}
				}
			}
			/**** robust variance estimation ********************************************/
		} 
		else 
		{
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							if (sva_item.Miss_[nstudy](j) == 1.)
							{
								if (flag_strata_)
								{
									sva_item.Bhat_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.het_miss_inv_sigma2_[nstudy](j);
								}
								else
								{
									sva_item.Bhat_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)/sva_item.sigma2_;
								}
							}
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** unweighted analysis *******************************************************/
	} 
	else if (flag_pairwise_inclusion_prob_)
	{
		/**** weighted analysis *********************************************************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/		
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];	
			}
			/**** robust variance estimation ********************************************/		
		} 
		else 
		{
			/**** model-based variance estimation ***************************************/
			if (flag_strata_)
			{
				for (int nstudy=0; nstudy<N_study_; nstudy++)
				{
					sva_item.Bhat_.noalias() += W_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()
						*W_[nstudy];
				}				
			}
			else
			{
				for (int nstudy=0; nstudy<N_study_; nstudy++)
				{
					sva_item.Bhat_.noalias() += (W_[nstudy].transpose())*sva_item.wtd_[nstudy].asDiagonal()*sva_item.Miss_[nstudy].asDiagonal()*W_[nstudy];
				}	
				sva_item.Bhat_ /= sva_item.sigma2_;
			}
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];	
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis *********************************************************/
	} 
	else
	{
		/**** weighted analysis without pairwise inclusion probabilities ****************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
						}
					}
				}
			}
			/**** robust variance estimation ********************************************/
		} 
		else 
		{
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							if (sva_item.Miss_[nstudy](j) == 1.)
							{
								if (flag_strata_)
								{
									sva_item.Bhat_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.het_miss_inv_sigma2_[nstudy](j)
										*wt_[nstudy](j)*wt_[nstudy](j);
								}
								else
								{
									sva_item.Bhat_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)*wt_[nstudy](j)*wt_[nstudy](j)/sva_item.sigma2_;
								}
							}
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis without pairwise inclusion probabilities ****************/		
	}
	sva_item.Ahat_ = sva_item.WtW_.inverse();
	sva_item.vartheta_.noalias() = sva_item.Ahat_*sva_item.Bhat_*sva_item.Ahat_;
	if (flag_uw_ && sva_item.vartheta_(0,0) < (sva_item.Ahat_(0,0)-ERROR_MARGIN))
	{
		sva_item.vartheta_ = sva_item.Ahat_;
	}
	/**** variance estimation ***********************************************************/	
} // SUGEN::LinearWald_

void SUGEN::LogisticWald_ (SVA_UTILS& sva_item) 
{	
	/**** GEE for logistic models *******************************************************/ 
	/**** independent working correlation matrix ****************************************/
	/**** Wald statistics ***************************************************************/
	
	double tol;
	int iter;
	
	sva_item.theta0_.setZero();
	sva_item.theta_.setZero();
	
	sva_item.flag_logistic_converge_ = false;
	
	/**** effect estimation *************************************************************/
	for (iter=1; iter<=MAX_ITER; iter++) 
	{			
		sva_item.WtW_.setZero();
		sva_item.WtY_.setZero();		
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Wttheta_[nstudy].noalias() = W_[nstudy]*sva_item.theta0_;
			sva_item.e_Wttheta_[nstudy] = sva_item.Wttheta_[nstudy].array().exp();
			sva_item.logi_Wttheta_[nstudy] = sva_item.e_Wttheta_[nstudy].array()/(sva_item.e_Wttheta_[nstudy].array()+1.);
			sva_item.logi2_Wttheta_[nstudy] = sva_item.logi_Wttheta_[nstudy].array()/(sva_item.e_Wttheta_[nstudy].array()+1.);			
			sva_item.resi_[nstudy] = (Y_[nstudy]-sva_item.logi_Wttheta_[nstudy]).array()*sva_item.Miss_[nstudy].array();
			sva_item.Uhat_[nstudy].noalias() = sva_item.resi_[nstudy].asDiagonal()*W_[nstudy];
			if (flag_uw_) 
			{
				sva_item.WtY_.noalias() += sva_item.Uhat_[nstudy].transpose().rowwise().sum();
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.Miss_[nstudy].asDiagonal()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
			} 
			else 
			{
				sva_item.WtY_.noalias() += sva_item.Uhat_[nstudy].transpose()*wt_[nstudy];
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.Miss_[nstudy].asDiagonal()
					*wt_[nstudy].asDiagonal()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
			}
		}
		sva_item.theta_ = sva_item.WtW_.selfadjointView<Eigen::Upper>().ldlt().solve(sva_item.WtY_);
		
		tol = sva_item.theta_.array().abs().sum();
		sva_item.theta_ += sva_item.theta0_;		
		if (tol < TOL) 
		{
			sva_item.flag_logistic_converge_ = true;
			break;
		} 
		else 
		{
			sva_item.theta0_ = sva_item.theta_;
		}
	}
	/**** effect estimation *************************************************************/
	
	if (iter == MAX_ITER+1) 
	{
		sva_item.flag_logistic_converge_ = false;
		sva_item.vartheta_.setZero();
	} 
	else 
	{
		/**** variance estimation *******************************************************/
		sva_item.WtW_.setZero();
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Wttheta_[nstudy].noalias() = W_[nstudy]*sva_item.theta0_;
			sva_item.e_Wttheta_[nstudy] = sva_item.Wttheta_[nstudy].array().exp();
			sva_item.logi_Wttheta_[nstudy] = sva_item.e_Wttheta_[nstudy].array()/(sva_item.e_Wttheta_[nstudy].array()+1.);
			sva_item.logi2_Wttheta_[nstudy] = sva_item.logi_Wttheta_[nstudy].array()/(sva_item.e_Wttheta_[nstudy].array()+1.);			
			sva_item.resi_[nstudy] = (Y_[nstudy]-sva_item.logi_Wttheta_[nstudy]).array()*sva_item.Miss_[nstudy].array();
			sva_item.Uhat_[nstudy].noalias() = sva_item.resi_[nstudy].asDiagonal()*W_[nstudy];
			if (flag_uw_) 
			{	
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.Miss_[nstudy].asDiagonal()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
			} 
			else 
			{
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.Miss_[nstudy].asDiagonal()*wt_[nstudy].asDiagonal()
					*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
			}
		}
		
		sva_item.Bhat_.setZero();
		if (flag_uw_) 
		{	
			/**** unweighted analysis ***************************************************/			
			if (flag_robust_) 
			{
				/**** robust variance estimation ****************************************/
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{			
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
							}
						}
					}
				}
				/**** robust variance estimation ****************************************/				
			} 
			else 
			{				
				/**** model-based variance estimation ***********************************/				
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{	
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{
							if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
							{
								if (sva_item.Miss_[nstudy](j) == 1.)
								{
									sva_item.Bhat_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.logi2_Wttheta_[nstudy](j);
								}
							}
							else
							{
								for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
								{
									sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
								}							
							}
						}
					}
				}
				/**** model-based variance estimation ***********************************/
			}
			/**** unweighted analysis ***************************************************/
		} 
		else if (flag_pairwise_inclusion_prob_)
		{
			/**** weighted analysis *****************************************************/
			if (flag_robust_) 
			{
				/**** robust variance estimation ****************************************/
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{
					sva_item.Bhat_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
				}
				/**** robust variance estimation ****************************************/
			} 
			else 
			{
				/**** model-based variance estimation ***********************************/
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{
					sva_item.Bhat_.noalias() += W_[nstudy].transpose()*sva_item.Miss_[nstudy].asDiagonal()*sva_item.wtd_[nstudy].asDiagonal()
						*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
					sva_item.Bhat_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];				
				}
				/**** model-based variance estimation ***********************************/
			}
			/**** weighted analysis *****************************************************/
		}
		else
		{
			/**** weighted analysis without pairwise inclusion probabilities ************/			
			if (flag_robust_) 
			{
				/**** robust variance estimation ****************************************/
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{			
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							}
						}
					}
				}
				/**** robust variance estimation ****************************************/				
			} 
			else 
			{				
				/**** model-based variance estimation ***********************************/				
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{	
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{
							if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
							{
								if (sva_item.Miss_[nstudy](j) == 1.)
								{
									sva_item.Bhat_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.logi2_Wttheta_[nstudy](j)*wt_[nstudy](j)*wt_[nstudy](j);
								}
							}
							else
							{
								for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
								{
									sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
								}							
							}
						}
					}
				}
				/**** model-based variance estimation ***********************************/
			}
			/**** weighted analysis without pairwise inclusion probabilities ************/			
		}
		sva_item.Ahat_ = sva_item.WtW_.inverse();
		sva_item.vartheta_.noalias() = sva_item.Ahat_*sva_item.Bhat_*sva_item.Ahat_;
		if (flag_uw_ && sva_item.vartheta_(0,0) < (sva_item.Ahat_(0,0)-ERROR_MARGIN))
		{
			sva_item.vartheta_ = sva_item.Ahat_;
		}
		/**** variance estimation *******************************************************/
	}	
} // SUGEN::LogisticWald_

void SUGEN::CoxphWald_ (SVA_UTILS& sva_item)
{
	/**** GEE for Cox proportional hazards models ***************************************/ 
	/**** independent working correlation matrix ****************************************/
	/**** Wald statistics ***************************************************************/
	double tol;
	int iter, stop_sign = 0;
	bool small_step_ind;
	double last_tol = 1E20;
	// // RT
	// time_t timer_start, timer_end;
	// // RT end;
	
	sva_item.theta0_ = sva_item.theta_initilization_;
	sva_item.theta_ = sva_item.theta_initilization_;	
	sva_item.flag_coxph_converge_ = false;
	
	/**** effect estimation *************************************************************/
	for (iter=1; iter<=MAX_ITER; iter++) 
	{
		// // RT
		// time(&timer_start);
		// // RT end
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Wttheta_[nstudy].noalias() = W_[nstudy]*sva_item.theta0_;
			sva_item.e_Wttheta_[nstudy] = sva_item.Wttheta_[nstudy].array().exp();
		}
		
		sva_item.WtW_.setZero();
		sva_item.WtY_.setZero();

		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{			
			for (int i=0; i<F_[nstudy].size(); i++)
			{
				if (!Y_cox_[nstudy][i].is_alive_ && sva_item.Miss_[nstudy](i) == 1.)
				{
					sva_item.S0_ = 0.0;
					sva_item.S1_.setZero();
					sva_item.S2_.setZero();
					
					for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
					{
						sva_item.riskset_[nstudy1].setZero();
						for (int j=0; j<F_[nstudy1].size(); j++)
						{
							if (sva_item.Miss_[nstudy1](j) == 1. && Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][i].survival_time_)
							{
								if (Y_cox_[nstudy1][j].is_alive_)
								{
									sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][i].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_) ? sva_item.e_Wttheta_[nstudy1](j) : 0.;
								}
								else
								{
									sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][i].survival_time_ <= Y_cox_[nstudy1][j].survival_time_) ? sva_item.e_Wttheta_[nstudy1](j) : 0.;
								}
							}
						}
						
						if (flag_uw_)
						{
							sva_item.S0_ += sva_item.riskset_[nstudy1].sum();
							sva_item.S1_ += W_[nstudy1].transpose()*sva_item.riskset_[nstudy1];
							sva_item.S2_ += W_[nstudy1].transpose()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
						}
						else
						{
							sva_item.S0_ += sva_item.riskset_[nstudy1].dot(wt_[nstudy1]);
							sva_item.S1_ += W_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1];
							sva_item.S2_ += W_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
						}
					}
					
					if (flag_uw_)
					{
						sva_item.WtW_.noalias() += sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_);
						sva_item.WtY_.noalias() += W_[nstudy].row(i).transpose()-sva_item.S1_/sva_item.S0_;
					}
					else
					{
						sva_item.WtW_.noalias() += (sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](i);
						sva_item.WtY_.noalias() += (W_[nstudy].row(i).transpose()-sva_item.S1_/sva_item.S0_)*wt_[nstudy](i);						
					}
				}
			}
		}
		
		sva_item.theta_ = sva_item.WtW_.selfadjointView<Eigen::Upper>().ldlt().solve(sva_item.WtY_);
		
		tol = sva_item.theta_.array().abs().sum();
		
		small_step_ind = false;
		while (tol > sva_item.newton_raphson_step_size_)
		{
			small_step_ind = true;
			sva_item.theta_ *= 0.5;
			tol = sva_item.theta_.array().abs().sum();
		}
			
		sva_item.theta_ += sva_item.theta0_;
		sva_item.theta0_ = sva_item.theta_;
				
		if (tol >= last_tol-TOL && !small_step_ind)
		{
			stop_sign ++;
		}
		last_tol = tol;

		// // RT
		// time(&timer_end);
		// cout << "Iteration=" << iter << "; time=" << difftime(timer_end, timer_start) << " seconds; error=" << tol << endl;
		// cout << "theta=" << sva_item.theta_.transpose() << endl;
		// cout << "WtW=" << sva_item.WtW_ << endl;
		// cout << "WtY=" << sva_item.WtY_.transpose() << endl;
		// cout << endl;
		// // RT end
		
		if (tol < TOL) 
		{
			sva_item.flag_coxph_converge_ = true;
			break;
		}
		else if (::isnan(tol))
		{
			sva_item.newton_raphson_step_size_ /= 10.;
			iter = 1;
			sva_item.theta0_ = sva_item.theta_initilization_;
			sva_item.theta_ = sva_item.theta_initilization_;
		}
		else if (stop_sign > STOP_MAXIMUM)
		{
			break;
		}
	}
	/**** effect estimation *************************************************************/

	if (iter == MAX_ITER+1 || ::isnan(tol) || stop_sign > STOP_MAXIMUM)
	{
		sva_item.flag_coxph_converge_ = false;
		sva_item.vartheta_.setZero();
	} 
	else 
	{
		/**** update initial values of theta ********************************************/
		sva_item.theta_initilization_.tail(ncov_) = sva_item.theta_.tail(ncov_);
		/**** update initial values of theta ********************************************/
		
		/**** variance estimation *******************************************************/
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Wttheta_[nstudy].noalias() = W_[nstudy]*sva_item.theta0_;
			sva_item.e_Wttheta_[nstudy] = sva_item.Wttheta_[nstudy].array().exp();
			sva_item.Uhat_[nstudy].setZero();
		}
		
		sva_item.WtW_.setZero();
		sva_item.Bhat_.setZero();

		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{			
			for (int i=0; i<fam_ind_[nstudy].size()-1; i++)
			{
				for (int ii=fam_ind_[nstudy](i)+1; ii<=fam_ind_[nstudy](i+1); ii++)
				{
					if (!Y_cox_[nstudy][ii].is_alive_ && sva_item.Miss_[nstudy](ii) == 1.)
					{
						sva_item.S0_ = 0.0;
						sva_item.S1_.setZero();
						sva_item.S2_.setZero();
						
						for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
						{
							sva_item.riskset_[nstudy1].setZero();
							for (int j=0; j<F_[nstudy1].size(); j++)
							{
								if (sva_item.Miss_[nstudy1](j) == 1. && Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
								{
									if (Y_cox_[nstudy1][j].is_alive_)
									{
										sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_) 
											? sva_item.e_Wttheta_[nstudy1](j) : 0.;
									}
									else
									{
										sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_) 
											? sva_item.e_Wttheta_[nstudy1](j) : 0.;
									}
								}
							}
							
							if (flag_uw_)
							{
								sva_item.S0_ += sva_item.riskset_[nstudy1].sum();
								sva_item.S1_ += W_[nstudy1].transpose()*sva_item.riskset_[nstudy1];
								sva_item.S2_ += W_[nstudy1].transpose()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
							}
							else
							{
								sva_item.S0_ += sva_item.riskset_[nstudy1].dot(wt_[nstudy1]);
								sva_item.S1_ += W_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1];
								sva_item.S2_ += W_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
							}
						}

						sva_item.Uhat_[nstudy].row(ii) += W_[nstudy].row(ii)-sva_item.S1_.transpose()/sva_item.S0_;

						if (flag_uw_)
						{
							sva_item.WtW_.noalias() += sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_);
								
							for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
							{
								for (int j=0; j<F_[nstudy1].size(); j++)
								{
									if (sva_item.Miss_[nstudy1](j) == 1. && Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
									{
										if (Y_cox_[nstudy1][j].is_alive_)
										{
											if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_)
											{
												sva_item.Uhat_[nstudy1].row(j) -= (W_[nstudy1].row(j)-sva_item.S1_.transpose()/sva_item.S0_)
													*sva_item.e_Wttheta_[nstudy1](j)/sva_item.S0_;
											}
										}
										else
										{
											if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_)
											{
												sva_item.Uhat_[nstudy1].row(j) -= (W_[nstudy1].row(j)-sva_item.S1_.transpose()/sva_item.S0_)
													*sva_item.e_Wttheta_[nstudy1](j)/sva_item.S0_;
											}
										}
									}
								}								
							}
							
							if (!flag_robust_ && (fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
							{
								sva_item.Bhat_.noalias() += sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_);
							}
							
						}
						else
						{
							sva_item.WtW_.noalias() += (sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](ii);
							
							for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
							{
								for (int j=0; j<F_[nstudy1].size(); j++)
								{
									if (sva_item.Miss_[nstudy1](j) == 1. && Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
									{
										if (Y_cox_[nstudy1][j].is_alive_)
										{
											if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_)
											{
												sva_item.Uhat_[nstudy1].row(j) -= (W_[nstudy1].row(j).transpose()-sva_item.S1_.transpose()/sva_item.S0_)
													*sva_item.e_Wttheta_[nstudy1](j)*wt_[nstudy](ii)/sva_item.S0_;
											}
										}
										else
										{
											if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_)
											{
												sva_item.Uhat_[nstudy1].row(j) -= (W_[nstudy1].row(j).transpose()-sva_item.S1_.transpose()/sva_item.S0_)
													*sva_item.e_Wttheta_[nstudy1](j)*wt_[nstudy](ii)/sva_item.S0_;
											}
										}
									}
								}								
							}

							if (!flag_robust_ && (fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
							{
								if (flag_pairwise_inclusion_prob_)
								{
									sva_item.Bhat_.noalias() += (sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))
										*sva_item.wtd_[nstudy](ii);
								}
								else
								{
									sva_item.Bhat_.noalias() += (sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))
										*wt_[nstudy](ii)*wt_[nstudy](ii);
								}
							}							
						}
					}
				}
			}
		}
		
		if (flag_uw_) 
		{	
			/**** unweighted analysis ***************************************************/			
			if (flag_robust_) 
			{
				/**** robust variance estimation ****************************************/
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{			
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
							}
						}
					}
				}
				/**** robust variance estimation ****************************************/				
			} 
			else 
			{				
				/**** model-based variance estimation ***********************************/				
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{	
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{
							if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) > 1)
							{
								for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
								{
									sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
								}							
							}
						}
					}
				}
				/**** model-based variance estimation ***********************************/
			}
			/**** unweighted analysis ***************************************************/
		} 
		else if (flag_pairwise_inclusion_prob_)
		{
			/**** weighted analysis *****************************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
			}
			/**** weighted analysis *****************************************************/
		}
		else
		{
			/**** weighted analysis without pairwise inclusion **************************/			
			if (flag_robust_) 
			{
				/**** robust variance estimation ****************************************/
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{			
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							}
						}
					}
				}
				/**** robust variance estimation ****************************************/				
			} 
			else 
			{				
				/**** model-based variance estimation ***********************************/				
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{	
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{
							if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) > 1)
							{
								for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
								{
									sva_item.Bhat_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
								}							
							}
						}
					}
				}
				/**** model-based variance estimation ***********************************/
			}
			/**** weighted analysis without pairwise inclusion **************************/			
		}
		
		sva_item.Ahat_ = sva_item.WtW_.inverse();
		sva_item.vartheta_.noalias() = sva_item.Ahat_*sva_item.Bhat_*sva_item.Ahat_;
		// // RT
		// cout << sqrt(sva_item.vartheta_(0,0)) << endl << endl;
		// // RT end
		if (flag_uw_ && sva_item.vartheta_(0,0) < (sva_item.Ahat_(0,0)-ERROR_MARGIN))
		{
			sva_item.vartheta_ = sva_item.Ahat_;
		}
		/**** variance estimation *******************************************************/
		
		// // RT
		// cout << fam_ind_[0].size()-1 << endl << endl;
		// cout << sva_item.WtW_ << endl << endl;
		// cout << sva_item.Bhat_ << endl << endl;
		// cout << sva_item.theta_ << endl << endl;
		// cout << sva_item.vartheta_ << endl << endl;
		// // RT end
	}
	
	// // RT
	// ofstream FO_test;
	// FO_test.open("test_data.tab");
	// for (int i=0; i<Y_cox_[0].size(); i++)
	// {
		// FO_test << F_[0][i] << "\t";
		// if (!Y_cox_[0][i].is_alive_)
		// {
			// FO_test << Y_cox_[0][i].survival_time_;
		// }
		// else
		// {
			// FO_test << Y_cox_[0][i].censoring_time_;
		// }
		// FO_test << "\t" << 1-Y_cox_[0][i].is_alive_;
		// FO_test << "\t" << Y_cox_[0][i].left_truncation_time_;
		// for (int j=0; j<W_[0].cols(); j++)
		// {
			// FO_test << "\t" << W_[0](i,j);
		// }
		// FO_test << endl;
	// }
	// FO_test.close();
	// // RT end
} // SUGEN::CoxphWald_

void SUGEN::SingleVariantAnalysis_PerSNPAnalysis_ (SVA_UTILS& sva_item) 
{
	SingleVariantAnalysis_GetSNP_(sva_item);
	
	if (sva_item.flag_multiallelic_) 
	{
		FO_log_ << VCF_record_.getChromStr() << ":" << VCF_record_.get1BasedPosition() << ": multiallelic SNP!" << endl;
	} 
	else 
	{
		if (method_ == LS) 
		{
			LinearWald_(sva_item);
			if (flag_strata_ && !sva_item.flag_het_converge_) 
			{
				FO_log_ << VCF_record_.getChromStr() << ":" << VCF_record_.get1BasedPosition() 
					<< ": In linear regression allowing heterogeneous variance, algorithm does not converge!" << endl;
				SingleVariantAnalysis_Output_(sva_item, sva_results_miss); 
			}
			else if (CheckSingularVar(sva_item.vartheta_))				
			{
				FO_log_ << VCF_record_.getChromStr() << ":" << VCF_record_.get1BasedPosition() << ": Singular Variance Estimation!" << endl;
				SingleVariantAnalysis_Output_(sva_item, sva_results_miss);
			}
			else {
				SingleVariantAnalysis_Output_(sva_item, sva_no_miss);
			}
		} 
		else if (method_ == logistic) 
		{
			LogisticWald_(sva_item);
			if (!sva_item.flag_logistic_converge_) 
			{
				FO_log_ << VCF_record_.getChromStr() << ":" << VCF_record_.get1BasedPosition() << ": In logistic regression, algorithm does not converge!" << endl;
				SingleVariantAnalysis_Output_(sva_item, sva_results_miss); 
			}
			else if (CheckSingularVar(sva_item.vartheta_)) 
			{
				FO_log_ << VCF_record_.getChromStr() << ":" << VCF_record_.get1BasedPosition() << ": Singular Variance Estimation!" << endl;
				SingleVariantAnalysis_Output_(sva_item, sva_results_miss);
			} else {			
				SingleVariantAnalysis_Output_(sva_item, sva_no_miss);
			}
		}
		else if (method_ == right_censored)
		{
			CoxphWald_(sva_item);
			if (!sva_item.flag_coxph_converge_) 
			{
				FO_log_ << VCF_record_.getChromStr() << ":" << VCF_record_.get1BasedPosition() << ": In Cox proportional hazards regression, algorithm does not converge!" << endl;
				SingleVariantAnalysis_Output_(sva_item, sva_results_miss); 
			}
			else if (CheckSingularVar(sva_item.vartheta_)) 
			{
				FO_log_ << VCF_record_.getChromStr() << ":" << VCF_record_.get1BasedPosition() << ": Singular Variance Estimation!" << endl;
				SingleVariantAnalysis_Output_(sva_item, sva_results_miss);
			} else {			
				SingleVariantAnalysis_Output_(sva_item, sva_no_miss);
			}			
		}
	}	
} // SUGEN::SingleVariantAnalysis_PerSNPAnalysis_

void SUGEN::SingleVariantAnalysis_ () 
{	
	SVA_UTILS sva_item;
	
	SingleVariantAnalysis_Initialization_(sva_item);
	
	/**** open output file **************************************************************/
	if (flag_out_zip_) 
	{
		sva_item.FO_out_ = ifopen(FN_out_.c_str(), "w", InputFile::GZIP);
	} 
	else 
	{
		sva_item.FO_out_ = ifopen(FN_out_.c_str(), "w", InputFile::UNCOMPRESSED);
	}
	if (!(*sva_item.FO_out_).isOpen()) 
	{
		error(FO_log_, "Error: Cannot open file "+FN_out_+"!");
	} 
	else 
	{
		SingleVariantAnalysis_Output_(sva_item, sva_header); 
	}		
	/**** open output file **************************************************************/
	
	/**** SNP analysis ******************************************************************/
	VCF_reader_.open(FN_geno_.c_str(), VCF_header_);
	VCF_reader_.readVcfIndex();	
	FO_log_ << "Start variant by variant analysis..." << endl;
	if (extract_type_ == EXTRACT_TYPE_CHR) 
	{	
		VCF_reader_.setReadSection(extract_chr_.c_str());
		while (VCF_reader_.readRecord(VCF_record_)) 
		{			
			SingleVariantAnalysis_PerSNPAnalysis_(sva_item);
		}		
	} 
	else if (extract_type_ == EXTRACT_TYPE_RANGE) 
	{	
		VCF_reader_.set1BasedReadSection(extract_chr_.c_str(), extract_start_, extract_end_+1);
		while (VCF_reader_.readRecord(VCF_record_)) 
		{
			SingleVariantAnalysis_PerSNPAnalysis_(sva_item);
		}	
	} 
	else if (extract_type_ == EXTRACT_TYPE_FILE) 
	{
		unsigned long long NSNP, NSNP_final;
		int32_t pos;
		string SNP_name;
		vector<string> chr_pos;
		ifstream FI;
				
		nrow(NSNP, FN_extract_, false, FO_log_);
		
		FI.open(FN_extract_);
		NSNP_final = 0;
		for (unsigned long long i=0; i<NSNP; i++) 
		{			
			FI >> SNP_name;
			chr_pos.clear();
			if (!Split(SNP_name, ":", &chr_pos)) 
			{
				stdError("Error: Cannot parse SNP "+SNP_name+" in "+FN_extract_+"!\n");
			}
			pos = atoi(chr_pos[1].c_str());
			VCF_reader_.set1BasedReadSection(chr_pos[0].c_str(), pos, pos+1);
			if (VCF_reader_.readRecord(VCF_record_)) 
			{
				SingleVariantAnalysis_PerSNPAnalysis_(sva_item);
				NSNP_final ++;
			}
		}
		FI.close();
		if (NSNP_final == 0) 
		{
			FO_log_ << "Warning: No variants in "+FN_extract_+" are present in the VCF file!" << endl;
			FO_log_ << "Warning: Therefore, no analysis has been performed!" << endl;
		}			
	}
	else
	{
		while (VCF_reader_.readRecord(VCF_record_)) 
		{			
			SingleVariantAnalysis_PerSNPAnalysis_(sva_item);
		}
	}
	VCF_reader_.close();
	ifclose(sva_item.FO_out_);
	FO_log_ << "Done!" << endl << endl;
	/**** SNP analysis ******************************************************************/			
} // SUGEN::SingleVariantAnalysis_

void SUGEN::ScoreTests_GlobalInitialization_ (SVA_UTILS& sva_item) 
{
	int N1;
	if (p_ > 0)
	{
		sva_item.theta_.resize(p_);
		sva_item.WtW_.resize(p_, p_);
		sva_item.WtY_.resize(p_);
		sva_item.Bhat_cov_.resize(p_, p_);
	}
	sva_item.Ahat_rescale_.resize(p_+1, p_+1);
	sva_item.Bhat_rescale_.resize(p_+1, p_+1);
	
	sva_item.resi_.resize(N_study_);
	sva_item.rawG_.resize(N_study_);
	sva_item.Uhat_.resize(N_study_);
	if (!flag_robust_ && !flag_uw_ && flag_pairwise_inclusion_prob_) 
	{
		sva_item.wtd_.resize(N_study_);
	}
	if (flag_strata_)
	{
		sva_item.het_sigma2_.resize(N_study_);
		sva_item.het_sigma2_0_.resize(N_study_);
		sva_item.theta0_.resize(p_);
		sva_item.theta_initilization_.setZero(p_);
		sva_item.het_miss_inv_sigma2_.resize(N_study_);
	}	
	if (method_ == logistic) 
	{
		sva_item.Wttheta_.resize(N_study_);
		sva_item.e_Wttheta_.resize(N_study_);
		sva_item.logi_Wttheta_.resize(N_study_);
		sva_item.logi2_Wttheta_.resize(N_study_);
		sva_item.theta0_.resize(p_);
		sva_item.theta_initilization_.setZero(p_);
	}
	else if (method_ == right_censored)
	{
		sva_item.Wttheta_.resize(N_study_);
		sva_item.e_Wttheta_.resize(N_study_);
		sva_item.riskset_.resize(N_study_);
		if (p_ > 0)
		{
			sva_item.theta0_.resize(p_);
			sva_item.theta_initilization_.setZero(p_);
			sva_item.S1_.resize(p_);
			sva_item.S2_.resize(p_, p_);
		}		
	}		
	N_total_ = 0;
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		N1 = F_[nstudy].size();
		N_total_ += N1;
		sva_item.resi_[nstudy].resize(N1);
		sva_item.rawG_[nstudy].resize(N1);
		if (p_ > 0)
		{
			sva_item.Uhat_[nstudy].resize(N1, p_);
		}
		if (!flag_robust_ && !flag_uw_ && flag_pairwise_inclusion_prob_) 
		{
			sva_item.wtd_[nstudy].resize(N1);
			for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
			{
				for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
				{
					if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
					{
						sva_item.wtd_[nstudy](j) = wtds_[nstudy](j,j);
						wtds_[nstudy](j,j) = 0.;						
					}
					else
					{
						sva_item.wtd_[nstudy](j) = 0.;
					}
				}
			}
		}
		if (flag_strata_)
		{
			sva_item.het_sigma2_[nstudy].resize(strata_ind_[nstudy].cols());
			sva_item.het_sigma2_0_[nstudy].resize(strata_ind_[nstudy].cols());
			sva_item.het_miss_inv_sigma2_[nstudy].resize(N1);
		}
		if (method_ == logistic) 
		{
			sva_item.Wttheta_[nstudy].resize(N1);
			sva_item.e_Wttheta_[nstudy].resize(N1);
			sva_item.logi_Wttheta_[nstudy].resize(N1);
			sva_item.logi2_Wttheta_[nstudy].resize(N1);
		}
		else if (method_ == right_censored)
		{
			sva_item.Wttheta_[nstudy].resize(N1);
			sva_item.e_Wttheta_[nstudy].resize(N1);
			sva_item.riskset_[nstudy].resize(N1);
		}
	}
	sva_item.G_.resize(N_study_);
	sva_item.Uhat_G_.resize(N_study_);
} // SUGEN::ScoreTests_GlobalInitialization_

void SUGEN::LinearScoreNull_ (SVA_UTILS& sva_item) 
{
	double numerator, denominator;
	
	/**** effect estimation *************************************************************/
	sva_item.WtW_.setZero();
	sva_item.WtY_.setZero();
	if (flag_uw_) 
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.WtW_.noalias() += W_[nstudy].transpose()*W_[nstudy];
			sva_item.WtY_.noalias() += W_[nstudy].transpose()*Y_[nstudy];
		}		
	} 
	else 
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.WtW_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*W_[nstudy];
			sva_item.WtY_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*Y_[nstudy];
		}
	}	
	sva_item.theta_ = sva_item.WtW_.selfadjointView<Eigen::Upper>().ldlt().solve(sva_item.WtY_);
	/**** effect estimation *************************************************************/
	
	/**** calculate sigma2 **************************************************************/
	numerator = 0.0;
	denominator = 0.0;	
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		sva_item.resi_[nstudy] = Y_[nstudy]-W_[nstudy]*sva_item.theta_;
		if (flag_uw_) 
		{
			numerator += sva_item.resi_[nstudy].squaredNorm();
			denominator += sva_item.resi_[nstudy].size();
		}
		else {
			numerator += (sva_item.resi_[nstudy].array()*wt_[nstudy].array()*sva_item.resi_[nstudy].array()).sum();
			denominator += wt_[nstudy].sum();			
		}			
	}
	sva_item.sigma2_ = numerator/denominator;
	/**** calculate sigma2 **************************************************************/
	
	if (flag_strata_)
	{
		double tol;
		int iter;
		
		sva_item.theta0_ = sva_item.theta_;
		for (int nstudy=0; nstudy<N_study_; nstudy++)
		{
			for (int i=0; i<strata_ind_[nstudy].cols(); i++)
			{	
				sva_item.het_sigma2_[nstudy](i) = sva_item.sigma2_;
				sva_item.het_sigma2_0_[nstudy](i) = sva_item.sigma2_;
			}
		}
		sva_item.flag_het_converge_ = false;
		
		for (iter=1; iter<=MAX_ITER; iter++) 
		{	
			/**** effect estimation *****************************************************/
			sva_item.WtW_.setZero();
			sva_item.WtY_.setZero();
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.het_miss_inv_sigma2_[nstudy].setZero();
				for (int i=0; i<strata_ind_[nstudy].cols(); i++)
				{
					sva_item.het_miss_inv_sigma2_[nstudy].noalias() += strata_ind_[nstudy].col(i)/sva_item.het_sigma2_[nstudy](i);
				}
				if (flag_uw_)
				{
					sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
					sva_item.WtY_.noalias() += W_[nstudy].transpose()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*Y_[nstudy];
				}
				else
				{
					sva_item.WtW_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
					sva_item.WtY_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*Y_[nstudy];						
				}
			}		
			sva_item.theta_ = sva_item.WtW_.selfadjointView<Eigen::Upper>().ldlt().solve(sva_item.WtY_);
			/**** effect estimation *****************************************************/
			
			/**** calculate residual variance *******************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{
				sva_item.resi_[nstudy] = Y_[nstudy]-W_[nstudy]*sva_item.theta_;
				for (int i=0; i<strata_ind_[nstudy].cols(); i++)
				{	
					if (flag_uw_) 
					{
						numerator = (sva_item.resi_[nstudy].array()*strata_ind_[nstudy].col(i).array()).matrix().squaredNorm();
						denominator = strata_ind_[nstudy].col(i).sum();
					}
					else
					{
						numerator = (sva_item.resi_[nstudy].array()*wt_[nstudy].array()*strata_ind_[nstudy].col(i).array()*sva_item.resi_[nstudy].array()).sum();
						denominator = (strata_ind_[nstudy].col(i).array()*wt_[nstudy].array()).sum();					
					}
					sva_item.het_sigma2_[nstudy](i) = numerator/denominator;
				}
			}		
			/**** calculate residual variance *******************************************/
						
			tol = (sva_item.theta_-sva_item.theta0_).array().abs().sum();
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{
				tol += (sva_item.het_sigma2_[nstudy]-sva_item.het_sigma2_0_[nstudy]).array().abs().sum();
			}
			if (tol < TOL) 
			{
				sva_item.flag_het_converge_ = true;
				break;
			} 
			else 
			{
				sva_item.theta0_ = sva_item.theta_;
				for (int nstudy=0; nstudy<N_study_; nstudy++)
				{
					sva_item.het_sigma2_0_[nstudy] = sva_item.het_sigma2_[nstudy];
				}				
			}
		}
		
		if (!sva_item.flag_het_converge_)
		{
			error(FO_log_, "Error: In estimating the null model, algorithm does not converge!");
		}
		
		/**** calculate WtW_, U_hat *****************************************************/
		sva_item.WtW_.setZero();
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.het_miss_inv_sigma2_[nstudy].setZero();
			for (int i=0; i<strata_ind_[nstudy].cols(); i++)
			{
				sva_item.het_miss_inv_sigma2_[nstudy].noalias() += strata_ind_[nstudy].col(i)/sva_item.het_sigma2_[nstudy](i);
			}
			if (flag_uw_)
			{
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
			}
			else
			{
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()
					*W_[nstudy];					
			}
			sva_item.resi_[nstudy] = Y_[nstudy]-W_[nstudy]*sva_item.theta_;
			sva_item.Uhat_[nstudy] = (sva_item.resi_[nstudy].array()*sva_item.het_miss_inv_sigma2_[nstudy].array()).matrix().asDiagonal()*W_[nstudy];
		}		
		/**** calculate WtW_, U_hat *****************************************************/
	}
	else
	{
		/**** calculate Uhat_ ***********************************************************/
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Uhat_[nstudy] = sva_item.resi_[nstudy].asDiagonal()*W_[nstudy]/sva_item.sigma2_;
		}
		sva_item.WtW_ /= sva_item.sigma2_;
		/**** calculate Uhat_ ***********************************************************/		
	}

	/**** calculate Bhat_cov_ ***********************************************************/
	sva_item.Bhat_cov_.setZero();
	if (flag_uw_) 
	{
		/**** unweighted analysis *******************************************************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
						}
					}
				}
			}			
			/**** robust variance estimation ********************************************/
		} 
		else 
		{
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							if (flag_strata_)
							{
								sva_item.Bhat_cov_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.het_miss_inv_sigma2_[nstudy](j);
							}
							else
							{
								sva_item.Bhat_cov_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)/sva_item.sigma2_;								
							}
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** unweighted analysis *******************************************************/
	} 
	else if (flag_pairwise_inclusion_prob_)
	{
		/**** weighted analysis *********************************************************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/	
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_cov_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];	
			}
			/**** robust variance estimation ********************************************/		
		} 
		else 
		{
			/**** model-based variance estimation ***************************************/
			if (flag_strata_)
			{
				for (int nstudy=0; nstudy<N_study_; nstudy++)
				{
					sva_item.Bhat_cov_.noalias() += W_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
				}
			}
			else
			{
				for (int nstudy=0; nstudy<N_study_; nstudy++)
				{
					sva_item.Bhat_cov_.noalias() += W_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()*W_[nstudy];
				}
				sva_item.Bhat_ /= sva_item.sigma2_;
			}
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_cov_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis *********************************************************/
	}
	else
	{
		/**** weighted analysis without pairwise inclusion probabilities ****************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
						}
					}
				}
			}			
			/**** robust variance estimation ********************************************/
		} 
		else 
		{
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							if (flag_strata_)
							{
								sva_item.Bhat_cov_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.het_miss_inv_sigma2_[nstudy](j)
									*wt_[nstudy](j)*wt_[nstudy](j);
							}
							else
							{
								sva_item.Bhat_cov_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)/sva_item.sigma2_
									*wt_[nstudy](j)*wt_[nstudy](j);;								
							}
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);;
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis without pairwise inclusion probabilities ****************/		
	}
	/**** calculate Bhat_cov_ ***********************************************************/
} // SUGEN::LinearScoreNull_

void SUGEN::LogisticScoreNull_ (SVA_UTILS& sva_item)
{
	double tol;
	int iter, stop_sign = 0;
	bool small_step_ind;
	double last_tol = 1E20;
	// // RT
	// time_t timer_start, timer_end;
	// // RT end
	
	sva_item.theta0_.setZero();
	sva_item.theta_.setZero();
	
	/**** effect estimation *************************************************************/
	for (iter=1; iter<=MAX_ITER; iter++) 
	{
		// // RT
		// time(&timer_start);
		// // RT end		
		sva_item.WtW_.setZero();
		sva_item.WtY_.setZero();		
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Wttheta_[nstudy].noalias() = W_[nstudy]*sva_item.theta0_;
			sva_item.e_Wttheta_[nstudy] = sva_item.Wttheta_[nstudy].array().exp();
			sva_item.logi_Wttheta_[nstudy] = sva_item.e_Wttheta_[nstudy].array()/(sva_item.e_Wttheta_[nstudy].array()+1.);
			sva_item.logi2_Wttheta_[nstudy] = sva_item.logi_Wttheta_[nstudy].array()/(sva_item.e_Wttheta_[nstudy].array()+1.);			
			sva_item.resi_[nstudy] = Y_[nstudy]-sva_item.logi_Wttheta_[nstudy];
			sva_item.Uhat_[nstudy].noalias() = sva_item.resi_[nstudy].asDiagonal()*W_[nstudy];
			if (flag_uw_) 
			{
				sva_item.WtY_.noalias() += sva_item.Uhat_[nstudy].transpose().rowwise().sum();
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
			} 
			else 
			{
				sva_item.WtY_.noalias() += sva_item.Uhat_[nstudy].transpose()*wt_[nstudy];
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
			}
		}
		sva_item.theta_ = sva_item.WtW_.selfadjointView<Eigen::Upper>().ldlt().solve(sva_item.WtY_);
		
		tol = sva_item.theta_.array().abs().sum();
		
		small_step_ind = false;
		while (tol > sva_item.newton_raphson_step_size_)
		{
			small_step_ind = true;
			sva_item.theta_ *= 0.5;
			tol = sva_item.theta_.array().abs().sum();
		}
		
		sva_item.theta_ += sva_item.theta0_;
		sva_item.theta0_ = sva_item.theta_;
		
		if (tol >= last_tol-TOL && !small_step_ind)
		{
			stop_sign ++;
		}
		last_tol = tol;

		// // RT
		// time(&timer_end);
		// cout << "Iteration=" << iter << "; time=" << difftime(timer_end, timer_start) << " seconds; error=" << tol << endl;
		// cout << "theta=" << sva_item.theta_.transpose() << endl;
		// cout << endl;
		// // RT end
				
		if (tol < TOL) 
		{
			break;
		}
		else if (::isnan(tol))
		{
			sva_item.newton_raphson_step_size_ /= 10.;
			iter = 1;
			sva_item.theta0_ = sva_item.theta_initilization_;
			sva_item.theta_ = sva_item.theta_initilization_;
		}
		else if (stop_sign > 5)
		{
			break;
		}
	
	}
	/**** effect estimation *************************************************************/
	
	if (iter == MAX_ITER+1 || ::isnan(tol) || stop_sign > 5) 
	{
		error(FO_log_, "Error: Logistic regression under the null does not converge!");
	} 
	else 
	{
		/**** calculate Uhat_, resi_ ****************************************************/
		sva_item.WtW_.setZero();
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Wttheta_[nstudy].noalias() = W_[nstudy]*sva_item.theta0_;
			sva_item.e_Wttheta_[nstudy] = sva_item.Wttheta_[nstudy].array().exp();
			sva_item.logi_Wttheta_[nstudy] = sva_item.e_Wttheta_[nstudy].array()/(sva_item.e_Wttheta_[nstudy].array()+1.);
			sva_item.logi2_Wttheta_[nstudy] = sva_item.logi_Wttheta_[nstudy].array()/(sva_item.e_Wttheta_[nstudy].array()+1.);			
			sva_item.resi_[nstudy] = Y_[nstudy]-sva_item.logi_Wttheta_[nstudy];
			sva_item.Uhat_[nstudy].noalias() = sva_item.resi_[nstudy].asDiagonal()*W_[nstudy];
			if (flag_uw_) 
			{
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
			} 
			else 
			{
				sva_item.WtW_.noalias() += W_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
			}
		}
		/**** calculate Uhat_, resi_ ****************************************************/
	}

	/**** calculate Bhat_cov_ ***********************************************************/
	sva_item.Bhat_cov_.setZero();
	if (flag_uw_) 
	{	
		/**** unweighted analysis *******************************************************/			
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{							
							sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
						}
					}
				}
			}
			/**** robust variance estimation ********************************************/			
		} 
		else 
		{				
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{					
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							sva_item.Bhat_cov_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.logi2_Wttheta_[nstudy](j);
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{							
								sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** unweighted analysis *******************************************************/
	} 
	else if (flag_pairwise_inclusion_prob_)
	{
		/**** weighted analysis *********************************************************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_cov_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
			}
			/**** robust variance estimation ********************************************/
		} else {
			/**** model-based variance estimation ***************************************/					
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{
				sva_item.Bhat_cov_.noalias() += W_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
				sva_item.Bhat_cov_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis *********************************************************/
	}
	else
	{
		/**** weighted analysis without pairwise inclusion probabilities ****************/			
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{							
							sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
						}
					}
				}
			}
			/**** robust variance estimation ********************************************/			
		} 
		else 
		{				
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{					
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							sva_item.Bhat_cov_.noalias() += W_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.logi2_Wttheta_[nstudy](j)*wt_[nstudy](j)*wt_[nstudy](j);
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{							
								sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis without pairwise inclusion probabilities ****************/
	}
	/**** calculate Bhat_cov_ ***********************************************************/
} // SUGEN::LogisticScoreNull_

void SUGEN::CoxphScoreNull_ (SVA_UTILS& sva_item)
{
	double tol;
	int iter, stop_sign = 0;
	bool small_step_ind;
	double last_tol = 1E20;
	// // RT
	// time_t timer_start, timer_end;
	// // RT end
		
	sva_item.theta0_ = sva_item.theta_initilization_;
	sva_item.theta_ = sva_item.theta_initilization_;	
	
	/**** effect estimation *************************************************************/
	for (iter=1; iter<=MAX_ITER; iter++)		
	{
		// // RT
		// time(&timer_start);
		// // RT end
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Wttheta_[nstudy].noalias() = W_[nstudy]*sva_item.theta0_;
			sva_item.e_Wttheta_[nstudy] = sva_item.Wttheta_[nstudy].array().exp();
		}
		
		sva_item.WtW_.setZero();
		sva_item.WtY_.setZero();

		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{			
			for (int i=0; i<F_[nstudy].size(); i++)
			{
				if (!Y_cox_[nstudy][i].is_alive_)
				{
					sva_item.S0_ = 0.0;
					sva_item.S1_.setZero();
					sva_item.S2_.setZero();
					
					for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
					{
						sva_item.riskset_[nstudy1].setZero();
						for (int j=0; j<F_[nstudy1].size(); j++)
						{
							if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][i].survival_time_)
							{
								if (Y_cox_[nstudy1][j].is_alive_)
								{
									sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][i].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_) ? sva_item.e_Wttheta_[nstudy1](j) : 0.;
								}
								else
								{
									sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][i].survival_time_ <= Y_cox_[nstudy1][j].survival_time_) ? sva_item.e_Wttheta_[nstudy1](j) : 0.;
								}
							}
						}
						
						if (flag_uw_)
						{
							sva_item.S0_ += sva_item.riskset_[nstudy1].sum();
							sva_item.S1_.noalias() += W_[nstudy1].transpose()*sva_item.riskset_[nstudy1];
							sva_item.S2_.noalias() += W_[nstudy1].transpose()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
						}
						else
						{
							sva_item.S0_ += sva_item.riskset_[nstudy1].dot(wt_[nstudy1]);
							sva_item.S1_.noalias() += W_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1];
							sva_item.S2_.noalias() += W_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
						}
					}
					
					if (flag_uw_)
					{
						sva_item.WtW_.noalias() += sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_);
						sva_item.WtY_.noalias() += W_[nstudy].row(i).transpose()-sva_item.S1_/sva_item.S0_;
					}
					else
					{
						sva_item.WtW_.noalias() += (sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](i);
						sva_item.WtY_.noalias() += (W_[nstudy].row(i).transpose()-sva_item.S1_/sva_item.S0_)*wt_[nstudy](i);						
					}
				}
			}
		}
		
		sva_item.theta_ = sva_item.WtW_.selfadjointView<Eigen::Upper>().ldlt().solve(sva_item.WtY_);
		
		tol = sva_item.theta_.array().abs().sum();

		small_step_ind = false;
		while (tol > sva_item.newton_raphson_step_size_)
		{
			small_step_ind = true;
			sva_item.theta_ *= 0.5;
			tol = sva_item.theta_.array().abs().sum();
		}
				
		sva_item.theta_ += sva_item.theta0_;
		sva_item.theta0_ = sva_item.theta_;	

		if (tol >= last_tol-TOL && !small_step_ind)
		{
			stop_sign ++;
		}
		last_tol = tol;
		
		// // RT
		// time(&timer_end);
		// cout << "Iteration=" << iter << "; time=" << difftime(timer_end, timer_start) << " seconds; error=" << tol << endl;
		// cout << "theta=" << sva_item.theta_.transpose() << endl;
		// cout << endl;
		// // RT end
		
		if (tol < TOL) 
		{
			break;
		}
		else if (::isnan(tol))
		{
			sva_item.newton_raphson_step_size_ /= 10.;
			iter = 1;
			sva_item.theta0_ = sva_item.theta_initilization_;
			sva_item.theta_ = sva_item.theta_initilization_;
		}
		else if (stop_sign > 5)
		{
			break;
		}
	}
	/**** effect estimation *************************************************************/

	if (iter == MAX_ITER+1 || ::isnan(tol) || stop_sign > 5) 
	{
		error(FO_log_, "Error: Cox proportional hazards regression under the null does not converge!");
	} 
	else 
	{
		/**** variance estimation *******************************************************/
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Wttheta_[nstudy].noalias() = W_[nstudy]*sva_item.theta0_;
			sva_item.e_Wttheta_[nstudy] = sva_item.Wttheta_[nstudy].array().exp();
			sva_item.Uhat_[nstudy].setZero();
		}
		
		sva_item.WtW_.setZero();
		sva_item.Bhat_cov_.setZero();

		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{			
			for (int i=0; i<fam_ind_[nstudy].size()-1; i++)
			{
				for (int ii=fam_ind_[nstudy](i)+1; ii<=fam_ind_[nstudy](i+1); ii++)
				{
					if (!Y_cox_[nstudy][ii].is_alive_)
					{
						sva_item.S0_ = 0.0;
						sva_item.S1_.setZero();
						sva_item.S2_.setZero();
						
						for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
						{
							sva_item.riskset_[nstudy1].setZero();
							for (int j=0; j<F_[nstudy1].size(); j++)
							{
								if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
								{
									if (Y_cox_[nstudy1][j].is_alive_)
									{
										sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_) 
											? sva_item.e_Wttheta_[nstudy1](j) : 0.;
									}
									else
									{
										sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_) 
											? sva_item.e_Wttheta_[nstudy1](j) : 0.;
									}
								}
							}
							
							if (flag_uw_)
							{
								sva_item.S0_ += sva_item.riskset_[nstudy1].sum();
								sva_item.S1_.noalias() += W_[nstudy1].transpose()*sva_item.riskset_[nstudy1];
								sva_item.S2_.noalias() += W_[nstudy1].transpose()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
							}
							else
							{
								sva_item.S0_ += sva_item.riskset_[nstudy1].dot(wt_[nstudy1]);
								sva_item.S1_.noalias() += W_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1];
								sva_item.S2_.noalias() += W_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
							}
						}

						sva_item.Uhat_[nstudy].row(ii) += W_[nstudy].row(ii)-sva_item.S1_.transpose()/sva_item.S0_;

						if (flag_uw_)
						{
							sva_item.WtW_.noalias() += sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_);								
							for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
							{
								for (int j=0; j<F_[nstudy1].size(); j++)
								{
									if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
									{
										if (Y_cox_[nstudy1][j].is_alive_)
										{
											if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_)
											{
												sva_item.Uhat_[nstudy1].row(j) -= (W_[nstudy1].row(j)-sva_item.S1_.transpose()/sva_item.S0_)
													*sva_item.e_Wttheta_[nstudy1](j)/sva_item.S0_;
											}
										}
										else
										{
											if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_)
											{
												sva_item.Uhat_[nstudy1].row(j) -= (W_[nstudy1].row(j)-sva_item.S1_.transpose()/sva_item.S0_)
													*sva_item.e_Wttheta_[nstudy1](j)/sva_item.S0_;
											}
										}
									}
								}								
							}
							
							if (!flag_robust_ && (fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
							{
								sva_item.Bhat_cov_.noalias() += sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_);
							}
						}
						else
						{
							sva_item.WtW_.noalias() += (sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](ii);							
							for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
							{
								for (int j=0; j<F_[nstudy1].size(); j++)
								{
									if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
									{
										if (Y_cox_[nstudy1][j].is_alive_)
										{
											if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_)
											{
												sva_item.Uhat_[nstudy1].row(j) -= (W_[nstudy1].row(j).transpose()-sva_item.S1_.transpose()/sva_item.S0_)
													*sva_item.e_Wttheta_[nstudy1](j)*wt_[nstudy](ii)/sva_item.S0_;
											}
										}
										else
										{
											if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_)
											{
												sva_item.Uhat_[nstudy1].row(j) -= (W_[nstudy1].row(j).transpose()-sva_item.S1_.transpose()/sva_item.S0_)
													*sva_item.e_Wttheta_[nstudy1](j)*wt_[nstudy](ii)/sva_item.S0_;
											}
										}
									}
								}								
							}

							if (!flag_robust_ && (fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
							{
								if (flag_pairwise_inclusion_prob_)
								{
									sva_item.Bhat_cov_.noalias() += 
										(sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))*sva_item.wtd_[nstudy](ii);
								}
								else
								{
									sva_item.Bhat_cov_.noalias() += 
										(sva_item.S2_/sva_item.S0_-sva_item.S1_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](ii)*wt_[nstudy](ii);									
								}
							}							
						}
					}
				}
			}
		}
		
		if (flag_uw_) 
		{	
			/**** unweighted analysis ***************************************************/			
			if (flag_robust_) 
			{
				/**** robust variance estimation ****************************************/
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{			
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
							}
						}
					}
				}
				/**** robust variance estimation ****************************************/				
			} 
			else 
			{				
				/**** model-based variance estimation ***********************************/				
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{	
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{
							if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) > 1)
							{
								for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
								{
									sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
								}							
							}
						}
					}
				}
				/**** model-based variance estimation ***********************************/
			}
			/**** unweighted analysis ***************************************************/
		} 
		else if (flag_pairwise_inclusion_prob_)
		{
			/**** weighted analysis *****************************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_cov_.noalias() += sva_item.Uhat_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
			}
			/**** weighted analysis *****************************************************/
		}
		else
		{
			/**** weighted analysis without pairwise inclusion probabilities ************/			
			if (flag_robust_) 
			{
				/**** robust variance estimation ****************************************/
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{			
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							}
						}
					}
				}
				/**** robust variance estimation ****************************************/				
			} 
			else 
			{				
				/**** model-based variance estimation ***********************************/				
				for (int nstudy=0; nstudy<N_study_; nstudy++) 
				{	
					for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
					{
						for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
						{
							if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) > 1)
							{
								for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
								{
									sva_item.Bhat_cov_.noalias() += (sva_item.Uhat_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
								}							
							}
						}
					}
				}
				/**** model-based variance estimation ***********************************/
			}
			/**** weighted analysis without pairwise inclusion probabilities ************/			
		}
	}			
} // SUGEN::CoxphScoreNull_

void SUGEN::ScoreTests_Output_ (SVA_UTILS& sva_item, const SVA_OUTPUT_TYPE_ sva_output_type) 
{
	if (sva_output_type == sva_header) 
	{
		*sva_item.FO_score_snp_ << "GROUP_ID\tCHROM\tPOS\tVCF_ID\tREF\tALT\tALT_AF\tALT_AC\tN_INFORMATIVE\tN_REF\tN_HET\tN_ALT\tN_DOSE";
		if (method_ == logistic)
		{
			*sva_item.FO_score_snp_ << "\tALT_AF_CASE\tN_CASE";
		}
		else if (method_ == right_censored)
		{
			*sva_item.FO_score_snp_ << "\tALT_AF_EVENT\tN_EVENT";
		}
		*sva_item.FO_score_snp_ << "\tU\tV\tBETA\tSE\tPVALUE\n";
		*sva_item.FO_score_mass_ << "#Samples = " << N_total_ << "\n";
	}
	else if (sva_output_type == sva_no_miss) 
	{
		for (int i=0; i<sva_item.nSNP_; i++) 
		{
			*sva_item.FO_score_snp_ << sva_item.gene_ID_ << "\t" << sva_item.SNP_chr_[i] << "\t" << sva_item.SNP_pos_[i] << "\t" << sva_item.SNP_ID_[i]
				<< "\t" << sva_item.SNP_ref_[i] << "\t" << sva_item.SNP_alt_[i]
				<< "\t" << sva_item.SNP_maf_[i] << "\t" << sva_item.SNP_mac_[i] << "\t" << N_total_ << "\t" << sva_item.SNP_n0_count_[i] 
				<< "\t"  << sva_item.SNP_n1_count_[i] << "\t" << sva_item.SNP_n2_count_[i] << "\t" << sva_item.SNP_n_dose_[i];
			
			if (method_ == logistic || method_ == right_censored)
			{
				*sva_item.FO_score_snp_ << "\t" << sva_item.SNP_maf_case_[i] << "\t" << sva_item.SNP_n_case_[i];
			}
			
			*sva_item.FO_score_snp_ << "\t" << sva_item.U_(i) << "\t" << sva_item.V_(i,i);
			if (sva_item.V_(i,i) > 0.)
			{
				sva_item.pvalue_ = gammq(0.5, sva_item.U_(i)*sva_item.U_(i)/(2.*sva_item.V_(i,i)));	
				*sva_item.FO_score_snp_ << "\t" << sva_item.beta_(i) << "\t" << sva_item.se_(i) << "\t" << sva_item.pvalue_;
			}
			else
			{
				*sva_item.FO_score_snp_ << "\tNA\tNA\tNA";
			}
			*sva_item.FO_score_snp_ << "\n";

			*sva_item.FO_score_mass_ << sva_item.gene_ID_ << "\t" << sva_item.SNP_chr_[i] << ":" << sva_item.SNP_pos_[i]
				<< "\t" << sva_item.SNP_maf_[i] << "\t" << sva_item.SNP_mac_[i] << "\t" << sva_item.SNP_n_[i] << "\t" << sva_item.SNP_n0_count_[i]
				<< "\t" << sva_item.SNP_n1_count_[i] << "\t" << sva_item.SNP_n2_count_[i] << "\t" << sva_item.U_(i);				
			for (int j=0; j<=i; j++) 
			{
				*sva_item.FO_score_mass_ << "\t" << sva_item.V_(j,i);
			}
			for (int j=i+1; j<sva_item.nSNP_; j++) 
			{
				*sva_item.FO_score_mass_ << "\t" << 0;
			}	
			*sva_item.FO_score_mass_ << "\n";
		}
	}	
} // SUGEN::ScoreTests_Output_

void SUGEN::ScoreTests_GetSNP_ (SVA_UTILS& sva_item) 
{	
	if (VCF_record_.getNumAlts() > 1) 
	{
		sva_item.flag_multiallelic_ = true;
		return;
	} 
	else 
	{
		sva_item.flag_multiallelic_ = false;
	}
	VcfRecordGenotype & genoInfo = VCF_record_.getGenotypeInfo();
	
	sva_item.mac_ = 0;
	sva_item.maf_ = 0.;
	sva_item.n0_count_ = 0;
	sva_item.n1_count_ = 0;
	sva_item.n2_count_ = 0;
	sva_item.n_dose_ = 0;
	sva_item.n_ = 0;
	if (method_ == logistic || method_ == right_censored)
	{
		sva_item.maf_case_ = 0.;
		sva_item.n_case_ = 0.;
	}
			
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		for (int i=0; i<F_[nstudy].size(); i++) 
		{
			sva_item.rawG_[nstudy](i) = ReadOneVariant_(geno_pheno_linker_[nstudy](i), genoInfo);
			if (sva_item.rawG_[nstudy](i) != MISSIND) 
			{
				sva_item.maf_ += sva_item.rawG_[nstudy](i);
				if (flag_dosage_)
				{
					sva_item.n_dose_++;
				}
				else
				{
					if (fabs(sva_item.rawG_[nstudy](i)-0.0) < ERROR_MARGIN) 
					{
						sva_item.n0_count_++;
					} 
					else if (fabs(sva_item.rawG_[nstudy](i)-1.0) < ERROR_MARGIN) 
					{
						sva_item.n1_count_++;
					} 
					else if (fabs(sva_item.rawG_[nstudy](i)-2.0) < ERROR_MARGIN) 
					{
						sva_item.n2_count_++;
					}
				}
				if (method_ == logistic && Y_[nstudy](i) == 1)
				{
					sva_item.maf_case_ += sva_item.rawG_[nstudy](i);
					sva_item.n_case_ += 1;
				}
				if (method_ == right_censored && !Y_cox_[nstudy][i].is_alive_)
				{
					sva_item.maf_case_ += sva_item.rawG_[nstudy](i);
					sva_item.n_case_ += 1;					
				}				
			}
		}
	}
						
	sva_item.n_ = sva_item.n0_count_+sva_item.n1_count_+sva_item.n2_count_+sva_item.n_dose_;
	if (sva_item.n_ > 0) 
	{
		sva_item.maf_ /= 2.*sva_item.n_;
		sva_item.mac_ = sva_item.n1_count_+2*sva_item.n2_count_;
	}
	if (method_ == logistic || method_ == right_censored)
	{
		if (sva_item.n_case_ > 0)
		{
			sva_item.maf_case_ /= 2.*sva_item.n_case_;
		}
	}
	
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		for (int i=0; i<F_[nstudy].size(); i++) 
		{
			if (sva_item.rawG_[nstudy](i) == MISSIND) sva_item.rawG_[nstudy](i) = 2.*sva_item.maf_;
		}
	}
} // SUGEN::ScoreTests_GetSNP_

void SUGEN::CalculateUV_ (SVA_UTILS& sva_item) 
{
	sva_item.U_.setZero();
	if (flag_uw_) 
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++)
		{
			for (int i=0; i<sva_item.nSNP_; i++) 
			{
				sva_item.U_(i) += sva_item.Uhat_G_[nstudy].col(i).sum();
			}
		}
	}
	else
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++)
		{
			for (int i=0; i<sva_item.nSNP_; i++) 
			{
				sva_item.U_(i) += sva_item.Uhat_G_[nstudy].col(i).dot(wt_[nstudy]);
			}
		}
	}

	Eigen::MatrixXd invArr_Arb = sva_item.Ahat_.bottomRightCorner(p_, p_).selfadjointView<Eigen::Upper>().ldlt().solve(sva_item.Ahat_.bottomLeftCorner(p_, sva_item.nSNP_));
	Eigen::MatrixXd Bbr_invArr_Arb = sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_)*invArr_Arb;
	sva_item.V_ = sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_);
	sva_item.V_ -= Bbr_invArr_Arb;
	sva_item.V_ -= Bbr_invArr_Arb.transpose();
	sva_item.V_.noalias() += invArr_Arb.transpose()*sva_item.Bhat_.bottomRightCorner(p_, p_)*invArr_Arb;
	
	sva_item.V_A_ = sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_);
	sva_item.V_A_ -= sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_)*invArr_Arb;
		
	sva_item.A_snplist_.clear();
	sva_item.B_snplist_.clear();
	for (int i=0; i<sva_item.nSNP_; i++) 
	{
		if (flag_uw_ && ((1./sva_item.V_(i,i)) < ((1./sva_item.V_A_(i,i)-ERROR_MARGIN)) || (std::isnan(sva_item.V_(i,i)))))
		{
			sva_item.A_snplist_.push_back(i);
		}
		else
		{
			sva_item.B_snplist_.push_back(i);
		}
	}
	
	if (sva_item.A_snplist_.size()+sva_item.B_snplist_.size() == sva_item.nSNP_)
	{
		for (int i=0; i<sva_item.A_snplist_.size(); i++)
		{
			for (int j=0; j<sva_item.A_snplist_.size(); j++)
			{
				sva_item.V_(sva_item.A_snplist_[i],sva_item.A_snplist_[j]) = sva_item.V_A_(sva_item.A_snplist_[i],sva_item.A_snplist_[j]);
			}
			for (int j=0; j<sva_item.B_snplist_.size(); j++)
			{
				sva_item.V_(sva_item.A_snplist_[i],sva_item.B_snplist_[j]) = sva_item.V_A_(sva_item.A_snplist_[i],sva_item.B_snplist_[j]);
				sva_item.V_(sva_item.B_snplist_[j],sva_item.A_snplist_[i]) = sva_item.V_A_(sva_item.B_snplist_[j],sva_item.A_snplist_[i]);
			}
		}
		for (int i=0; i<sva_item.nSNP_; i++) 
		{
			sva_item.beta_(i) = sva_item.U_(i)/sva_item.V_(i,i);
			sva_item.se_(i) = sqrt(1./sva_item.V_(i,i));
		}		
	}
	else
	{
		error(FO_log_, "Error: In SUGEN::CalculateUV_, sva_item.A_snplist_.size()+sva_item.B_snplist_.size() != sva_item.nSNP_!");
	}
	
	double scale;
	if (rescale_type_ == NAIVE) 
	{
		if (!flag_uw_)
		{	
			scale = 0.;
			for (int nstudy=0; nstudy<N_study_; nstudy++) {
				scale += wt_[nstudy].sum();
			}
			scale = N_total_/scale;
			sva_item.U_ *= scale;
			sva_item.V_ *= scale*scale;			
		}
	}
	else 
	{
		sva_item.Ahat_rescale_(0,0) = sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).sum();
		sva_item.Ahat_rescale_.topRightCorner(1, p_) = sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).colwise().sum();
		sva_item.Ahat_rescale_.bottomLeftCorner(p_, 1) = sva_item.Ahat_rescale_.topRightCorner(1, p_).transpose();
		sva_item.Ahat_rescale_.bottomRightCorner(p_, p_) = sva_item.Ahat_.bottomRightCorner(p_, p_);
		
		sva_item.Bhat_rescale_(0,0) = sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).sum();
		sva_item.Bhat_rescale_.topRightCorner(1, p_) = sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).colwise().sum();
		sva_item.Bhat_rescale_.bottomLeftCorner(p_, 1) = sva_item.Bhat_rescale_.topRightCorner(1, p_).transpose();
		sva_item.Bhat_rescale_.bottomRightCorner(p_, p_) = sva_item.Bhat_.bottomRightCorner(p_, p_);
		
		LDLT<MatrixXd> chol_A(sva_item.Ahat_rescale_);
		MatrixXd invA_cholB = chol_A.solve(sva_item.Bhat_rescale_);
		MatrixXd invA_B_invA = chol_A.solve(invA_cholB.transpose());
		double invV = invA_B_invA(0,0);
		
		MatrixXd Abr_invArr_Arb = sva_item.Ahat_rescale_.topRightCorner(1, p_)*(invArr_Arb.rowwise().sum());
		double invAbb = sva_item.Ahat_rescale_(0,0)-Abr_invArr_Arb(0,0);

		if (invV > 0. && invAbb > 0.)
		{
			scale = 1./(invV*invAbb);
			sva_item.U_ *= scale;
			sva_item.V_ *= scale*scale;
		}
		else
		{
			if (!flag_uw_)
			{	
				scale = 0.;
				for (int nstudy=0; nstudy<N_study_; nstudy++) {
					scale += wt_[nstudy].sum();
				}
				scale = N_total_/scale;
				sva_item.U_ *= scale;
				sva_item.V_ *= scale*scale;				
			}
		}
	}
} // SUGEN::CalculateUV_

void SUGEN::LinearScore_ (SVA_UTILS& sva_item) 
{	 
	/**** calculate Uhat_G_, Ahat_ ******************************************************/
	if (flag_strata_)
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++)
		{
			sva_item.Uhat_G_[nstudy] = (sva_item.resi_[nstudy].array()*sva_item.het_miss_inv_sigma2_[nstudy].array()).matrix().asDiagonal()*sva_item.G_[nstudy];
		}
	}
	else
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Uhat_G_[nstudy] = sva_item.resi_[nstudy].asDiagonal()*sva_item.G_[nstudy];
			sva_item.Uhat_G_[nstudy] /= sva_item.sigma2_;
		}
	}
	
	sva_item.Ahat_.setZero();
	if (flag_strata_)
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			if (flag_uw_) 
			{
				sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
					sva_item.G_[nstudy].transpose()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*sva_item.G_[nstudy];
				sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
					sva_item.G_[nstudy].transpose()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
			}
			else
			{
				sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
					sva_item.G_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*sva_item.G_[nstudy];
				sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
					sva_item.G_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
			}
		}
	}
	else
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			if (flag_uw_) 
			{
				sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += sva_item.G_[nstudy].transpose()*sva_item.G_[nstudy];
				sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).noalias() += sva_item.G_[nstudy].transpose()*W_[nstudy];
			}
			else
			{
				sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
					sva_item.G_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.G_[nstudy];
				sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).noalias() += sva_item.G_[nstudy].transpose()*wt_[nstudy].asDiagonal()*W_[nstudy];
			}
		}
		sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_) /= sva_item.sigma2_;
		sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_) /= sva_item.sigma2_;
	}
	sva_item.Ahat_.bottomRightCorner(p_, p_) = sva_item.WtW_;
	sva_item.Ahat_.bottomLeftCorner(p_, sva_item.nSNP_) = sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).transpose(); 
	/**** calculate Uhat_G_, Ahat_ ******************************************************/
	
	/**** calculate Bhat_ ***************************************************************/
	sva_item.Bhat_.setZero();
	if (flag_uw_) 
	{
		/**** unweighted analysis *******************************************************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj);
							sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
						}
					}
				}
			}			
			/**** robust variance estimation ********************************************/
		} 
		else 
		{
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							if (flag_strata_)
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
									sva_item.G_[nstudy].row(j).transpose()*sva_item.G_[nstudy].row(j)*sva_item.het_miss_inv_sigma2_[nstudy](j);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
									sva_item.G_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.het_miss_inv_sigma2_[nstudy](j);
							}
							else
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
									sva_item.G_[nstudy].row(j).transpose()*sva_item.G_[nstudy].row(j)/sva_item.sigma2_;
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
									sva_item.G_[nstudy].row(j).transpose()*W_[nstudy].row(j)/sva_item.sigma2_;							
							}
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
									(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
									(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** unweighted analysis *******************************************************/
	} 
	else if (flag_pairwise_inclusion_prob_)
	{
		/**** weighted analysis *********************************************************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/	
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
					sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_G_[nstudy];
				sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
					sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];	
			}
			/**** robust variance estimation ********************************************/		
		} 
		else 
		{
			/**** model-based variance estimation ***************************************/
			if (flag_strata_)
			{
				for (int nstudy=0; nstudy<N_study_; nstudy++)
				{
					sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
						sva_item.G_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()
						*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*sva_item.G_[nstudy];
					sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
						sva_item.G_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()*sva_item.het_miss_inv_sigma2_[nstudy].asDiagonal()*W_[nstudy];
				}
			}
			else
			{
				for (int nstudy=0; nstudy<N_study_; nstudy++)
				{
					sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
						sva_item.G_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()*sva_item.G_[nstudy];
					sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
						sva_item.G_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()*W_[nstudy];
				}
				sva_item.Bhat_ /= sva_item.sigma2_;
			}
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
					sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_G_[nstudy];
				sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis *********************************************************/
	}
	else
	{
		/**** weighted analysis without pairwise inclusion probabilities ****************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
						}
					}
				}
			}			
			/**** robust variance estimation ********************************************/
		} 
		else 
		{
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							if (flag_strata_)
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
									sva_item.G_[nstudy].row(j).transpose()*sva_item.G_[nstudy].row(j)*sva_item.het_miss_inv_sigma2_[nstudy](j)*wt_[nstudy](j)*wt_[nstudy](j);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
									sva_item.G_[nstudy].row(j).transpose()*W_[nstudy].row(j)*sva_item.het_miss_inv_sigma2_[nstudy](j)*wt_[nstudy](j)*wt_[nstudy](j);
							}
							else
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
									sva_item.G_[nstudy].row(j).transpose()*sva_item.G_[nstudy].row(j)*wt_[nstudy](j)*wt_[nstudy](j)/sva_item.sigma2_;
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
									sva_item.G_[nstudy].row(j).transpose()*W_[nstudy].row(j)*wt_[nstudy](j)*wt_[nstudy](j)/sva_item.sigma2_;							
							}
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
									(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
									(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis without pairwise inclusion probabilities ****************/
	}
	sva_item.Bhat_.bottomRightCorner(p_, p_) = sva_item.Bhat_cov_;
	sva_item.Bhat_.bottomLeftCorner(p_, sva_item.nSNP_) = sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).transpose();
	/**** calculate Bhat_ ***************************************************************/
	
	CalculateUV_(sva_item);
} // SUGEN::LinearScore_

void SUGEN::LogisticScore_ (SVA_UTILS& sva_item) 
{
	/**** calculate Uhat_G_, Ahat_ ******************************************************/
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		sva_item.Uhat_G_[nstudy] = sva_item.resi_[nstudy].asDiagonal()*sva_item.G_[nstudy];
	}
	sva_item.Ahat_.setZero();
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		if (flag_uw_) 
		{
			sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
				sva_item.G_[nstudy].transpose()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*sva_item.G_[nstudy];
			sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
				sva_item.G_[nstudy].transpose()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
		}
		else
		{
			sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
				sva_item.G_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*sva_item.G_[nstudy];
			sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
				sva_item.G_[nstudy].transpose()*wt_[nstudy].asDiagonal()*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
		}
	}
	sva_item.Ahat_.bottomRightCorner(p_, p_) = sva_item.WtW_;
	sva_item.Ahat_.bottomLeftCorner(p_, sva_item.nSNP_) = sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).transpose();	
	/**** calculate Uhat_G_, Ahat_ ******************************************************/
	
	/**** calculate Bhat_ ***************************************************************/
	sva_item.Bhat_.setZero();
	if (flag_uw_) 
	{	
		/**** unweighted analysis *******************************************************/			
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{							
							sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += (sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj);
							sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += (sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
						}
					}
				}
			}
			/**** robust variance estimation ********************************************/			
		} 
		else 
		{				
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{					
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += sva_item.G_[nstudy].row(j).transpose()*sva_item.G_[nstudy].row(j)
								*sva_item.logi2_Wttheta_[nstudy](j);
							sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += sva_item.G_[nstudy].row(j).transpose()*W_[nstudy].row(j)
								*sva_item.logi2_Wttheta_[nstudy](j);
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{							
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += (sva_item.Uhat_G_[nstudy].row(j).transpose())
									*sva_item.Uhat_G_[nstudy].row(jj);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += (sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** unweighted analysis *******************************************************/
	} 
	else if (flag_pairwise_inclusion_prob_)
	{
		/**** weighted analysis *********************************************************/
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_G_[nstudy];
				sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
			}
			/**** robust variance estimation ********************************************/
		} else {
			/**** model-based variance estimation ***************************************/					
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{
				sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += sva_item.G_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()
					*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*sva_item.G_[nstudy];
				sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += sva_item.G_[nstudy].transpose()*sva_item.wtd_[nstudy].asDiagonal()
					*sva_item.logi2_Wttheta_[nstudy].asDiagonal()*W_[nstudy];
				sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_G_[nstudy];
				sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis *********************************************************/
	}
	else
	{
		/**** weighted analysis without pairwise inclusion probabilities ****************/			
		if (flag_robust_) 
		{
			/**** robust variance estimation ********************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{							
							sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
						}
					}
				}
			}
			/**** robust variance estimation ********************************************/			
		} 
		else 
		{				
			/**** model-based variance estimation ***************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{					
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += sva_item.G_[nstudy].row(j).transpose()*sva_item.G_[nstudy].row(j)
								*sva_item.logi2_Wttheta_[nstudy](j)*wt_[nstudy](j)*wt_[nstudy](j);
							sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += sva_item.G_[nstudy].row(j).transpose()*W_[nstudy].row(j)
								*sva_item.logi2_Wttheta_[nstudy](j)*wt_[nstudy](j)*wt_[nstudy](j);
						}
						else
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{							
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += (sva_item.Uhat_G_[nstudy].row(j).transpose())
									*sva_item.Uhat_G_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += (sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)
									*wt_[nstudy](j)*wt_[nstudy](jj);
							}
						}
					}
				}
			}
			/**** model-based variance estimation ***************************************/
		}
		/**** weighted analysis without pairwise inclusion probabilities ****************/		
	}
	sva_item.Bhat_.bottomRightCorner(p_, p_) = sva_item.Bhat_cov_;
	sva_item.Bhat_.bottomLeftCorner(p_, sva_item.nSNP_) = sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).transpose();				
	/**** calculate Bhat_ ***************************************************************/
	
	CalculateUV_(sva_item);
} // SUGEN::LogisticScore_

void SUGEN::CoxphScore_ (SVA_UTILS& sva_item)
{
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		sva_item.Uhat_G_[nstudy].setZero();
	}	
	sva_item.Ahat_.setZero();
	sva_item.Bhat_.setZero();
	
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{			
		for (int i=0; i<fam_ind_[nstudy].size()-1; i++)
		{
			for (int ii=fam_ind_[nstudy](i)+1; ii<=fam_ind_[nstudy](i+1); ii++)
			{
				if (!Y_cox_[nstudy][ii].is_alive_)
				{
					sva_item.S0_ = 0.0;
					sva_item.S1_.setZero();
					sva_item.S1_G_.setZero();
					sva_item.S2_GW_.setZero();
					sva_item.S2_GG_.setZero();
					
					for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
					{
						sva_item.riskset_[nstudy1].setZero();
						for (int j=0; j<F_[nstudy1].size(); j++)
						{
							if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
							{
								if (Y_cox_[nstudy1][j].is_alive_)
								{
									sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_) 
										? sva_item.e_Wttheta_[nstudy1](j) : 0.;
								}
								else
								{
									sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_) 
										? sva_item.e_Wttheta_[nstudy1](j) : 0.;
								}
							}
						}
						
						if (flag_uw_)
						{
							sva_item.S0_ += sva_item.riskset_[nstudy1].sum();
							sva_item.S1_.noalias() += W_[nstudy1].transpose()*sva_item.riskset_[nstudy1];
							sva_item.S1_G_.noalias() += sva_item.G_[nstudy1].transpose()*sva_item.riskset_[nstudy1];
							sva_item.S2_GW_.noalias() += sva_item.G_[nstudy1].transpose()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
							sva_item.S2_GG_.noalias() += sva_item.G_[nstudy1].transpose()*sva_item.riskset_[nstudy1].asDiagonal()*sva_item.G_[nstudy1];					
						}
						else
						{
							sva_item.S0_ += sva_item.riskset_[nstudy1].dot(wt_[nstudy1]);
							sva_item.S1_.noalias() += W_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1];
							sva_item.S1_G_.noalias() += sva_item.G_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1];
							sva_item.S2_GW_.noalias() += sva_item.G_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1].asDiagonal()*W_[nstudy1];
							sva_item.S2_GG_.noalias() += sva_item.G_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1].asDiagonal()*sva_item.G_[nstudy1];
						}
					}

					sva_item.Uhat_G_[nstudy].row(ii) += sva_item.G_[nstudy].row(ii)-sva_item.S1_G_.transpose()/sva_item.S0_;

					if (flag_uw_)
					{
						sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
							sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_);
						sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).noalias() +=
							sva_item.S2_GW_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_);
							
						for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
						{
							for (int j=0; j<F_[nstudy1].size(); j++)
							{
								if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
								{
									if (Y_cox_[nstudy1][j].is_alive_)
									{
										if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_)
										{
											sva_item.Uhat_G_[nstudy1].row(j) -= (sva_item.G_[nstudy1].row(j)-sva_item.S1_G_.transpose()/sva_item.S0_)
												*sva_item.e_Wttheta_[nstudy1](j)/sva_item.S0_;
										}
									}
									else
									{
										if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_)
										{
											sva_item.Uhat_G_[nstudy1].row(j) -= (sva_item.G_[nstudy1].row(j)-sva_item.S1_G_.transpose()/sva_item.S0_)
												*sva_item.e_Wttheta_[nstudy1](j)/sva_item.S0_;
										}
									}
								}
							}								
						}
						
						if (!flag_robust_ && (fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
								sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_);
							sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() +=
								sva_item.S2_GW_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_);							
						}						
					}
					else
					{
						sva_item.Ahat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
							(sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](ii);
						sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).noalias() +=
							(sva_item.S2_GW_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](ii);
						
						for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
						{
							for (int j=0; j<F_[nstudy1].size(); j++)
							{
								if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
								{
									if (Y_cox_[nstudy1][j].is_alive_)
									{
										if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_)
										{
											sva_item.Uhat_G_[nstudy1].row(j) -= (sva_item.G_[nstudy1].row(j).transpose()-sva_item.S1_G_.transpose()/sva_item.S0_)
												*sva_item.e_Wttheta_[nstudy1](j)*wt_[nstudy](ii)/sva_item.S0_;
										}
									}
									else
									{
										if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_)
										{
											sva_item.Uhat_G_[nstudy1].row(j) -= (sva_item.G_[nstudy1].row(j).transpose()-sva_item.S1_G_.transpose()/sva_item.S0_)
												*sva_item.e_Wttheta_[nstudy1](j)*wt_[nstudy](ii)/sva_item.S0_;
										}
									}
								}
							}								
						}

						if (!flag_robust_ && (fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							if (flag_pairwise_inclusion_prob_)
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
									(sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_))*sva_item.wtd_[nstudy](ii);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() +=
									(sva_item.S2_GW_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))*sva_item.wtd_[nstudy](ii);
							}
							else
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
									(sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](ii)*wt_[nstudy](ii);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() +=
									(sva_item.S2_GW_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](ii)*wt_[nstudy](ii);
							}
						}							
					}
				}
			}
		}
	}
	
	if (flag_uw_) 
	{	
		/**** unweighted analysis ***************************************************/			
		if (flag_robust_) 
		{
			/**** robust variance estimation ****************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj);
							sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
						}
					}
				}
			}
			/**** robust variance estimation ****************************************/				
		} 
		else 
		{				
			/**** model-based variance estimation ***********************************/				
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) > 1)
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() +=
									(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
									(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj);
							}							
						}
					}
				}
			}
			/**** model-based variance estimation ***********************************/
		}
		/**** unweighted analysis ***************************************************/
	} 
	else if (flag_pairwise_inclusion_prob_)
	{
		/**** weighted analysis *****************************************************/
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_G_[nstudy];
			sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_[nstudy];
		}
		/**** weighted analysis *****************************************************/
	}
	else
	{
		/**** weighted analysis without pairwise inclusion probabilities ************/			
		if (flag_robust_) 
		{
			/**** robust variance estimation ****************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
						}
					}
				}
			}
			/**** robust variance estimation ****************************************/				
		} 
		else 
		{				
			/**** model-based variance estimation ***********************************/				
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) > 1)
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() +=
									(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
								sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).noalias() += 
									(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							}							
						}
					}
				}
			}
			/**** model-based variance estimation ***********************************/
		}
		/**** weighted analysis without pairwise inclusion probabilities ************/		
	}

	sva_item.Ahat_.bottomRightCorner(p_, p_) = sva_item.WtW_;
	sva_item.Ahat_.bottomLeftCorner(p_, sva_item.nSNP_) = sva_item.Ahat_.topRightCorner(sva_item.nSNP_, p_).transpose(); 	
	sva_item.Bhat_.bottomRightCorner(p_, p_) = sva_item.Bhat_cov_;
	sva_item.Bhat_.bottomLeftCorner(p_, sva_item.nSNP_) = sva_item.Bhat_.topRightCorner(sva_item.nSNP_, p_).transpose();
	
	// // RT
	// MatrixXd Ahat_inverse = sva_item.Ahat_.inverse();
	// cout << sva_item.Ahat_ << endl << endl;
	// cout << sva_item.Bhat_ << endl << endl;
	// cout << sva_item.theta_ << endl << endl;
	// cout << Ahat_inverse*sva_item.Bhat_*Ahat_inverse << endl << endl;
	// ofstream FO_test;
	// FO_test.open("test_data_score.tab");
	// for (int i=0; i<Y_cox_[0].size(); i++)
	// {
		// FO_test << F_[0][i] << "\t";
		// if (!Y_cox_[0][i].is_alive_)
		// {
			// FO_test << Y_cox_[0][i].survival_time_;
		// }
		// else
		// {
			// FO_test << Y_cox_[0][i].censoring_time_;
		// }
		// FO_test << "\t" << 1-Y_cox_[0][i].is_alive_;
		// for (int j=0; j<W_[0].cols(); j++)
		// {
			// FO_test << "\t" << W_[0](i,j);
		// }
		// for (int j=0; j<sva_item.G_[0].cols(); j++)
		// {
			// FO_test << "\t" << sva_item.G_[0](i,j);
		// }
		// FO_test << endl;
	// }
	// FO_test.close();
	// // RT end

	CalculateUV_(sva_item);	
} // SUGEN::CoxphScore_

void SUGEN::CoxphLogRank_ (SVA_UTILS& sva_item)
{
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{
		sva_item.Uhat_G_[nstudy].setZero();
	}	
	sva_item.Ahat_.setZero();
	sva_item.Bhat_.setZero();
	
	for (int nstudy=0; nstudy<N_study_; nstudy++) 
	{			
		for (int i=0; i<fam_ind_[nstudy].size()-1; i++)
		{
			for (int ii=fam_ind_[nstudy](i)+1; ii<=fam_ind_[nstudy](i+1); ii++)
			{
				if (!Y_cox_[nstudy][ii].is_alive_)
				{
					sva_item.S0_ = 0.0;
					sva_item.S1_G_.setZero();
					sva_item.S2_GG_.setZero();
					
					for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
					{
						sva_item.riskset_[nstudy1].setZero();
						for (int j=0; j<F_[nstudy1].size(); j++)
						{
							if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
							{
								if (Y_cox_[nstudy1][j].is_alive_)
								{
									sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_) ? 1. : 0.;
								}
								else
								{
									sva_item.riskset_[nstudy1](j) = (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_) ? 1. : 0.;
								}
							}
						}
						
						if (flag_uw_)
						{
							sva_item.S0_ += sva_item.riskset_[nstudy1].sum();
							sva_item.S1_G_.noalias() += sva_item.G_[nstudy1].transpose()*sva_item.riskset_[nstudy1];
							sva_item.S2_GG_.noalias() += sva_item.G_[nstudy1].transpose()*sva_item.riskset_[nstudy1].asDiagonal()*sva_item.G_[nstudy1];					
						}
						else
						{
							sva_item.S0_ += sva_item.riskset_[nstudy1].dot(wt_[nstudy1]);
							sva_item.S1_G_.noalias() += sva_item.G_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1];
							sva_item.S2_GG_.noalias() += sva_item.G_[nstudy1].transpose()*wt_[nstudy1].asDiagonal()*sva_item.riskset_[nstudy1].asDiagonal()*sva_item.G_[nstudy1];
						}
					}

					sva_item.Uhat_G_[nstudy].row(ii) += sva_item.G_[nstudy].row(ii)-sva_item.S1_G_.transpose()/sva_item.S0_;

					if (flag_uw_)
					{
						sva_item.Ahat_.noalias() += sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_);
							
						for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
						{
							for (int j=0; j<F_[nstudy1].size(); j++)
							{
								if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
								{
									if (Y_cox_[nstudy1][j].is_alive_)
									{
										if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_)
										{
											sva_item.Uhat_G_[nstudy1].row(j) -= (sva_item.G_[nstudy1].row(j)-sva_item.S1_G_.transpose()/sva_item.S0_)/sva_item.S0_;
										}
									}
									else
									{
										if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_)
										{
											sva_item.Uhat_G_[nstudy1].row(j) -= (sva_item.G_[nstudy1].row(j)-sva_item.S1_G_.transpose()/sva_item.S0_)/sva_item.S0_;
										}
									}
								}
							}								
						}
						
						if (!flag_robust_ && (fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							sva_item.Bhat_.noalias() += sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_);							
						}						
					}
					else
					{
						sva_item.Ahat_.noalias() += (sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](ii);
						
						for (int nstudy1=0; nstudy1<N_study_; nstudy1++)
						{
							for (int j=0; j<F_[nstudy1].size(); j++)
							{
								if (Y_cox_[nstudy1][j].left_truncation_time_ <= Y_cox_[nstudy][ii].survival_time_)
								{
									if (Y_cox_[nstudy1][j].is_alive_)
									{
										if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].censoring_time_)
										{
											sva_item.Uhat_G_[nstudy1].row(j) -= (sva_item.G_[nstudy1].row(j).transpose()-sva_item.S1_G_.transpose()/sva_item.S0_)
												*wt_[nstudy](ii)/sva_item.S0_;
										}
									}
									else
									{
										if (Y_cox_[nstudy][ii].survival_time_ <= Y_cox_[nstudy1][j].survival_time_)
										{
											sva_item.Uhat_G_[nstudy1].row(j) -= (sva_item.G_[nstudy1].row(j).transpose()-sva_item.S1_G_.transpose()/sva_item.S0_)
												*wt_[nstudy](ii)/sva_item.S0_;
										}
									}
								}
							}								
						}

						if (!flag_robust_ && (fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) == 1)
						{
							if (flag_pairwise_inclusion_prob_)
							{
								sva_item.Bhat_.noalias() += 
									(sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_))*sva_item.wtd_[nstudy](ii);
							}
							else
							{
								sva_item.Bhat_.noalias() += 
									(sva_item.S2_GG_/sva_item.S0_-sva_item.S1_G_*sva_item.S1_G_.transpose()/(sva_item.S0_*sva_item.S0_))*wt_[nstudy](ii)*wt_[nstudy](ii);
							}
						}							
					}
				}
			}
		}
	}
	
	if (flag_uw_) 
	{	
		/**** unweighted analysis ***************************************************/			
		if (flag_robust_) 
		{
			/**** robust variance estimation ****************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_.noalias() += (sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj);
						}
					}
				}
			}
			/**** robust variance estimation ****************************************/				
		} 
		else 
		{				
			/**** model-based variance estimation ***********************************/				
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) > 1)
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.noalias() += (sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj);
							}							
						}
					}
				}
			}
			/**** model-based variance estimation ***********************************/
		}
		/**** unweighted analysis ***************************************************/
	} 
	else if (flag_pairwise_inclusion_prob_)
	{
		/**** weighted analysis *****************************************************/
		for (int nstudy=0; nstudy<N_study_; nstudy++) 
		{
			sva_item.Bhat_.noalias() += sva_item.Uhat_G_[nstudy].transpose()*wtds_[nstudy]*sva_item.Uhat_G_[nstudy];
		}
		/**** weighted analysis *****************************************************/
	}
	else
	{
		/**** weighted analysis without pairwise inclusion probabilities ************/			
		if (flag_robust_) 
		{
			/**** robust variance estimation ****************************************/
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{			
						for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
						{
							sva_item.Bhat_.noalias() += 
								(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
						}
					}
				}
			}
			/**** robust variance estimation ****************************************/				
		} 
		else 
		{				
			/**** model-based variance estimation ***********************************/				
			for (int nstudy=0; nstudy<N_study_; nstudy++) 
			{	
				for (int i=0; i<fam_ind_[nstudy].size()-1; i++) 
				{
					for (int j=fam_ind_[nstudy](i)+1; j<=fam_ind_[nstudy](i+1); j++) 
					{
						if ((fam_ind_[nstudy](i+1) - fam_ind_[nstudy](i)) > 1)
						{
							for (int jj=fam_ind_[nstudy](i)+1; jj<=fam_ind_[nstudy](i+1); jj++) 
							{
								sva_item.Bhat_.topLeftCorner(sva_item.nSNP_, sva_item.nSNP_).noalias() +=
									(sva_item.Uhat_G_[nstudy].row(j).transpose())*sva_item.Uhat_G_[nstudy].row(jj)*wt_[nstudy](j)*wt_[nstudy](jj);
							}							
						}
					}
				}
			}
			/**** model-based variance estimation ***********************************/
		}
		/**** weighted analysis without pairwise inclusion probabilities ************/		
	}
	
	/**** calculate U and V *********************************************************/
	sva_item.U_.setZero();
	if (flag_uw_) 
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++)
		{
			for (int i=0; i<sva_item.nSNP_; i++) 
			{
				sva_item.U_(i) += sva_item.Uhat_G_[nstudy].col(i).sum();
			}
		}
	}
	else
	{
		for (int nstudy=0; nstudy<N_study_; nstudy++)
		{
			for (int i=0; i<sva_item.nSNP_; i++) 
			{
				sva_item.U_(i) += sva_item.Uhat_G_[nstudy].col(i).dot(wt_[nstudy]);
			}
		}
	}

	sva_item.V_ = sva_item.Bhat_;	
	sva_item.V_A_ = sva_item.Ahat_;
		
	sva_item.A_snplist_.clear();
	sva_item.B_snplist_.clear();
	for (int i=0; i<sva_item.nSNP_; i++) 
	{
		if ((1./sva_item.V_(i,i)) < ((1./sva_item.V_A_(i,i)-ERROR_MARGIN)) || (std::isnan(sva_item.V_(i,i))))
		{
			sva_item.A_snplist_.push_back(i);
		}
		else
		{
			sva_item.B_snplist_.push_back(i);
		}
	}
	
	if (sva_item.A_snplist_.size()+sva_item.B_snplist_.size() == sva_item.nSNP_)
	{
		for (int i=0; i<sva_item.A_snplist_.size(); i++)
		{
			for (int j=0; j<sva_item.A_snplist_.size(); j++)
			{
				sva_item.V_(sva_item.A_snplist_[i],sva_item.A_snplist_[j]) = sva_item.V_A_(sva_item.A_snplist_[i],sva_item.A_snplist_[j]);
			}
			for (int j=0; j<sva_item.B_snplist_.size(); j++)
			{
				sva_item.V_(sva_item.A_snplist_[i],sva_item.B_snplist_[j]) = sva_item.V_A_(sva_item.A_snplist_[i],sva_item.B_snplist_[j]);
				sva_item.V_(sva_item.B_snplist_[j],sva_item.A_snplist_[i]) = sva_item.V_A_(sva_item.B_snplist_[j],sva_item.A_snplist_[i]);
			}
		}
		for (int i=0; i<sva_item.nSNP_; i++) 
		{
			sva_item.beta_(i) = sva_item.U_(i)/sva_item.V_(i,i);
			sva_item.se_(i) = sqrt(1./sva_item.V_(i,i));
		}		
	}
	else
	{
		error(FO_log_, "Error: In SUGEN::CalculateUV_, sva_item.A_snplist_.size()+sva_item.B_snplist_.size() != sva_item.nSNP_!");
	}
	
	double scale;
	if (rescale_type_ == NAIVE) 
	{
		if (!flag_uw_)
		{	
			scale = 0.;
			for (int nstudy=0; nstudy<N_study_; nstudy++) {
				scale += wt_[nstudy].sum();
				scale = N_total_/scale;
				sva_item.U_ *= scale;
				sva_item.V_ *= scale*scale;
			}				
		}
	}
	else 
	{
		sva_item.Ahat_rescale_(0,0) = sva_item.Ahat_.sum();
		sva_item.Bhat_rescale_(0,0) = sva_item.Bhat_.sum();
		
		LDLT<MatrixXd> chol_A(sva_item.Ahat_rescale_);
		MatrixXd invA_cholB = chol_A.solve(sva_item.Bhat_rescale_);
		MatrixXd invA_B_invA = chol_A.solve(invA_cholB.transpose());
		double invV = invA_B_invA(0,0);
		
		double invAbb = sva_item.Ahat_rescale_(0,0);

		if (invV > 0. && invAbb > 0.)
		{
			scale = 1./(invV*invAbb);
			sva_item.U_ *= scale;
			sva_item.V_ *= scale*scale;
		}
		else
		{
			if (!flag_uw_)
			{	
				scale = 0.;
				for (int nstudy=0; nstudy<N_study_; nstudy++) {
					scale += wt_[nstudy].sum();
					scale = N_total_/scale;
					sva_item.U_ *= scale;
					sva_item.V_ *= scale*scale;
				}				
			}
		}
	}
	/**** calculate U and V *********************************************************/
} // SUGEN::CoxphLogRank_

void SUGEN::ScoreTests_GroupAnalysis_ (SVA_UTILS& sva_item)
{
	int group_nrow, group_ncol;
	int32_t pos;
	vector<string> chr_pos, SNP_ID;
	string SNP_IDs;
	ifstream FI;
	
	dim(group_nrow, group_ncol, FN_group_, false, FO_log_);
	
	VCF_reader_.open(FN_geno_.c_str(), VCF_header_);
	VCF_reader_.readVcfIndex();
	FI.open(FN_group_.c_str());
	for (int ngene=0; ngene<group_nrow; ngene++) 
	{
		FI >> sva_item.gene_ID_ >> SNP_IDs;
		SNP_ID.clear();
		if (!Split(SNP_IDs, ",", &SNP_ID)) 
		{
			error(FO_log_, "Error: Cannot parse SNP IDs for group "+sva_item.gene_ID_+" in "+FN_group_+"!\n");
		}
		
		sva_item.rawGene_.clear();
		sva_item.nSNP_ = 0;
		sva_item.SNP_chr_.clear();
		sva_item.SNP_pos_.clear();
		sva_item.SNP_ID_.clear();
		sva_item.SNP_ref_.clear();
		sva_item.SNP_alt_.clear();
		sva_item.SNP_n_.clear();
		sva_item.SNP_mac_.clear();
		sva_item.SNP_n0_count_.clear();
		sva_item.SNP_n1_count_.clear();
		sva_item.SNP_n2_count_.clear(); 
		sva_item.SNP_n_dose_.clear();
		sva_item.SNP_maf_.clear();
		if (method_ == logistic || method_ == right_censored)
		{
			sva_item.SNP_n_case_.clear();
			sva_item.SNP_maf_case_.clear();
		}
		
		for (int nsnp=0; nsnp<SNP_ID.size(); nsnp++)
		{
			chr_pos.clear();
			if (!Split(SNP_ID[nsnp], ":", &chr_pos))
			{
				error(FO_log_, "Error: Cannot parse SNP "+SNP_ID[nsnp]+" in gene "+sva_item.gene_ID_+" in "+FN_group_+"!\n");
			}
			pos = atoi(chr_pos[1].c_str());
			VCF_reader_.set1BasedReadSection(chr_pos[0].c_str(), pos, pos+1);
			
			if (VCF_reader_.readRecord(VCF_record_)) 
			{
				ScoreTests_GetSNP_(sva_item);
				if (sva_item.maf_ <= group_maf_ && (sva_item.n_+0.)/(N_total_+0.) >= group_callrate_ && !sva_item.flag_multiallelic_)
				{
					sva_item.rawGene_.push_back(sva_item.rawG_);
					sva_item.SNP_chr_.push_back(VCF_record_.getChromStr());
					sva_item.SNP_pos_.push_back(VCF_record_.get1BasedPosition());
					sva_item.SNP_ID_.push_back(VCF_record_.getIDStr());
					sva_item.SNP_ref_.push_back(VCF_record_.getRefStr());
					sva_item.SNP_alt_.push_back(VCF_record_.getAltStr());
					sva_item.SNP_n_.push_back(sva_item.n_);
					sva_item.SNP_mac_.push_back(sva_item.mac_);
					sva_item.SNP_n0_count_.push_back(sva_item.n0_count_);
					sva_item.SNP_n1_count_.push_back(sva_item.n1_count_);
					sva_item.SNP_n2_count_.push_back(sva_item.n2_count_);
					sva_item.SNP_n_dose_.push_back(sva_item.n_dose_);
					sva_item.SNP_maf_.push_back(sva_item.maf_);
					if (method_ == logistic || method_ == right_censored)
					{
						sva_item.SNP_maf_case_.push_back(sva_item.maf_case_);
						sva_item.SNP_n_case_.push_back(sva_item.n_case_);
					}
					sva_item.nSNP_ ++;
				}
			}
		}
		
		if (sva_item.nSNP_ > 0) {
			for (int nstudy=0; nstudy<N_study_; nstudy++)
			{
				sva_item.G_[nstudy].resize(F_[nstudy].size(), sva_item.nSNP_);
				for (int nsnp=0; nsnp<sva_item.nSNP_; nsnp++)
				{
					sva_item.G_[nstudy].col(nsnp) = sva_item.rawGene_[nsnp][nstudy];
				}
				sva_item.Uhat_G_[nstudy].resize(F_[nstudy].size(), sva_item.nSNP_);
			}			
			sva_item.Ahat_.resize(sva_item.nSNP_+p_, sva_item.nSNP_+p_);
			sva_item.Bhat_.resize(sva_item.nSNP_+p_, sva_item.nSNP_+p_);
			sva_item.U_.resize(sva_item.nSNP_);
			sva_item.V_.resize(sva_item.nSNP_, sva_item.nSNP_);
			sva_item.V_A_.resize(sva_item.nSNP_, sva_item.nSNP_);
			sva_item.beta_.resize(sva_item.nSNP_);
			sva_item.se_.resize(sva_item.nSNP_);
			if (method_ == right_censored)
			{
				sva_item.S1_G_.resize(sva_item.nSNP_);
				sva_item.S2_GG_.resize(sva_item.nSNP_, sva_item.nSNP_);
				if (p_ > 0)
				{
					sva_item.S2_GW_.resize(sva_item.nSNP_, p_);
				}
			}
			
			if (method_ == LS) 
			{
				LinearScore_(sva_item);
			} 
			else if (method_ == logistic) 
			{
				LogisticScore_(sva_item);
			}
			else if (method_ == right_censored)
			{
				if (p_ > 0)
				{
					CoxphScore_(sva_item);
				}
				else
				{
					CoxphLogRank_(sva_item);
				}
			}
			ScoreTests_Output_(sva_item, sva_no_miss);
		} 
		else 
		{
			FO_log_ << sva_item.gene_ID_ << ": No variants eligible for analysis!" << endl;
		}			
	}
	FI.close();
	VCF_reader_.close();
} // SUGEN::ScoreTests_GroupAnalysis_

void SUGEN::ScoreTests_PerSNPAnalysis_ (SVA_UTILS& sva_item) 
{
	ScoreTests_GetSNP_(sva_item);
	for (int nstudy=0; nstudy<N_study_; nstudy++)
	{
		sva_item.G_[nstudy].col(sva_item.nSNP_-1) = sva_item.rawG_[nstudy];
	}
	
	sva_item.SNP_chr_[0] = VCF_record_.getChromStr();
	sva_item.SNP_pos_[0] = VCF_record_.get1BasedPosition(); 	
	if (sva_item.flag_multiallelic_) 
	{
		FO_log_ << sva_item.SNP_chr_[0] << ":" << sva_item.SNP_pos_[0] << ": multiallelic SNP!" << endl;
	} 
	else 
	{
		sva_item.gene_ID_ = sva_item.SNP_chr_[0]+":"+to_string(sva_item.SNP_pos_[0]);
		sva_item.SNP_ID_[0] = VCF_record_.getIDStr();
		sva_item.SNP_ref_[0] = VCF_record_.getRefStr();
		sva_item.SNP_alt_[0] = VCF_record_.getAltStr();
		sva_item.SNP_n_[0] = sva_item.n_;
		sva_item.SNP_mac_[0] = sva_item.mac_;
		sva_item.SNP_n0_count_[0] = sva_item.n0_count_;
		sva_item.SNP_n1_count_[0] = sva_item.n1_count_;
		sva_item.SNP_n2_count_[0] = sva_item.n2_count_;
		sva_item.SNP_n_dose_[0] = sva_item.n_dose_;
		sva_item.SNP_maf_[0] = sva_item.maf_;
		if (method_ == logistic || method_ == right_censored)
		{
			sva_item.SNP_maf_case_[0] = sva_item.maf_case_;
			sva_item.SNP_n_case_[0] = sva_item.n_case_;
		}
		if (method_ == LS) 
		{
			LinearScore_(sva_item);
		} 
		else if (method_ == logistic) 
		{
			LogisticScore_(sva_item);
		}
		else if (method_ == right_censored)
		{
			if (p_ > 0)
			{
				CoxphScore_(sva_item);
			}
			else
			{
				CoxphLogRank_(sva_item);
			}
		}
		ScoreTests_Output_(sva_item, sva_no_miss);
	}		
} // SUGEN::ScoreTests_PerSNPAnalysis_

void SUGEN::ScoreTests_SNPAnalysis_ (SVA_UTILS& sva_item)
{
	sva_item.nSNP_ = 1;
	sva_item.SNP_chr_.resize(sva_item.nSNP_);
	sva_item.SNP_pos_.resize(sva_item.nSNP_);
	sva_item.SNP_ID_.resize(sva_item.nSNP_);
	sva_item.SNP_ref_.resize(sva_item.nSNP_);
	sva_item.SNP_alt_.resize(sva_item.nSNP_);
	sva_item.SNP_n_.resize(sva_item.nSNP_);
	sva_item.SNP_mac_.resize(sva_item.nSNP_);
	sva_item.SNP_n0_count_.resize(sva_item.nSNP_);
	sva_item.SNP_n1_count_.resize(sva_item.nSNP_);
	sva_item.SNP_n2_count_.resize(sva_item.nSNP_); 
	sva_item.SNP_n_dose_.resize(sva_item.nSNP_);
	sva_item.SNP_maf_.resize(sva_item.nSNP_);
	if (method_ == logistic || method_ == right_censored)
	{
		sva_item.SNP_maf_case_.resize(sva_item.nSNP_);
		sva_item.SNP_n_case_.resize(sva_item.nSNP_);
	}
	
	for (int nstudy=0; nstudy<N_study_; nstudy++)
	{
		sva_item.G_[nstudy].resize(F_[nstudy].size(), sva_item.nSNP_);
		sva_item.Uhat_G_[nstudy].resize(F_[nstudy].size(), sva_item.nSNP_);
	}	
	sva_item.Ahat_.resize(sva_item.nSNP_+p_, sva_item.nSNP_+p_);
	sva_item.Bhat_.resize(sva_item.nSNP_+p_, sva_item.nSNP_+p_);
	sva_item.U_.resize(sva_item.nSNP_);
	sva_item.V_.resize(sva_item.nSNP_, sva_item.nSNP_);
	sva_item.V_A_.resize(sva_item.nSNP_, sva_item.nSNP_);
	sva_item.beta_.resize(sva_item.nSNP_);
	sva_item.se_.resize(sva_item.nSNP_);
	if (method_ == right_censored)
	{
		sva_item.S1_G_.resize(sva_item.nSNP_);
		sva_item.S2_GG_.resize(sva_item.nSNP_, sva_item.nSNP_);
		if (p_ > 0)
		{
			sva_item.S2_GW_.resize(sva_item.nSNP_, p_);
		}
	}
	
	VCF_reader_.open(FN_geno_.c_str(), VCF_header_);
	VCF_reader_.readVcfIndex();
		
	if (extract_type_ == EXTRACT_TYPE_CHR) 
	{	
		VCF_reader_.setReadSection(extract_chr_.c_str());
		while (VCF_reader_.readRecord(VCF_record_)) 
		{			
			ScoreTests_PerSNPAnalysis_(sva_item);
		}		
	} 
	else if (extract_type_ == EXTRACT_TYPE_RANGE) 
	{	
		VCF_reader_.set1BasedReadSection(extract_chr_.c_str(), extract_start_, extract_end_+1);
		while (VCF_reader_.readRecord(VCF_record_)) 
		{			
			ScoreTests_PerSNPAnalysis_(sva_item);
		}	
	} 
	else if (extract_type_ == EXTRACT_TYPE_FILE) 
	{
		unsigned long long NSNP, NSNP_final;
		int32_t pos;
		string SNP_name;
		vector<string> chr_pos;
		ifstream FI;
				
		nrow(NSNP, FN_extract_, false, FO_log_);
		
		FI.open(FN_extract_);
		NSNP_final = 0;
		for (unsigned long long i=0; i<NSNP; i++) 
		{			
			FI >> SNP_name;
			chr_pos.clear();
			if (!Split(SNP_name, ":", &chr_pos)) 
			{
				stdError("Error: Cannot parse SNP "+SNP_name+" in "+FN_extract_+"!\n");
			}
			pos = atoi(chr_pos[1].c_str());
			VCF_reader_.set1BasedReadSection(chr_pos[0].c_str(), pos, pos+1);
			if (VCF_reader_.readRecord(VCF_record_)) 
			{
				ScoreTests_PerSNPAnalysis_(sva_item);
				NSNP_final ++;
			}
		}
		FI.close();
		if (NSNP_final == 0) 
		{
			FO_log_ << "Warning: No variants in "+FN_extract_+" are present in the VCF file!" << endl;
			FO_log_ << "Warning: Therefore, no analysis has been performed!" << endl;
		}			
	}
	else
	{
		while (VCF_reader_.readRecord(VCF_record_)) 
		{			
			ScoreTests_PerSNPAnalysis_(sva_item);
		}		
	}
	
	VCF_reader_.close();
} // SUGEN::ScoreTests_SNPAnalysis_

void SUGEN::ScoreTests_ () 
{
	
	SVA_UTILS sva_item;
	ScoreTests_GlobalInitialization_(sva_item);
	FO_log_ << "Start estimating the parameters under the null..." << endl;
	if (method_ == LS)
	{
		LinearScoreNull_(sva_item);
	}
	else if (method_ == logistic)
	{
		LogisticScoreNull_(sva_item);
	}
	else if (method_ == right_censored)
	{
		if (p_ > 0)
		{
			CoxphScoreNull_(sva_item);
		}
	}
	FO_log_ << "Done!" << endl;
	
	/**** open output file **************************************************************/
	if (flag_out_zip_) 
	{
		sva_item.FO_score_mass_ = ifopen(FN_score_mass_.c_str(), "w", InputFile::GZIP);
		sva_item.FO_score_snp_ = ifopen(FN_score_snp_.c_str(), "w", InputFile::GZIP);
	} 
	else 
	{
		sva_item.FO_score_mass_ = ifopen(FN_score_mass_.c_str(), "w", InputFile::UNCOMPRESSED);
		sva_item.FO_score_snp_ = ifopen(FN_score_snp_.c_str(), "w", InputFile::UNCOMPRESSED);
	}
	if (!(*sva_item.FO_score_mass_).isOpen()) 
	{
		error(FO_log_, "Error: Cannot open file "+FN_score_mass_+"!");
	}
	else if (!(*sva_item.FO_score_snp_).isOpen())
	{
		error(FO_log_, "Error: Cannot open file "+FN_score_snp_+"!");
	}
	else 
	{
		ScoreTests_Output_(sva_item, sva_header);		
	}		
	/**** open output file **************************************************************/
	
	/**** analysis **********************************************************************/	
	FO_log_ << "Start score tests..." << endl;
	if (flag_group_)
	{
		ScoreTests_GroupAnalysis_(sva_item);
	}
	else 
	{
		ScoreTests_SNPAnalysis_(sva_item);
	}
	FO_log_ << "Done!" << endl << endl;
	/**** analysis **********************************************************************/

	ifclose(sva_item.FO_score_mass_);
	ifclose(sva_item.FO_score_snp_);	
} // SUGEN::ScoreTests_

void SUGEN::Analysis_ ()
{
	if (test_type_ == WALD)
	{	
		SingleVariantAnalysis_();
	}
	else if (test_type_ == SCORE)
	{
		ScoreTests_();
	}
} // SUGEN::Analysis_
