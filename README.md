**Table of Contents**

- [GENERAL INFORMATION](#general-information)
- [CITATION](#citation)
- [DOWNLOAD AND INSTALLATION](#download-and-installation)
  * [Download](#download)
  * [Installation](#installation)
- [SYNOPSIS](#synopsis)
- [OPTIONS](#options)
  * [Input Options](#input-options)
  * [Analysis Options](#analysis-options)
  * [Output Options](#output-options)
- [INPUT FILES](#input-files)
  * [Phenotype File](#phenotype-file)
  * [VCF File](#vcf-file)
  * [Pairwise Inclusion Probability Matrix](#pairwise-inclusion-probability-matrix)
  * [File that Contains the File Names of the Pairwise Inclusion Probability Matrices](#file-that-contains-the-file-names-of-the-pairwise-inclusion-probability-matrices)
  * [File that Contains the Variants for Conditional Analysis](#file-that-contains-the-variants-for-conditional-analysis)
  * [File that Contains Variants' Grouping Information in Gene-based Analysis](#file-that-contains-variants--grouping-information-in-gene-based-analysis)
  * [File that Contains the Subset of Variants to be Analyzed in Single-Variant Analysis](#file-that-contains-the-subset-of-variants-to-be-analyzed-in-single-variant-analysis)
- [OUTPUT FILES](#output-files)
  * [Wald Statistics](#wald-statistics)
    + [Single-Variant Analysis Results](#single-variant-analysis-results)
      - [Table 1: Column Description in Standard Association Analysis](#table-1--column-description-in-standard-association-analysis)
      - [Table 2: Column Description in Conditional Analysis](#table-2--column-description-in-conditional-analysis)
      - [Table 3: Column Description in Gene-Environment Interaction Analysis](#table-3--column-description-in-gene-environment-interaction-analysis)
  * [Score Statistics](#score-statistics)
    + [Single-Variant Analysis Results](#single-variant-analysis-results-1)
      - [Table 4: Column Description in Standard Association Analysis](#table-4--column-description-in-standard-association-analysis)
    + [Gene-Based Summary Statistics](#gene-based-summary-statistics)
- [VERSION HISTORY](#version-history)
- [CONTACT](#contact)

[![Build Status](https://travis-ci.org/dragontaoran/SUGEN.svg?branch=master)](https://travis-ci.org/dragontaoran/SUGEN)

# GENERAL INFORMATION
SUGEN is a command-line software program written in C++ to implement the weighted and unweighted approaches described by [Lin et al. (2014)](http://www.cell.com/ajhg/abstract/S0002-9297(14)00471-6) for various types of association analysis under complex survey sampling. The current version of the program can accommodate continuous, binary, and right-censored time-to-event traits. It can perform single-variant and gene-based association analysis. In single-variant analysis, it can perform standard association analysis, conditional analysis, and gene-environment interaction analysis using Wald statistics. In standard association analysis, we include the SNP of interest and other covariates (if any) as predictors in the regression model. In conditional analysis, we include the SNP of interest, the SNPs that are conditioned on, and other covariates (if any) as predictors in the regression model. In gene-environment interaction analysis, we include the SNP of interest, the environment variables, the interactions between the SNP and environment variables, and other covariates (if any) as predictors in the regression model. In gene-based analysis, it generates the score statistics and covariance
matrix for variants in each gene. These summary statistics can be loaded into the software program [MASS](http://dlin.web.unc.edu/software/mass/) to perform all commonly used gene-based association tests.

# CITATION
Lin, D. Y., Tao, R., Kalsbeek, W., Zeng, D., Gonzalez, F., Fernández-Rhodes, L., Graff, M., Koch, G., North, K. E., and Heiss, G. (2014). "[Genetic Association Analysis Under Complex Survey Sampling: The Hispanic Community Health Study/Study of Latinos](http://www.cell.com/ajhg/abstract/S0002-9297(14)00471-6)", American Journal of Human Genetics, 95(6): 675-688.

# DOWNLOAD AND INSTALLATION
## Download
The latest version of SUGEN can be downloaded from [github](https://github.com/dragontaoran/SUGEN/archive/master.zip) or [github page](https://github.com/dragontaoran/SUGEN).

## Installation
1. Unzip the package.

    ```
    unzip SUGEN-master.zip
    ```

2. Go to the SUGEN directory.

    ```
    cd ./SUGEN-master
    ```

3. Install SUGEN. When successful, an executable called "SUGEN" will be generated in ./SUGEN-master.
  
    ```
    make
    ```


# SYNOPSIS
```
SUGEN [--pheno pheno_file] [--formula formula] [--id-col iid] [--family-col fid] \
[--weight-col wt] [--vcf vcf_file.gz] [--dosage] [--probmatrix prob_file] \
[--subset subset_expression] [--unweighted] [--model model] [--robust-variance] \
[--left-truncation left_truncation_time] [--cond cond_file] [--ge envi_covs] [--score] \
[--score-rescale rescale_rule] [--group group_file] [--hetero-variance strata] [--out-prefix out_prefix] \
[--out-zip] [--extract-chr chr] [--extract-range range] [--extract-file extract_file] \
[--ge-output-detail][--group-maf maf_ub] [--group-callrate cr_lb]
```


# OPTIONS
## Input Options
* `--pheno pheno_file`  
Specifies the phenotype file. The default name is *pheno.txt*.

* `--formula formula`  
Specifies the regression formula. In linear or logistic regression, the format of `formula` is

    ```
    "trait=covariate_1+covariate_2+...+covariate_p"
    ```

    The trait and covariates must appear in `pheno_file`. If there is no covariate, then we specify the formula as  

    ```
    "trait="
    ```

    In Cox proportional hazards regression, the format of `formula` is  
    
    ```
    "(time, event)=covariate_1+covariate_2+...+covariate_p"
    ```

    The time, event indicator, and covariates must appear in `pheno_file`. If there is no covariate, then we specify the formula as  

    ```
    "(time, event)="
    ```

* `--id-col iid`  
Specifies the subject ID column in `pheno_file`. The default column name is *IID*.

* `--family-col fid`  
Specifies the family ID column in `pheno_file`. The default column name is *FID*. If study subjects are independent, then we specify the family ID column to be the same as the subject ID column.

* `--weight-col wt`  
Specifies the weight column in `pheno_file`. The default column name is *WT*. This option is ignored if `--unweighted` is specified.

* `--vcf vcf_file.gz`  
Specifies the [block compressed and indexed](http://www.htslib.org/doc/tabix.html) VCF file. The default name is *geno.vcf.gz*.

* `--dosage`  
Analyzes dosage data in the VCF file. The dosages must be stored in the *DS* field of the VCF file. This requirement is the same as [RAREMETALWORKER](http://genome.sph.umich.edu/wiki/RAREMETALWORKER).

* `--probmatrix prob_file`  
Specifies the file that contains the file names of the pairwise inclusion probability matrices. The default name is *probmatrix.txt*. This option is optional in weighted analysis and ignored in unweighted analysis.

* `--subset subset_expression`  
Restricts analysis to a subset of subjects in `pheno_file`. For example, if one wants to restrict the analysis to subjects whose *var_a* equals *level_1*, where *var_a* is a column in `pheno_file`, and *level_1* is one of the values of *var_a*, then we can specify `--subset_expression "var_a=level_1"`.

## Analysis Options
* `--unweighted`  
Uses the unweighted approach.

* `--model model`  
Specifies the regression model. There are three options: *linear* (linear regression), *logistic* (logistic regression), and *coxph* (Cox proportional hazards regression). The default value is *linear*. In linear or logistic regression, the trait is continuous or binary (0/1), respectively. In Cox proportional hazards regression, the event time is positive, and the event indicator is binary (0/1).

* `--robust-variance`  
If this option is specified, then the robust variance estimator will be used. Otherwise, the model-based variance estimator will be used.

* `--left-truncation left_truncation_time`  
Specifies the left truncation time (if any) in Cox proportional hazards regression. 

* `--cond cond_file`  
In single-variant analysis, performs conditional analysis conditioning on the variants included in `cond_file`. There is no default value for `cond_file`. The format of the variant IDs in `cond_file` is *chromosome:position*. This option is valid only when `--score` is not specified. In this situation, either `--cond cond_file` or `--ge envi_covs` can be specified, but not both. If neither is specified, then standard association analysis is performed.

* `--ge envi_covs`  
In single-variant analysis, performs gene-environment interaction analysis. `envi_covs` are the names of the environment variables. The format of `envi_covs` is *covariate_1,covariate_2,...,covariate_k*. That is, multiple environment variables are separately by commas. There is no default value for `envi_covs`. This option is valid only when `--score` is not specified. In this situation, either `--cond cond_file` or `--ge envi_covs` can be specified, but not both. If neither is specified, then standard association analysis is performed.

* `--score`  
Uses score statistics.

* `--score-rescale rescale_rule`  
Specifies the method to rescale the score statistics. There are two options: *naive* and *optimal*. The default value is *naive*. This option is valid only when `--score` is specified.

* `--group group_file`  
Performs gene-based association analysis. Gene memberships of variants are defined in `group_file`. There is no default value for `group_file`. This option is valid only when `--score` is specified.

* `--hetero-variance strata`  
Allows the residual variance in linear regression to be different in different levels of `strata`.

## Output Options
* `--out-prefix prefix`  
Specifies the prefix of the output files. The default prefix is *results*.

* `--out-zip`  
Zips the output files.

* `--extract-chr chr`  
Restricts single-variant analysis to variants in chromosome `chr`. This option is valid only when `--group group_file` is not specified.

* `--extract-range range`
Restricts single-variant analysis to variants in chromosome `chr` and positions in `range`. The format of `range` is *1000000-2000000*. 
This option is valid only when `--group group_file` is not specified and `--extract-chr chr` is specified.

* `--extract-file extract_file`  
Restricts single-variant analysis to variants in `extract_file`. The format of the variant IDs in `extract_file` is *chromosome:position*. This option is valid only when `--group group_file`, `--extract-chr chr`, and `--extract-range range` are not specified.

* `--ge-output-detail`
In gene-environment interaction analysis, output the covariances between the genetic variant, environment variables, and gene-environment interaction variables. Otherwise, only output the variances of the genetic variant, environment variables, and gene-environment variables.

* `--group-maf maf_ub`  
Specifies the minor allele frequency (MAF) upper bound for gene-based association analysis. `maf_ub` is a real number between 0 and 1. Its default value is *0.05*. Variants with MAFs greater than `maf_ub` will not be included in the analysis.

* `--group-callrate cr_lb`  
Specifies the call rate lower bound for gene-based association analysis. `cr_lb` is a real number between 0 and 1. Its default value is *0*. Variants with call rates less than `cr_lb` will not be included in the analysis.


# INPUT FILES
## Phenotype File
The phenotype file should be tab-delimited. Missing data are denoted by *NA*. The rows represent study subjects. The 1st row is the header line. This file should include the subject ID column, family ID column (unless the subjects are independent), weight column (unless the unweighted approach is used, i.e., when `--unweighted` is specified), trait column (with trait values being continuous or binary if `model=linear` or `model=logistic`, respectively), event time and indicator columns (if `model=coxph`), and covariates columns (unless there is no covariate in `formula`). Subjects with missing values in any of the columns specified by `--formula formula`,
`--id-col iid`, `--family-col fid`, or `--weight-col wt`are excluded from the analysis.

## VCF File
The VCF file contains the genotype data. The format specifications of a VCF file can be found [here](http://www.1000genomes.org/wiki/Analysis/Variant\%20Call\%20Format/vcf-variant-call-format-version-41). The VCF file should be compressed and indexed by [bgzip and tabix](http://www.htslib.org/doc/tabix.html), respectively, using the following commands:
```
bgzip vcf_file
tabix -p vcf -f vcf_file.gz  
```

## Pairwise Inclusion Probability Matrix
The files that contain the pairwise inclusion probability matrices should be tab-delimited. The 1st row is the header line containing the subject IDs. The remaining rows constitute a symmetric square matrix. That is to say, the number of rows equals the number of columns plus 1 (for the header line). The marginal inclusion probability of the ith subject is in the (i+1)th row and ith column. The pairwise inclusion probability of the ith and jth subjects is in the (i+1)th row and jth column, as well as in the (j+1)th row and ith column. All inclusion probabilities are strictly greater than 0 and less than or equal to 1. Missing values are not allowed. Note that there can be multiple pairwise inclusion probability matrices. Subjects in different pairwise inclusion probability matrices are assumed to be independent. Note that these pairwise inclusion probability matrices are optional in the weighted approach and not needed in the unweighted approach.

## File that Contains the File Names of the Pairwise Inclusion Probability Matrices
Each row is the file name of one pairwise inclusion probability matrix. Note that this file is optional in the weighted approach and not needed in the unweighted approach.

## File that Contains the Variants for Conditional Analysis
Each row is a variant ID, which should be in *chromosome:position* format. Note that this file is needed
only when we perform conditional analysis (i.e., when `--cond cond_file` is specified).

## File that Contains Variants' Grouping Information in Gene-based Analysis
Each row is a gene, which should be in the following format:
```
gene_1  variant_1,variant_2  
gene_2  variant_3,variant_4,variant_5
```
The gene and variant IDs are separated by a tab. The variant IDs in the same gene are separated by commas. Variant IDs should be in *chromosome:position* format. Note that this file is needed only when we perform gene-based analysis (i.e., when `--group group_file` is specified).

## File that Contains the Subset of Variants to be Analyzed in Single-Variant Analysis
Each row is a variant ID, which should be in *chromosome:position* format. Note that this file is needed only when `--extract-file extract_file` is specified.


# OUTPUT FILES
## Wald Statistics
### Single-Variant Analysis Results
The rows represent varaints. The first row is the header line. Missing values are denoted by *NA*. Tables 1-3 describe the columns of `prefix.wald.out` in standard association analysis, conditional analysis, and gene-environment interaction analysis, respectively.

#### Table 1: Column Description in Standard Association Analysis
| Column Name   | Description                                                        |
|---------------|--------------------------------------------------------------------|
| CHROM         | Chromosome.                                                        |
| POS           | Position.                                                          |
| VCF_ID        | Varaint ID in the VCF file.                                        |
| REF           | Reference allele.                                                  |
| ALT           | Alternative allele.                                                |
| ALT_AF        | Alternative allele frequency.                                      |
| ALT_AC        | Alternative allele count.                                          |
| N_INFORMATIVE | Number of subjects included in the analysis.                       |
| N_REF         | Number of subjects with two reference alleles.                     |
| N_HET         | Number of subjects with one reference and one alternative alleles. |
| N_ALT         | Number of subjects with two alternative alleles.                   |
| N_DOSE        | Number of subjects with genotype dosages.                          |
| ALT_AF_CASE   | Alternative allele frequency among cases included in logistic regression analysis.|
| N_CASE        | Number of cases included in logistic regression analytsis.         |
| ALT_AF_EVENT  | Alternative allele frequency among cases included in Cox proportional hazards regression analysis. |
| N_EVENT       | Number of cases included in Cox proportional hazards regression analysis.
| BETA          | Effect estimate.                                                   |
| SE            | Standard error estimate of BETA.                                   |
| PVALUE        | *p*-value.                                                         |

#### Table 2: Column Description in Conditional Analysis
| Column Name      | Description                                                        |
|------------------|--------------------------------------------------------------------|
| CHROM            | Chromosome.                                                        |
| POS              | Position.                                                          |
| VCF_ID           | Varaint ID in the VCF file.                                        |
| REF              | Reference allele.                                                  |
| ALT              | Alternative allele.                                                |
| ALT_AF           | Alternative allele frequency.                                      |
| ALT_AC           | Alternative allele count.                                          |
| N_INFORMATIVE    | Number of subjects included in the analysis.                       |
| N_REF            | Number of subjects with two reference alleles.                     |
| N_HET            | Number of subjects with one reference and one alternative alleles. |
| N_ALT            | Number of subjects with two alternative alleles.                   |
| N_DOSE           | Number of subjects with genotype dosages.                          |
| BETA             | Effect estimate.                                                   |
| SE               | Standard error estimate of BETA.                                   |
| PVALUE           | *p*-value.                                                         |
| BETA_*variant*   | Effect estimate of *variant* that is conditioned on.               |
| SE_*variant*     | Standard error estimate of BETA_*variant*.                         |
| PVALUE_*variant* | *p*-value of *variant* that is conditioned on.                     |

#### Table 3: Column Description in Gene-Environment Interaction Analysis
| Column Name           | Description                                                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|
| CHROM                 | Chromosome.                                                                                                       |
| POS                   | Position.                                                                                                         |
| VCF_ID                | Varaint ID in the VCF file.                                                                                       |
| REF                   | Reference allele.                                                                                                 |
| ALT                   | Alternative allele.                                                                                               |
| ALT_AF                | Alternative allele frequency.                                                                                     |
| ALT_AC                | Alternative allele count.                                                                                         |
| N_INFORMATIVE         | Number of subjects included in the analysis.                                                                      |
| N_REF                 | Number of subjects with two reference alleles.                                                                    |
| N_HET                 | Number of subjects with one reference and one alternative alleles.                                                |
| N_ALT                 | Number of subjects with two alternative alleles.                                                                  |
| N_DOSE                | Number of subjects with genotype dosages.                                                                         |
| PVALUE_G              | *p*-value of the variant.                                                                                         |
| PVALUE_INTER          | *p*-value of the interaction term(s) between the variant and environment variable(s).                             |
| PVALUE_BOTH           | *p*-value of both the variant and gene-environment interaction terms.                                             |
| BETA_G                | Effect estimate of the variant.                                                                                   |
| BETA_*envi*           | Effect estimate of environment variable *envi*.                                                                   |
| BETA_G:*envi*         | Effect estimate of the interaction term between the variant and environment variable *envi*, denoted by G:*envi*. |
| COV_G_G               | Variance estimate of BETA_G.                                                                                      |
| COV_*envi*_*envi*     | Variance estimate of BETA_*envi*.                                                                                 |
| COV_G:*envi*_G:*envi* | Variance estimate of BETA_G:*envi*.                                                                               |
| COV_G_*envi*          | Covariance estimate between BETA_G and BETA_*envi*.                                                               |
| COV_G_G:*envi*        | Covariance estimate between BETA_G and BETA_G:*envi*.                                                             |
| COV_*envi*_G:*envi*   | Covariance estimate between BETA_*envi* and BETA_G:*envi*.                                                        |

## Score Statistics
### Single-Variant Analysis Results
The rows represent SNPs. The first row is the header line. Missing values are denoted by *NA*. Tables 4 describes the columns of `prefix.score.snp.out` in standard association analysis.

#### Table 4: Column Description in Standard Association Analysis
| Column Name   | Description                                                                                                  |
|---------------|--------------------------------------------------------------------------------------------------------------|
| GENE_ID       | Gene ID. In single-variant analysis (i.e., `--group group_file` is not specified), GENE_ID equals CHROM:POS. |
| CHROM         | Chromosome.                                                                                                  |
| POS           | Position.                                                                                                    |
| VCF_ID        | Varaint ID in the VCF file.                                                                                  |
| REF           | Reference allele.                                                                                            |
| ALT           | Alternative allele.                                                                                          |
| ALT_AF        | Alternative allele frequency.                                                                                |
| ALT_AC        | Alternative allele count.                                                                                    |
| N_INFORMATIVE | Number of subjects included in the analysis.                                                                 |
| N_REF         | Number of subjects with two reference alleles.                                                               |
| N_HET         | Number of subjects with one reference and one alternative alleles.                                           |
| N_ALT         | Number of subjects with two alternative alleles.                                                             |
| N_DOSE        | Number of subjects with genotype dosages.                                                                    |
| U             | Score statistic.                                                                                             |
| V             | Variance estimate of U.                                                                                      |
| BETA          | Effect estimate.                                                                                             |
| SE            | Standard error estimate of BETA.                                                                             |
| PVALUE        | *p*-value.                                                                                                   |

### Gene-Based Summary Statistics
The gene-based summary statistics are stored in [MASS format](http://dlin.web.unc.edu/software/mass/).
They can be loaded into the software program [MASS](http://dlin.web.unc.edu/software/mass/) to perform all commonly used gene-based association tests. They can also be converted by the software program [PreMeta](http://dlin.web.unc.edu/software/premeta/) to files that are compatible with other commonly used rare-variant meta-analysis software programs, including [RAREMETAL](http://genome.sph.umich.edu/wiki/RAREMETAL_Documentation), [seqMeta](https://cran.r-project.org/web/packages/seqMeta/index.html), 
and [MetaSKAT](https://cran.r-project.org/web/packages/MetaSKAT/index.html).

# VERSION HISTORY
* 1.0 (released on May 29th, 2013)  
    First version released.

* 2.0 (released on Nov 12nd, 2013)  
    1. Added the capability to perform gene-environment interaction analysis.
    2. Deleted the tab delimiter at the end of each row in the output file.

* 3.0 (released on Dec 7th, 2013)  
    Added the capability to perform logistic regression for binary (0/1) traits.

* 4.0 (released on Feb 9th, 2014)  
    Added the capability to analyze data with multiple pairwise inclusion probability matrices.

* 4.1 (released on Mar 13rd, 2014)  
    Added the capability to deal with imputed genotype dosages.

* 5.0 (released on May 21st, 2014)  
    1. Modified the variance estimation formula. Included both the model-based and robust variance estimators.
    2. Changed the format of the phenotype file.

* 5.1 (released on Aug 14th, 2014)  
    Added the capability to perform conditional analysis.

* 5.2 (released on Sep 21st, 2014)  
    Modified the variance estimation formula. Used a new approach to trim the pairwise inclusion probabilities.

* 6.0 (released on Oct 1st, 2014)  
    Added the unweighted approach.

* 6.1 (released on Oct 6th, 2014)  
    Changed some option names. Changed some column names in output files.

* 6.2 (released on Nov 18th, 2014)  
    Changed the name of the software program from "SOLReg" to "SUGEN".

* 6.3 (released on Nov 13rd, 2015)  
    Improved the computational efficiency of unweighted analysis.

* 7.0 (released on March 30th, 2016)  
    Improved the user interface. Changed the genotype file format from plain text to VCF. Added the capability to perform gene-based association analysis.

* 7.1 (released on May 2nd, 2016)  
    Added the capability to handle dosage data. 

* 7.2 (released on May 5th, 2016)  
    Fixed a bug in reading the phenotype file when it contains redundant columns.

* 7.3 (released on May 30th, 2016)  
    1. Fixed a bug in gene-environment interaction analysis where the environment variable is the last
covariate in the model.
    2. Added the `--subset` option.
    3. Added the `--hetero-variance` option.
    4. Modified the model-based variance estimator so that it is stable for rare variants.

* 8 (released on September 29, 2016)  
    1.  Added the capability to perform Cox proportional hazards regression.
    2. Modified the model-based covariance matrix estimator in gene-based tests so that it is more accurate for rare variants.
    3. Fixed a bug in reading the phenotype file when the subject ID or family ID column is the last column of the phenotype file.

* 8.1 (released on November 2, 2016)  
    1. Added *p*-values in the gene-environment interaction analysis output file.
    2. Fixed a bug in the weighted approach.

* 8.2 (released on January 5, 2017)
	1. Added columns ALT_AF_CASE (ALT_AF_EVENT) and N_CASE (N_EVENT) to the single-variant analysis results file in logistic (Cox proportional hazards) regression.
	
* 8.3 (current version, released on January 18, 2017)
	1. Added the `--ge-output-detail` option.
    
# CONTACT
For questions, please contact [Ran Tao](https://sites.google.com/site/dragontaoran/home).
