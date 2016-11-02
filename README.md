# SUGEN

# GENERAL INFORMATION
SUGEN is a command-line software program written in C++ to implement the weighted and unweighted approaches described by [Lin et al. (2014)](http://www.cell.com/ajhg/abstract/S0002-9297(14)00471-6) for various types of association analysis under complex survey sampling. The current version of the program can accommodate continuous, binary, and right-censored time-to-event traits. It can perform single-variant and gene-based association analysis. In single-variant analysis, it can perform standard association analysis, conditional analysis, and gene-environment interaction analysis using Wald statistics. In standard association analysis, we include the SNP of interest and other covariates (if any) as predictors in the regression model. In conditional analysis, we include the SNP of interest, the SNPs that are conditioned on, and other covariates (if any) as predictors in the regression model. In gene-environment interaction analysis, we include the SNP of interest, the environment variables, the interactions between the SNP and environment variables, and other covariates (if any) as predictors in the regression model. In gene-based analysis, it generates the score statistics and covariance
matrix for variants in each gene. These summary statistics can be loaded into the software program [MASS](http://dlin.web.unc.edu/software/mass/) to perform all commonly used gene-based association tests.


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
[--group-maf maf_ub] [--group-callrate cr_lb]
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

    In this case, the double quotes in `formula` cannot be omitted. The time, event indicator, and covariates must appear in `pheno_file`. If there is no covariate, then we specify the formula as  

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
Analyzes dosage data in the VCF file. The dosages must be stored in the "DS" field of the VCF file. An example can be found [here](http://genome.sph.umich.edu/wiki/RAREMETALWORKER).

* `--probmatrix prob_file`  
Specifies the file that contains the file names of the pairwise inclusion probability matrices. The default name is *probmatrix.txt*. This option is optional in weighted analysis and ignored in unweighted analysis.

* `--subset subset_expression`  
Restricts analysis to a subset of subjects in `pheno_file`. For example, if one wants to restrict the analysis to subjects whose *var_a* equals *level_1*, where *var_a* is a column in `pheno_file`, and *level_1* is one of the values of *var_a*, then we can specify `subset_expression` as `"var_a=level_1"`.

## Analysis Options
* `--unweighted`  
Uses the unweighted approach.

* `--model model`  
Specifies the regression model. There are three options: *linear* (linear regression), *logistic* (logistic regression), and *coxph* (Cox proportional hazard regression). The default value is *linear*. In linear or logistic regression, the trait is continuous or binary (0/1), respectively. In Cox proportional hazard regression, the time is positive, and the event indicator is binary (0/1).

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
* `--out-prefix out_prefix`  
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

* `--group-maf maf_ub`  
Specifies the minor allele frequency (MAF) upper bound for gene-based association analysis. `maf_ub` is a real number between 0 and 1. Its default value is *0.05*. Variants with MAFs greater than *maf_ub* will not be included in the analysis.

* `--group-callrate cr_lb`  
Specifies the call rate lower bound for gene-based association analysis. `cr_lb` is a real number between 0 and 1. Its default value is *0*. Variants with call rates less than `cr_lb` will not be included in the analysis.
