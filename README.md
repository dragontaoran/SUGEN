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
