# SUGEN

# GENERAL INFORMATION
SUGEN is a command-line software program written in C++ to implement the weighted and unweighted approaches described by [Lin et al. (2014)](http://www.cell.com/ajhg/abstract/S0002-9297(14)00471-6) for various types of association analysis under complex survey sampling. The current version of the program can accommodate continuous, binary, and right-censored time-to-event traits. It can perform single-variant and gene-based association analysis. In single-variant analysis, it can perform standard association analysis, conditional analysis, and gene-environment interaction analysis using Wald statistics. In standard association analysis, we include the SNP of interest and other covariates (if any) as predictors in the regression model. In conditional analysis, we include the SNP of interest, the SNPs that are conditioned on, and other covariates (if any) as predictors in the regression model. In gene-environment interaction analysis, we include the SNP of interest, the environment variables, the interactions between the SNP and environment variables, and other covariates (if any) as predictors in the regression model. In gene-based analysis, it generates the score statistics and covariance
matrix for variants in each gene. These summary statistics can be loaded into the software program [MASS](http://dlin.web.unc.edu/software/mass/) to perform all commonly used gene-based association tests.

# DOWNLOAD AND INSTALLATION
## Download
The latest version of SUGEN can be downloaded from [github](https://github.com/dragontaoran/SUGEN/archive/master.zip) or [github page](https://github.com/dragontaoran/SUGEN).

## Installation
Step 1: Unzip the package.
```sh
  unzip SUGEN-master.zip
```
Step 2: Go to the SUGEN directory.
```sh
  cd ./SUGEN-master
```
Step 3: Install SUGEN. When successful, an executable called "SUGEN" will be generated in ./SUGEN.
```sh
  make
```

