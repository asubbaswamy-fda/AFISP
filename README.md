# AFISP: Software to identify patient characteristics and data acquisition parameters associated with high error rates in predictive AI/ML models

This repository contains code for the Regulatory Science Tool titled AFISP. The tool allows users to probe a trained machine learning (ML) or AI model for subgroups on which it underperforms.


# Installation

The code uses the R and python programming languages. To install an environment with code-compatible versions of these languages, first run

```
conda env create -f environment.yml
```

Activate the new conda environment by running

```
conda activate afisp
```

Then, to install the requisite R packages, run

```
Rscript install_R_packages.R
```

The code can now be run when the 'afisp' conda environment is activated.


# Repo Organization

The /afisp directory contains the source code for the tool.

A demo jupyter notebook showcasing the core functionality of the tool can be found in 'demo.ipynb'.
