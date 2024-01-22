# AFISP

This repository contains code for the submission of the manuscript titled "A Data-Driven Framework for Identifying Patient Subgroups on Which an AI/Machine Learning Model May Underperform". The manuscript describes an Algorithmic Framework for Identifying Subgroups with potential Performance disparities (AFISP), which allows users to probe a trained machine learning or AI model for subgroups on which it underperforms.


# Installation

The code uses the R and python programming languages. To install an environment with code-compatible versions of these languages, first run

```
conda env create -f conda_environment.yml
```

To install the requisite python packages, then run

```
pip install -r pip_packages.txt
```

To install the requisite R packages, run

```
Rscript install_R_packages.R
```

# Repo Organization

The /src directory contains the source code for the AFISP method.

The /slicefinder directory contains a slightly modified version of the SliceFinder source code (https://github.com/yeounoh/slicefinder). SliceFinder is an algorithmic, search-based subgroup discovery method that serves as a state-of-the-art baseline in the experiments.

The /simulation directory contains code used to generate results on synthetic data for which ground truth problematic subgroups are known.

The /aam_experiments directory contains code used to generate the results and figures in the main text of the manuscript for the illustration of AFISP on the AAM model, a hospital in-patient triage model.

The /mimic_experiments directory contains code used to generate results and figures in the supplementary material of the manuscript to apply AFISP on the publicly available MIMIC dataset.
