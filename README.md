# AFISP: Software to identify patient characteristics and data acquisition parameters associated with high error rates in predictive AI/ML models

This repository contains code for the Regulatory Science Tool titled AFISP. The tool allows users to probe a trained machine learning (ML) or AI model for subgroups on which it underperforms.

Full documentation can be viewed at [https://afisp.readthedocs.io/](https://afisp.readthedocs.io/).

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

# Tool Description

AFISP is a tool used to evaluate predictive AI models for patient subgroups with high error rates. The tool is applied after AI models have been trained and is applicable across many types of diagnostic and prognostic AI applications and modalities. Specifically, for a given AI model, AFISP requires three inputs: First, it requires per-sample loss function values (i.e., per-sample error values) for an independent test dataset (i.e., a dataset consisting of data samples that were not used to train or tune the AI model). Second, it requires meta-data such as measured patient characteristics (e.g., demographic information, comorbidities, etc.) or data acquisition parameters (e.g., X-ray radiation dose, scanner type) for each sample in the evaluation dataset. Third, it uses a performance threshold, which can be selected using prior knowledge or using a reference model. Given these inputs, AFISP identifies the worst-performing data subset: the largest subset of the evaluation dataset that produces performance below the threshold. The worst-performing data subset contains samples with the highest expected error rates (based on the meta-data features) and will have a different distribution of the meta-data features than the full evaluation dataset.

After selecting the worst-performing data subset, AFISP identifies interpretable subgroups (combinations of meta-data values such as “male patients with dementia”) present within the subset. This allows the user to understand specific characteristics of the data subset that are associated with the model’s lower performance. AFISP outputs a table of subgroups with low AI model performance, along with their sizes and performance values.


# Purpose

AFISP is a software tool intended for exploratory analysis of patient subgroups to help identify which subgroups the predictive AI model is inaccurate. After a predictive AI model has been trained, model developers can use AFISP to test their models for the existence of subgroups on whom their model performs poorly. This is important for understanding the range of populations and data acquisition parameters for which the model may be unreliable. When AFISP identifies a subgroup, this can prompt model developers to perform further analysis to better understand the subgroup and to update or improve their model (for example, by collecting additional data from the underperforming subgroup and fine-tuning). 

