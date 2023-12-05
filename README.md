# AI for Health Equity in Lung Cancer Screening

## Table of Contents

### Chapter 1. Sybil Evaluation

This Chapter serves as documentation of the steps performed and tools used to evaluate Sybil, the AI architecture developed by Mikhael and Wohlwend et al., and published [here](https://ascopubs.org/doi/full/10.1200/JCO.22.01345) in the Journal of Clinical Oncology.

Sybil is an AI architecture created based on a 3D convolutional neural network which accepts a Computed Tomography (CT) 3D image of the chest, and returns the probability of being diagnosed with lung cancer by year N (1 to 6).

Specifically, the validation results on CT chests from the National Lung Screening Trial were validated.

Sybil is available on Github [here](https://github.com/reginabarzilaygroup/Sybil). 

Units:

1. Setting up a Python container [↗](docs/doc_setup_python.md)

2. How to Download NLST CT Data from the Cancer Imaging Archive [↗](docs/doc_nbia_download.md)

3. How to run Sybil on the UIC Extreme Cluster [↗](docs/doc_run_sybil.md)

4. Using Sybil to evaluate every CT chest in the NLST data [↗](docs/doc_sybil_main_py.md)

5. Preparing NLST Clinical Data for Sybil Evaluation [↗](docs/nlst_actual.md)

6. Filtering NLST data, then generating ROC curves and confusion matrices based on Sybil predictions [↗](docs/doc_sybil_eval.md)
