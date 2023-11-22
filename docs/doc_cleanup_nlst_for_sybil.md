# Documentation: `cleanup_nlst_for_sybil.py` 

*Last updated 11/20/2023 by Abdul Zakkar*

Find the Python script `cleanup_nlst_for_sybil.py` [here](../scripts/cleanup_nlst_for_sybil.py).

`Usage: cleanup_nlst_for_sybil.py data_split.csv nlst_clinical_data_dir out_dir`

This Python executable generates the following tabular output which will be used to validate the Sybil neural network classification model.
Each row represents an individual CT scan.

| pid    | study_yr | days_to_diagnosis | gender | race |
|--------|----------|-------------------|--------|------|
| 100012 | 0        | 438               | 2      | 1    |

Table columns continued...

| canc_yr1 | canc_yr2 | canc_yr3 | canc_yr4 | canc_yr5 | canc_yr6 | data_split |
|----------|----------|----------|----------|----------|----------|------------|
| 0        | 1        | 1        | 1        | 1        | 1        | 2          |

- ***pid*** is a unique identifier for each patient.
- ***study_yr***- Each patient has one initial CT and up to 2 follow-ups.
	- 0 = first study year, the initial CT.
	- 1 = second study year, the first follow-up CT.
	- 2 = third study year, the second follow-up CT.
- ***days_to_diagnosis*** was calculated using this formula:
	- [diagnosis day] - [screening day (the day the CT was performed)] = days to diagnosis.
	- It represents the number of days remaining from the time of the CT scan until the time of diagnosis with lung cancer.
	- This value is -1 if the patient did not develop cancer during the study.
- ***gender*** is the patient's gender:
	- 1 = Male
	- 2 = Female
- ***race*** is the patient's race:
	- 1= White
	- 2 = Black or African American
	- 3 = Asian
	- 4 = American Indian or Alaskan Native
	- 5 = Native Hawaiian or Other Pacific Islander  
	- 6 = More than one race
	- 7 = Participant refused to answer
	- 95 = Missing data form - form is not expected to ever be completed
	- 96 = Missing - no response
	- 98 = Missing - form was submitted and the answer was left blank
	- 99 = Unknown/decline to answer
- ***canc_yrN*** signifies whether the patient had cancer with N years of the CT scan.
	- This is calculated based on *days_to_diagnosis*, for example:
		- *canc_yr3* is 1 if *days_to_diagnosis* is </= 365 * 3, otherwise it is 0. [^1]
	- 0 = Cancer is **NOT** present N years since CT scan.
	- 1 = Cancer is present N years since CT scan.
- ***data_split*** describes how this patient's data was used during the development of the Sybil neural network classification model.
	- 0 = data was used for **training** the model.
	- 1 = data was used for **developing** the model.
	- 2 = data was used for **testing/validating** the model.
	- 99 = data was not used in the Sybil study.

## This Python executable requires 3 inputs:
### 1.  The Sybil data split as a CSV, formatted as such:
| pid    | split |
|--------|-------|
| 122361 | test  |
| 113845 | train |
| 128046 | dev   |
- *pid* is a unique identifier for each patient.
- *split* shows how this patient's data was handled in the Sybil study.
	- *train* = used to train the classification model.
	- *dev* = used in the process of developing the model.
	- *test* = used to test/validate the model's performance.
- This data set is provided by the Sybil authors [here](https://drive.google.com/drive/folders/1nBp05VV9mf5CfEO6W5RY4ZpcpxmPDEeR).

### 2. The downloadable directory of NLST clinical data, found [here](https://wiki.cancerimagingarchive.net/display/NLST).
- After downloading, the directory and subdirectories must all be extracted.
- Below is an example of the directory structure, showing only the required files:
```
nlst_780
|
+-- nlst_780_prsn_idc_20210527.csv
|
+-- nlst_780_screen_idc_20210527.csv
```
### 3.  A directory to save the output CSV file.
- The output CSV file will be named `cleanup_nlst_for_sybil_out.csv`
 
[^1]: This formula assumes that every year has 365 days, neglecting leap years, which may result in very slight inaccuracies.

