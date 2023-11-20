# Documentation: Sybil `main.py`

*Last updated 11/15/2023 by Abdul Zakkar*

### 1. Directory structure of all NLST data

```
root
|
+---metadata.csv (contents described below)
|
+---NLST
    |
    +---pid (e.g. 100002)
    |   |
    :   +---study year (e.g. 01-02-1999-NLST-LSS-16408)
        |   |
        :   +---DICOM dir (e.g. 1.0000000OPLSEVZOOMT20s...)
            |   |
            :   +---dcm files (e.g. 1-001.dcm)
```
### 2. metadata.csv contents
| Column name | Example |
|---|---|
| Series UID | 1.2.840.113654.2.55.108942278744480339540309181071819421924 |
| Collection| NLST |
| 3rd Party Analysis | |
| Data Description URI | |
| Subject ID | 121338 |
| Study UID | 1.2.840.113654.2.55.336579435982133093583913888335494870457 |
| Study Description | NLST-LSS |
| Study Date | 01-02-1999 |
| Series Description | 0OPASEVZOOMB50f340212080.040.0null |
| Manufacturer | SIEMENS |
| Modality | CT |
| SOP Class Name | CT Image Storage |
| SOP Class UID | 1.2.840.10008.5.1.4.1.1.2 |
| Number of Images | 152 |
| File Size | 80.03 MB |
| File Location | ./NLST/121338/01-02-1999-NLST-LSS-70457/2.000000-0OPASEVZOOMB50f340212080.040.0null-84839 |
| Download Timestamp | 2023-11-12T06:39:47.783 |

## The Plan
- Iterate through each row in metadata.csv.
- Since we have the image count for each directory, we can use it to filter out the CT scout images. We can include a default image count filter of <=10 images by default, list this value as a constant in the beginning on the script as:
```
MINIMUM_IMAGE_COUNT = 10 
# Please adjust to desired value.
include_bool = row["Number of Images"].values[0] > MINIMUM_IMAGE_COUNT
    and other_filters
```
- Other filters?
- If it is determined that a CT scan be used for prediction, 3 values will be used for identification: 
	- `pid` = Subject ID column from metadata.csv.
	- `study year` = Using the Study Date column in metadata.csv, 0 if the year is 1999, 1 if 2000, and 2 if 2001, to align with the values used in the NLST clinical data tables.
	- `unique_id` = Series Description column from metadata.csv
- In code, `study_year` could be mapped in code as follows:
```
STUDY_YEAR_INDEX = ["1999", "2000", "2001"]
study_year = STUDY_YEAR_INDEX.index(
    row["Study Date"].values[0].split("-")[-1]
) # Should result in study_year = 0, 1, or 2.
```
- Now that we have a way to identify this individual CT scan and link it to the truth table generated from the NLST clinical data, the next step is to pass the CT into Sybil.
- Sybil would receive the location of `root` in the directory tree defined above. This will be stored in a variable.
```
# Example root directory
root_dir = "/projects/com_shared/azakka2/nlst/NLST_all_CTs/manifest-NLST_1_of_20"
```
- When at a specific row in metadata.csv, we can add the value in the File Location column to the root directory to find a specific CT with some string manipulation.
- This directory can be passed into a function that creates a list of dcm files.
- Sybil will eventually return 6 values, and a row for the final table can be constructed:

| pid | study_year | unique_id | 
|---|---|---|
| 100002 | 0 | 0OPASEVZOOMB50f340212080.040.0null | 
Table columns continued...
pred_yr1 | pred_yr2 | pred_yr3 | pred_yr4 | pred_yr5 | pred_yr6 |
|---|---|---|---|---|---|
| 0.0038567 | 0.0064984 | 0.0134987 | 0.0173409 | 0.0214857 | 0.259987 |

- This script will be executed across 20 groups of DICOMs so they can be executed in parallel on different CPU cores.
- Each DICOM group will get its own output `sybil_pred_[dir_name].csv`
