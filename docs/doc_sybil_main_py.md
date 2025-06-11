# Documentation: Sybil `main.py`

*Last updated 12/06/2023 by Abdul Zakkar*

Find the Python script `main.py` [here](../scripts/main.py).

## Usage

`main.py [-h] [-p PORTION] [-m MINIMAGES] dicomdir`

This script is automatically called by the Sybil container image found [here](https://hub.docker.com/r/mitjclinic/sybil). In other words, when the Sybil container image is executed (e.g. `./sybil_latest.sif`), it looks for a script in its directory called `main.py` to run.

### Positional arguments:

| Argument | Description |
|---|---|
| dicomdir | A directory downloaded from the Cancer Imaging Archive via the NBIA data retriever tool. Please see relevant documentation. This script parses this directory according to its original structure, please do not modify its contents.|

### Optional Arguments:

| Shortened identifier | Identifier | Description | Default |
|---|---|---|---|
| -p | --portion | Identifies the fraction of the data to be evaluated. This option allows for concurrent instances of Sybil to evaluate different portions of the same directory in parallel. Examples: 1/5 is the first 20% of the data. 5/5 is the last 20% of the data. | keep_all |
| -m | --minimages | Identifies the minimum number of images required for the DICOM to be included for evaluation. If the value is below this minimum, it is considered to be a scout image. | 10 images |

### Example usage:

`./sybil_dir.sif path/to/nlst_dicom_dir -p 1/5`

This example shows `./sybil_dir.sif` used instead of `main.py` since `main.py` is called by `sybil_dir.sif` when running, and allows `main.py` to access the Sybil libraries.

### Sybil Container Modification

The Sybil container was modified to allow terminal arguments (explained above) to be utilized by this `main.py` script. The steps to perform this modification are as follows:

- Download the original Sybil container image from Docker. 
    - In this documentation, the Apptainer program is used to manage containers. You can read more about Apptainer [here](https://apptainer.org/docs/user/main/index.html).
- Use the following command:
    - `apptainer pull docker://mitjclinic/sybil`
- Next, create a definition file that will allow the modification this container image:
    - `nano sybil_dir.def`
    - It is named `sybil_dir.def` because it is a version of the Sybil container which allows the user to pass a directory of CT scans as an argument in the command line. 
- With `sybil_dir.def` open in Nano, paste the contents of this [document](../extras/sybil_dir.def) into the terminal. Save with Ctrl-X, then Y, then hit Enter.
- Now build the new container image using this definition file:
    - `apptainer build sybil_dir.sif sybil_dir.def`
- The new file `sybil_dir.sif` should now be generated, which is now used for script execution (see example usage above).

## Description

### Positional argument `dicomdir`: Directory structure of NLST data

When NLST data is downloaded from the Cancer Imaging Archive using the NBIA Data Retriever tool, the download directory structure is as follows:

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

This directory structure is important for the execution of this script.

### metadata.csv contents

The `metadata.csv` file is located in the outermost directory of the downloaded NLST data. It contains some details about each CT scan present in the download. This document is used to iterate through each CT scan in the download since it references their file locations.

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

### Identifying a portion of the CT scan DICOMs

- First, the entire metadata.csv file is read into a DataFrame.
- Then, a portion is selected depending on the `--portion` identified by the user in the command line (see above). For example, if `1/5` is entered, the first 20% of the metadata.csv file will be used.

### Exclusion criteria

- The DICOMs are evaluated one by one.
- Prior to passing the DICOM to the Sybil CNN for evaluation, certain DICOMs must be filtered out due to not meeting specific criteria.

Exclusion Criteria List:
- Directory does not exist: On rare occasions, a directory file location listed in metadata.csv does not exist among the downloaded files.
- Scout image exclusion: CT scout images are distinguished by setting a cutoff for the number of images in the DICOM file. By default, this cutoff is 10. This means that if a DICOM file has less than 10 images, it is assumed to be a scout image, and is excluded.
    - The value of the minimum image count can be modified in the terminal with the identifier `-m`. See Optional arguments above.
- Slice thickness is >5 mm: This limitation is in place through Sybil, so if a CT scan has a slice thickness too large, it will not be evaluated.
- Unable to convert pixel data: On rare occasions, the file is unable to be converted by the Pydicom library, and thus cannot be evaluated.

## Output

- For each eligible DICOM, Sybil will eventually return 6 values, and a row for the final table can be constructed:

| pid | study_year | unique_id | 
|---|---|---|
| 100002 | 0 | 0OPASEVZOOMB50f340212080.040.0null | 

Table columns continued...

pred_yr1 | pred_yr2 | pred_yr3 | pred_yr4 | pred_yr5 | pred_yr6 |
|---|---|---|---|---|---|
| 0.0038567 | 0.0064984 | 0.0134987 | 0.0173409 | 0.0214857 | 0.259987 |

- The output data will be stored in `sybil_predictions_start_end.csv`, where start and end are the indexes of the metadata.csv file which signify the range of the DICOMs evaluated in this document, based on the portion selected by the user. The output CSV file will be located in the same directory as chosen in the terminal.
- An additional output will be found called `progress_start_end.txt`, so progress can be monitored during the execution of this script.
