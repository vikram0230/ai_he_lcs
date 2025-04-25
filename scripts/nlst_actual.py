import os
import sys
import time
import numpy as np
import pandas as pd
import argparse
"""
Prior to this script, Sybil should be used to generate a prediction.csv file.
Sybil's CSV output should include 8 columns: pid | study_yr | pred_yr1-6

This script will perform the following steps:

- A DataFrame will be initialized, with the following headers:

- pid | study_yr | days_to_diagnosis | gender | race | canc_yr1-6
(total 11 columns)

- There will be about 75,000 entries, just as there is in the screen metadata
file, which has one entry per CT.

- We will iterate through screen.csv.

- The CSV metadata sets will be queried to pull the following columns, which
will correspond to the headers above:

- pid(screen.csv) corresponds with `pid`
- study_yr(screen.csv) corresponds to `study_yr`
- `days_to_diagnosis` will use candx_days(prsn.csv) MINUS scr_days0-2(prsn.csv)
    - If candx_days is absent (no cancer), this value will be -1.
- gender(prsn.csv) corresponds with `gender`
- race(prsn.csv) corresponds with `race`

- The next step is to fill the canc_yr1-6 columns:
- Each column will represent if a patient had cancer by a given number of years
since the CT scan.
- This means that if `Days to diagnosis` is <=365, all 6 columns will have the
value 1.
- If `Days to diagnosis` is 1000, then diagnosis has occurred within 3 years-
so, the 6 columns would be:
0 | 0 | 1 | 1 | 1 | 1
- If `Days to diagnosis` is -1 (no cancer), then all columns should be 0.

- ROC curve generation will be managed by a different script, which can accept
possibly a JSON file to identify which data to filter when creating the ROC
curve.

"""
#Constants
DAYS_IN_YEAR = 365 # Slight error due to leap years

def main():
    print("NLST Actual (Truth values regarding cancer diagnosis by year N)")
    
    # ArgParse library is used to manage command line arguments.
    parser = argparse.ArgumentParser(
        epilog="Example: nlst_actual.py \
        path/to/data_split.csv path/to/nlst_clinical_data_dir -o output_dir"
    )
    parser.add_argument("datasplit", help="a CSV file provided by Sybil \
        authors which identifies the split of each CT by patient ID. \
        Available on Google drive here: \
        https://drive.google.com/drive/folders\
        /1nBp05VV9mf5CfEO6W5RY4ZpcpxmPDEeR as pid2split.csv.")
    parser.add_argument("clinical", help="a directory which can be downloaded \
        from the Cancer Imaging Archive which contains clinical metadata \
        regarding the NLST patients and the CT scans. Available here: \
        https://wiki.cancerimagingarchive.net/display/NLST")
    parser.add_argument("-o", "--outdir", help="A directory in which to \
        generate the output. \
        Default: script current working directory.",
        default=os.getcwd())
    args = parser.parse_args()
    print("Data split:", args.datasplit)
    print("Clinical data directory:", args.clinical)
    print("Output directory:", args.outdir)

    # Read in data split file provided by Sybil authors.
    data_split = pd.read_csv(args.datasplit)
    
    # Find the CT screen and prsn file.
    metadata_files = os.listdir(args.clinical)
    screen_file = ""
    prsn_file = ""
    for file_name in metadata_files:
        if "nlst" in file_name and "screen" in file_name:
            screen_file = args.clinical + file_name
        if "nlst" in file_name and "prsn" in file_name:
            prsn_file = args.clinical + file_name

    # Read in metadata CSVs
    screen = pd.read_csv(screen_file) 
    # We will iterate through these rows
    
    prsn = pd.read_csv(prsn_file)

    # This list will be a list of lists, each list being a row.
    output = [] 

    # Iterate through each CT
    for index, row in screen.iterrows():
        # Initialize current row
        current_row = []

        # Add pid
        current_row.append(row["pid"])

        # Add study year
        study_yr = row["study_yr"]
        current_row.append(study_yr)
        scr_day_str = "scr_days" + str(study_yr)
        screen_day = prsn.loc[prsn["pid"] == row["pid"]][scr_day_str]
        # Make sure there is a day for this CT scan, otherwise do not record.
        if screen_day.isnull().values[0]:
            print(f">>> " + str(row["pid"]) + ":\n" +
                "Missing " + scr_day_str + " value.\n" +
                "Unable to include data for study year " +
                str(study_yr) + ".")
            continue 

        # Add days to diagnosis (-1 for no cancer)
        if (prsn.loc[prsn["pid"] == row["pid"]]["candx_days"
            ].isnull().values[0]
        ):
            current_row.append(-1)
        else:
            screen_day = screen_day.astype(int).values[0]
            days_to_diagnosis = prsn.loc[prsn["pid"] == row["pid"]][
                "candx_days"].astype(int).values[0] - screen_day
            current_row.append(days_to_diagnosis)

        # Add age
        age = prsn.loc[prsn["pid"] == row["pid"]]["age"].values[0]
        current_row.append(age)

        # Add gender
        gender = prsn.loc[prsn["pid"] == row["pid"]]["gender"].values[0]
        current_row.append(gender)
    
        # Add race
        race = prsn.loc[prsn["pid"] == row["pid"]]["race"].values[0]
        current_row.append(race)

        # Add yearly cancer columns
        for year in range(1,7):
            if (current_row[2] <= year * DAYS_IN_YEAR and
                current_row[2] != -1
            ):
                current_row.append(1)
            else:
                current_row.append(0)

        # Add the sybil data split.
        # 0=train, 1=development, 2=test, 3=unseen
        split = data_split.loc[data_split["PID"] == row["pid"]][
            "SPLIT"]
        if split.size == 0: # Not found in the split
            current_row.append(99)
        else:
            split = split.values[0]
            split_int = -1 
            # if -1 found in output, 
            # then there was an erroneous split table entry.
            if split == "train": split_int = 0
            elif split == "dev": split_int = 1
            elif split == "test": split_int = 2
            current_row.append(split_int)

        # Add the whole complete row to the table.
        output.append(current_row)      

    # Convert output into a Pandas DataFrame.
    output_df = pd.DataFrame(output, columns=[
        "pid",
        "study_yr",
        "days_to_diagnosis",
        "age",
        "gender",
        "race",
        "canc_yr1",
        "canc_yr2",
        "canc_yr3",
        "canc_yr4",
        "canc_yr5",
        "canc_yr6",
        "sybil_data_split"
    ]) 
    # Save output CSV in output directory
    output_df.to_csv(args.outdir + "/nlst_actual.csv",
        index = False)

start = time.perf_counter()
main()
end = time.perf_counter()
print(f"{sys.argv[0]} completed in {end - start:.4f} seconds.")
