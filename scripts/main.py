from sybil import Serie, Sybil
from pydicom import dcmread
import pandas as pd
from os import listdir, path
import time
import sys
from math import ceil

"""
This script is utilized by sybil_dir.sif, and must be in the same directory.
Please enter below the directory containing DICOMs.

The DICOMs must be in the same directory structure as if they were downloaded
via the NBIA retriever tool. See README for more details.
"""

#CONSTANTS
STUDY_YEAR_INDEX = ["1999", "2000", "2001"]
MINIMUM_IMAGE_COUNT = 10

def main():
    # Check args
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} nbia_download_dir portion(optional)" +
            "\nportion format: " +
            "\n\tn/n e.g. 1/5 will use the first 20% of the data."
        )

    # Read in arguments
    root_dir = sys.argv[1]
    portion = [int(i) for i in sys.argv[2].split("/")]
    
    # Simple directory check:
    root_dir_contents = listdir(root_dir)
    if ("metadata.csv" not in root_dir_contents or
        "NLST" not in root_dir_contents
    ):
        print("Invalid directory structure. Please see README.")

    # Load a trained model
    model = Sybil("sybil_ensemble")

    # Set up output dataframe
    output = pd.DataFrame(columns=[
        "pid", "study_year", "unique_id", 
        "yr1", "yr2", "yr3", "yr4", "yr5", "yr6"
    ])

    # Read in metadata CSV file
    metadata = pd.read_csv(root_dir + "/metadata.csv")
    row_count = metadata.shape[0]
    start_index = int(ceil(row_count / portion[1]) * (portion[0] - 1))
    end_index = int(min(
        start_index + ceil(row_count / portion[1]) - 1, row_count - 1
    ))
    metadata = metadata.loc[start_index:end_index,:]
    row_count = metadata.shape[0]

    output = [] 
    n_excluded = 0  

    # Logging
    print("Sybil prediction to be performed on contents of:" +
        f"\n{root_dir}" +
        f"\nFrom index {start_index} to index {end_index}."
    )

    for index, row in metadata.iterrows():
        # Get path from CSV.
        file_path = row["File Location"][1:]
        
        #Logging
        print(f"Evaluating {file_path}.")
        
        # Exclusion criteria: directory does not exist
        full_dir = root_dir + file_path
        if not path.exists(full_dir):
            print("Directory does not exist. Skipping.")
            n_excluded += 1
            continue

        # Exclusion criteria: scout image, made up of 1-2 images.
        n_slices = row["Number of Images"]
        if n_slices < MINIMUM_IMAGE_COUNT:
            print(f"This DICOM has too few slices (< {MINIMUM_IMAGE_COUNT})." +
                "Skipping.")
            n_excluded += 1
            continue

        # Exclusion criteria: slice thickness is greater than 5 mm.
        dcm = dcmread(
            full_dir + "/" + listdir(root_dir + file_path)[0]
        )
        slice_thickness = float(dcm.SliceThickness)
        if slice_thickness > 5.0:
            print("Slice thickness is too large (> 5 mm). Skipping.")
            n_excluded += 1
            continue

        # Exclusion criteria: Pydicom is unable to convert pixel data.
        try:
            dcm.convert_pixel_data()
        except NotImplementedError:
            print("Pydicom unable to convert pixel data. Skipping.")
            n_excluded += 1
            continue

        # Initialize empty row for the output.
        output_row = []     

        # Add pid
        pid = row["Subject ID"]
        output_row.append(pid)

        # Add study_year
        study_yr = STUDY_YEAR_INDEX.index(
            row["Study Date"].split("-")[-1]
        )
        output_row.append(study_yr)
        
        # Add unique_id
        unique_id = row["Series Description"]
        output_row.append(unique_id)

        # Evaluate probabilities with Sybil.
        scores = evaluate(root_dir + file_path, model)
        # Rounding for legibility
        scores = [round(i, 5) for i in scores]
        
        # Output current progress to text file
        write_progress(index + 1 - start_index, row_count, n_excluded, 
            root_dir + f"/progress_{start_index}_{end_index}.out",
            start_index, end_index
        )

        # Add row to final output.
        output.append(output_row + scores)

    # Convert output into a Pandas DataFrame.
    output_df = pd.DataFrame(output, columns=[
        "pid",
        "study_yr",
        "unique_id",
        "pred_yr1",
        "pred_yr2",
        "pred_yr3",
        "pred_yr4",
        "pred_yr5",
        "pred_yr6"
    ])
    # Save output CSV in output directory
    output_df.to_csv(root_dir + 
        f"/sybil_predictions_{start_index}_{end_index}.csv",
        index = False)      

def evaluate(file_path, model):
    start = time.perf_counter()
    serie = Serie([file_path + "/" + 
        i for i in listdir(file_path)])
    scores = model.predict([serie])
    end = time.perf_counter()
    print(f"Prediction on {file_path} in {end - start:0.4f} seconds.")
    return scores.scores[0]

def write_progress(current, total, excluded, file_name, start_i, end_i):
    f = open(file_name, 'w')
    f.write(f"metadata.csv {start_i} to {end_i}:\n")
    f.write(f"Current progress: {current} / {total}. " +
        f"{excluded} excluded.\n")
    f.close()

start = time.perf_counter()
main()
end = time.perf_counter()
print(f"{sys.argv[0]} Completed in {end - start:0.4f} seconds.")
