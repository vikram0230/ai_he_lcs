import os
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml

def process_patient_scan(root_dir, patient_id, selected_study_yr, labels_df):
    """Process a single patient scan and return slice thickness information"""
    patient_dir = os.path.join(root_dir, str(patient_id))
    
    # Get all scan dates for this patient
    scan_dates = [d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))]
    
    # Parse and order dates
    date_folders = {}
    for scan_date in scan_dates:
        date_str = scan_date[:10]
        date_parts = date_str.split('-')
        formatted_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
        date_folders[scan_date] = formatted_date
    
    # Sort dates and assign study years
    sorted_dates = sorted(date_folders.items(), key=lambda x: x[1])
    study_yr_map = {folder: idx for idx, (folder, _) in enumerate(sorted_dates)}
    
    # Get patient records
    patient_records = labels_df[labels_df['pid'] == int(patient_id)]
    if patient_records.empty:
        return None
    
    # Process each study year
    for date_folder, study_yr in study_yr_map.items():
        if study_yr in patient_records['study_yr'].values and study_yr == selected_study_yr:
            scan_path = os.path.join(patient_dir, date_folder)
            recon_folders = [f for f in os.listdir(scan_path) if os.path.isdir(os.path.join(scan_path, f))]
            
            if not recon_folders:
                continue
            
            # Find reconstruction with highest number of valid slices
            max_valid_slices = 0
            best_recon = None
            slice_thicknesses = []
            
            for recon_folder in recon_folders:
                recon_path = os.path.join(scan_path, recon_folder)
                dicom_files = sorted([f for f in os.listdir(recon_path) if f.endswith('.dcm')],
                                   key=lambda x: int(x.split('-')[1].split('.')[0]))
                
                valid_slices = []
                current_thicknesses = []
                
                for dicom_file in dicom_files:
                    dicom_path = os.path.join(recon_path, dicom_file)
                    try:
                        dicom = pydicom.dcmread(dicom_path)
                        slice_thickness = float(dicom.SliceThickness)
                        if 1.5 <= slice_thickness <= 3.0:
                            valid_slices.append(dicom_file)
                            current_thicknesses.append(slice_thickness)
                    except (AttributeError, ValueError) as e:
                        continue
                
                if len(valid_slices) >= 50 and len(valid_slices) > max_valid_slices:
                    max_valid_slices = len(valid_slices)
                    best_recon = recon_folder
                    slice_thicknesses = current_thicknesses
            
            if best_recon:
                return {
                    'patient_id': patient_id,
                    'study_yr': study_yr,
                    'slice_thicknesses': slice_thicknesses,
                    'num_valid_slices': max_valid_slices
                }
    
    return None

def main():
    # Load configuration
    with open('src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load inference results
    inference_df = pd.read_csv('/mmfs1/projects/dom_ameen_chi/common/SENTINL0/dinov2/outputs/inference_results.csv')
    
    # Load labels
    labels_df = pd.read_csv(config['data']['labels_file'])
    
    # Process all patients
    results = []
    for _, row in tqdm(inference_df.iterrows(), total=len(inference_df)):
        patient_id = int(row['patient_id'])
        study_yr = int(row['study_yr'])
        # print(f"Processing patient {patient_id} with study year {study_yr}", flush=True)
        
        result = process_patient_scan(config['data']['test_data_dir'], patient_id, study_yr, labels_df)
        if result:
            result['true_label'] = row['true_label']
            result['prediction'] = row['prediction']
            results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate classification accuracy
    results_df['correct'] = (results_df['true_label'] == 1) == (results_df['prediction'] > 0.5)
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Plot slice thickness distribution
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Slice thickness distribution
    plt.subplot(2, 2, 1)
    all_thicknesses = [thickness for thicknesses in results_df['slice_thicknesses'] for thickness in thicknesses]
    sns.histplot(all_thicknesses, bins=30)
    plt.title('Distribution of Slice Thicknesses')
    plt.xlabel('Slice Thickness (mm)')
    plt.ylabel('Count')
    
    # Plot 2: Classification accuracy by slice thickness
    plt.subplot(2, 2, 2)
    avg_thickness = results_df['slice_thicknesses'].apply(np.mean)
    sns.boxplot(x='correct', y=avg_thickness, data=results_df)
    plt.title('Classification Accuracy by Average Slice Thickness')
    plt.xlabel('Correctly Classified')
    plt.ylabel('Average Slice Thickness (mm)')
    
    # Plot 3: Number of valid slices vs classification accuracy
    plt.subplot(2, 2, 3)
    sns.boxplot(x='correct', y='num_valid_slices', data=results_df)
    plt.title('Number of Valid Slices vs Classification Accuracy')
    plt.xlabel('Correctly Classified')
    plt.ylabel('Number of Valid Slices')
    
    # Plot 4: Prediction confidence distribution
    plt.subplot(2, 2, 4)
    sns.histplot(data=results_df, x='prediction', hue='correct', bins=30)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('outputs/slice_thickness_analysis.png')
    plt.close()
    
    # Print statistics
    print("\nAnalysis Results:")
    print(f"Total scans analyzed: {len(results_df)}")
    print(f"Average slice thickness: {np.mean(all_thicknesses):.2f} mm")
    print(f"Classification accuracy: {results_df['correct'].mean():.2%}")
    print(f"Average number of valid slices: {results_df['num_valid_slices'].mean():.2f}")
    
    # Calculate additional statistics
    correct_predictions = results_df[results_df['correct']]
    incorrect_predictions = results_df[~results_df['correct']]
    
    print("\nDetailed Statistics:")
    print("\nCorrectly Classified Scans:")
    print(f"Average slice thickness: {correct_predictions['slice_thicknesses'].apply(np.mean).mean():.2f} mm")
    print(f"Average number of valid slices: {correct_predictions['num_valid_slices'].mean():.2f}")
    
    print("\nIncorrectly Classified Scans:")
    print(f"Average slice thickness: {incorrect_predictions['slice_thicknesses'].apply(np.mean).mean():.2f} mm")
    print(f"Average number of valid slices: {incorrect_predictions['num_valid_slices'].mean():.2f}")
    
    # Save detailed results
    results_df.to_csv('outputs/slice_thickness_analysis.csv', index=False)

if __name__ == "__main__":
    main() 