import os
import pandas as pd
import torchvision.transforms as transforms
from dataset_loader import PatientDicomDataset

def extract_patient_data(root_dir, labels_file, patients_count, output_dir):
    """
    Extract positive and negative patient data and save to separate files.
    
    Args:
        root_dir (str): Directory containing patient folders
        labels_file (str): Path to CSV file containing patient labels
        patients_count (int): Number of patients to process
        output_dir (str): Directory to save the output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load patient labels from CSV
    labels_df = pd.read_csv(labels_file)
    print(f"Loaded {len(labels_df)} records from CSV")
    
    # Get list of patient folders
    patient_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    patient_folders.sort()
    
    # Split patients into positive and negative cases
    positive_patients = []
    negative_patients = []
    
    for patient_id in patient_folders:
        patient_records = labels_df[labels_df['pid'] == int(patient_id)]
        if not patient_records.empty:
            # Check if any record has days_to_diagnosis between 0 and 365 (1 year)
            if ((patient_records['days_to_diagnosis'] > 0) & (patient_records['days_to_diagnosis'] <= 365)).any():
                positive_patients.append(patient_id)
            else:
                negative_patients.append(patient_id)
    
    print(f"Found {len(positive_patients)} positive cases and {len(negative_patients)} negative cases")
    
    # Calculate how many patients to take from each group
    half_count = patients_count // 2
    positive_patients = positive_patients[:half_count]
    negative_patients = negative_patients[:half_count]
    
    # Create dataframes for positive and negative patients
    # For positive patients, only include records where days_to_diagnosis is between 0 and 365
    positive_df = labels_df[
        (labels_df['pid'].astype(str).isin(positive_patients)) & 
        (labels_df['days_to_diagnosis'] > 0) & 
        (labels_df['days_to_diagnosis'] <= 365)
    ]
    
    # For negative patients, include all records
    negative_df = labels_df[labels_df['pid'].astype(str).isin(negative_patients)]
    
    # Save the dataframes
    positive_output = os.path.join(output_dir, 'positive_patients.csv')
    negative_output = os.path.join(output_dir, 'negative_patients.csv')
    
    positive_df.to_csv(positive_output, index=False)
    negative_df.to_csv(negative_output, index=False)
    
    print(f"\nSaved positive patients data to: {positive_output}")
    print(f"Number of positive patient records: {len(positive_df)}")
    print(f"\nSaved negative patients data to: {negative_output}")
    print(f"Number of negative patient records: {len(negative_df)}")
    
    # Print additional statistics
    print("\nPositive Cases Statistics:")
    print(f"Number of unique positive patients: {len(positive_df['pid'].unique())}")
    print(f"Average days to diagnosis: {positive_df['days_to_diagnosis'].mean():.2f}")
    print(f"Min days to diagnosis: {positive_df['days_to_diagnosis'].min()}")
    print(f"Max days to diagnosis: {positive_df['days_to_diagnosis'].max()}")
    
    return positive_df, negative_df

def main():
    # Base directories
    base_dir = '/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2'
    labels_file = os.path.join(base_dir, 'nlst_actual.csv')
    
    # Process training data
    print("\nProcessing Training Data:")
    print("-" * 50)
    train_dir = os.path.join(base_dir, 'nlst_train_data')
    train_output_dir = os.path.join(base_dir, 'train_patient_data')
    train_positive_df, train_negative_df = extract_patient_data(
        root_dir=train_dir,
        labels_file=labels_file,
        patients_count=200,
        output_dir=train_output_dir
    )
    
    # Process test data
    print("\nProcessing Test Data:")
    print("-" * 50)
    test_dir = os.path.join(base_dir, 'nlst_test_data')
    test_output_dir = os.path.join(base_dir, 'test_patient_data')
    test_positive_df, test_negative_df = extract_patient_data(
        root_dir=test_dir,
        labels_file=labels_file,
        patients_count=50,
        output_dir=test_output_dir
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print("\nTraining Data:")
    print(f"Total positive patients: {len(train_positive_df['pid'].unique())}")
    print(f"Total negative patients: {len(train_negative_df['pid'].unique())}")
    print(f"Total training records: {len(train_positive_df) + len(train_negative_df)}")
    
    print("\nTest Data:")
    print(f"Total positive patients: {len(test_positive_df['pid'].unique())}")
    print(f"Total negative patients: {len(test_negative_df['pid'].unique())}")
    print(f"Total test records: {len(test_positive_df) + len(test_negative_df)}")

if __name__ == "__main__":
    main() 