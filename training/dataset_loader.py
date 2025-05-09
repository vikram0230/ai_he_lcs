import torch
import pydicom
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PatientDicomDataset(Dataset):
    def __init__(self, root_dir, labels_file, patients_count=500, transform=None):
        """
        Args:
            root_dir (str): Directory containing patient folders
            labels_file (str): Path to CSV file containing patient labels
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load patient labels from CSV
        self.labels_df = pd.read_csv(labels_file)
        print(f"Loaded {len(self.labels_df)} records from CSV")
        
        # Create a mapping of patient scans
        self.patient_scans = {}
        
        # Get list of patient folders
        patient_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        patient_folders.sort()
        
        # Split patients into positive and negative cases
        positive_patients = []
        negative_patients = []
        
        for patient_id in patient_folders:
            patient_records = self.labels_df[self.labels_df['pid'] == int(patient_id)]
            if not patient_records.empty:
                # Check if any record has days_to_diagnosis > 0
                if (patient_records['days_to_diagnosis'] > 0).any():
                    positive_patients.append(patient_id)
                else:
                    negative_patients.append(patient_id)
        
        print(f"Found {len(positive_patients)} positive cases and {len(negative_patients)} negative cases")
        
        # Calculate how many patients to take from each group
        half_count = patients_count // 2
        positive_patients = positive_patients[:half_count]
        negative_patients = negative_patients[:half_count]
        
        # Combine the selected patients
        selected_patients = positive_patients + negative_patients
        print(f"Selected {len(selected_patients)} patients ({len(positive_patients)} positive, {len(negative_patients)} negative)")
        
        for patient_id in selected_patients:
            patient_dir = os.path.join(root_dir, patient_id)
            
            # Get all scan dates for this patient
            scan_dates = [d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))]
            print(f"Patient {patient_id}: Found {len(scan_dates)} scan dates")
            
            # Initialize patient in mapping
            self.patient_scans[patient_id] = {}
            
            # Parse and order dates
            date_folders = {}
            for scan_date in scan_dates:
                # Extract date from folder name (MM-DD-YYYY format)
                date_str = scan_date[:10]  # Get '01-02-1999' from '01-02-1999-NLST-LSS-55322'
                date_parts = date_str.split('-')
                # Convert to YYYY-MM-DD format for proper sorting
                formatted_date = f"{date_parts[2]}-{date_parts[0]}-{date_parts[1]}"
                date_folders[scan_date] = formatted_date
            
            # Sort dates and assign study years
            sorted_dates = sorted(date_folders.items(), key=lambda x: x[1])  # Sort by YYYY-MM-DD
            study_yr_map = {folder: idx for idx, (folder, _) in enumerate(sorted_dates)}
            
            # Get patient records
            patient_records = self.labels_df[self.labels_df['pid'] == int(patient_id)]
            if patient_records.empty:
                print(f"Warning: No records found in CSV for patient {patient_id}")
                continue
            
            # Store reconstructions for each study year
            for date_folder, study_yr in study_yr_map.items():
                if study_yr in patient_records['study_yr'].values:
                    scan_path = os.path.join(patient_dir, date_folder)
                    recon_folders = [f for f in os.listdir(scan_path) if os.path.isdir(os.path.join(scan_path, f))]
                    
                    if not recon_folders:
                        print(f"Warning: No reconstruction folders found for patient {patient_id}, date {date_folder}")
                        continue
                    
                    # Check DICOM counts for each reconstruction
                    dicom_counts = []
                    valid_recons = []
                    for recon_folder in recon_folders:
                        recon_path = os.path.join(scan_path, recon_folder)
                        dicom_files = [f for f in os.listdir(recon_path) if f.endswith('.dcm')]
                        if len(dicom_files) >= 50:
                            dicom_counts.append(len(dicom_files))
                            valid_recons.append(recon_folder)
                    
                    # Only store scan if all valid reconstructions have same number of DICOMs
                    if valid_recons and len(set(dicom_counts)) == 1:
                        self.patient_scans[patient_id][study_yr] = {
                            'date': date_folder,
                            'reconstructions': valid_recons
                        }
                        print(f"Patient {patient_id}, date {date_folder}: {len(valid_recons)} reconstructions with {dicom_counts[0]} slices each")
                    else:
                        print(f"Skipping scan - Patient {patient_id}, date {date_folder}: Unequal DICOM counts {dicom_counts}")
                else:
                    print(f"Warning: Could not determine study year for patient {patient_id}, date {date_folder}")
        
        # Create flat list of (patient_id, study_yr) pairs
        self.scan_list = [(pid, yr) for pid in self.patient_scans 
                         for yr in self.patient_scans[pid].keys()]
        print(f"Final dataset size: {len(self.scan_list)} scans")
        print(f"Positive cases: {len([pid for pid, _ in self.scan_list if pid in positive_patients])}")
        print(f"Negative cases: {len([pid for pid, _ in self.scan_list if pid in negative_patients])}")
    
    def __len__(self):
        return len(self.scan_list)
    
    def __getitem__(self, idx):
        patient_id, study_yr = self.scan_list[idx]
        scan_info = self.patient_scans[patient_id][study_yr]
        
        # Get path to this scan
        scan_path = os.path.join(self.root_dir, patient_id, scan_info['date'])

        print("Collecting reconstructions for Patient ID: ", patient_id, "Study Year: ", study_yr)
        
        # Process all reconstructions for this scan
        all_reconstructions = []
        for recon_folder in scan_info['reconstructions']:
            recon_path = os.path.join(scan_path, recon_folder)
            
            # Get all DICOM files for this reconstruction and sort by slice number
            dicom_files = sorted([f for f in os.listdir(recon_path) if f.endswith('.dcm')],
                               key=lambda x: int(x.split('-')[1].split('.')[0]))  # For files like '1-001.dcm'
            
            # Skip reconstructions with fewer than 50 slices
            if len(dicom_files) < 50:
                print(f"Skipping reconstruction {recon_folder} with only {len(dicom_files)} slices")
                continue
            
            reconstruction_images = []
            for dicom_file in dicom_files:
                dicom_path = os.path.join(recon_path, dicom_file)
                dicom = pydicom.dcmread(dicom_path)
                
                # Convert DICOM to PIL Image
                image = dicom.pixel_array
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                image = Image.fromarray(image).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                reconstruction_images.append(image)
            
            reconstruction_tensor = torch.stack(reconstruction_images)
            print(f"Single reconstruction tensor shape: {reconstruction_tensor.shape}")  # [num_slices, C, H, W]
            all_reconstructions.append(reconstruction_tensor)
        
        # Check if we have any valid reconstructions after filtering
        if not all_reconstructions:
            raise ValueError(f"No valid reconstructions found for patient {patient_id}, study year {study_yr} (all had <50 slices)")
            
        patient_tensor = torch.stack(all_reconstructions)
        print(f"Final patient tensor shape: {patient_tensor.shape}")  # [num_reconstructions, num_slices, C, H, W]
        
        # Create normalized slice positions based on slice numbers
        num_slices = len(dicom_files)
        slice_positions = torch.linspace(0, 1, num_slices, dtype=torch.float32)
        
        # Get labels for this patient and study year
        patient_labels = self.labels_df[
            (self.labels_df['pid'] == int(patient_id)) & 
            (self.labels_df['study_yr'] == int(study_yr))
        ].iloc[0, [6,10]].values
        
        patient_labels = torch.tensor(patient_labels, dtype=torch.float32)
        print(f"Got labels for Patient ID: {patient_id}, Study Year: {study_yr} => Patient Labels: {patient_labels}")
        
        return patient_tensor, slice_positions, patient_labels