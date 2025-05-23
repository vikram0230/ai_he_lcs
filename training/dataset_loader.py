import torch
import pydicom
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
from torch.utils.data import Dataset

class PatientDicomDataset(Dataset):
    def __init__(self, root_dir, labels_file, patient_scan_count=500, transform=None):
        """
        Args:
            root_dir (str): Directory containing patient folders
            labels_file (str): Path to CSV file containing patient labels
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load patient labels from CSV
        try:
            self.labels_df = pd.read_csv(labels_file)
            print(f"Loaded {len(self.labels_df)} records from CSV")
        except Exception as e:
            print(f"Error loading labels file: {e}")
            raise
        
        # Create a mapping of patient scans
        self.patient_scans = {}
        
        # Get list of patient folders
        try:
            patient_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
            patient_folders.sort()
        except Exception as e:
            print(f"Error reading patient directories: {e}")
            raise
        
        # Split patients into positive and negative cases
        positive_patient_scans = []
        negative_patient_scans = []
        
        for patient_id in patient_folders:
            patient_records = self.labels_df[self.labels_df['pid'] == int(patient_id)]
            for index, row in patient_records.iterrows():
                if (row['canc_yr1'] == 1):
                    positive_patient_scans.append((patient_id, row['study_yr']))
                else:
                    negative_patient_scans.append((patient_id, row['study_yr']))
        
        print(f"Found {len(positive_patient_scans)} positive study and {len(negative_patient_scans)} negative study")
        
        # Process all scans first and collect valid ones
        valid_positive_scans = []
        valid_negative_scans = []
        
        # Process positive scans
        for patient_id, selected_study_yr in positive_patient_scans:
            if self._process_patient_scan(patient_id, selected_study_yr):
                valid_positive_scans.append((patient_id, selected_study_yr))
        
        # Process negative scans
        for patient_id, selected_study_yr in negative_patient_scans:
            if self._process_patient_scan(patient_id, selected_study_yr):
                valid_negative_scans.append((patient_id, selected_study_yr))
        
        print(f"Found {len(valid_positive_scans)} valid positive scans and {len(valid_negative_scans)} valid negative scans")
        
        # Calculate how many to take from each group after validation
        half_count = min(patient_scan_count // 2, len(valid_positive_scans), len(valid_negative_scans))
        
        # Randomly select from valid scans
        random.shuffle(valid_positive_scans)
        random.shuffle(valid_negative_scans)
        
        selected_positive = valid_positive_scans[:half_count]
        selected_negative = valid_negative_scans[:half_count]
        
        # Combine the selected patients
        selected_patient_scans = selected_positive + selected_negative
        print(f"Selected {len(selected_patient_scans)} patients ({len(selected_positive)} positive, {len(selected_negative)} negative)")
        
        # Create flat list of (patient_id, study_yr) pairs
        self.scan_list = [(pid, yr) for pid, yr in selected_patient_scans]
        
        print(f"Final dataset size: {len(self.scan_list)} scans")
        print('Scan list: ', self.scan_list)
    
    def _process_patient_scan(self, patient_id, selected_study_yr):
        """Helper method to process a single patient scan and return True if valid"""
        patient_dir = os.path.join(self.root_dir, patient_id)
        
        # Get all scan dates for this patient
        scan_dates = [d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))]
        
        # Initialize patient in mapping
        if patient_id not in self.patient_scans:
            self.patient_scans[patient_id] = {}
        
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
        patient_records = self.labels_df[self.labels_df['pid'] == int(patient_id)]
        if patient_records.empty:
            print(f"Warning: No records found in CSV for patient {patient_id}")
            return False
        
        # Store reconstructions for each study year
        for date_folder, study_yr in study_yr_map.items():
            if study_yr in patient_records['study_yr'].values and study_yr == selected_study_yr:
                scan_path = os.path.join(patient_dir, date_folder)
                recon_folders = [f for f in os.listdir(scan_path) if os.path.isdir(os.path.join(scan_path, f))]
                
                if not recon_folders:
                    print(f"Warning: No reconstruction folders found for patient {patient_id}, date {date_folder}")
                    continue
                
                # Find reconstruction with highest number of slices
                max_slices = 0
                best_recon = None
                
                for recon_folder in recon_folders:
                    recon_path = os.path.join(scan_path, recon_folder)
                    dicom_files = [f for f in os.listdir(recon_path) if f.endswith('.dcm')]
                    num_slices = len(dicom_files)
                    
                    if num_slices >= 50 and num_slices > max_slices:
                        max_slices = num_slices
                        best_recon = recon_folder
                
                # Store scan if we found a valid reconstruction
                if best_recon:
                    self.patient_scans[patient_id][study_yr] = {
                        'date': date_folder,
                        'reconstructions': [best_recon]  # Store only the best reconstruction
                    }
                    print(f"Patient {patient_id}, date {date_folder}: Selected reconstruction {best_recon} with {max_slices} slices")
                    return True
                else:
                    print(f"Skipping scan - Patient {patient_id}, date {date_folder}: No reconstruction with >= 50 slices found")
        
        return False
    
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
                print(f"Skipping reconstruction {recon_folder} with {len(dicom_files)} slices (< 50)")
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
        
        # Get label for this patient and study year
        patient_label = self.labels_df[
            (self.labels_df['pid'] == int(patient_id)) & 
            (self.labels_df['study_yr'] == int(study_yr))
        ].iloc[0, 6]  # Get year 1 label (index 6)
        
        patient_label = torch.tensor(patient_label, dtype=torch.float32).unsqueeze(-1)  # Add extra dimension
        print(f"Got labels for Patient ID: {patient_id}, Study Year: {study_yr} => Patient Label: {patient_label}")
        
        return patient_tensor, slice_positions, patient_label