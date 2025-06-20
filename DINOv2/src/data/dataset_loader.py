import torch
import pydicom
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
from torch.utils.data import Dataset
import torchio as tio
from typing import List, Tuple

class PatientDicomDataset(Dataset):
    def __init__(self, config, is_train=True, transform=None):
        """
        Args:
            config (dict): Configuration dictionary
            is_train (bool): Whether this is training dataset
            transform: Image transformations
        """
        self.config = config
        self.transform = transform
        self.check_slice_thickness = config['data']['check_slice_thickness']
        self.use_clinical_features = config['data'].get('use_clinical_features', False)
        
        # Resampling parameters
        self.target_slice_thickness = config['data'].get('target_slice_thickness', 2.5)  # mm
        self.target_num_slices = config['data'].get('target_num_slices', 200)
        self.voxel_spacing = tuple(config['data']['voxel_spacing'])  # Convert list to tuple
        
        # Set data directory based on whether this is training or test
        self.root_dir = config['data']['train_data_dir'] if is_train else config['data']['test_data_dir']
        self.patient_scan_count = config['data']['train_patient_scan_count'] if is_train else config['data']['test_patient_scan_count']
        
        # Load patient labels from CSV
        try:
            self.labels_df = pd.read_csv(config['data']['labels_file'])
            print(f"Loaded {len(self.labels_df)} records from CSV")
        except Exception as e:
            print(f"Error loading labels file: {e}")
            raise
        
        # Load clinical features if enabled
        if self.use_clinical_features:
            try:
                clinical_features_path = config['data']['clinical_features']['file_path']
                if not clinical_features_path:
                    raise ValueError("Clinical features file path not specified in config")
                self.clinical_features_df = pd.read_csv(clinical_features_path)
                print(f"Loaded {len(self.clinical_features_df)} clinical feature records")
                
                # Validate required features
                required_features = config['data']['clinical_features']['features']
                missing_features = [f for f in required_features if f not in self.clinical_features_df.columns]
                if missing_features:
                    raise ValueError(f"Missing required clinical features: {missing_features}")
                
                # Set feature dimension in config
                config['data']['clinical_features']['feature_dim'] = len(required_features)
            except Exception as e:
                print(f"Error loading clinical features: {e}")
                raise
        
        # Create a mapping of patient scans
        self.patient_scans = {}
        
        # Get list of patient folders
        try:
            patient_folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]
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
                elif row['days_to_diagnosis'] == -1:
                    negative_patient_scans.append((patient_id, row['study_yr']))
        
        print(f"Found {len(positive_patient_scans)} positive study and {len(negative_patient_scans)} negative study")
        
        # Process all scans first and collect valid ones
        valid_positive_scans = []
        valid_negative_scans = []
        
        # Process positive scans
        print(f"Processing {len(positive_patient_scans)} positive scans")
        for patient_id, selected_study_yr in positive_patient_scans:
            if self._process_patient_scan(patient_id, selected_study_yr):
                valid_positive_scans.append((patient_id, selected_study_yr))
        
        # Process negative scans
        print(f"Processing {len(negative_patient_scans)} negative scans")
        for patient_id, selected_study_yr in negative_patient_scans:
            if self._process_patient_scan(patient_id, selected_study_yr):
                valid_negative_scans.append((patient_id, selected_study_yr))
        
        print(f"Found {len(valid_positive_scans)} valid positive scans and {len(valid_negative_scans)} valid negative scans")
        
        # Calculate how many to take from each group after validation
        if is_train:
            # Use config ratio for training
            positive_ratio = self.config['data']['train_positive_ratio']
            num_positive = int(self.patient_scan_count * positive_ratio)
            num_negative = self.patient_scan_count - num_positive
        else:
            # Use config ratio for test
            positive_ratio = self.config['data']['test_positive_ratio']
            num_positive = int(self.patient_scan_count * positive_ratio)
            num_negative = self.patient_scan_count - num_positive
            num_positive = min(num_positive, len(valid_positive_scans))
            num_negative = min(num_negative, len(valid_negative_scans))

        # Randomly select from valid scans
        random.shuffle(valid_positive_scans)
        random.shuffle(valid_negative_scans)
        self.selected_positive = valid_positive_scans[:num_positive]
        self.selected_negative = valid_negative_scans[:num_negative]

        # Combine the selected patients
        selected_patient_scans = self.selected_positive + self.selected_negative
        print(f"Selected {len(selected_patient_scans)} patients ({len(self.selected_positive)} positive, {len(self.selected_negative)} negative)")

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
                
                # Find reconstruction with highest number of valid slices
                max_valid_slices = 0
                best_recon = None
                
                for recon_folder in recon_folders:
                    recon_path = os.path.join(scan_path, recon_folder)
                    dicom_files = sorted([f for f in os.listdir(recon_path) if f.endswith('.dcm')],
                                       key=lambda x: int(x.split('-')[1].split('.')[0]))
                    
                    # Check slice thickness if enabled
                    if self.check_slice_thickness:
                        valid_slices = []
                        for dicom_file in dicom_files:
                            dicom_path = os.path.join(recon_path, dicom_file)
                            try:
                                dicom = pydicom.dcmread(dicom_path)
                                slice_thickness = float(dicom.SliceThickness)
                                if 1.5 <= slice_thickness <= 3.0:
                                    valid_slices.append(dicom_file)
                            except (AttributeError, ValueError) as e:
                                print(f"Warning: Could not read slice thickness for {dicom_file}: {e}")
                                continue
                        
                        if len(valid_slices) >= 50 and len(valid_slices) > max_valid_slices:
                            max_valid_slices = len(valid_slices)
                            best_recon = recon_folder
                    else:
                        # If not checking thickness, just use total number of slices
                        num_slices = len(dicom_files)
                        if num_slices >= 50 and num_slices > max_valid_slices:
                            max_valid_slices = num_slices
                            best_recon = recon_folder
                
                # Store scan if we found a valid reconstruction
                if best_recon:
                    self.patient_scans[patient_id][study_yr] = {
                        'date': date_folder,
                        'reconstructions': [best_recon]  # Store only the best reconstruction
                    }
                    # print(f"Patient {patient_id}, date {date_folder}: Selected reconstruction {best_recon} with {max_valid_slices} valid slices")
                    return True
                else:
                    pass
                    # print(f"Skipping scan - Patient {patient_id}, date {date_folder}: No reconstruction with >= 50 valid slices found")
        
        return False
    
    def _resample_reconstruction(self, dicom_files: List[str], recon_path: str) -> torch.Tensor:
        """
        Resample a reconstruction to have consistent slice thickness and number of slices.
        
        Args:
            dicom_files: List of DICOM file paths
            recon_path: Path to reconstruction directory
            
        Returns:
            torch.Tensor: Resampled reconstruction with shape [num_slices, C, H, W]
        """
        # Create 3D volume from DICOM slices
        slices = []
        for dicom_file in dicom_files:
            dicom = pydicom.dcmread(os.path.join(recon_path, dicom_file))
            image = dicom.pixel_array
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            slices.append(image)
        
        # Stack slices into 3D volume
        volume = np.stack(slices, axis=0)  # [Z, H, W]
        
        # Create TorchIO subject
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(volume).unsqueeze(0))  # Add channel dimension
        )
        
        # Use voxel spacing from config
        target_spacing = self.voxel_spacing  # (0.703125, 0.703125, 2.5)
        
        # Create resampling transform
        resample = tio.transforms.Resample(
            target=target_spacing,
            image_interpolation='linear'
        )
        
        # Apply resampling
        resampled = resample(subject)
        resampled_volume = resampled.image.data.squeeze(0)  # Remove channel dimension
        
        # If we have more slices than target, take evenly spaced slices
        if resampled_volume.shape[0] > self.target_num_slices:
            indices = np.linspace(0, resampled_volume.shape[0]-1, self.target_num_slices, dtype=int)
            resampled_volume = resampled_volume[indices]
        # If we have fewer slices, pad with zeros
        elif resampled_volume.shape[0] < self.target_num_slices:
            padding = torch.zeros(
                (self.target_num_slices - resampled_volume.shape[0], 
                 resampled_volume.shape[1], 
                 resampled_volume.shape[2]),
                dtype=resampled_volume.dtype
            )
            resampled_volume = torch.cat([resampled_volume, padding], dim=0)
        
        # Convert to RGB by repeating the channel
        resampled_volume = resampled_volume.unsqueeze(1).repeat(1, 3, 1, 1)  # [Z, 3, H, W]
        
        return resampled_volume
    
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
                               key=lambda x: int(x.split('-')[1].split('.')[0]))
            
            # Resample the reconstruction
            reconstruction_tensor = self._resample_reconstruction(dicom_files, recon_path)
            print(f"Single reconstruction tensor shape: {reconstruction_tensor.shape}")  # [num_slices, C, H, W]
            all_reconstructions.append(reconstruction_tensor)
        
        # Check if we have any valid reconstructions after filtering
        if not all_reconstructions:
            print(f"No valid reconstructions found for patient {patient_id}, study year {study_yr}")
            
        patient_tensor = torch.stack(all_reconstructions)
        print(f"Final patient tensor shape: {patient_tensor.shape}")  # [num_reconstructions, num_slices, C, H, W]
        
        # Create normalized slice positions based on target number of slices
        slice_positions = torch.linspace(0, 1, self.target_num_slices, dtype=torch.float32)
        
        # Get label for this patient and study year
        patient_label = self.labels_df[
            (self.labels_df['pid'] == int(patient_id)) & 
            (self.labels_df['study_yr'] == int(study_yr))
        ].iloc[0, 6]  # Get year 1 label (index 6)
        
        patient_label = torch.tensor(patient_label, dtype=torch.float32).unsqueeze(-1)  # Add extra dimension
        print(f"Got labels for Patient ID: {patient_id}, Study Year: {study_yr} => Patient Label: {patient_label}")
        
        # Get clinical features if enabled
        if self.use_clinical_features:
            clinical_features = self.clinical_features_df[
                (self.clinical_features_df['pid'] == int(patient_id)) & 
                (self.clinical_features_df['study_yr'] == int(study_yr))
            ]
            
            if clinical_features.empty:
                raise ValueError(f"No clinical features found for patient {patient_id}, study year {study_yr}")
            
            # Select only the required features
            required_features = self.config['data']['clinical_features']['features']
            clinical_features = clinical_features[required_features].values[0]
            clinical_features = torch.tensor(clinical_features, dtype=torch.float32)
            
            return patient_tensor, slice_positions, clinical_features, patient_label
        else:
            return patient_tensor, slice_positions, patient_label