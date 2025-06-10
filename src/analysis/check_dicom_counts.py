import os

def check_dicom_counts():
    root_dir = '/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_cancer_imaging_archive'
    
    # Get all patient folders
    patient_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    
    for patient_id in patient_folders:
        patient_dir = os.path.join(root_dir, patient_id)
        scan_dates = [d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))]
        
        for scan_date in scan_dates:
            scan_path = os.path.join(patient_dir, scan_date)
            recon_folders = [f for f in os.listdir(scan_path) if os.path.isdir(os.path.join(scan_path, f))]
            
            if not recon_folders:
                continue
                
            # Count DICOMs in each reconstruction
            dicom_counts = []
            for recon_folder in recon_folders:
                recon_path = os.path.join(scan_path, recon_folder)
                dicom_files = [f for f in os.listdir(recon_path) if f.endswith('.dcm')]
                if len(dicom_files) >= 50:
                    dicom_counts.append(len(dicom_files))
            
            # Check if counts are equal
            if len(set(dicom_counts)) > 1:
                print(f"Unequal DICOM counts found:")
                print(f"Patient: {patient_id}")
                print(f"Scan date: {scan_date}")
                for recon, count in zip(recon_folders, dicom_counts):
                    print(f"  Reconstruction {recon}: {count} DICOMs")
                print("-" * 50)

if __name__ == "__main__":
    check_dicom_counts() 