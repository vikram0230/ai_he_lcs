#!/usr/bin/env venv/bin/python3

import pydicom
import sys

def print_dicom_info(dicom_file_path):
    """
    Read and print all details from a DICOM file
    """
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_file_path)
        
        # Print basic file information
        print("\n=== DICOM File Information ===")
        print(f"Storage Type: {ds.SOPClassUID.name if hasattr(ds, 'SOPClassUID') else 'Unknown'}")
        print(f"Patient Name: {ds.PatientName if hasattr(ds, 'PatientName') else 'Unknown'}")
        print(f"Patient ID: {ds.PatientID if hasattr(ds, 'PatientID') else 'Unknown'}")
        print(f"Modality: {ds.Modality if hasattr(ds, 'Modality') else 'Unknown'}")
        print(f"Study Date: {ds.StudyDate if hasattr(ds, 'StudyDate') else 'Unknown'}")
        
        # Print slice position information
        print("\n=== Slice Position Information ===")
        print(f"Image Position (Patient): {ds.ImagePositionPatient if hasattr(ds, 'ImagePositionPatient') else 'Unknown'}")
        print(f"Image Orientation (Patient): {ds.ImageOrientationPatient if hasattr(ds, 'ImageOrientationPatient') else 'Unknown'}")
        print(f"Slice Location: {ds.SliceLocation if hasattr(ds, 'SliceLocation') else 'Unknown'}")
        print(f"Slice Thickness: {ds.SliceThickness if hasattr(ds, 'SliceThickness') else 'Unknown'}")
        print(f"Spacing Between Slices: {ds.SpacingBetweenSlices if hasattr(ds, 'SpacingBetweenSlices') else 'Unknown'}")
        print(f"Pixel Spacing: {ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else 'Unknown'}")
        
        # Print all DICOM elements
        print("\n=== Complete DICOM Dataset ===")
        print(ds)

    except Exception as e:
        print(f"Error reading DICOM file: {str(e)}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python dicom_reader.py <path_to_dicom_file>")
        sys.exit(1)
    
    dicom_file_path = sys.argv[1]
    print_dicom_info(dicom_file_path)

if __name__ == "__main__":
    main() 