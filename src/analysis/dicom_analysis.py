"""
DICOM Analysis Tool

This script analyzes DICOM files in a directory and generates various statistics and visualizations.
It can analyze:
- Distribution of slices per patient
- Distribution of slices per scan
- Slice thickness distribution
- Slice spacing distribution
- Body parts distribution
- Relationship between slice thickness and number of slices
- Inconsistent slice thickness detection within scans

Usage:
    python dicom_analysis.py [options]

Options:
    --dicom_path PATH    Path to DICOM files (default: /mmfs1/projects/dom_ameen_chi/common/SENTINL0/dinov2/nlst_train_data)
    --output FILE        Output file for plots (default: dicom_analysis_results.png)
    --patient-slices     Analyze slices per patient (default: True)
    --scan-slices        Analyze slices per scan (default: True)
    --thickness          Analyze slice thickness (default: True)
    --spacing            Analyze slice spacing (default: True)
    --body-parts         Analyze body parts (default: True)
    --thickness-vs-slices Analyze relationship between thickness and slice count (default: True)
    --check-inconsistency Check for inconsistent slice thicknesses within scans (default: True)
    --num-workers N      Number of worker processes (default: 4)

Example:
    # Run all analyses
    python dicom_analysis.py

    # Only analyze slice thickness and spacing
    python dicom_analysis.py --patient-slices --scan-slices --body-parts --thickness-vs-slices

    # Only analyze body parts
    python dicom_analysis.py --body-parts
"""

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

def read_dicom_file(file_path):
    """Read a single DICOM file and return relevant information."""
    try:
        dcm = pydicom.dcmread(file_path)
        info = {
            'has_thickness': hasattr(dcm, 'SliceThickness'),
            'has_spacing': hasattr(dcm, 'SpacingBetweenSlices'),
            'has_body_part': hasattr(dcm, 'BodyPartExamined'),
        }
        
        if info['has_thickness']:
            info['thickness'] = float(dcm.SliceThickness)
        if info['has_spacing']:
            info['spacing'] = float(dcm.SpacingBetweenSlices)
        if info['has_body_part']:
            info['body_part'] = dcm.BodyPartExamined
            
        return info
    except Exception as e:
        print(f"Error reading DICOM file {file_path}: {e}")
        return None

def process_directory(dir_info, analyze_patient_slices=True, analyze_scan_slices=True, 
                     analyze_thickness=True, analyze_spacing=True, analyze_body_parts=True,
                     analyze_thickness_vs_slices=True, check_inconsistency=True):
    """Process a single directory of DICOM files."""
    root, files = dir_info
    patient_id = os.path.basename(root)
    results = {}
    
    # Filter DICOM files
    dicom_files = [f for f in files if f.endswith('.dcm')]
    if not dicom_files or len(dicom_files) < 50:  # Skip directories with fewer than 50 slices
        return results
    
    # Read all DICOM files in parallel
    file_paths = [os.path.join(root, f) for f in dicom_files]
    with ProcessPoolExecutor(max_workers=min(len(file_paths), 8)) as executor:
        dicom_infos = list(executor.map(read_dicom_file, file_paths))
    dicom_infos = [info for info in dicom_infos if info is not None]
    
    if not dicom_infos:
        return results
    
    # Process results
    if analyze_scan_slices:
        results['scan_slices'] = [len(dicom_infos)]  # Store as a list
    if analyze_patient_slices:
        results['patient_slices'] = {patient_id: len(dicom_infos)}
    
    # Get first valid DICOM info for basic stats
    first_info = next((info for info in dicom_infos if any(info.values())), None)
    if first_info:
        if analyze_thickness and first_info['has_thickness']:
            results['slice_thicknesses'] = [first_info['thickness']]
            if analyze_thickness_vs_slices:
                results['scan_thicknesses'] = [first_info['thickness']]
                results['scan_slice_counts'] = [len(dicom_infos)]
        if analyze_spacing and first_info['has_spacing']:
            results['slice_spacings'] = [first_info['spacing']]
        if analyze_body_parts and first_info['has_body_part']:
            results['body_parts'] = {first_info['body_part']: 1}
    
    # Check for inconsistent thicknesses
    if check_inconsistency and analyze_thickness:
        thicknesses = [info['thickness'] for info in dicom_infos if info['has_thickness']]
        if thicknesses:
            unique_thicknesses = set(thicknesses)
            if len(unique_thicknesses) > 1:
                results['inconsistent_scans'] = [patient_id]
                results['inconsistency_details'] = [{
                    'patient_id': patient_id,
                    'thicknesses': list(unique_thicknesses),
                    'counts': [thicknesses.count(t) for t in unique_thicknesses]
                }]
    
    return results

def merge_results(results_list):
    """Merge results from multiple directories."""
    merged = {}
    
    for result in results_list:
        for key, value in result.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, dict):
                if isinstance(merged[key], defaultdict):
                    for k, v in value.items():
                        merged[key][k] += v
                else:
                    for k, v in value.items():
                        merged[key][k] = merged[key].get(k, 0) + v
            elif isinstance(value, list):
                merged[key].extend(value)
    
    return merged

def analyze_dicoms(dicom_path, analyze_patient_slices=True, analyze_scan_slices=True, 
                  analyze_thickness=True, analyze_spacing=True, analyze_body_parts=True,
                  analyze_thickness_vs_slices=True, check_inconsistency=True, num_workers=4):
    """Analyze DICOM files in parallel."""
    # Initialize results
    results = {}
    if analyze_patient_slices:
        results['patient_slices'] = defaultdict(int)
    if analyze_scan_slices:
        results['scan_slices'] = []
    if analyze_thickness:
        results['slice_thicknesses'] = []
    if analyze_spacing:
        results['slice_spacings'] = []
    if analyze_body_parts:
        results['body_parts'] = defaultdict(int)
    if analyze_thickness_vs_slices:
        results['scan_thicknesses'] = []
        results['scan_slice_counts'] = []
    if check_inconsistency:
        results['inconsistent_scans'] = []
        results['inconsistency_details'] = []
    
    # Get all directories with DICOM files
    dirs_to_process = []
    for root, dirs, files in os.walk(dicom_path):
        if any(f.endswith('.dcm') for f in files):
            dirs_to_process.append((root, files))
    
    # Process directories in parallel
    process_func = partial(
        process_directory,
        analyze_patient_slices=analyze_patient_slices,
        analyze_scan_slices=analyze_scan_slices,
        analyze_thickness=analyze_thickness,
        analyze_spacing=analyze_spacing,
        analyze_body_parts=analyze_body_parts,
        analyze_thickness_vs_slices=analyze_thickness_vs_slices,
        check_inconsistency=check_inconsistency
    )
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_list = list(tqdm(
            executor.map(process_func, dirs_to_process),
            total=len(dirs_to_process),
            desc="Analyzing DICOM files"
        ))
    
    # Merge results
    return merge_results(results_list)

def plot_distributions(results, output_file='dicom_analysis_results.png'):
    # Determine number of plots needed
    num_plots = sum([
        'patient_slices' in results,
        'scan_slices' in results,
        'slice_thicknesses' in results,
        'slice_spacings' in results,
        'body_parts' in results,
        'scan_thicknesses' in results and 'scan_slice_counts' in results,
        'inconsistent_scans' in results
    ])
    
    # Calculate grid dimensions
    rows = (num_plots + 1) // 2
    fig = plt.figure(figsize=(18, 6 * rows))
    gs = fig.add_gridspec(rows, 2)
    
    plot_idx = 0
    row = 0
    col = 0
    
    if 'patient_slices' in results:
        ax = fig.add_subplot(gs[row, col])
        sns.histplot(data=list(results['patient_slices'].values()), ax=ax, bins=30)
        ax.set_title('Distribution of Slices per Patient')
        ax.set_xlabel('Number of Slices')
        ax.set_ylabel('Count')
        plot_idx += 1
        row = plot_idx // 2
        col = plot_idx % 2
    
    if 'scan_slices' in results:
        ax = fig.add_subplot(gs[row, col])
        sns.histplot(data=results['scan_slices'], ax=ax, bins=30)
        ax.set_title('Distribution of Slices per Scan')
        ax.set_xlabel('Number of Slices')
        ax.set_ylabel('Count')
        plot_idx += 1
        row = plot_idx // 2
        col = plot_idx % 2
    
    if 'slice_thicknesses' in results:
        ax = fig.add_subplot(gs[row, col])
        sns.histplot(data=results['slice_thicknesses'], ax=ax, bins=30)
        ax.set_title('Distribution of Slice Thickness')
        ax.set_xlabel('Slice Thickness (mm)')
        ax.set_ylabel('Count')
        plot_idx += 1
        row = plot_idx // 2
        col = plot_idx % 2
    
    if 'slice_spacings' in results:
        ax = fig.add_subplot(gs[row, col])
        sns.histplot(data=results['slice_spacings'], ax=ax, bins=30)
        ax.set_title('Distribution of Slice Spacing')
        ax.set_xlabel('Slice Spacing (mm)')
        ax.set_ylabel('Count')
        plot_idx += 1
        row = plot_idx // 2
        col = plot_idx % 2
    
    if 'body_parts' in results:
        ax = fig.add_subplot(gs[row, col])
        body_parts_data = list(results['body_parts'].items())
        body_parts_data.sort(key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in body_parts_data]
        values = [x[1] for x in body_parts_data]
        ax.bar(range(len(values)), values)
        ax.set_title('Distribution of Body Parts Examined')
        ax.set_xlabel('Body Part')
        ax.set_ylabel('Count')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        plot_idx += 1
        row = plot_idx // 2
        col = plot_idx % 2
    
    if 'scan_thicknesses' in results and 'scan_slice_counts' in results:
        ax = fig.add_subplot(gs[row, col])
        ax.scatter(results['scan_thicknesses'], results['scan_slice_counts'], alpha=0.5)
        ax.set_title('Slice Thickness vs Number of Slices per Scan')
        ax.set_xlabel('Slice Thickness (mm)')
        ax.set_ylabel('Number of Slices')
        
        # Add trend line
        z = np.polyfit(results['scan_thicknesses'], results['scan_slice_counts'], 1)
        p = np.poly1d(z)
        ax.plot(results['scan_thicknesses'], p(results['scan_thicknesses']), "r--", alpha=0.8)
        plot_idx += 1
        row = plot_idx // 2
        col = plot_idx % 2
    
    if 'inconsistent_scans' in results:
        ax = fig.add_subplot(gs[row, col])
        if results['inconsistent_scans']:
            # Create a bar plot showing the number of different thicknesses per scan
            num_thicknesses = [len(detail['thicknesses']) for detail in results['inconsistency_details']]
            ax.hist(num_thicknesses, bins=range(min(num_thicknesses), max(num_thicknesses) + 2))
            ax.set_title('Distribution of Inconsistent Thicknesses per Scan')
            ax.set_xlabel('Number of Different Thicknesses')
            ax.set_ylabel('Number of Scans')
        else:
            ax.text(0.5, 0.5, 'No inconsistent scans found', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Inconsistent Thickness Analysis')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def print_statistics(results):
    if 'patient_slices' in results:
        print("\nPatient slices statistics:")
        print(f"Total number of patients: {len(results['patient_slices'])}")
        print(f"Mean slices per patient: {np.mean(list(results['patient_slices'].values())):.2f}")
        print(f"Median slices per patient: {np.median(list(results['patient_slices'].values())):.2f}")
        print(f"Min slices per patient: {min(results['patient_slices'].values())}")
        print(f"Max slices per patient: {max(results['patient_slices'].values())}")
    
    if 'scan_slices' in results:
        print(f"\nScan slices statistics:")
        print(f"Total number of scans: {len(results['scan_slices'])}")
        print(f"Mean slices per scan: {np.mean(results['scan_slices']):.2f}")
        print(f"Median slices per scan: {np.median(results['scan_slices']):.2f}")
        print(f"Min slices per scan: {min(results['scan_slices'])}")
        print(f"Max slices per scan: {max(results['scan_slices'])}")
    
    if 'slice_thicknesses' in results:
        print(f"\nSlice thickness statistics (mm):")
        print(f"Mean thickness: {np.mean(results['slice_thicknesses']):.2f}")
        print(f"Median thickness: {np.median(results['slice_thicknesses']):.2f}")
        print(f"Min thickness: {min(results['slice_thicknesses']):.2f}")
        print(f"Max thickness: {max(results['slice_thicknesses']):.2f}")
    
    if 'slice_spacings' in results:
        print(f"\nSlice spacing statistics (mm):")
        print(f"Mean spacing: {np.mean(results['slice_spacings']):.2f}")
        print(f"Median spacing: {np.median(results['slice_spacings']):.2f}")
        print(f"Min spacing: {min(results['slice_spacings']):.2f}")
        print(f"Max spacing: {max(results['slice_spacings']):.2f}")
    
    if 'body_parts' in results:
        print(f"\nBody Parts Examined:")
        for body_part, count in sorted(results['body_parts'].items(), key=lambda x: x[1], reverse=True):
            print(f"{body_part}: {count}")
    
    if 'inconsistent_scans' in results:
        print(f"\nInconsistent Slice Thickness Analysis:")
        print(f"Number of scans with inconsistent thickness: {len(results['inconsistent_scans'])}")
        if results['inconsistent_scans']:
            print("\nDetails of inconsistent scans:")
            for detail in results['inconsistency_details']:
                print(f"\nPatient ID: {detail['patient_id']}")
                print("Thicknesses found (mm):")
                for thickness, count in zip(detail['thicknesses'], detail['counts']):
                    print(f"  {thickness:.2f} mm: {count} slices")

def main():
    parser = argparse.ArgumentParser(description='Analyze DICOM files with various metrics')
    parser.add_argument('--dicom_path', type=str, 
                      default="/mmfs1/projects/dom_ameen_chi/common/SENTINL0/dinov2/nlst_train_data",
                      help='Path to DICOM files')
    parser.add_argument('--output', type=str, default='dicom_analysis_results.png',
                      help='Output file for plots')
    parser.add_argument('--patient-slices', action='store_true', default=True,
                      help='Analyze slices per patient')
    parser.add_argument('--scan-slices', action='store_true', default=True,
                      help='Analyze slices per scan')
    parser.add_argument('--thickness', action='store_true', default=True,
                      help='Analyze slice thickness')
    parser.add_argument('--spacing', action='store_true', default=True,
                      help='Analyze slice spacing')
    parser.add_argument('--body-parts', action='store_true', default=True,
                      help='Analyze body parts')
    parser.add_argument('--thickness-vs-slices', action='store_true', default=True,
                      help='Analyze relationship between thickness and slice count')
    parser.add_argument('--check-inconsistency', action='store_true', default=True,
                      help='Check for inconsistent slice thicknesses within scans')
    parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count(),
                      help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Analyze DICOM files
    results = analyze_dicoms(
        args.dicom_path,
        analyze_patient_slices=args.patient_slices,
        analyze_scan_slices=args.scan_slices,
        analyze_thickness=args.thickness,
        analyze_spacing=args.spacing,
        analyze_body_parts=args.body_parts,
        analyze_thickness_vs_slices=args.thickness_vs_slices,
        check_inconsistency=args.check_inconsistency,
        num_workers=args.num_workers
    )
    
    # Print statistics
    print_statistics(results)
    
    # Plot distributions
    plot_distributions(results, args.output)
    print(f"\nAnalysis complete. Results saved to '{args.output}'")

if __name__ == "__main__":
    main()
