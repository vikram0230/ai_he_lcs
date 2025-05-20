#!/usr/bin/env python3

"""
NLST Data Transfer Script

This script facilitates the transfer of NLST (National Lung Screening Trial) data between Globus endpoints.
It supports transferring both training and test datasets with configurable set sizes and diagnosis status.

Usage:
    python nlst_transfer.py -type <data_type> (-size <set_size> | -range <start>-<end>) [-diagnosis <diagnosis>] [-output <output_dir>]
    
    Arguments:
        -type, --type: Specify 'train' or 'test' to select the dataset type
        -size, --size: Number of PIDs to transfer from the selected dataset (e.g., 100)
        -range, --range: Range of PIDs to transfer (e.g., 100-200)
        -diagnosis, --diagnosis: Optional. Specify 'positive' or 'negative' to filter by diagnosis status
        -output, --output: Optional. Custom output directory path (default: nlst_train_data or nlst_test_data)

Example:
    python nlst_transfer.py -type train -size 100  # Transfer first 100 PIDs from training set
    python nlst_transfer.py -type test -range 100-200  # Transfer PIDs from index 100 to 200
    python nlst_transfer.py -type train -size 50 -diagnosis positive  # Transfer first 50 positive PIDs
    python nlst_transfer.py -type test -range 50-150 -diagnosis negative -output custom_dir  # Transfer to custom directory
"""

import os
import pandas as pd
import logging
import sys
import argparse
from dotenv import load_dotenv
from globus_connect import GlobusConnect
from globus_sdk import TransferAPIError
from datetime import datetime

# Configure logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'logs/nlst_transfer/{os.path.basename(__file__)}_{timestamp}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {os.path.abspath(log_file)}")

def get_train_pids():
    """
    Get the list of PIDs for the training set.
    
    Returns:
        list: List of PIDs in the training set
    """
    pid_df = pd.read_csv('pid2split_sorted.csv')
    train_pids = pid_df[pid_df['SPLIT'] == 'train']['PID'].tolist()
    logger.info(f"Found {len(train_pids)} PIDs in training set")
    return train_pids

def get_test_pids():
    """
    Get the list of PIDs for the test set.
    
    Returns:
        list: List of PIDs in the test set
    """
    pid_df = pd.read_csv('pid2split_sorted.csv')
    test_pids = pid_df[pid_df['SPLIT'] == 'test']['PID'].tolist()
    logger.info(f"Found {len(test_pids)} PIDs in test set")
    return test_pids

def get_diagnosis_pids(diagnosis: str) -> list[int]:
    """
    Get the list of PIDs based on diagnosis status.
    
    Args:
        diagnosis: 'positive' or 'negative' to filter PIDs
        
    Returns:
        list: List of PIDs matching the diagnosis criteria
    """
    actual_df = pd.read_csv('nlst_actual.csv')
    # Get unique PIDs where days_to_diagnosis matches the criteria
    if diagnosis == 'positive':
        pids = actual_df[actual_df['days_to_diagnosis'] > 0]['pid'].unique().tolist()
    else:  # negative
        pids = actual_df[actual_df['days_to_diagnosis'] < 0]['pid'].unique().tolist()
    logger.info(f"Found {len(pids)} PIDs with {diagnosis} diagnosis")
    return pids

def transfer_data(pids: list[int], dest_dir: str):
    ENV_PATH = "/home/vhari/dom_ameen_chi_link/vhari/.env"
    load_dotenv(dotenv_path=ENV_PATH)

    # Access environment variables
    CLIENT_ID = os.getenv('CLIENT_ID')
    
    # Initialize the transfer client
    globus_connect = GlobusConnect(CLIENT_ID)
    
    # Authenticate
    globus_connect.authenticate()
    logger.info("Globus Authenticated")
    
    source_endpoint = os.getenv('DATA_LAKE_ID')
    dest_endpoint = os.getenv('DATA_RAPIDS_ID')
    
    total_pids = len(pids)
    logger.info(f"Starting transfer of {total_pids} PIDs")
    
    for idx, pid in enumerate(pids, 1):
        progress = (idx / total_pids) * 100
        logger.info(f"Processing PID {idx}/{total_pids} ({progress:.1f}%): {pid}")
        
        source_path = f'/dom_ameen/common/NLST/manifest-NLST_allCT/NLST/{pid}'
        dest_path = f'/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/{dest_dir}/{pid}'
        
        # Check if destination directory already exists
        try:
            dest_contents = globus_connect.list_directory(dest_endpoint, dest_path)
            if dest_contents["files"] or dest_contents["folders"]:
                logger.info(f"Directory for PID {pid} already exists at destination. Skipping transfer.")
                continue
        except Exception as e:
            logger.info(f"Destination directory for PID {pid} does not exist or is empty. Proceeding with transfer.")
        
        # List directory contents with error handling
        try:
            logger.info(f"Attempting to list contents of: {source_path}")
            contents = globus_connect.list_directory(source_endpoint, source_path, recursive=False)
            
            if not contents["files"] and not contents["folders"]:
                logger.warning(f"No contents found in {source_path}")
                continue
                
            logger.info(f"Found {len(contents['files'])} files and {len(contents['folders'])} folders")
            
            # Log first few files and folders for verification
            if contents["files"]:
                logger.info("Sample files:")
                for file in contents["files"][:3]:
                    logger.info(f"- {file['name']} ({file['size']} bytes)")
                if len(contents["files"]) > 3:
                    logger.info(f"... and {len(contents['files']) - 3} more files")
            
            if contents["folders"]:
                logger.info("Sample folders:")
                for folder in contents["folders"][:3]:
                    logger.info(f"- {folder['name']}")
                if len(contents["folders"]) > 3:
                    logger.info(f"... and {len(contents['folders']) - 3} more folders")
            
        except TransferAPIError as e:
            logger.error(f"Error accessing directory {source_path}: {str(e)}")
            logger.error("Skipping this PID due to directory access error")
            continue
        except Exception as e:
            logger.error(f"Unexpected error while listing directory {source_path}: {str(e)}")
            logger.error("Skipping this PID due to unexpected error")
            continue
        
        # Transfer files
        try:
            task_id = globus_connect.transfer_files(
                source_endpoint,
                dest_endpoint,
                source_path,
                dest_path,
                recursive=True
            )
            
            # Check transfer status
            globus_connect.check_transfer_status(task_id)
            logger.info(f"Completed transfer of {pid} ({progress:.1f}% complete)")
        except Exception as e:
            logger.error(f"Error during transfer of {pid}: {str(e)}")
            logger.error("Continuing with next PID")
            continue
    
    logger.info("All transfers completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer data for a specified set size and type.")
    parser.add_argument("-type", "--type", type=str, required=True, choices=['train', 'test'],
                      help="Specify the data type as 'train' or 'test'.")
    
    # Create a mutually exclusive group for size and range
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-size", "--size", type=int,
                      help="Number of PIDs to transfer from the start of the dataset")
    group.add_argument("-range", "--range", type=str,
                      help="Range of PIDs to transfer (e.g., '100-200')")
    
    parser.add_argument("-diagnosis", "--diagnosis", type=str, choices=['positive', 'negative'],
                      help="Optional: Specify 'positive' or 'negative' to filter by diagnosis status")
    parser.add_argument("-output", "--output", type=str,
                      help="Optional: Custom output directory path (default: nlst_train_data or nlst_test_data)")
    args = parser.parse_args()

    # Log the command being used
    command = f"python {os.path.basename(__file__)}"
    command += f" -type {args.type}"
    if args.size is not None:
        command += f" -size {args.size}"
    else:
        command += f" -range {args.range}"
    if args.diagnosis:
        command += f" -diagnosis {args.diagnosis}"
    if args.output:
        command += f" -output {args.output}"
    logger.info(f"Executing command: {command}")

    data_type = args.type.lower()
    diagnosis = args.diagnosis

    # Get base PIDs based on data type
    if data_type == 'train':
        pids = get_train_pids()
        default_dest_dir = 'nlst_train_data'
    elif data_type == 'test':
        pids = get_test_pids()
        default_dest_dir = 'nlst_test_data'
    else:
        logger.error("Invalid data type. Please specify 'train' or 'test'.")
        sys.exit(1)

    # Use custom output directory if provided, otherwise use default
    dest_dir = args.output if args.output else default_dest_dir
    logger.info(f"Using output directory: {dest_dir}")

    # Filter by diagnosis if specified
    if diagnosis:
        diagnosis_pids = set(get_diagnosis_pids(diagnosis))
        pids = [pid for pid in pids if pid in diagnosis_pids]
        logger.info(f"Filtered to {len(pids)} PIDs with {diagnosis} diagnosis")

    # Handle size or range selection
    if args.size is not None:
        selected_pids = pids[:args.size]
        logger.info(f"Selected first {args.size} PIDs")
    else:
        try:
            start, end = map(int, args.range.split('-'))
            if start < 0 or end >= len(pids) or start > end:
                logger.error(f"Invalid range: {args.range}. Must be within 0-{len(pids)-1} and start <= end")
                sys.exit(1)
            selected_pids = pids[start:end+1]
            logger.info(f"Selected PIDs from index {start} to {end}")
        except ValueError:
            logger.error(f"Invalid range format: {args.range}. Must be in format 'start-end'")
            sys.exit(1)

    transfer_data(selected_pids, dest_dir)