#!/usr/bin/env python3

"""
NLST Data Transfer Script

This script facilitates the transfer of NLST (National Lung Screening Trial) data between Globus endpoints.
It supports transferring both training and test datasets with configurable set sizes.

Usage:
    python nlst_transfer.py <data_type> <set_size>
    
    Arguments:
        data_type: Specify 'train' or 'test' to select the dataset type
        set_size: Number of PIDs to transfer from the selected dataset

Example:
    python nlst_transfer.py train 100  # Transfer first 100 PIDs from training set
    python nlst_transfer.py test 50   # Transfer first 50 PIDs from test set
"""

import os
import pandas as pd
import logging
import sys
import argparse
from dotenv import load_dotenv
from globus_connect import GlobusConnect
from globus_sdk import TransferAPIError

# Configure logging
log_file = f'logs/nlst_transfer/{os.path.basename(__file__)}_{os.getpid()}.log'
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
    parser.add_argument("data_type", type=str, help="Specify the data type as 'train' or 'test'.")
    parser.add_argument("set_size", type=int, help="Specify the size of the set.")
    args = parser.parse_args()

    data_type = args.data_type.lower()
    set_size = args.set_size

    if data_type == 'train':
        pids = get_train_pids()
        dest_dir = 'nlst_train_data'
        transfer_data(pids[:set_size], dest_dir)
    elif data_type == 'test':
        pids = get_test_pids()
        dest_dir = 'nlst_test_data'
        transfer_data(pids[:set_size], dest_dir)
    else:
        logger.error("Invalid data type. Please specify 'train' or 'test'.")