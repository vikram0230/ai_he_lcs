"""
This Python script is used for data transfer and management of NLST data. 
It sets up the necessary environment, initializes the Globus client, 
and defines a function to transfer files to Data Lake and delete local copies.
"""

import subprocess
import os
import glob
import globus_sdk
import shutil
from dotenv import dotenv_values
from globus_sdk.scopes import GCSCollectionScopeBuilder, TransferScopes
from loguru import logger

config = dotenv_values(".env")
logger.add("data_transfer.log")

os.makedirs("nlst_cancer_imaging_archive", exist_ok=True)

# Globus setup
CLIENT_ID = config.get('CLIENT_ID')
SOURCE_ENDPOINT = config.get('DATA_RAPIDS_ID')
DESTINATION_ENDPOINT = config.get('DATA_LAKE_ID')
DESTINATION_PATH = "NLST/nlst_cancer_imaging_archive"

transfer_scope = TransferScopes.make_mutable("all")
transfer_scope.add_dependency(
    GCSCollectionScopeBuilder(SOURCE_ENDPOINT).data_access, optional=True
)
transfer_scope.add_dependency(
    GCSCollectionScopeBuilder(DESTINATION_ENDPOINT).data_access, optional=True
)

# Initialize Globus client
client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
client.oauth2_start_flow(requested_scopes=transfer_scope, refresh_tokens=True)

print(f"Please go to this URL and login: {client.oauth2_get_authorize_url()}")
auth_code = input("Enter the auth code: ")

token_response = client.oauth2_exchange_code_for_tokens(auth_code)
transfer_client = globus_sdk.TransferClient(authorizer=globus_sdk.AccessTokenAuthorizer(
    token_response.by_resource_server['transfer.api.globus.org']['access_token']))

# Function to transfer files and delete local copies
def transfer_and_delete_batch(local_dir):
    transfer_data = globus_sdk.TransferData(transfer_client, SOURCE_ENDPOINT, DESTINATION_ENDPOINT)
    
    for file in glob.glob(f"{local_dir}/*"):
        relative_path = os.path.relpath(file, local_dir)
        transfer_data.add_item(file, f"{DESTINATION_PATH}/{relative_path}")
    
    result = transfer_client.submit_transfer(transfer_data)
    logger.info(f"Transfer submitted. Task ID: {result['task_id']}")
    
    # Wait for transfer to complete
    while not transfer_client.task_wait(result['task_id'], timeout=60):
        print("Transfer in progress...")

    for item in glob.glob(f"{local_dir}/*"):
        if os.path.isdir(item):
            shutil.rmtree(item)
        else:
            os.remove(item)
    logger.info(f"All files and directories in {local_dir} deleted.")
    
    
total_size = 0
tcia_files_dir = '/nlst/tcia_files'
tcia_files = os.listdir(tcia_files_dir)
tcia_files.sort()
for i in range(1300,1301):
    tcia_file = f"manifest-NLST_{i}.tcia"
    # logger.info(f"Processing {tcia_file}...")

    tcia_file_path = os.path.join(tcia_files_dir, tcia_file)

    # Run NBIA Data Retriever
    output_dir = "/nlst/nlst_cancer_imaging_archive"
    command = ["/nlst/nbia-retriever/opt/nbia-data-retriever/bin/nbia-data-retriever", 
                    "--cli", tcia_file_path, "-d", 
                    output_dir, "-f"]

    logger.info(f"Downloading {tcia_file} components...")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input='y\n')

    manifest_name = "manifest-NLST_allCT"
    for item in os.listdir(output_dir):
        # Check if the item is a directory and starts with "output"
        if item.startswith("manifest-NLST"):
            
            # Rename the directory
            os.rename(os.path.join(output_dir,item), os.path.join(output_dir,manifest_name))
            # print(f"Renamed '{item}' to '{manifest_name}'")
            break
        
    # Run the 'du' command to get directory size
    result = subprocess.run(['du', '-sb', os.path.join(output_dir,manifest_name)], capture_output=True, text=True)
    # Extract the size from the command output
    size = int(result.stdout.split()[0])
    total_size += size
    logger.info(f"Batch size: {size / (1024**3):.2f} GB")

    transfer_and_delete_batch(output_dir)
    logger.info(f"Completed processing {tcia_file}.")
    
logger.info("All data processed, transferred, and local copies deleted.")
total_size_gb = total_size / (1024**3)
logger.info(f"Total transfer size: {total_size_gb:.2f} GB")