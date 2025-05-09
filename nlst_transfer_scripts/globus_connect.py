from globus_sdk import (
    NativeAppAuthClient,
    TransferClient,
    TransferData,
    RefreshTokenAuthorizer,
)
from globus_sdk.exc import GlobusAPIError
import os

class GlobusConnect:
    def __init__(self, client_id):
        """
        Initialize the Globus transfer client.
        
        Args:
            client_id (str): Your Globus application client ID
        """
        self.client_id = client_id
        self.auth_client = NativeAppAuthClient(client_id)
        self.transfer_client = None
        self.authorizer = None

    def authenticate(self):
        """
        Authenticate with Globus and get the transfer client.
        """
        # Start the Native App authentication process
        self.auth_client.oauth2_start_flow(refresh_tokens=True)
        
        # Get the authorization URL
        authorize_url = self.auth_client.oauth2_get_authorize_url()
        print(f"Please go to this URL and login: {authorize_url}")
        
        # Get the authorization code from the user
        auth_code = input("Please enter the code you get after login here: ").strip()
        
        # Get the tokens
        token_response = self.auth_client.oauth2_exchange_code_for_tokens(auth_code)
        
        # Get the transfer token
        transfer_token = token_response.by_resource_server["transfer.api.globus.org"]["access_token"]
        refresh_token = token_response.by_resource_server["transfer.api.globus.org"]["refresh_token"]
        
        # Create the authorizer
        self.authorizer = RefreshTokenAuthorizer(
            refresh_token,
            self.auth_client,
            access_token=transfer_token,
            expires_at=token_response.by_resource_server["transfer.api.globus.org"]["expires_at_seconds"]
        )
        
        # Create the transfer client
        self.transfer_client = TransferClient(authorizer=self.authorizer)

    def transfer_files(self, source_endpoint, dest_endpoint, source_path, dest_path, recursive=False):
        """
        Transfer files from source to destination endpoint.
        
        Args:
            source_endpoint (str): UUID of the source endpoint
            dest_endpoint (str): UUID of the destination endpoint
            source_path (str): Path on the source endpoint
            dest_path (str): Path on the destination endpoint
            recursive (bool): Whether to transfer directories recursively
        """
        if not self.transfer_client:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        try:
            # Create a transfer task
            transfer_data = TransferData(
                self.transfer_client,
                source_endpoint,
                dest_endpoint,
                label="Globus Transfer",
                sync_level="checksum"
            )
            
            # Add the transfer items
            transfer_data.add_item(source_path, dest_path, recursive=recursive)
            
            # Submit the transfer
            transfer_result = self.transfer_client.submit_transfer(transfer_data)
            
            print(f"Transfer initiated with task ID: {transfer_result['task_id']}")
            return transfer_result['task_id']
            
        except GlobusAPIError as e:
            print(f"Error during transfer: {e}")
            raise

    def check_transfer_status(self, task_id):
        """
        Check the status of a transfer task.
        
        Args:
            task_id (str): The task ID to check
        """
        if not self.transfer_client:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        try:
            task = self.transfer_client.get_task(task_id)
            print(f"Task status: {task['status']}")
            return task['status']
        except GlobusAPIError as e:
            print(f"Error checking transfer status: {e}")
            raise

    def list_directory(self, endpoint_id, path, recursive=False, max_depth=1):
        """
        List files and folders in a given path on a Globus endpoint.
        
        Args:
            endpoint_id (str): UUID of the endpoint
            path (str): Path to list contents of
            recursive (bool): Whether to list contents recursively
            max_depth (int): Maximum depth for recursive listing (only used if recursive=True)
            
        Returns:
            dict: Dictionary containing lists of files and folders
        """
        if not self.transfer_client:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        try:
            # Initialize lists to store results
            files = []
            folders = []
            
            # Get the initial directory listing
            result = self.transfer_client.operation_ls(endpoint_id, path=path)
            
            # Process the results
            for entry in result:
                if entry["type"] == "file":
                    files.append({
                        "name": entry["name"],
                        "size": entry["size"],
                        "last_modified": entry["last_modified"]
                    })
                elif entry["type"] == "dir":
                    folder_path = os.path.join(path, entry["name"])
                    folders.append({
                        "name": entry["name"],
                        "path": folder_path
                    })
                    
                    # If recursive is True and we haven't reached max depth
                    if recursive and max_depth > 0:
                        # Recursively list the subdirectory
                        sub_result = self.list_directory(
                            endpoint_id,
                            folder_path,
                            recursive=True,
                            max_depth=max_depth - 1
                        )
                        files.extend(sub_result["files"])
                        folders.extend(sub_result["folders"])
            
            return {
                "files": files,
                "folders": folders
            }
            
        except GlobusAPIError as e:
            print(f"Error listing directory: {e}")
            raise
