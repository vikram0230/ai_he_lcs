# Documentation: How to run Sybil on the UIC Lakeshore

*Last updated 03/07/2025 by Vikram Harikrishnan*

**Disclaimer:** NetID access to the Extreme Cluster is required to follow this tutorial. If you do not have access, you may request access [here](https://acer.uic.edu/get-started/request-access/).

### 1. (Optional) Getting a sample DICOM from Sybil authors.
- There is a sample DICOM available provided by the Sybil authors [here](https://www.dropbox.com/sh/addq480zyguxbbg/AACJRVsKDL0gpq-G9o3rfCBQa?dl=0).
- Download sybil_demo_data.
- This file contains an ordered list of many .dcm files and represents the format of input which should be provided to the Sybil neural network.

### 2. Uploading a DICOM to the Lakeshore
- If your DICOM is a local file on your hardware, these are the steps for moving them to lakeshore (if your NetID has access).
- Using your terminal/command line, we will use **Secure Copy (scp)**.  

**Windows:**
- Windows does not have scp support by default, but this could be mitigated by using [openSSH](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui), or by using the Git command line. Many other options exist.
- I personally found [WinSCP](https://winscp.net/eng/index.php) to be easy to setup.
- [Git](https://git-scm.com/) command line should also be useful for this purpose.

**Mac/Linux:**
- Type the following: 
```
scp -r path/to/directory netid@lakeshore.acer.uic.edu:/path/to/directory
```
- Example:
```
scp -r sybil_demo_data azakka2@lakeshore.acer.uic.edu:/home/azakka2
```
- Follow the password prompts.

### 3. Accessing Lakeshore
Assuming your NetID has been granted access to lakeshore, you can access it on **Terminal** or **VS Code**.

The steps are as follows:

**Windows:**

- Open Terminal/command line.
- Type the following: 

```bash
ssh -m hmac-sha2-512 netid@lakeshore.acer.uic.edu
```
- Follow password prompts.

OR

- Download PuTTY. [PuTTY download page here.](https://putty.org/)
- Make sure the connection type is SSH.
- Enter the following into the Host Name field: 
```
netid@lakeshore.acer.uic.edu
```
- Follow password prompts.
  
For more information on PuTTY, refer to [ACER Documentation](https://confluence.acer.uic.edu/display/KB/Logging+into+the+Cluster). *Make sure you are connected to [UIC VPN](https://it.uic.edu/services/faculty-staff/uic-network/uic-vpn/) to access this page.*

- If [Git](https://git-scm.com/) command line is used, then you can follow the same steps as Mac/Linux.

**Mac/Linux:**
- Open Terminal/command line.
- Type the following: 
 
```
ssh netid@lakeshore.acer.uic.edu
```

- Follow password prompts.

**VS Code:**

- Press `F1` or `Ctrl+Shift+P` (Windows/Linux) / `Cmd+Shift+P` (macOS) to open the Command Palette.
- Type "Remote-SSH: Connect to Host..." and select it from the list.
- Choose "+ Add New SSH Host..." if your server is not listed.
- Enter the SSH connection command in the format: 
  
```bash
ssh netid@lakeshore.acer.uic.edu
```

- Select a configuration file to update (usually the first option).
- Click on the blue "><" icon in the lower-left corner of VS Code.
- Select "Remote-SSH: Connect to Host..." from the menu.
- Choose `lakeshore.acer.uic.edu` from the list.
- Enter your password when prompted.

### 4. Clone the Sybil Repository:
```bash
git clone https://github.com/reginabarzilaygroup/Sybil.git
cd Sybil
```

### 5. Setup the Environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Or .\.venv\Scripts\activate on Windows
pip install build
python -m build
pip install -e .  # Or pip install dist/*.whl
```

### 6. Verify Installation:

After installation, try running:

```bash
sybil-predict --help
```

- If the command is recognized, it confirms that the installation and entry point setup were successful.
- You should also see sybil_demo_data if you followed the prior optional step.

### 7. Running the model:

After verifing the installation of Sybil, run the following command to get inference from the model:

```bash
sybil-predict ../sybil_demo_data --output-dir ../output --file-type dicom
```

The resulting output should include the following:
```
{
  "predictions": [
    [
      0.37779576815811844,
      0.46321693657219976,
      0.5267677947638203,
      0.5485210415170675,
      0.5876559385527592,
      0.6707708482179324
    ]
  ]
}
```

## Running Sybil Model as a Slurm Job
Lakeshore allows users to submit scripts as jobs, and this performs multiple high-performance tasks simultaneously.
Learn more about creating a Slurm job [here](https://slurm.schedmd.com/documentation.html).

- Create a file `sybil_inference.sh`

```bash
nano sybil_inference.sh
```

- Copy-paste the following code and make necessary changes.

```bash
#!/bin/bash

#SBATCH --job-name=sybil
#SBATCH --output=path/to/output.out
#SBATCH --error=path/to/error_log.err
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# Activate virtual environment
source .venv/bin/activate

dicom_dir="path/to/dicom/images"
output_dir="path/to/output"

# Running Sybil inference
sybil-predict "$dicom_dir" --output-dir "$output_dir" --file-type dicom

echo "Sybil predict completed successfully."
```

- Run this script using this command:

```bash
sbatch sybil_inference.sh
```

To know the status of the running Slurm job, run this command:

``` bash
squeue
```