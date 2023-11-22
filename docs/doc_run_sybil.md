# Documentation: How to run Sybil on the UIC Extreme Cluster

*Last updated 11/20/2023 by Abdul Zakkar*

**Disclaimer:** NetID access to the Extreme Cluster is required to follow this tutorial. If you do not have access, you may request access [here](https://acer.uic.edu/get-started/request-access/).

### 1. (Optional) Getting a sample DICOM from Sybil authors.
- There is a sample DICOM available provided by the Sybil authors [here](https://www.dropbox.com/sh/addq480zyguxbbg/AACJRVsKDL0gpq-G9o3rfCBQa?dl=0).
- Download sybil_demo_data.
- This file contains an ordered list of many .dcm files and represents the format of input which should be provided to the Sybil neural network.

### 2. Uploading a DICOM to the Extreme Cluster
- If your DICOM is a local file on your hardware, these are the steps for moving them to the extreme cluster (if your NetID has access).
- Using your terminal/command line, we will use **Secure Copy (scp)**.  
- Windows does not have scp support by default, but this could be mitigated by using [openSSH](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui), or by using the Git command line. Many other options exist.
- **I use [Git](https://git-scm.com/) command line, since I find it to be the most straightforward for my purposes.**
- Type the following: 
```
scp -r path/to/directory netid@login-1.extreme.acer.uic.edu:/path
```
- Example:
```
scp -r sybil_demo_data azakka2@login-1.extreme.acer.uic.edu:/home/azakka2
```
- Follow the password prompts.

### 3. Accessing the Extreme Cluster
Assuming your NetID has been granted access to the extreme cluster, the steps are as follows:

**Windows:**
- First, download PuTTY. [PuTTY download page here.](https://putty.org/)
- Enter the following into the Host Name field: 
```
netid@login-1.extreme.acer.uic.edu
```
- Follow password prompts.
- If [Git](https://git-scm.com/) command line is used, then you can follow the same steps as Mac/Linux.

**Mac/Linux:**
- Open Terminal/command line.
- Type the following: 
```
ssh netid@login-1.extreme.acer.uic.edu
```
- Follow password prompts.

### 4. Getting the latest Sybil Image
- Enter the following lines of code:
```
module load Apptainer
apptainer pull docker://mitjclinic/sybil
```
- This should result in having `sybil_latest.sif` available on the Extreme Cluster.
- This can be verified by typing the following:
```
ls
```
- You should also see sybil_demo_data if you followed the prior optional step.

### 5. Setting up `main.py`
- Sybil is built in Python and requires a Python script called `main.py` to be used as a starting point for program execution.
- First, we will create a new file in the extreme cluster. Type the following:
```
nano main.py
```
- Then, copy-paste the contents **below** into your terminal/command line.
```
# main.py

from sybil import Serie, Sybil
from os import listdir

# Load a trained model
model = Sybil("sybil_ensemble") 
# can also use "sybil_base" (1 model vs ensemble of 5)

dir_name = "sybil_demo_data" # EDIT THIS TO MATCH YOUR DIRECTORY
# This is the name of the directory containing the dicoms.

# Get risk scores
serie = Serie([dir_name + "/" + i for i in listdir(dir_name)])
scores = model.predict([serie])

print(scores)
```
- Press Ctrl-X, then Y to confirm saving the file.

### 6. Running the Sybil Model as a PBS job
- The Extreme Cluster allows users to submit scripts as jobs, and this perform multiple high-performance tasks simultaneously.
- The best way to create a job is to start with a PBS script. Let's create one.
```
nano sybil_job.pbs
```
- Copy-paste the contents below. Each line is also explained. [This](https://latisresearch.umn.edu/creating-a-PBS-script) is also a good resource to learn more about PBS job scripts.
```
# Specifies that the job be submitted to the batch queue.
#PBS -q batch

# Requests 1 node and 1 processor per node.
#PBS -l nodes=1:ppn=1

# Sets max walltime for the job to 1 hour.
#PBS -l walltime=1:00:00

# Sets the name of the job as displayed by qstat.
#PBS -N sybil

# Sends standard output to sybil.out.
#PBS -o sybil.out

# Merge output and error files.
#PBS -j oe

# Sends email on job abort, begin, and end.
#PBS -m abe

# Specifies email address to which mail should be sent.
#PBS -M netid@uic.edu

# Start the job in the current working directory.
cd $PBS_O_WORKDIR/

# Load the Apptainer module which allows us to use Sybil.
module load Apptainer

# Run Sybil.
./sybil_latest.sif
```
- Press Ctrl-X, then Y to confirm saving the file.
- Next, run this job with the following command:
```
qsub sybil_job.pbs
```
- Sybil should now be running, calculating prediction scores for your DICOM.
- You can check the status of your job with this command:
```
qstat
```
- Under the Job State column, you will see `R`  if it is running, or `C` if it is complete.
- The result of the job should be stored in `sybil.out`, since this is what we named our output file in the PBS job.
- We can view the contents of this output with this command:
```
cat sybil.out
``` 
- The resulting output should include the following:
```
Prediction(scores=[[0.0033378278896217693, 0.01461589983420139,
0.02436322603858815, 0.033144887056502856, 0.040198648409035156,
0.060580642990126575]])
```
- The 6 numbers listed are each probability of cancer diagnosis 1 year, 2 years, … , and 6 years after diagnosis.

### Addendum:
- In order to submit Sybil jobs in batches, I needed a way to pass a directory to Sybil via the terminal.
- I created a new Apptainer image called `sybil_dir.sif`, which allows the user to run the image as such:
```
./sybil_dir.sif /path/to/dicom/directory
```
- This allows multiple PBS jobs to be set up with each job referencing a different directory.
- I was able to create this custom image by using a new definition file.
	- This new file uses local image bootstrapping and applies a new runscript which facilitates the inclusion of a directory argument.
- Sybil's `main.py` (see `doc_sybil_main_py.md` [here](doc_sybil_main_py.md)) was then also modified to be able to handle a directory argument, by using `sys.argv[1]`.

