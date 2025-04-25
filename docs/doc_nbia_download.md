# Documentation: How to download NLST CT data from the Cancer Imaging Archive

*Last edited 03/11/2025 by Vikram Harikrishnan*

This document details the process of obtaining all CT scans of patients who were part of the low-dose CT scan arm of the National Lung Screening Trial (NLST). The instructions here are specific for servers with the PBS job service, as well as available use of the Apptainer container software. This is available to users of the University of Illinois at Chicago's Extreme Cluster.

### Downloading the manifest file

- The first step is to download this file from [here](https://wiki.cancerimagingarchive.net/display/NLST).
	- Find the Data Access table- click the Download button in the Radiology CT Images row.
- You should now have a manifest file with the extension `.tcia`. This file contains every entry to be used by the NBIA data retriever tool, for every CT scan.

### Setting up the NBIA data retriever tool

- This tool can be deployed in an Apptainer container.
- This requires setting up a container definition file.
- This definition file uses CentOS 7 as a base to install the tool onto, and allows the user to provide a `.tcia` manifest file as an argument.
- First, this definition file must be created.
- Create a new file in your terminal:
```
nano nbia_retriever.def
```
- Next, paste the contents of [this document](../extras/nbia_retriever.def), and save your document using Ctrl-X, then y.
- Start an interative session with compute node with the following command. Only compute node has apptainer for some reason.
```
salloc --nodes=1 --cpus-per-task=1 --time=00:10:00 -p batch
```
- Ensure that Apptainer is available for use by loading the module:
```
module load apptainer
```
- Next, create the container image:
```
apptainer build nbia_retriever.sif nbia_retriever.def
```
- Now, the NBIA data retriever tool is ready for use.

### Downloading the data

- The next step is to begin the download using the NBIA data retriever tool. The `.tcia` manifest file must be provided to the tool, as well as a directory to place the download into.
- Type the following command, remembering to exchange certain conponents with ones specific to your setup.
```
printf '%s\n' y | ./nbia_retriever.sif -c /path/to/manifest.tcia -d output_dir -f
```
- Here is an example of the command:
```
printf '%s\n' y | ./nbia_retriever.sif -c /projects/com_shared/azakka2/nlst/manifests/manifest-NLST_allCT.tcia -d /projects/com_shared/azakka2/nlst/nlst_cancer_imaging_archive -f
```
- Command description:
	- `printf '%s\n' y`- This portion passes a `y` or yes to the NBIA retriever tool when asked if the user agrees with the terms.
	- `./nbia_retriever.sif`- Executing the Apptainer image created above.
	- `-c /projects/com_shared/azakka2/nlst/manifests/manifest-NLST_allCT.tcia`- Passing in the manifest file as an argument.
	- `-d /projects/com_shared/azakka2/nlst/nlst_cancer_imaging_archive`- Passing the output directory, the download will be contained here.
	- `-f` This argument signifies that the tool will skip downloading files to which the user does not have access.
- The download will now begin. The download duration will be quite large.

### Downloading the data as a PBS job

- Due to the extremely large size of the data, it may be better to divide the download into multiple jobs that can all run simultaneously, downloading only a portion of the full dataset.
- Linked [here](../extras/nbia_nlst_job.pbs) is an example of a PBS job to perform this download.
- You can use a manifest `.tcia` file which does not contain the full dataset, but selecting only certain files from the [NBIA Search](https://nlst.cancerimagingarchive.net/nbia-search/), which can be accessed on [this page](https://wiki.cancerimagingarchive.net/display/NLST) for NLST, as the Search button in the "Radiology CT Images" row of the Data Access table. 



*Refer to [this document](ai_he_lcs/blob/main/docs/doc_nbia_download_chunked.md) to download NLST data in chunks.*