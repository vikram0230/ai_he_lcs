# Documentation: Setting up a Python container

*Last edited on 11/22/2023 by Abdul Zakkar*

This document describes how a user may set up Python as a container, which would allow including any libraries which the user wishes to add. This is especially useful is a server environment where the user cannot install libraries of their choice on Python otherwise.

### Setting up Python container definition file

- We will utilize Apptainer to create a Python container to which we can add our own Python libraries.
- To load Apptainer, use the following command:
```
module load Apptainer
```
- The next step is create the definition file.
- First, create a new file in your terminal:
```
nano python.def
```
- Now that the file is open, paste the contents of [this document](../extras/python.def), and save your document using Ctrl-X, then y.

### Setting up the requirements file

- The next step is to provide a requirements file. This file lists all the libraries which the user plans to incorporate into their Python container.
- First, create this file in the terminal:
```
nano requirements.txt
```
- Next, either paste the contents of this document (link to be added), or enter your own libraries in the following format:
```
library=version # e.g. numpy=1.26.2
```
- Now save your document using Ctrl-X, then y.

### Creating the container image

- The next step is to use Apptainer to generate the Python container image `python.sif`
- Use the following command:
```
apptainer build python.sif python.def
```
- The file `python.sif` should now be available for use.

### Executing python scripts with the container

- Now that you have created this container, you have acceess to the all the specified Python libraries. 
- To execute a Python script, you can simply replace the keyword `python` with `./python.sif`.
- For example, instead of:
```
python hello.py arg1 arg2
```
- You will now enter:
```
./python.sif hello.py arg1 arg2
```
