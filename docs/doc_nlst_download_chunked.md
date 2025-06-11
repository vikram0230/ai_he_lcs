# Documentation: How to download NLST CT data from the Cancer Imaging Archive in Chunks

*Last edited 05/127/2025 by Vikram Harikrishnan*

This document is a continuation of [NBIA Download Document](doc_nbia_download.md). Please refer to the above document to setup NBIA and try out starter code to download a sample of NLST data. This document details the process of chunking NLST's huge dataset and downloading it to Lakeshore and uploading it to Data Lake using Globus.

### Chunking Patients Data

Run the following code to break down `manifest-NLST_allCT.tcia` file into smaller files to download the data in chunks. This creates `2031` files.

```python
import math

# Read the original manifest file
with open('manifest-NLST_allCT.tcia', 'r') as file:
    content = file.read()

# Extract the header and series list
header, series_list = content.split('ListOfSeriesToDownload=\n')
series_list = series_list.strip().split('\n')

# Define batch size
batch_size = 100  # Adjust as needed

# Calculate number of batches
num_batches = math.ceil(len(series_list) / batch_size)

# Split series list into batches and create new manifest files
for i in range(num_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(series_list))
    batch = series_list[start:end]
    
    # Create new manifest file
    new_filename = f'tcia_files/manifest-NLST_{i+1}.tcia'
    with open(new_filename, 'w') as file:
        file.write(header)
        file.write('ListOfSeriesToDownload=\n')
        file.write('\n'.join(batch))
    
    print(f'Created {new_filename} with {len(batch)} series')

print(f'Split completed. Created {num_batches} manifest files.')
```

### Globus setup

Establish a `.env` file containing the specified Globus Connect information.

```bash
# Client ID of Python cli app in Globus
CLIENT_ID=<client_id>

# Client ID of Collections
DATA_RAPIDS_ID=<client_id>
DATA_LAKE_ID=<client_id>
```

### Starting the download and transfer of NLST data

This script facilitates the batched download of NLST data, followed by its transfer to DataLake, and concludes by removing the local data copies.

Download this [script](../scripts/nlst_download.py) and run it with the following command.

```bash
python nlst_download.py
```




