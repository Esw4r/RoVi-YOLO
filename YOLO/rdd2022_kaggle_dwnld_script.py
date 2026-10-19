import os
import kaggle

# 2. Authenticate the API
kaggle.api.authenticate()

# 3. Download the dataset
dataset_slug = "aliabdelmenam/rdd-2022" 

kaggle.api.dataset_download_files(
    dataset_slug, 
    path='./rdd2022',     # Directory where data will download
    unzip=True,         # Automatically extract the zip file
    quiet=False
)

print("Dataset downloaded and unzipped successfully!")