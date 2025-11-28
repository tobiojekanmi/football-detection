import os
from modules.utils import download_from_gdrive, unzip_file


# Create dataset folder
base_dataset_path = "./datasets/"
if not os.path.exists(base_dataset_path):
    os.mkdir(base_dataset_path)

# Download dataset from google drive
filepath = os.path.join(base_dataset_path, "football_dataset.zip")
link = "https://drive.google.com/file/d/1p6sHdhY7OhNCbO2wpWmfuil-5hYl82n8/view?usp=drive_link"
if not os.path.exists(filepath):
    download_from_gdrive(link=link, output_path=filepath)

# Unzip dataset
unzip_file(filepath, base_dataset_path)
