import os
import urllib.request
from zipfile import ZipFile


KITTI_URLS = {
    "images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "calibration": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
    "labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "lidar": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",  # LiDAR point clouds
}

# Directory to save the dataset
dataset_dir = "kitti_3d_object_detection"
os.makedirs(dataset_dir, exist_ok=True)

def download_and_extract(url, dest_folder):
    """Downloads and extracts a zip file from a URL."""
    filename = url.split("/")[-1]
    zip_path = os.path.join(dest_folder, filename)
    
    # Download the file
    if not os.path.exists(zip_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already downloaded.")
    
    # Extract the file
    print(f"Extracting {filename}...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    print(f"Extracted {filename}")

# Download and extract each part of the dataset
for key, url in KITTI_URLS.items():
    download_and_extract(url, dataset_dir)

print("KITTI 3D Object Detection dataset with LiDAR data downloaded and extracted successfully.")
