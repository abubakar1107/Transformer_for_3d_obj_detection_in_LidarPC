import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader
import os


# Set a custom data directory for TFDS
data_dir = r'C:\Users\abuba\Desktop\ENPM703\Final project\project_v1\data'  # Set to your preferred local directory
os.makedirs(data_dir, exist_ok=True)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\abuba\Desktop\ENPM703\Final project\project_v1\data-3d-obj-detection-b1955ec6bb94.json'

# Load Waymo Open Dataset using tensorflow_datasets
train_dataset, train_info = tfds.load('waymo_open_dataset/v1.1', split='train', data_dir=data_dir, with_info=True)
validation_dataset, val_info = tfds.load('waymo_open_dataset/v1.1', split='validation', data_dir=data_dir, with_info=True)

print(train_info)


#PyTorch Dataset Class for Waymo LiDAR Data
class WaymoLidarDataset(Dataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = list(tf_dataset)  

    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, idx):
       
        example = self.tf_dataset[idx]
        
        # Extract LiDAR data and labels
        lidar_data = example['lidar']['laser_data']
        labels = example['labels']
        
        # Convert to PyTorch tensors
        lidar_data = torch.tensor(lidar_data.numpy(), dtype=torch.float32)
        labels = torch.tensor(labels.numpy(), dtype=torch.long)
        
        return lidar_data, labels

train_lidar_dataset = WaymoLidarDataset(train_dataset)
val_lidar_dataset = WaymoLidarDataset(validation_dataset)

#DataLoaders for batching and shuffling
train_dataloader = DataLoader(train_lidar_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_lidar_dataset, batch_size=32, shuffle=False)

for batch_idx, (lidar_data, labels) in enumerate(train_dataloader):

    print(f'Train Batch {batch_idx}: LiDAR Data Shape: {lidar_data.shape}, Labels Shape: {labels.shape}')
