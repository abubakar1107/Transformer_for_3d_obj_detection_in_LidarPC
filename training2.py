import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from Networks.network5 import ObjectDetectionModel  
from loss_functions.loss_function import CombinedLoss 
import matplotlib.pyplot as plt

# Define the KITTI Dataset class (same as before)
class KITTIDataset(Dataset):
    def __init__(self, velodyne_dir, label_dir, calib_dir, num_points=5000):
        self.velodyne_dir = velodyne_dir
        self.label_dir = label_dir
        self.calib_dir = calib_dir
        self.files = [f.split('.')[0] for f in os.listdir(velodyne_dir) if f.endswith('.bin')]
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]

        # Load point cloud data
        velodyne_path = os.path.join(self.velodyne_dir, file_id + '.bin')
        point_cloud = self.load_point_cloud(velodyne_path)

        # Load labels (bounding boxes and classes)
        label_path = os.path.join(self.label_dir, file_id + '.txt')
        class_label, bbox = self.load_labels(label_path)

        # Load calibration data
        calib_path = os.path.join(self.calib_dir, file_id + '.txt')
        calib_data = self.load_calibration(calib_path)
        Tr_cam_to_velo = self.get_velo_to_cam_transform(calib_data)

        return point_cloud, class_label, bbox, Tr_cam_to_velo

    def load_point_cloud(self, file_path):
        # Load the entire point cloud
        point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # Use x, y, z only
        return torch.from_numpy(point_cloud).float()


    def load_labels(self, file_path):
        boxes = []
        with open(file_path, 'r') as f:
            for line in f:
                elements = line.split()
                if len(elements) < 15:
                    continue  # Skip malformed lines
                
                # Parse bounding box parameters
                _, _, _, _, _, _, _, h, w, l, x, y, z, ry = map(float, elements[1:])
                
                # Filter invalid boxes
                if x < -500 or y < -500 or z < -500 or h < 0 or w < 0 or l < 0:
                    continue  # Skip unrealistic or placeholder values
                
                boxes.append((h, w, l, x, y, z, ry))

        if len(boxes) == 0:
            boxes.append((0, 0, 0, 0, 0, 0, 0))  # Add a placeholder if no valid boxes are found
        class_labels = [0]  # Replace with actual mapping logic as needed

        return torch.tensor(class_labels[0]), torch.tensor(boxes[0])

    def load_calibration(self, file_path):
        calib_data = {}
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    key, *values = line.split()
                    calib_data[key.rstrip(':')] = np.array(values, dtype=np.float32)
        return calib_data

    def get_velo_to_cam_transform(self, calib_data):
        Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
        Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])  # Make it a 4x4 matrix
        return torch.from_numpy(np.linalg.inv(Tr_velo_to_cam)).float()  # Invert to go from camera to LiDAR frame


def custom_collate_fn(batch):
    # Extract each item in the batch
    point_clouds, class_labels, bbox_labels, Tr_cam_to_velo = zip(*batch)

    # Find the maximum number of points in the batch
    max_points = max(p.shape[0] for p in point_clouds)

    # Pad each point cloud to the size of the largest one
    padded_clouds = [torch.cat([p, torch.zeros(max_points - p.shape[0], 3)], dim=0) for p in point_clouds]

    # Stack all data into batched tensors
    padded_clouds = torch.stack(padded_clouds)
    class_labels = torch.stack(class_labels)
    bbox_labels = torch.stack(bbox_labels)
    Tr_cam_to_velo = torch.stack(Tr_cam_to_velo)

    return padded_clouds, class_labels, bbox_labels, Tr_cam_to_velo


# Paths to data directories
velodyne_dir = 'kitti_3d_object_detection/training/velodyne'
label_dir = 'kitti_3d_object_detection/training/label_2'
calib_dir = 'kitti_3d_object_detection/training/calib'

# Create the dataset and split into training and validation sets
full_dataset = KITTIDataset(velodyne_dir, label_dir, calib_dir)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)


# Initialize model, loss function, and optimizer
model = ObjectDetectionModel(num_classes=3, feature_dim=64).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
criterion = CombinedLoss(alpha=1.0, beta=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation and model checkpointing
num_epochs = 20
best_val_loss = float('inf')
best_model_path = 'best_object_detection_model.pth'

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    print("Epoch: ", epoch)
    for i, (inputs, class_labels, bbox_labels, Tr_cam_to_velo) in enumerate(train_loader):
        inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        class_labels = class_labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        bbox_labels = bbox_labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Zero the gradient buffers
        optimizer.zero_grad()

        # Forward pass
        class_outputs, bbox_outputs = model(inputs)

        # Compute the combined loss
        loss = criterion(bbox_outputs, bbox_labels, class_outputs, class_labels)
        loss = loss.sum()  # Compute the sum of the loss tensor

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_losses.append(loss.item())

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, class_labels, bbox_labels, Tr_cam_to_velo) in enumerate(val_loader):
            inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            class_labels = class_labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            bbox_labels = bbox_labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Forward pass
            class_outputs, bbox_outputs = model(inputs)

            # Compute the combined loss
            v_loss = criterion(bbox_outputs, bbox_labels, class_outputs, class_labels)
            v_loss = v_loss.sum()  
            val_loss += loss.item()
            val_losses.append(loss.item())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Checkpoint the model if validation loss is the best so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with validation loss {best_val_loss:.4f}")

print("Training complete. Best model saved as 'best_object_detection_model.pth'.")

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
