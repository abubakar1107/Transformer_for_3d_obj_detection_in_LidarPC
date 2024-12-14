# Custum Transformer model for 3d object detection

This project focuses on 3D object detection using LiDAR data from the KITTI dataset. It includes data preprocessing, model training, and visualization of results.

## Models

The models are located in the `Networks` folder. The final model is implemented in `network5.py`. You can modify the model architecture, such as the number of heads and blocks, directly in this file.

## Training

The training script is located in `training3.py`. This script handles both the training and validation of the model using the KITTI dataset.

### Training the Model

1. Ensure you have the required dependencies installed.
2. Download the KITTI dataset using the script `download_kitti_dataset.py`.
3. Train the model by running the following command:

   ```bash
   python training3.py