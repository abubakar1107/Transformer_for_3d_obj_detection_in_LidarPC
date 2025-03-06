# Custom Transformer Model for 3D Object Detection

This project focuses on 3D object detection using **3D perception data** from the KITTI dataset. It includes data preprocessing, model training, and visualization of results.

## Model Architecture

The following image illustrates the architecture of the **Transformer-based 3D Object Detection Model** used in this project:

![Model Architecture](Picture4.png)

The following image illustrates the 3D objects detected 
![3Ddetection](Picture3.jpg)

The model consists of:
- A **pretrained PointNet++** encoder for extracting higher-dimensional features.
- **Multi-head attention blocks** for learning spatial relationships in 3D data.
- **Fully connected layers** for classification and bounding box regression.
- Outputs include **bounding boxes and classification results**.

## Models

The models are located in the `Networks` folder. The final model is implemented in `network5.py`.  
You can modify the model architecture, such as the number of heads and blocks, directly in this file.

## Training

The training script is located in `training3.py`. This script handles both the training and validation of the model using the KITTI dataset.

### Training the Model

1. Ensure you have the required dependencies installed.
2. Download the KITTI dataset using the script `download_kitti_dataset.py`.
3. Train the model by running the following command:

   ```bash
   python training3.py
