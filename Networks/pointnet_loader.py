# PROJECT_V1/pointnet_loader.py
import sys
import os
import torch
from Pointnet_Pointnet2_pytorch.models.pointnet2_cls_msg import get_model

sys.path.append(os.path.abspath('./Pointnet_Pointnet2_pytorch/models'))

def load_pretrained_pointnet():
    model = get_model(num_class=40, normal_channel=False)
    checkpoint_path = './Pointnet_Pointnet2_pytorch/log/classification/pointnet2_msg_normals/best_model.pth'  # Adjust path as needed
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Use 'cuda' if on a GPU
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    
    return model
