import numpy as np


point_cloud = np.fromfile(r'C:\Users\abuba\Desktop\ENPM703\Final project\project_v1\kitti_3d_object_detection\training\velodyne\000001.bin', dtype=np.float32).reshape(-1, 4)
num_points = point_cloud.shape[0]

print(f"Number of points in the point cloud: {num_points}")
