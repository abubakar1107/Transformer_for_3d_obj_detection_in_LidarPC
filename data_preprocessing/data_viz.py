import numpy as np
import open3d as o3d

def load_kitti_lidar_data(file_path):
    # Load point cloud from .bin file
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud

def load_kitti_labels(label_file):
    # Read labels from .txt file
    boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            elements = line.split()
            if len(elements) < 15:
                continue  # Skip malformed lines
            
            # Parse bounding box parameters (e.g., class, height, width, length, etc.)
            _, _, _, _, _, _, _, h, w, l, x, y, z, ry = map(float, elements[1:])
            
            # Filter out boxes with clearly invalid coordinates
            if x < -500 or y < -500 or z < -500 or h < 0 or w < 0 or l < 0:
                continue  # Skip unrealistic or placeholder values
            
            boxes.append((h, w, l, x, y, z, ry))
    return boxes



def load_calibration(calib_file):
    calib_data = {}
    with open(calib_file, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                key, *values = line.split()
                calib_data[key.rstrip(':')] = np.array(values, dtype=np.float32)
    return calib_data


def visualize_point_cloud(points):
    # Convert points to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Only x, y, z coordinates
    o3d.visualization.draw_geometries([pcd])

def get_velo_to_cam_transform(calib_data):
    # Tr_velo_to_cam is a 3x4 transformation matrix
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
    # Append [0, 0, 0, 1] to make it a 4x4 matrix for easy transformation
    Tr_velo_to_cam = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    return np.linalg.inv(Tr_velo_to_cam)  # Invert to go from camera to LiDAR frame


def create_bounding_box(h, w, l, x, y, z, ry, Tr_cam_to_velo):
    # Define the eight corners of the bounding box in the camera frame
    corners = np.array([
        [ l/2,  h/2,  w/2],
        [ l/2,  h/2, -w/2],
        [-l/2,  h/2, -w/2],
        [-l/2,  h/2,  w/2],
        [ l/2, -h/2,  w/2],
        [ l/2, -h/2, -w/2],
        [-l/2, -h/2, -w/2],
        [-l/2, -h/2,  w/2]
    ]).T  # Transpose for easier multiplication

    # Rotate and translate the corners
    rot_mat = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    corners = rot_mat @ corners + np.array([[x], [y], [z]])

    # Transform from camera to LiDAR frame
    corners = np.vstack((corners, np.ones((1, 8))))  # Add 1s for homogeneous coordinates
    corners = Tr_cam_to_velo @ corners  # Apply transform
    return corners[:3, :].T 

def visualize_point_cloud_with_boxes(points, boxes):
    # Convert points to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    print("Boxes:", boxes)
    # Create bounding boxes and add them to the visualization
    geometries = [pcd]
    for (h, w, l, x, y, z, ry, Tr_cam_to_velo) in boxes:
        corners = create_bounding_box(h, w, l, x, y, z, ry, Tr_cam_to_velo)
        print("Bounding Box Corners:\n", corners)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Top face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Bottom face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set color of the bounding box to black
        line_set.paint_uniform_color([0, 0, 0])  # Black color for visibility on white background
        geometries.append(line_set)
    
    # Visualize with thicker lines by setting `line_width`
    o3d.visualization.draw_geometries(geometries, width=800, height=600)



def visualize_only_boxes(boxes):
    geometries = []
    for (h, w, l, x, y, z, ry) in boxes:
        corners = create_bounding_box(h, w, l, x, y, z, ry, shift=np.array([0, 0, 10]))  # Adjust shift as needed
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Top face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Bottom face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Side edges
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0, 0, 0])  # Red for visibility
        geometries.append(line_set)
    
    # Add coordinate frame to help with visibility
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0]))
    
    o3d.visualization.draw_geometries(geometries,
                                      zoom=0.8,
                                      front=[0.5, 0.5, -0.5],
                                      lookat=[0, 0, 0],
                                      up=[0, 1, 0])

file_path = r'C:\Users\abuba\Desktop\ENPM703\Final project\project_v1\kitti_3d_object_detection\training\velodyne\007453.bin'
label_file = r'C:\Users\abuba\Desktop\ENPM703\Final project\project_v1\kitti_3d_object_detection\training\label_2\007453.txt'
calib_file = r'C:\Users\abuba\Desktop\ENPM703\Final project\project_v1\kitti_3d_object_detection\training\calib\007453.txt'

point_cloud_data = load_kitti_lidar_data(file_path)
label_boxes = load_kitti_labels(label_file)
calib_data = load_calibration(calib_file)
Tr_cam_to_velo = get_velo_to_cam_transform(calib_data)

# visualize_point_cloud(point_cloud_data)
# visualize_point_cloud_with_boxes(point_cloud_data, label_boxes)
# visualize_only_boxes(label_boxes)
visualize_point_cloud_with_boxes(point_cloud_data, 
                                 [(h, w, l, x, y, z, ry, Tr_cam_to_velo) for (h, w, l, x, y, z, ry) in label_boxes])