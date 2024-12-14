import torch
from torchviz import make_dot
from Networks.network5 import ObjectDetectionModel

# Define your model
model = ObjectDetectionModel(num_classes=3, feature_dim=64)

# Create a sample input tensor (ensure gradients are enabled for visualization)
sample_input = torch.randn(1, 5000, 3, requires_grad=True)  # Adjust dimensions if needed

# Perform a forward pass
class_output, bbox_output = model(sample_input)

# Generate the computational graph
dot = make_dot((class_output, bbox_output), params=dict(model.named_parameters()))

# Save the visualization as PNG
output_path = "object_detection_network"
dot.render(output_path, format="png", cleanup=True)

print(f"Network visualization saved as '{output_path}.png'")
