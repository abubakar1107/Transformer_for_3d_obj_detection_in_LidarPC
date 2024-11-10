import torch
import torch.nn as nn

class CustomMambaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super(CustomMambaModel, self).__init__()
        
        # Convolutional Layers to extract spatial features
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Recurrent layer to handle sequences and mimic long-range dependencies
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer for output prediction (e.g., bounding box coordinates)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        
        # Permute back to (batch, sequence, features) for RNN
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        
        # Predict output
        x = self.fc(x)
        return x

# Example Usage
# Define the model
input_dim = 3    # e.g., x, y, z for LiDAR data
hidden_dim = 64  # Number of features in hidden layers
output_dim = 7   # e.g., bounding box (x, y, z, dx, dy, dz, rotation)
num_layers = 4   # Number of GRU layers

model = CustomMambaModel(input_dim, hidden_dim, output_dim, num_layers)

# Dummy input
batch_size = 8
sequence_length = 100  # Length of each sequence (e.g., number of LiDAR points)
x = torch.randn(batch_size, sequence_length, input_dim)

# Forward pass
output = model(x)
print("Output shape:", output.shape)  # Expecting (batch_size, sequence_length, output_dim)
