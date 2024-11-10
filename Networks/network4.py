import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Shape: (1, max_len, dim)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class PointTransformerBlockWithPE(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(PointTransformerBlockWithPE, self).__init__()
        self.self_attention = nn.MultiheadAttention(output_dim, num_heads)
        self.positional_encoding = PositionalEncoding(output_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Add positional encoding to the input
        x_pe = self.positional_encoding(x)
        
        # Transpose for multi-head attention: (batch_size, N, embed_dim) -> (N, batch_size, embed_dim)
        x_pe = x_pe.permute(1, 0, 2)
        
        # Self-attention layer
        attn_output, _ = self.self_attention(x_pe, x_pe, x_pe)
        
        # Transpose back: (N, batch_size, embed_dim) -> (batch_size, N, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)
        
        # Residual connection and normalization
        x = x + self.norm1(attn_output)
        
        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = x + self.norm2(ff_output)  # Residual connection and normalization
        return x

def load_pretrained_pointnet():
    # Replace this with the actual code to download and load the pre-trained PointNet
    # This could be from a GitHub repository or a model hub
    pointnet = torch.hub.load('username/repo_name', 'pointnet_model_name', pretrained=True)
    return pointnet


class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=3, feature_dim=64, chunk_size=5000):
        super(ObjectDetectionModel, self).__init__()
        
        # Load the pre-trained PointNet as the embedding layer
        self.pointnet = load_pretrained_pointnet()
        
        # Freeze the PointNet weights
        for param in self.pointnet.parameters():
            param.requires_grad = False
        
        # Check if the output feature dimension matches your feature_dim, adapt if needed
        self.adapt_feature_dim = nn.Linear(self.pointnet.output_dim, feature_dim) if self.pointnet.output_dim != feature_dim else nn.Identity()
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([PointTransformerBlockWithPE(feature_dim, feature_dim) for _ in range(4)])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # x, y, z, width, height, depth, yaw
        )
        
        # Set chunk size
        self.chunk_size = chunk_size

    def forward(self, x):
        batch_size, total_points, input_dim = x.shape
        
        # If the total points exceed the chunk size, process in chunks
        if total_points > self.chunk_size:
            outputs = []
            for start in range(0, total_points, self.chunk_size):
                end = min(start + self.chunk_size, total_points)
                chunk = x[:, start:end, :]
                if chunk.size(1) < self.chunk_size:  # Pad if the last chunk is smaller
                    padding = torch.zeros((batch_size, self.chunk_size - chunk.size(1), input_dim), device=chunk.device)
                    chunk = torch.cat([chunk, padding], dim=1)
                
                # Process the chunk through PointNet and adapt feature dimensions if needed
                chunk = self.pointnet(chunk)
                chunk = self.adapt_feature_dim(chunk)
                
                # Pass through transformer blocks
                for block in self.transformer_blocks:
                    chunk = block(chunk)
                
                outputs.append(chunk)

            # Combine outputs (e.g., using mean pooling)
            x = torch.cat(outputs, dim=1)
            x = x.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1, feature_dim)
        else:
            # Process normally if points are within the chunk size
            x = self.pointnet(x)
            x = self.adapt_feature_dim(x)
            for block in self.transformer_blocks:
                x = block(x)
        
        # Classification and regression heads
        class_output = self.classifier(x.mean(dim=1))  # Aggregating features across points
        bbox_output = self.regressor(x.mean(dim=1))
        
        return class_output, bbox_output
