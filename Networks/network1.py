import torch
import torch.nn as nn

class PointTransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointTransformerBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(output_dim, num_heads=4)
        self.feed_forward = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # Self-attention layer
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.norm1(attn_output)  # Residual connection and normalization
        # Feed-forward layer
        ff_output = self.feed_forward(x)
        x = x + self.norm2(ff_output)  # Residual connection and normalization
        return x

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=3, input_dim=3, feature_dim=64):
        super(ObjectDetectionModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, feature_dim)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([PointTransformerBlock(feature_dim, feature_dim) for _ in range(4)])

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

    def forward(self, x):
        # Initial embedding
        x = self.embedding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Classification and regression heads
        class_output = self.classifier(x.mean(dim=1))  # Aggregating features across points
        bbox_output = self.regressor(x.mean(dim=1))
        
        return class_output, bbox_output
