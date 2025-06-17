import torch
import torch.nn as nn


class AnatomicalPositionEncoding(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.position_mlp = nn.Sequential(
            nn.Linear(1, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )
    
    def forward(self, z_positions):
        # z_positions: (batch_size, num_slices, 1)
        return self.position_mlp(z_positions)  # Output: (batch_size, num_slices, feature_dim)


class ReconstructionAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, recon_features):
        # recon_features: [batch, num_recon, features]
        attention_scores = self.attention(recon_features)  # [batch, num_recon, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # normalize across reconstructions
        return attention_weights 