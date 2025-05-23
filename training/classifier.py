import torch
import math
import torch.nn as nn
from copy import deepcopy

# Download DINOv2 model
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


class AnatomicalPositionEncoding(nn.Module):
    def __init__(self, feature_dim=384):
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
    def __init__(self, feature_dim=384):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, recon_features):
        # recon_features: [batch, num_recon, features]
        print(f"recon_features shape: {recon_features.shape}", flush=True)
        attention_scores = self.attention(recon_features)  # [batch, num_recon, 1]
        print(f"attention_scores shape: {attention_scores.shape}", flush=True)
        attention_weights = torch.softmax(attention_scores, dim=1)  # normalize across reconstructions
        print(f"attention_weights shape: {attention_weights.shape}", flush=True)
        return attention_weights


class DinoVisionTransformerCancerPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = deepcopy(dinov2_vits14)
        
        # Freeze all parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
        print(f"Running with frozen DINOv2", flush=True)

        # Unfreeze the last few layers of DINOv2
        # for name, param in self.transformer.named_parameters():
        #     if 'blocks.11' in name or 'blocks.10' in name or 'blocks.9' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # print(f"Running with unfrozen last few layers of DINOv2", flush=True)
        
        self.position_encoding = AnatomicalPositionEncoding()
        
        # Process sequence of slice features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True  # Important for shape handling
        )
        self.slice_processor = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        
        self.reconstruction_attention = ReconstructionAttention(384)

    def forward(self, x, slice_positions, attention_mask=None):
        """
        Args:
            x: Input tensor (batch_size, num_reconstructions, num_slices, channels, height, width)
            slice_positions: Z-coordinates of slices (batch_size, num_slices)
            attention_mask: Attention mask for padding (batch_size, num_reconstructions, num_slices)
        """
        B, R, S, C, H, W = x.shape  # batch, reconstructions, slices, channels, height, width
        
        # Reshape to batch all images together
        x_reshaped = x.view(B * R * S, C, H, W)
        print(f"x_reshaped shape: {x_reshaped.shape}", flush=True)
        # print(f"x_reshaped: {x_reshaped}", flush=True)
        
        # Process all images in one go
        features = self.transformer(x_reshaped)  # [B*R*S, feature_dim]
        features = features.view(B, R, S, -1)  # [B, R, S, feature_dim]
        features = self.transformer.norm(features)
        print(f"features shape: {features.shape}", flush=True)
        # print(f"features: {features}", flush=True)
        
        # Process each reconstruction separately
        recon_features = []
        for r in range(R):
            slice_features = features[:, r]  # [B, S, feature_dim]
            print(f"slice_features shape: {slice_features.shape}", flush=True)
            # print(f"slice_features: {slice_features}", flush=True)
            
            # Add positional encoding
            positions = slice_positions.unsqueeze(-1)
            position_encoding = self.position_encoding(positions)
            print(f"position_encoding shape: {position_encoding.shape}", flush=True)
            # print(f"position_encoding: {position_encoding}", flush=True)
            slice_features_with_position_encoding = slice_features + position_encoding
            print(f"slice_features_with_position_encoding shape: {slice_features_with_position_encoding.shape}", flush=True)
            # print(f"slice_features_with_position_encoding: {slice_features_with_position_encoding}", flush=True)
            
            # Handle masking
            if attention_mask is not None:
                mask = attention_mask[:, r]  # [B, S]
                key_padding_mask = ~mask.bool()  # For transformer
            else:
                key_padding_mask = None
            print(f"key_padding_mask shape: {key_padding_mask.shape}", flush=True)
            # print(f"key_padding_mask: {key_padding_mask}", flush=True)
            
            # Process through slice transformer
            processed_slice_sequence = self.slice_processor(
                slice_features_with_position_encoding,
                src_key_padding_mask=key_padding_mask
            )
            print(f"processed_slice_sequence shape: {processed_slice_sequence.shape}", flush=True)
            # print(f"processed_slice_sequence: {processed_slice_sequence}", flush=True)
            
            # Attention pooling with masking
            if attention_mask is not None:
                mask = mask.unsqueeze(1)  # [B, 1, S]
                processed_slice_sequence = processed_slice_sequence * mask.unsqueeze(-1)
                print(f"processed_slice_sequence with mask shape: {processed_slice_sequence.shape}", flush=True)
                # print(f"processed_slice_sequence with mask: {processed_slice_sequence}", flush=True)
            
            # Pool features
            recon_feature = processed_slice_sequence.mean(dim=1)  # [B, feature_dim]
            recon_features.append(recon_feature)
        
        # Stack and apply reconstruction attention
        recon_features = torch.stack(recon_features, dim=1)  # [B, R, feature_dim]
        print(f"recon_features shape: {recon_features.shape}", flush=True)
        # print(f"recon_features: {recon_features}", flush=True)
        # Apply reconstruction attention
        if attention_mask is not None:
            # print(f"attention_mask shape: {attention_mask.shape}", flush=True)
            recon_mask = (attention_mask.sum(dim=-1) > 0).float()  # [B, R]
            print(f"recon_mask shape: {recon_mask.shape}", flush=True)
            attention_weights = self.reconstruction_attention(recon_features)
            print(f"attention_weights shape: {attention_weights.shape}", flush=True)
            # print(f"attention_weights: {attention_weights}", flush=True)
            attention_weights = attention_weights * recon_mask.unsqueeze(-1).unsqueeze(-1)
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
            print(f"attention_weights shape after masking: {attention_weights.shape}", flush=True)
        else:
            print(f"recon_features shape: {recon_features.shape}", flush=True)
            attention_weights = self.reconstruction_attention(recon_features)
            print(f"attention_weights shape: {attention_weights.shape}", flush=True)
            # print(f"attention_weights: {attention_weights}", flush=True)
        
        combined_features = torch.sum(attention_weights * recon_features, dim=(1,2))
        print(f"combined_features shape: {combined_features.shape}", flush=True)
        # print(f"combined_features: {combined_features}", flush=True)
        return self.predictor(combined_features)

    def predict_probabilities(self, x):
        """
        Returns probability score for cancer prediction for 1st year.
        
        Args:
            x: Input tensor of shape (batch_size, num_reconstructions, num_slices, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, 1) containing probability score between 0 and 1
            for 1st year cancer prediction
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities

    def extract_dinov2_features(self, x):
        """
        Extract and analyze DINOv2 features for visualization.
        
        Args:
            x: Input tensor of shape (batch_size, num_reconstructions, num_slices, channels, height, width)
            
        Returns:
            Dictionary containing:
            - raw_features: Raw DINOv2 features
            - feature_stats: Statistics about the features
            - feature_visualization: Visualization of feature patterns
        """
        self.eval()
        with torch.no_grad():
            B, R, S, C, H, W = x.shape
            
            # Reshape to batch all images together
            x_reshaped = x.view(B * R * S, C, H, W)
            
            # Get DINOv2 features
            # Process in smaller batches to avoid memory issues
            batch_size = 32
            features_list = []
            
            for i in range(0, B * R * S, batch_size):
                batch_x = x_reshaped[i:i + batch_size]
                batch_features = self.transformer(batch_x)
                features_list.append(batch_features)
            
            features = torch.cat(features_list, dim=0)
            features = features.view(B, R, S, -1)  # [B, R, S, feature_dim]
            features = self.transformer.norm(features)
            
            # Calculate statistics
            feature_stats = {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'min': features.min().item(),
                'max': features.max().item(),
                'feature_dim': features.shape[-1],
                'num_slices': S,
                'num_reconstructions': R
            }
            
            # Get feature patterns (first 10 dimensions)
            feature_patterns = features[0, 0, :, :10]  # First batch, first reconstruction, all slices, first 10 features
            
            return {
                'raw_features': features,
                'feature_stats': feature_stats,
                'feature_patterns': feature_patterns
            }
