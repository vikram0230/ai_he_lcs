import sys
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import math
from common_components import AnatomicalPositionEncoding, ReconstructionAttention
import numpy as np


class RadDinoClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        print("\nInitializing RadDinoClassifier...", flush=True)
        
        # Load RAD-DINO model
        self.model_name = "microsoft/rad-dino"
        print(f"Loading RAD-DINO model: {self.model_name}", flush=True)
        self.transformer = AutoModel.from_pretrained(self.model_name)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        
        # Force model to float32 to avoid xFormers issues
        self.transformer = self.transformer.float()
        print(f"Transformer device: {next(self.transformer.parameters()).device}", flush=True)
        
        # Set feature dimensions based on model
        feature_dim = self.transformer.config.hidden_size
        config['model']['feature_dim'] = feature_dim
        config['model']['position_encoding_dim'] = feature_dim
        
        # Freeze all parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
        print(f"Running with frozen RAD-DINO", flush=True)

        # Unfreeze the last few layers if specified in config
        if config['model'].get('unfreeze_last_layers', False):
            for name, param in self.transformer.named_parameters():
                if any(f'layer.{i}' in name for i in config['model'].get('unfreeze_layers', [9, 10, 11])):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f"Running with unfrozen last few layers of RAD-DINO", flush=True)
        
        self.position_encoding = AnatomicalPositionEncoding(config['model']['position_encoding_dim'])
        print(f"Position encoding device: {next(self.position_encoding.parameters()).device}", flush=True)
        
        # Process sequence of slice features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model']['feature_dim'],
            nhead=config['model']['transformer']['nhead'],
            dim_feedforward=config['model']['transformer']['dim_feedforward'],
            dropout=config['model']['transformer']['dropout'],
            batch_first=True  # Important for shape handling
        )
        self.slice_processor = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['model']['transformer']['num_layers']
        )
        
        # Build predictor layers from config
        predictor_layers = []
        prev_dim = config['model']['feature_dim']
        
        # Process all layers except the last one
        for i, dim in enumerate(config['model']['predictor']['layers'][:-1]):
            predictor_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(config['model']['predictor']['dropout'])
            ])
            prev_dim = dim
        
        # Add the final layer without normalization, activation, and dropout
        final_dim = config['model']['predictor']['layers'][-1]
        predictor_layers.append(nn.Linear(prev_dim, final_dim))
        
        self.predictor = nn.Sequential(*predictor_layers)
        
        self.reconstruction_attention = ReconstructionAttention(config['model']['feature_dim'])

    def forward(self, x, slice_positions, attention_mask=None):
        """
        Args:
            x: Input tensor (batch_size, num_reconstructions, num_slices, channels, height, width)
            slice_positions: Z-coordinates of slices (batch_size, num_slices)
            attention_mask: Attention mask for padding (batch_size, num_reconstructions, num_slices)
        """
        # Print device information for debugging
        print(f"Input device: {x.device}", flush=True)
        print(f"Model device: {next(self.parameters()).device}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available() and x.device.type == 'cuda':
            print(f"Current CUDA device: {torch.cuda.current_device()}", flush=True)
            print(f"Device name: {torch.cuda.get_device_name(x.device)}", flush=True)
        
        B, R, S, C, H, W = x.shape  # batch, reconstructions, slices, channels, height, width
        
        # Ensure input is on the same device as the model and in float32
        model_device = next(self.parameters()).device
        x = x.to(model_device).float()  # Force float32
        slice_positions = slice_positions.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        
        # Reshape to batch all images together
        x_reshaped = x.view(B * R * S, C, H, W)
        
        # Process all images in one go
        try:
            # Convert to PIL images for RAD-DINO processor
            pil_images = []
            for img in x_reshaped:
                # Convert to numpy and transpose to (H, W, C)
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                # Normalize to 0-255 range and convert to uint8
                img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            inputs = self.processor(images=pil_images, return_tensors="pt").to(model_device)
            features = self.transformer(**inputs).last_hidden_state  # [B*R*S, num_patches+1, hidden_size]
            features = features[:, 0, :]  # [B*R*S, hidden_size]  # CLS token only
            features = features.view(B, R, S, -1)  # [B, R, S, hidden_size]
        except Exception as e:
            print(f"Error in transformer forward pass: {e}", flush=True)
            print(f"Input shape: {x_reshaped.shape}, Device: {x_reshaped.device}", flush=True)
            print(f"Model device: {next(self.parameters()).device}", flush=True)
            raise e
        
        apply_positional_encoding = True
        if apply_positional_encoding:
            # Process each reconstruction separately
            recon_features = []
            for r in range(R):
                slice_features = features[:, r]  # [B, S, feature_dim]
                
                # Add positional encoding
                positions = slice_positions.unsqueeze(-1)
                position_encoding = self.position_encoding(positions)
                slice_features_with_position_encoding = slice_features + position_encoding
                
                # Handle masking
                if attention_mask is not None:
                    mask = attention_mask[:, r]  # [B, S]
                    key_padding_mask = ~mask.bool()  # For transformer
                else:
                    key_padding_mask = None
                
                # Process through slice transformer
                processed_slice_sequence = self.slice_processor(
                    slice_features_with_position_encoding,
                    src_key_padding_mask=key_padding_mask
                )
                
                # Attention pooling with masking
                if attention_mask is not None:
                    mask = mask.unsqueeze(1)  # [B, 1, S]
                    processed_slice_sequence = processed_slice_sequence * mask.unsqueeze(-1)
                
                # Pool features
                recon_feature = processed_slice_sequence.mean(dim=1)  # [B, feature_dim]
                recon_features.append(recon_feature)
            
            # Stack and apply reconstruction attention
            recon_features = torch.stack(recon_features, dim=1)  # [B, R, feature_dim]
            
            # Apply reconstruction attention
            if attention_mask is not None:
                recon_mask = (attention_mask.sum(dim=-1) > 0).float()  # [B, R]
                attention_weights = self.reconstruction_attention(recon_features)
                attention_weights = attention_weights * recon_mask.unsqueeze(-1).unsqueeze(-1)
                attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
            else:
                attention_weights = self.reconstruction_attention(recon_features)
            
            combined_features = torch.sum(attention_weights * recon_features, dim=[1,2])
            final_features = combined_features
        
        else:
            # Reshape features from [1, 1, 168, 384] to [168, 384]
            final_features = features.squeeze(0).squeeze(0)  # Remove first two dimensions
            
        print(f"final_features shape: {final_features.shape}", flush=True)
        return self.predictor(final_features)

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

    def extract_features(self, x):
        """
        Extract and analyze RAD-DINO features for visualization.
        
        Args:
            x: Input tensor of shape (batch_size, num_reconstructions, num_slices, channels, height, width)
            
        Returns:
            Dictionary containing:
            - raw_features: Raw RAD-DINO features
            - feature_stats: Statistics about the features
            - feature_visualization: Visualization of feature patterns
        """
        self.eval()
        with torch.no_grad():
            B, R, S, C, H, W = x.shape
            
            # Reshape to batch all images together
            x_reshaped = x.view(B * R * S, C, H, W)
            
            # Get RAD-DINO features
            # Process in smaller batches to avoid memory issues
            batch_size = 32
            features_list = []
            
            for i in range(0, B * R * S, batch_size):
                batch_x = x_reshaped[i:i + batch_size]
                # Convert to PIL images for RAD-DINO processor
                pil_images = []
                for img in batch_x:
                    # Convert to numpy and transpose to (H, W, C)
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    # Normalize to 0-255 range and convert to uint8
                    img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(img_np))
                inputs = self.processor(images=pil_images, return_tensors="pt").to(next(self.parameters()).device)
                batch_features = self.transformer(**inputs).last_hidden_state
                features_list.append(batch_features)
            
            features = torch.cat(features_list, dim=0)
            features = features.view(B, R, S, -1)  # [B, R, S, feature_dim]
            
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

    def get_attention_maps(self, x):
        """
        Extract attention maps from the RAD-DINO transformer.
        
        Args:
            x: Input tensor of shape (batch_size, num_reconstructions, num_slices, channels, height, width)
            
        Returns:
            Dictionary containing:
            - rad_dino_attention: Attention maps from RAD-DINO transformer layers
            - slice_attention: Attention maps from slice processor
            - reconstruction_attention: Attention weights for reconstructions
        """
        self.eval()
        with torch.no_grad():
            B, R, S, C, H, W = x.shape
            
            # Reshape to batch all images together
            x_reshaped = x.view(B * R * S, C, H, W)
            
            # Get attention maps from RAD-DINO transformer
            rad_dino_attention_maps = []
            for layer in self.transformer.encoder.layer:
                # Get attention weights from the layer
                attention_output = layer.attention(x_reshaped)
                rad_dino_attention_maps.append(attention_output.attn_weights)
            
            # Process through the model to get slice and reconstruction attention
            features = self.transformer(x_reshaped).last_hidden_state
            features = features.view(B, R, S, -1)  # [B, R, S, feature_dim]
            
            # Get slice attention maps
            slice_attention_maps = []
            for r in range(R):
                slice_features = features[:, r]  # [B, S, feature_dim]
                # Get attention maps from slice processor
                for layer in self.slice_processor.layers:
                    # Compute attention weights for slice processor
                    qkv = layer.self_attn.qkv(slice_features)  # [B, S, 3*num_heads*head_dim]
                    # Calculate head_dim for slice processor
                    head_dim = layer.self_attn.qkv.out_features // (3 * layer.self_attn.num_heads)
                    # Split into Q, K, V
                    qkv = qkv.reshape(B, -1, 3, layer.self_attn.num_heads, head_dim)
                    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, S, head_dim]
                    q, k, v = qkv.unbind(0)  # Each is [B, num_heads, S, head_dim]
                    
                    # Compute attention scores
                    attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                    attn = attn.softmax(dim=-1)
                    slice_attention_maps.append(attn)
            
            # Get reconstruction attention weights
            recon_features = []
            for r in range(R):
                slice_features = features[:, r]  # [B, S, feature_dim]
                recon_feature = slice_features.mean(dim=1)  # [B, feature_dim]
                recon_features.append(recon_feature)
            
            recon_features = torch.stack(recon_features, dim=1)  # [B, R, feature_dim]
            reconstruction_attention = self.reconstruction_attention(recon_features)
            
            return {
                'attention': rad_dino_attention_maps,
                'slice_attention': slice_attention_maps,
                'reconstruction_attention': reconstruction_attention
            }