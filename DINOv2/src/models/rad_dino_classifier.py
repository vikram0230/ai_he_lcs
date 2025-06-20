import sys
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import math
from src.utils.common_components import AnatomicalPositionEncoding, ReconstructionAttention
import numpy as np


class RadDinoClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
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
        
        if config['model'].get('apply_positional_encoding', False):
            self.position_encoding = AnatomicalPositionEncoding(config['model']['position_encoding_dim'])
            print(f"Position encoding device: {next(self.position_encoding.parameters()).device}", flush=True)
        
        # Build slice processor from config
        slice_processor_layers = []
        for layer_config in config['model']['slice_processor']['layers']:
            layer_type = layer_config['type']
            if layer_type == 'conv1d':
                layer = nn.Conv1d(
                    in_channels=feature_dim if layer_config['in_channels'] == 'feature_dim' else layer_config['in_channels'],
                    out_channels=feature_dim if layer_config['out_channels'] == 'feature_dim' else layer_config['out_channels'],
                    kernel_size=layer_config['kernel_size'],
                    padding=layer_config['padding']
                )
            elif layer_type == 'batchnorm1d':
                layer = nn.BatchNorm1d(feature_dim if layer_config['num_features'] == 'feature_dim' else layer_config['num_features'])
            elif layer_type == 'relu':
                layer = nn.ReLU()
            elif layer_type == 'adaptive_avg_pool1d':
                layer = nn.AdaptiveAvgPool1d(layer_config['output_size'])
            slice_processor_layers.append(layer)
        
        self.slice_processor = nn.Sequential(*slice_processor_layers)
        
        # Initialize clinical features processing if enabled
        self.use_clinical_features = config['data'].get('use_clinical_features', False)
        if self.use_clinical_features:
            print("\nInitializing clinical features processing...", flush=True)
            clinical_feature_dim = config['data']['clinical_features']['feature_dim']
            
            # Build clinical processor from config
            clinical_processor_layers = []
            for layer_config in config['model']['clinical_processor']['layers']:
                layer_type = layer_config['type']
                if layer_type == 'linear':
                    layer = nn.Linear(
                        in_features=clinical_feature_dim if layer_config['in_features'] == 'clinical_feature_dim' else layer_config['in_features'],
                        out_features=layer_config['out_features']
                    )
                elif layer_type == 'layernorm':
                    layer = nn.LayerNorm(layer_config['normalized_shape'])
                elif layer_type == 'relu':
                    layer = nn.ReLU()
                elif layer_type == 'dropout':
                    layer = nn.Dropout(layer_config['p'])
                clinical_processor_layers.append(layer)
            
            self.clinical_processor = nn.Sequential(*clinical_processor_layers)
            # Adjust predictor input dimension to include clinical features
            predictor_input_dim = config['model']['feature_dim'] + 128
        else:
            predictor_input_dim = config['model']['feature_dim']
        
        # Build predictor layers from config
        predictor_layers = []
        prev_dim = predictor_input_dim
        
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
        
        # self.reconstruction_attention = ReconstructionAttention(config['model']['feature_dim'])

    def forward(self, x, slice_positions, attention_mask=None, clinical_features=None):
        """
        Args:
            x: Input tensor (batch_size, num_reconstructions, num_slices, channels, height, width)
            slice_positions: Z-coordinates of slices (batch_size, num_slices)
            attention_mask: Attention mask for padding (batch_size, num_reconstructions, num_slices)
            clinical_features: Optional clinical features tensor (batch_size, num_clinical_features)
        """
        # Print device information for debugging
        print(f"Input device: {x.device}", flush=True)
        print(f"Model device: {next(self.parameters()).device}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available() and x.device.type == 'cuda':
            print(f"Current CUDA device: {torch.cuda.current_device()}", flush=True)
            print(f"Device name: {torch.cuda.get_device_name(x.device)}", flush=True)
        
        B, R, S, C, H, W = x.shape  # batch, reconstructions, slices, channels, height, width
        
        model_device = next(self.parameters()).device
        x = x.to(model_device).float()  # Force float32
        slice_positions = slice_positions.to(model_device)
        
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
        
        # Remove the hardcoded variable and use config
        apply_positional_encoding = self.config['model'].get('apply_positional_encoding', False)
        
        # Process single reconstruction
        slice_features = features[:, 0]  # [B, S, feature_dim] - take first (and only) reconstruction
        
        # Add positional encoding if enabled
        if apply_positional_encoding:
            positions = slice_positions.unsqueeze(-1)
            position_encoding = self.position_encoding(positions)
            slice_features = slice_features + position_encoding
        
        # Transpose to [B, feature_dim, S] for Conv1d
        slice_features = slice_features.transpose(1, 2)
        processed_slice_sequence = self.slice_processor(slice_features)
        # Remove the extra dimension from adaptive pooling
        final_features = processed_slice_sequence.squeeze(-1)  # [B, feature_dim]
        
        # Process clinical features if enabled and provided
        if self.use_clinical_features and clinical_features is not None:
            clinical_features = clinical_features.to(model_device)
            clinical_embeddings = self.clinical_processor(clinical_features)
            final_features = torch.cat([final_features, clinical_embeddings], dim=1)
            
        print(f"final_features shape: {final_features.shape}", flush=True)
        
        # Get prediction for the entire scan
        return self.predictor(final_features)  # [B, 1]


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
            
            # Convert to PIL images for RAD-DINO processor
            pil_images = []
            for img in x_reshaped:
                # Convert to numpy and transpose to (H, W, C)
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                # Normalize to 0-255 range and convert to uint8
                img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            
            # Process through DINOv2 processor
            inputs = self.processor(images=pil_images, return_tensors="pt").to(next(self.parameters()).device)
            
            # Get attention maps from RAD-DINO transformer
            rad_dino_attention_maps = []
            # First get the hidden states from the transformer
            outputs = self.transformer(**inputs)
            hidden_states = outputs.last_hidden_state
            
            for layer in self.transformer.encoder.layer:
                # Get attention weights from the layer
                attention_output = layer.attention(hidden_states)
                rad_dino_attention_maps.append(attention_output.attn_weights)
            
            # Process through the model to get slice and reconstruction attention
            features = hidden_states
            features = features.view(B, R, S, -1)  # [B, R, S, feature_dim]
            
            # Get slice attention maps
            slice_attention_maps = []
            for r in range(R):
                slice_features = features[:, r]  # [B, S, feature_dim]
                # Get attention maps from slice processor
                for layer in self.slice_processor:
                    # Compute attention weights for slice processor
                    qkv = layer(slice_features.unsqueeze(1))  # [B, 1, feature_dim]
                    qkv = qkv.squeeze(1)  # [B, feature_dim]
                    
                    # Compute attention scores
                    attn = (qkv @ slice_features) * (1.0 / math.sqrt(slice_features.size(-1)))
                    attn = attn.softmax(dim=-1)
                    slice_attention_maps.append(attn)
            
            # Get reconstruction attention weights
            recon_features = []
            for r in range(R):
                slice_features = features[:, r]  # [B, S, feature_dim]
                recon_feature = slice_features.mean(dim=1)  # [B, feature_dim]
                recon_features.append(recon_feature)
            
            recon_features = torch.stack(recon_features, dim=1)  # [B, R, feature_dim]
            # reconstruction_attention = self.reconstruction_attention(recon_features)
            
            return {
                'attention': rad_dino_attention_maps,
                'slice_attention': slice_attention_maps,
                # 'reconstruction_attention': reconstruction_attention
            }