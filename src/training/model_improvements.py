import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
import seaborn as sns

class ModelImprovements:
    def __init__(self, model: nn.Module, config: Dict):
        """
        Initialize model improvement techniques.
        
        Args:
            model: The base model to improve
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention maps from the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention maps tensor
        """
        self.model.eval()
        with torch.no_grad():
            # Get attention weights from the last layer
            attention_weights = self.model.transformer.get_attention_weights(x)
            return attention_weights
    
    def generate_gradcam(self, 
                        x: torch.Tensor,
                        target_layer: str = 'last_layer') -> np.ndarray:
        """
        Generate Grad-CAM visualization.
        
        Args:
            x: Input tensor
            target_layer: Layer to generate Grad-CAM for
            
        Returns:
            Grad-CAM heatmap
        """
        self.model.eval()
        x.requires_grad = True
        
        # Forward pass
        output = self.model(x)
        
        # Get gradients
        output.backward()
        
        # Get feature maps and gradients
        if target_layer == 'last_layer':
            feature_maps = self.model.transformer.last_hidden_state
            gradients = x.grad
        else:
            # Get specific layer features and gradients
            feature_maps = self.model.transformer.get_layer_features(target_layer)
            gradients = x.grad
        
        # Generate heatmap
        weights = torch.mean(gradients, dim=(2, 3))
        cam = torch.zeros(feature_maps.shape[2:], dtype=torch.float32)
        
        for i, w in enumerate(weights[0]):
            cam += w * feature_maps[0, i, :, :]
        
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                          size=x.shape[2:],
                          mode='bilinear',
                          align_corners=False)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def visualize_attention(self,
                          x: torch.Tensor,
                          save_path: str = None) -> None:
        """
        Visualize attention maps for the input.
        
        Args:
            x: Input tensor
            save_path: Path to save visualization
        """
        attention_maps = self.get_attention_maps(x)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(x[0].cpu().permute(1, 2, 0))
        plt.title('Original Image')
        
        # Attention map
        plt.subplot(1, 3, 2)
        plt.imshow(attention_maps[0].cpu(), cmap='jet')
        plt.title('Attention Map')
        
        # Overlay
        plt.subplot(1, 3, 3)
        overlay = self._create_attention_overlay(x[0], attention_maps[0])
        plt.imshow(overlay)
        plt.title('Attention Overlay')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def _create_attention_overlay(self,
                                image: torch.Tensor,
                                attention: torch.Tensor) -> np.ndarray:
        """Create overlay of attention map on original image."""
        # Convert image to numpy
        image = image.cpu().permute(1, 2, 0).numpy()
        attention = attention.cpu().numpy()
        
        # Normalize attention
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # Create overlay
        overlay = 0.6 * image + 0.4 * heatmap
        overlay = overlay / overlay.max()
        
        return overlay
    
    def estimate_uncertainty(self,
                           x: torch.Tensor,
                           num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate prediction uncertainty using Monte Carlo dropout.
        
        Args:
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Mean predictions and uncertainty estimates
        """
        self.model.train()  # Enable dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty
    
    def analyze_feature_importance(self,
                                 dataloader: torch.utils.data.DataLoader,
                                 feature_names: List[str]) -> Dict[str, float]:
        """
        Analyze feature importance using permutation importance.
        
        Args:
            dataloader: DataLoader containing evaluation data
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_scores = {}
        baseline_score = self._evaluate_model(dataloader)
        
        for feature_idx, feature_name in enumerate(feature_names):
            # Permute feature
            permuted_score = self._evaluate_with_permuted_feature(
                dataloader, feature_idx
            )
            
            # Calculate importance
            importance = baseline_score - permuted_score
            importance_scores[feature_name] = importance
        
        return importance_scores
    
    def _evaluate_model(self,
                       dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate model performance."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(inputs)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        return total_correct / total_samples
    
    def _evaluate_with_permuted_feature(self,
                                      dataloader: torch.utils.data.DataLoader,
                                      feature_idx: int) -> float:
        """Evaluate model with permuted feature."""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Permute feature
                permuted_inputs = inputs.clone()
                permuted_inputs[:, :, :, feature_idx] = torch.randn_like(
                    permuted_inputs[:, :, :, feature_idx]
                )
                
                outputs = self.model(permuted_inputs)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        return total_correct / total_samples
    
    def visualize_feature_importance(self,
                                   importance_scores: Dict[str, float],
                                   save_path: str = None) -> None:
        """
        Visualize feature importance scores.
        
        Args:
            importance_scores: Dictionary of feature importance scores
            save_path: Path to save visualization
        """
        plt.figure(figsize=(10, 6))
        
        # Sort features by importance
        features = list(importance_scores.keys())
        scores = list(importance_scores.values())
        sorted_idx = np.argsort(scores)
        
        plt.barh(range(len(features)), [scores[i] for i in sorted_idx])
        plt.yticks(range(len(features)), [features[i] for i in sorted_idx])
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def create_interpretability_report(self,
                                     x: torch.Tensor,
                                     save_dir: str) -> None:
        """
        Create a comprehensive interpretability report.
        
        Args:
            x: Input tensor
            save_dir: Directory to save report
        """
        # Generate attention visualization
        self.visualize_attention(x, f"{save_dir}/attention.png")
        
        # Generate Grad-CAM
        gradcam = self.generate_gradcam(x)
        plt.figure(figsize=(8, 8))
        plt.imshow(gradcam, cmap='jet')
        plt.title('Grad-CAM Visualization')
        plt.savefig(f"{save_dir}/gradcam.png")
        plt.close()
        
        # Estimate uncertainty
        mean_pred, uncertainty = self.estimate_uncertainty(x)
        
        # Create uncertainty plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mean_pred[0].cpu(), cmap='viridis')
        plt.title('Mean Prediction')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(uncertainty[0].cpu(), cmap='hot')
        plt.title('Uncertainty')
        plt.colorbar()
        
        plt.savefig(f"{save_dir}/uncertainty.png")
        plt.close()
        
        # Save numerical results
        results = {
            'mean_prediction': mean_pred.mean().item(),
            'uncertainty': uncertainty.mean().item(),
            'max_uncertainty': uncertainty.max().item(),
            'min_uncertainty': uncertainty.min().item()
        }
        
        with open(f"{save_dir}/results.txt", 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n") 