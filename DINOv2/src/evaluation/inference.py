import os
import torch
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dinov2_classifier import DinoVisionTransformerCancerPredictor
from utils.helpers import HelperUtils
from src.data.dataset_loader import PatientDicomDataset

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_dinov2_attention(attention_maps, input_image=None, save_path='dinov2_attention.png'):
    """Visualize attention maps from DINOv2 transformer layers without averaging heads."""
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].shape[1]  # Number of attention heads
    
    # Calculate grid dimensions
    grid_cols = num_heads + 1  # +1 for input image
    grid_rows = num_layers
    
    # Create figure with subplots
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5*grid_cols, 5*grid_rows))
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx, attn in enumerate(attention_maps):
        # Get grid size for this layer
        grid_size = int(np.sqrt(attn.shape[-1]))
        
        # Plot input image in first column
        if input_image is not None:
            axes[layer_idx, 0].imshow(input_image[0].cpu().numpy().transpose(1, 2, 0))
            axes[layer_idx, 0].set_title('Input Image')
            axes[layer_idx, 0].axis('off')
        
        # Plot each attention head
        for head_idx in range(num_heads):
            attn_grid = attn[0, head_idx].reshape(grid_size, grid_size)
            sns.heatmap(attn_grid, ax=axes[layer_idx, head_idx + 1], cmap='viridis')
            axes[layer_idx, head_idx + 1].set_title(f'Head {head_idx + 1}')
            axes[layer_idx, head_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_slice_attention(attention_maps, save_path='slice_attention.png'):
    """Visualize attention maps from slice processor without averaging heads."""
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].shape[1]  # Number of attention heads
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(5*num_heads, 5*num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx, attn in enumerate(attention_maps):
        # Plot each attention head
        for head_idx in range(num_heads):
            # Take first batch and first reconstruction
            attn_matrix = attn[0, head_idx]  # [num_slices, num_slices]
            
            sns.heatmap(attn_matrix, ax=axes[layer_idx, head_idx], cmap='viridis')
            axes[layer_idx, head_idx].set_title(f'Layer {layer_idx + 1}, Head {head_idx + 1}')
            axes[layer_idx, head_idx].set_xlabel('Slice Index')
            axes[layer_idx, head_idx].set_ylabel('Slice Index')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_reconstruction_attention(attention_weights, save_path='reconstruction_attention.png'):
    """Visualize attention weights for reconstructions."""
    plt.figure(figsize=(8, 4))
    # Take first batch
    weights = attention_weights[0, :, 0, 0].cpu().numpy()
    plt.bar(range(len(weights)), weights)
    plt.title('Reconstruction Attention Weights')
    plt.xlabel('Reconstruction Index')
    plt.ylabel('Attention Weight')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_attention_visualization(model: DinoVisionTransformerCancerPredictor, test_dataset: PatientDicomDataset, device: torch.device):
    """Run attention map visualization for a single scan."""
    # Get the single test scan
    patient_tensor, slice_positions, true_label = test_dataset[0]
    
    # Add batch dimension
    patient_tensor = patient_tensor.unsqueeze(0).to(device)
    slice_positions = slice_positions.unsqueeze(0).to(device)
    
    # Get attention maps
    print("\nExtracting attention maps...")
    attention_maps = model.get_attention_maps(patient_tensor)
    
    # Visualize attention maps
    print("\nVisualizing attention maps...")
    visualize_dinov2_attention(attention_maps['attention'], patient_tensor)
    visualize_slice_attention(attention_maps['slice_attention'])
    visualize_reconstruction_attention(attention_maps['reconstruction_attention'])
    
    # Get prediction
    with torch.no_grad():
        output = model(patient_tensor, slice_positions)
        prediction = torch.sigmoid(output).item()
    
    print("\nResults:")
    print(f"True Label: {true_label.item():.2f}")
    print(f"Prediction: {prediction:.2f}")
    print("\nAttention maps have been saved as:")
    print("- attention.png")
    print("- slice_attention.png")
    print("- reconstruction_attention.png")


def main():
    # Parse command line arguments
    args = HelperUtils.parse_args(mode='inference')
    
    # Initialize helper utils
    helper = HelperUtils()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Set the MLflow run context to the same run as the model
    with mlflow.start_run(run_id=args.run_id):
        try:
            model = HelperUtils.load_model_from_mlflow(args.run_id, 'best_model')
            model = model.to(device)
            model.eval()
            
            # Set up data transformations from config
            test_transform = HelperUtils.create_transforms(config['transforms']['test'])
            
            # Load test dataset
            test_dataset = PatientDicomDataset(
                config=config,
                is_train=False,
                transform=test_transform
            )
            
            if args.visualize_attention:
                run_attention_visualization(model, test_dataset, device)
            else:
                helper.run_inference_and_log_metrics(model, test_dataset, device, mlflow.active_run(), logging=True)
                
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

if __name__ == "__main__":
    main() 