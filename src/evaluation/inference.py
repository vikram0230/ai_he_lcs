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
from src.utils.helper_functions import HelperUtils
from src.data.dataset_loader import PatientDicomDataset

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_dinov2_attention(attention_maps, save_path='dinov2_attention.png'):
    """Visualize attention maps from DINOv2 transformer layers."""
    num_layers = len(attention_maps)
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for i, attn in enumerate(attention_maps):
        # Average attention across heads
        attn = attn.mean(dim=1)  # Average across attention heads
        # Reshape to 2D grid (assuming square input)
        grid_size = int(np.sqrt(attn.shape[-1]))
        attn_grid = attn[0].reshape(grid_size, grid_size)  # Take first batch
        
        sns.heatmap(attn_grid, ax=axes[i], cmap='viridis')
        axes[i].set_title(f'Layer {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_slice_attention(attention_maps, save_path='slice_attention.png'):
    """Visualize attention maps from slice processor."""
    num_layers = len(attention_maps)
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for i, attn in enumerate(attention_maps):
        # Average attention across heads
        attn = attn.mean(dim=1)  # Average across attention heads
        # Take first batch and first reconstruction
        attn_matrix = attn[0, 0]  # [num_slices, num_slices]
        
        sns.heatmap(attn_matrix, ax=axes[i], cmap='viridis')
        axes[i].set_title(f'Slice Layer {i+1}')
        axes[i].set_xlabel('Slice Index')
        axes[i].set_ylabel('Slice Index')
    
    plt.tight_layout()
    plt.savefig(save_path)
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
    visualize_dinov2_attention(attention_maps['attention'])
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

def run_inference(model, test_dataset, device, mlflow_run, helper):
    """Run inference on the test dataset."""
    metrics = helper.run_inference_and_log_metrics(model, test_dataset, device, mlflow_run, logging=True)
    
    print("\nInference complete!")
    print("\nTest Metrics Summary:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics['auc']:.4f}")

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
                run_inference(model, test_dataset, device, mlflow.active_run(), helper)
                
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

if __name__ == "__main__":
    main() 