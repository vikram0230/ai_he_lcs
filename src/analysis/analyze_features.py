import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from training.dinov2_classifier import DinoVisionTransformerCancerPredictor
from training.dataset_loader import PatientDicomDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import submitit
import datetime

def collate_fn(batch):
    max_reconstructions = max(x[0].size(0) for x in batch)
    max_slices = max(x[0].size(1) for x in batch)
    
    images = []
    positions = []
    labels = []
    attention_masks = []
    
    for img, pos, label in batch:
        # Create 2D attention mask
        recon_mask = torch.zeros((max_reconstructions, max_slices))
        
        # Set 1s for actual data positions
        recon_mask[:img.size(0), :img.size(1)] = 1
        
        # Pad image and positions
        if img.size(0) < max_reconstructions or img.size(1) < max_slices:
            # Pad image
            padded_img = torch.zeros((
                max_reconstructions,
                max_slices,
                *img.shape[2:]  # channels, height, width
            ))
            # Copy actual data
            padded_img[:img.size(0), :img.size(1)] = img
            img = padded_img
            
            # Pad positions
            padded_pos = torch.zeros(max_slices)
            padded_pos[:pos.size(0)] = pos
            pos = padded_pos
        
        images.append(img)
        positions.append(pos)
        labels.append(label)
        attention_masks.append(recon_mask)
    
    return {
        'images': torch.stack(images),           # [batch, max_recon, max_slices, C, H, W]
        'positions': torch.stack(positions),     # [batch, max_slices]
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_masks)  # [batch, max_recon, max_slices]
    }

def visualize_features(model, dataloader, device, num_samples=5):
    """Visualize DINOv2 features for a few samples."""
    model.eval()
    
    # Create output directory
    os.makedirs('feature_analysis', exist_ok=True)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_samples:
            break
            
        inputs = batch['images'].to(device)
        labels = batch['labels'].float()
        
        # Extract features
        feature_dict = model.extract_dinov2_features(inputs)
        features = feature_dict['raw_features']
        stats = feature_dict['feature_stats']
        patterns = feature_dict['feature_patterns']
        
        # Print statistics
        print(f"\nSample {batch_idx + 1} Statistics:")
        print(f"Label: {labels[0].item()}")
        print(f"Feature mean: {stats['mean']:.4f}")
        print(f"Feature std: {stats['std']:.4f}")
        print(f"Feature range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Plot feature patterns
        plt.figure(figsize=(15, 10))
        
        # Plot first 10 feature dimensions across slices
        plt.subplot(2, 2, 1)
        sns.heatmap(patterns.cpu().numpy(), cmap='viridis')
        plt.title('Feature Patterns Across Slices (First 10 Dimensions)')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Slice Index')
        
        # Plot feature distribution
        plt.subplot(2, 2, 2)
        sns.histplot(features.cpu().numpy().flatten(), bins=50)
        plt.title('Feature Value Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Count')
        
        # Plot feature correlation matrix (first 20 dimensions)
        plt.subplot(2, 2, 3)
        feature_corr = np.corrcoef(features[0, 0, :, :20].cpu().numpy().T)
        sns.heatmap(feature_corr, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix (First 20 Dimensions)')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Dimension')
        
        # Plot feature variance across slices
        plt.subplot(2, 2, 4)
        feature_var = features[0, 0].var(dim=0).cpu().numpy()
        plt.bar(range(len(feature_var)), feature_var)
        plt.title('Feature Variance Across Slices')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Variance')
        
        plt.tight_layout()
        plt.savefig(f'feature_analysis/sample_{batch_idx + 1}_label_{labels[0].item()}.png')
        plt.close()
        
        # Save raw features for further analysis
        np.save(f'feature_analysis/sample_{batch_idx + 1}_features.npy', features.cpu().numpy())
        np.save(f'feature_analysis/sample_{batch_idx + 1}_label.npy', labels.cpu().numpy())

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = DinoVisionTransformerCancerPredictor()
    model = model.to(device)
    
    # Load a small dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PatientDicomDataset(
        root_dir='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_test_data',
        labels_file='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_actual.csv',
        transform=transform,
        patient_scan_count=10  # Small subset for analysis
    )
    
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    
    # Visualize features
    visualize_features(model, dataloader, device)

if __name__ == "__main__":
    # Create submitit executor
    executor = submitit.AutoExecutor(folder="logs/submitit_logs")
    
    # Configure the job
    executor.update_parameters(
        name="feature_analysis",
        slurm_partition="batch_gpu",
        slurm_gres="gpu:1",
        slurm_time="2:00:00",
        slurm_mem="32G",
        slurm_cpus_per_task=4,
        slurm_array_parallelism=1,
    )
    
    # Submit the job
    job = executor.submit(main)
    
    print(f"Submitted job {job.job_id}")
    print(f"Job output will be in: {job.paths.stdout}")
    print(f"Job error will be in: {job.paths.stderr}")