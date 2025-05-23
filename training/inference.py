import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset_loader import PatientDicomDataset
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from sklearn.manifold import TSNE

from training.config import collate_fn, ensure_log_dir, load_model_from_mlflow

# def get_slurm_job_id():
#     """Get SLURM job ID from environment variable."""
#     return os.environ.get('SLURM_JOB_ID', 'local')

# def ensure_log_dir():
#     """Create logs directory with SLURM job ID if it doesn't exist."""
#     slurm_id = get_slurm_job_id()
#     log_dir = f'logs/predictions_{slurm_id}'
#     os.makedirs(log_dir, exist_ok=True)
#     return log_dir

# def load_model_from_mlflow(run_id, model_name="best_model"):
#     """Load a model from MLflow."""
#     model_uri = f"runs:/{run_id}/{model_name}"
#     model = mlflow.pytorch.load_model(model_uri)
#     return model

# def collate_fn(batch):
#     max_reconstructions = max(x[0].size(0) for x in batch)
#     max_slices = max(x[0].size(1) for x in batch)
    
#     images = []
#     positions = []
#     labels = []
#     attention_masks = []
    
#     for img, pos, label in batch:
#         # Create 2D attention mask
#         recon_mask = torch.zeros((max_reconstructions, max_slices))
        
#         # Set 1s for actual data positions
#         recon_mask[:img.size(0), :img.size(1)] = 1
        
#         # Pad image and positions
#         if img.size(0) < max_reconstructions or img.size(1) < max_slices:
#             # Pad image
#             padded_img = torch.zeros((
#                 max_reconstructions,
#                 max_slices,
#                 *img.shape[2:]  # channels, height, width
#             ))
#             # Copy actual data
#             padded_img[:img.size(0), :img.size(1)] = img
#             img = padded_img
            
#             # Pad positions
#             padded_pos = torch.zeros(max_slices)
#             padded_pos[:pos.size(0)] = pos
#             pos = padded_pos
        
#         # Verify shapes
#         assert img.size(1) == pos.size(0), \
#             f"Position shape {pos.shape} doesn't match image slices {img.size(1)}"
#         assert recon_mask.size() == (max_reconstructions, max_slices), \
#             f"Mask shape {recon_mask.size()} doesn't match expected {(max_reconstructions, max_slices)}"
        
#         images.append(img)
#         positions.append(pos)
#         labels.append(label)
#         attention_masks.append(recon_mask)
    
#     return {
#         'images': torch.stack(images),
#         'positions': torch.stack(positions),
#         'labels': torch.stack(labels),
#         'attention_mask': torch.stack(attention_masks)
#     }

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various classification metrics."""
    metrics = {}
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics for year 1 prediction only
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
    metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    metrics['confusion_matrix'] = cm
    
    return metrics

def plot_confusion_matrix(metrics, save_path):
    """Plot confusion matrix for year 1 prediction."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Year 1 Cancer Prediction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create log directory
    log_dir = ensure_log_dir()
    print(f"Storing outputs in: {log_dir}")
    
    # Load model from MLflow
    run_id = '6c307bbdae1e41128b1ff7e7ba8eaf2d'
    
    # Set the MLflow run context to the same run as the model
    with mlflow.start_run(run_id=run_id):
        model = load_model_from_mlflow(run_id)
        model = model.to(device)
        model.eval()
        
        # Set up data transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load test dataset
        test_dataset = PatientDicomDataset(
            root_dir='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_test_data',
            labels_file='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_actual.csv',
            transform=transform,
            patient_scan_count=50
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            collate_fn=collate_fn, 
            shuffle=False
        )
        
        # Initialize lists to store predictions and true labels
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        # all_patient_ids = []
        # all_years = []
        all_embeddings = []
        
        # Perform inference
        print("Performing inference on test data...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                inputs = batch['images'].to(device)
                positions = batch['positions'].to(device)
                labels = batch['labels'].float().to(device)
                attention_masks = batch['attention_mask'].to(device)
                
                # Get model predictions and embeddings
                outputs = model(inputs, positions, attention_masks)
                probabilities = torch.sigmoid(outputs)
                
                # Get embeddings before the predictor layer
                B, R, S, C, H, W = inputs.shape
                x_reshaped = inputs.view(B * R * S, C, H, W)
                features = model.transformer(x_reshaped)
                features = features.view(B, R, S, -1)
                features = model.transformer.norm(features)
                
                # Process each reconstruction
                recon_features = []
                for r in range(R):
                    slice_sequence = features[:, r]
                    positions_encoded = model.position_encoding(positions.unsqueeze(-1))
                    slice_sequence = slice_sequence + positions_encoded
                    
                    if attention_masks is not None:
                        mask = attention_masks[:, r]
                        key_padding_mask = ~mask.bool()
                    else:
                        key_padding_mask = None
                    
                    processed_sequence = model.slice_processor(
                        slice_sequence,
                        src_key_padding_mask=key_padding_mask
                    )
                    
                    if attention_masks is not None:
                        mask = mask.unsqueeze(1)
                        processed_sequence = processed_sequence * mask.unsqueeze(-1)
                    
                    recon_feature = processed_sequence.mean(dim=1)
                    recon_features.append(recon_feature)
                
                recon_features = torch.stack(recon_features, dim=1)
                attention_weights = model.reconstruction_attention(recon_features)
                if attention_masks is not None:
                    recon_mask = (attention_masks.sum(dim=-1) > 0).float()
                    attention_weights = attention_weights * recon_mask.unsqueeze(-1).unsqueeze(-1)
                    attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-9)
                
                embeddings = torch.sum(attention_weights * recon_features, dim=(1,2))
                
                # Store predictions, probabilities, embeddings and true labels
                all_predictions.append(outputs.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_true_labels.append(labels.cpu().numpy())
                all_embeddings.append(embeddings.cpu().numpy())
                
                # Store patient IDs and years
                # all_patient_ids.extend([test_dataset.patient_ids[i] for i in range(len(batch['labels']))])
                # all_years.extend([test_dataset.years[i] for i in range(len(batch['labels']))])
        
        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_probabilities = np.concatenate(all_probabilities, axis=0)
        all_true_labels = np.concatenate(all_true_labels, axis=0)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Save predictions, probabilities, embeddings and true labels to CSV
        results_df = pd.DataFrame({
            # 'patient_id': all_patient_ids,
            # 'year': all_years,
            'raw_predictions': all_predictions.flatten(),
            'probabilities': all_probabilities.flatten(),
            'true_labels': all_true_labels.flatten(),
            'predicted_class': (all_probabilities > 0.5).flatten().astype(int)
        })
        
        # Add embedding columns
        # for i in range(all_embeddings.shape[1]):
        #     results_df[f'embedding_{i}'] = all_embeddings[:, i]
        
        # Save to log directory
        predictions_path = os.path.join(log_dir, 'model_predictions.csv')
        results_df.to_csv(predictions_path, index=False)
        print(f"\nPredictions and embeddings saved to '{predictions_path}'")
        
        # Log predictions CSV to MLflow
        mlflow.log_artifact(predictions_path)
        
        # Calculate metrics
        metrics = calculate_metrics(all_true_labels, all_predictions, all_probabilities)
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'test_accuracy': metrics['accuracy'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_f1': metrics['f1'],
            'test_auc': metrics['auc']
        })
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], 
                    annot=True, fmt='d', cmap='Blues')
        plt.title('Year 1 Cancer Prediction')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save confusion matrix plot to log directory
        confusion_matrix_path = os.path.join(log_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        mlflow.log_artifact(confusion_matrix_path)
        
        # Create and save embedding visualization
        n_samples = len(all_embeddings)
        perplexity = min(30, n_samples - 1)  # Use min of 30 or n_samples-1
        print(f"Number of samples: {n_samples}, Using perplexity: {perplexity}")

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(all_embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=all_true_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='True Label')
        plt.title('t-SNE Visualization of Model Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # Save embedding visualization to log directory
        embeddings_path = os.path.join(log_dir, 'embeddings_visualization.png')
        plt.savefig(embeddings_path)
        plt.close()
        mlflow.log_artifact(embeddings_path)
        print(f"\nEmbedding visualization saved to '{embeddings_path}'")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
            'Value': [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1'],
                metrics['auc']
            ]
        })
        
        # Save metrics to log directory
        metrics_path = os.path.join(log_dir, 'test_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nMetrics saved to '{metrics_path}'")
        
        # Log metrics CSV to MLflow
        mlflow.log_artifact(metrics_path)
        
        print("\nAll outputs have been saved to:", log_dir)
        print("\nAll metrics have been logged to MLflow in the same run")

if __name__ == "__main__":
    main() 