import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset_loader import PatientDicomDataset
from classifier import DinoVisionTransformerCancerPredictor
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_model_from_mlflow(run_id, model_name="best_model"):
    """Load a model from MLflow."""
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.pytorch.load_model(model_uri)
    return model

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
        'images': torch.stack(images),
        'positions': torch.stack(positions),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_masks)
    }

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various classification metrics."""
    metrics = {}
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics for each class (1st year and 5th year)
    for i, year in enumerate(['1st_year', '5th_year']):
        metrics[f'{year}_accuracy'] = accuracy_score(y_true[:, i], y_pred_binary[:, i])
        metrics[f'{year}_precision'] = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        metrics[f'{year}_recall'] = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        metrics[f'{year}_f1'] = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
        metrics[f'{year}_auc'] = roc_auc_score(y_true[:, i], y_prob[:, i])
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        metrics[f'{year}_confusion_matrix'] = cm
    
    return metrics

def plot_confusion_matrices(metrics, save_path):
    """Plot confusion matrices for both 1st and 5th year predictions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1st year confusion matrix
    sns.heatmap(metrics['1st_year_confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('1st Year Prediction')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Plot 5th year confusion matrix
    sns.heatmap(metrics['5th_year_confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('5th Year Prediction')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model from MLflow
    run_id = '72eb8cc3f14c4d8a80409b540ffbfd9c'
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
        patients_count=10
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=4, 
        collate_fn=collate_fn, 
        shuffle=False
    )
    
    # Initialize lists to store predictions and true labels
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    
    # Perform inference
    print("Performing inference on test data...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = batch['images'].to(device)
            positions = batch['positions'].to(device)
            labels = batch['labels'].float().to(device)
            attention_masks = batch['attention_mask'].to(device)
            
            # Get model predictions
            outputs = model(inputs, positions, attention_masks)
            probabilities = torch.sigmoid(outputs)
            
            # Store predictions and true labels
            all_predictions.append(outputs.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_true_labels.append(labels.cpu().numpy())
    
    # Concatenate all predictions and labels
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    
    # Print sample predictions and labels
    print("\nSample Predictions and Labels:")
    print("-" * 50)
    print("Format: [1st_year, 5th_year]")
    print("\nFirst 5 samples:")
    for i in range(min(5, len(all_predictions))):
        print(f"\nSample {i+1}:")
        print(f"Raw logits: {all_predictions[i]}")
        print(f"Probabilities: {all_probabilities[i]}")
        print(f"True labels: {all_true_labels[i]}")
        print(f"Predicted class: {all_probabilities[i] > 0.5}")
        print(f"Actual class: {all_true_labels[i]}")
    
    # Calculate metrics
    metrics = calculate_metrics(all_true_labels, all_predictions, all_probabilities)
    
    # Print metrics
    print("\nTest Results:")
    print("-" * 50)
    for year in ['1st_year', '5th_year']:
        print(f"\n{year.upper()} PREDICTIONS:")
        print(f"Accuracy: {metrics[f'{year}_accuracy']:.4f}")
        print(f"Precision: {metrics[f'{year}_precision']:.4f}")
        print(f"Recall: {metrics[f'{year}_recall']:.4f}")
        print(f"F1 Score: {metrics[f'{year}_f1']:.4f}")
        print(f"AUC-ROC: {metrics[f'{year}_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics[f'{year}_confusion_matrix'])
    
    # Plot confusion matrices
    plot_confusion_matrices(metrics, 'confusion_matrices.png')
    print("\nConfusion matrices saved to 'confusion_matrices.png'")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
        '1st Year': [
            metrics['1st_year_accuracy'],
            metrics['1st_year_precision'],
            metrics['1st_year_recall'],
            metrics['1st_year_f1'],
            metrics['1st_year_auc']
        ],
        '5th Year': [
            metrics['5th_year_accuracy'],
            metrics['5th_year_precision'],
            metrics['5th_year_recall'],
            metrics['5th_year_f1'],
            metrics['5th_year_auc']
        ]
    })
    metrics_df.to_csv('test_metrics.csv', index=False)
    print("\nMetrics saved to 'test_metrics.csv'")

if __name__ == "__main__":
    main() 