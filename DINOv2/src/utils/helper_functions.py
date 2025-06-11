import os
import argparse
from random import random
from matplotlib import pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
from torch import nn, optim
import seaborn as sns
from torch.utils.data import DataLoader
import datetime
import yaml

class HelperUtils:
    def __init__(self, config=None):
        """Initialize HelperUtils with optional config."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @staticmethod
    def parse_args(mode='train'):
        """
        Parse command line arguments for training or inference.
        
        Args:
            mode (str): Either 'train' or 'inference' to determine which arguments to parse
            
        Returns:
            argparse.Namespace: Parsed arguments
        """
        parser = argparse.ArgumentParser(description='DINOv2 Medical Imaging Project')
        
        # Common arguments
        parser.add_argument('--config', type=str, default='src/config/config.yaml',
                          help='Path to the configuration file (default: src/config/config.yaml)')
        
        if mode == 'train':
            # Training specific arguments
            parser.add_argument('--resume', action='store_true',
                              help='Resume training from checkpoint')
            parser.add_argument('--checkpoint', type=str,
                              help='Path to checkpoint file for resuming training')
            parser.add_argument('--debug', action='store_true',
                              help='Run in debug mode with smaller dataset')
        
        elif mode == 'inference':
            # Inference specific arguments
            parser.add_argument('--run_id', type=str, required=True,
                              help='MLflow run ID to load the model from')
            parser.add_argument('--visualize_attention', action='store_true',
                              help='Whether to visualize attention maps')
            parser.add_argument('--output_dir', type=str, default='outputs',
                              help='Directory to save inference outputs')
            parser.add_argument('--batch_size', type=int, default=1,
                              help='Batch size for inference')
        
        return parser.parse_args()

    @staticmethod
    def collate_fn(batch):
        """Create data loader with collate function to handle variable number of slices."""
        print("\nBatch contents:", flush=True)
        
        # Check if batch contains clinical features
        has_clinical_features = len(batch[0]) == 4
        
        max_reconstructions = max(x[0].size(0) for x in batch)
        max_slices = max(x[0].size(1) for x in batch)
        print(f"Max reconstructions: {max_reconstructions}", flush=True)
        print(f"Max slices: {max_slices}", flush=True)
        
        images = []
        positions = []
        labels = []
        attention_masks = []
        clinical_features = [] if has_clinical_features else None
        
        for item in batch:
            if has_clinical_features:
                img, pos, clin_feat, label = item
            else:
                img, pos, label = item
                
            print(f"Processing item: img shape {img.shape}, pos shape {pos.shape}", flush=True)

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
            
            # Verify shapes
            assert img.size(1) == pos.size(0), \
                f"Position shape {pos.shape} doesn't match image slices {img.size(1)}"
            assert recon_mask.size() == (max_reconstructions, max_slices), \
                f"Mask shape {recon_mask.size()} doesn't match expected {(max_reconstructions, max_slices)}"
            
            images.append(img)
            positions.append(pos)
            labels.append(label)
            attention_masks.append(recon_mask)
            
            if has_clinical_features:
                clinical_features.append(clin_feat)

        print("\nFinal batch shapes:", flush=True)
        print(f"Images: {torch.stack(images).shape}", flush=True)
        print(f"Positions: {torch.stack(positions).shape}", flush=True)
        print(f"Labels: {torch.stack(labels).shape}", flush=True)
        if has_clinical_features:
            print(f"Clinical Features: {torch.stack(clinical_features).shape}", flush=True)
        
        batch_dict = {
            'images': torch.stack(images),           # [batch, max_recon, max_slices, C, H, W]
            'positions': torch.stack(positions),     # [batch, max_slices]
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_masks)  # [batch, max_recon, max_slices]
        }
        
        if has_clinical_features:
            batch_dict['clinical_features'] = torch.stack(clinical_features)
        
        return batch_dict

    @staticmethod
    def create_transforms(transform_config):
        """Create transforms from config."""
        transform_list = []
        for t in transform_config:
            if t['name'] == 'Resize':
                transform_list.append(transforms.Resize(t['size']))
            elif t['name'] == 'RandomHorizontalFlip':
                transform_list.append(transforms.RandomHorizontalFlip())
            elif t['name'] == 'RandomRotation':
                transform_list.append(transforms.RandomRotation(t['degrees']))
            elif t['name'] == 'ColorJitter':
                transform_list.append(transforms.ColorJitter(
                    brightness=t.get('brightness', 0),
                    contrast=t.get('contrast', 0)
                ))
            elif t['name'] == 'ToTensor':
                transform_list.append(transforms.ToTensor())
            elif t['name'] == 'Normalize':
                transform_list.append(transforms.Normalize(
                    mean=t['mean'],
                    std=t['std']
                ))
        return transforms.Compose(transform_list)

    def resume_training(self, model, optimizer, checkpoint_path):
        """Resume training from a checkpoint.
        
        Args:
            model: The model to load the checkpoint into
            optimizer: The optimizer to load the checkpoint into
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            tuple: (start_epoch, best_loss)
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file {checkpoint_path} not found. Starting from epoch 0.")
            return 0, float('inf')
            
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get training state
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        
        print(f"Resumed training from epoch {start_epoch} with loss {best_loss:.4f}")
        
        # Log resume information to MLflow
        mlflow.log_params({
            'resumed_from_checkpoint': True,
            'checkpoint_path': checkpoint_path,
            'resume_epoch': start_epoch,
            'resume_loss': best_loss
        })
        
        return start_epoch, best_loss

    def save_checkpoint(self, model, optimizer, epoch, loss, path):
        """Save model checkpoint with training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}", flush=True)

    def load_checkpoint(self, model, optimizer, path):
        """Load model checkpoint and restore training state."""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
            print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss:.4f}", flush=True)
            return start_epoch, best_loss
        return 0, float('inf')

    @staticmethod
    def setup_mlflow(experiment_name=None):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Use provided experiment name or default
        if experiment_name is None:
            experiment_name = "DINOv2 Cancer Prediction"
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except:
            experiment_id = mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        return experiment_id

    def log_model_config(self, model, base_model, dataset, optimizer, scheduler, criterion, 
                        num_epochs, early_stopping_patience, batch_size, 
                        check_slice_thickness=False):
        """Log model configuration to MLflow."""
        try:
            # Get model type and feature dimensions
            if hasattr(model.transformer, 'config'):  # RAD-DINO
                model_type = 'RAD-DINO'
                feature_dim = model.transformer.config.hidden_size
            else:  # DINOv2
                model_type = 'DINOv2'
                feature_dim = model.transformer.patch_embed.embed_dim
            
            # Get number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Get model architecture details
            model_config = {
                'model_type': model_type,
                'base_model': base_model,
                'feature_dim': feature_dim,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'frozen_parameters': total_params - trainable_params,
                'num_epochs': num_epochs,
                'early_stopping_patience': early_stopping_patience,
                'batch_size': batch_size,
                'check_slice_thickness': check_slice_thickness,
                'optimizer': optimizer.__class__.__name__,
                'scheduler': scheduler.__class__.__name__,
                'criterion': criterion.__class__.__name__,
                'dataset_size': len(dataset),
                'num_classes': 1,  # Binary classification
                'input_shape': (3, 224, 224),  # Standard input shape
                'output_shape': (1,),  # Binary classification output
            }
            
            # Log model configuration
            mlflow.log_params(model_config)
            
            # Log model architecture as text
            model_architecture = str(model)
            mlflow.log_text(model_architecture, "model_architecture.txt")
            
            print("\nModel Configuration:", flush=True)
            for key, value in model_config.items():
                print(f"{key}: {value}", flush=True)
            print("\nModel Architecture:", flush=True)
            print(model_architecture, flush=True)
            
        except Exception as e:
            print(f"Error logging model configuration: {e}", flush=True)
            print(f"Model type: {type(model.transformer)}", flush=True)
            print(f"Available attributes: {dir(model.transformer)}", flush=True)
            raise e

    def log_dataset_info(self, dataset, dataset_name, mlflow_run):
        """Log dataset information to MLflow."""
        try:
            # Log dataset parameters
            mlflow.log_params({
                f'{dataset_name}_size': len(dataset),
                f'{dataset_name}_positive_count': len(dataset.selected_positive),
                f'{dataset_name}_negative_count': len(dataset.selected_negative),
            })
        except Exception as e:
            print(f"Warning: Could not log dataset parameters: {e}", flush=True)

    @staticmethod
    def load_model_from_mlflow(run_id, model_name="best_model"):
        """Load a model from MLflow."""
        model_uri = f"runs:/{run_id}/{model_name}"
        print(f"Loading model from {model_uri}")
        
        # Add current directory to Python path to ensure modules can be found
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        try:
            model = mlflow.pytorch.load_model(model_uri)
            print("Model loaded successfully")
            return model
        except ModuleNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Attempting to fix module import issue...")
            
            # Try loading with map_location to CPU first
            model = mlflow.pytorch.load_model(model_uri, map_location=torch.device('cpu'))
            print("Model loaded successfully with CPU map_location")
            return model

    @staticmethod
    def get_slurm_job_id():
        """Get SLURM job ID from environment variable."""
        return os.environ.get('SLURM_JOB_ID', 'local')

    def ensure_log_dir(self):
        """Create logs directory with SLURM job ID if it doesn't exist."""
        slurm_id = self.get_slurm_job_id()
        log_dir = f'logs/predictions/predictions_{slurm_id}'
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob):
        """Calculate various classification metrics with confidence intervals."""
        metrics = {}
        
        # Convert predictions to binary
        y_pred_binary = (y_prob > 0.5).astype(int)
        
        # Calculate metrics for year 1 prediction
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Calculate AUC with confidence intervals
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
            # Bootstrap confidence intervals for AUC
            n_bootstraps = 1000
            bootstrapped_aucs = []
            for i in range(n_bootstraps):
                indices = np.random.randint(0, len(y_prob), len(y_prob))
                if len(np.unique(y_true[indices])) < 2:
                    continue
                bootstrapped_auc = roc_auc_score(y_true[indices], y_prob[indices])
                bootstrapped_aucs.append(bootstrapped_auc)
            sorted_aucs = np.array(bootstrapped_aucs)
            metrics['auc_ci_lower'] = np.percentile(sorted_aucs, 2.5)
            metrics['auc_ci_upper'] = np.percentile(sorted_aucs, 97.5)
        except Exception as e:
            print(f"Could not calculate AUC: {e}", flush=True)
            metrics['auc'] = 0.0
            metrics['auc_ci_lower'] = 0.0
            metrics['auc_ci_upper'] = 0.0
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        metrics['confusion_matrix'] = cm
        
        # Calculate additional metrics
        metrics['specificity'] = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics

    @staticmethod
    def check_loss_saturation(outputs, labels):
        """Check for loss saturation in model outputs."""
        with torch.no_grad():
            # Get probabilities
            probs = torch.sigmoid(outputs)
            # Calculate individual losses
            pos_loss = -labels * torch.log(probs + 1e-10)
            neg_loss = -(1 - labels) * torch.log(1 - probs + 1e-10)
            total_loss = (pos_loss + neg_loss).mean()
            
            # Print diagnostics
            print("\nLoss Saturation Diagnostics:", flush=True)
            print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]", flush=True)
            print(f"Probability range: [{probs.min():.2f}, {probs.max():.2f}]", flush=True)
            print(f"Positive samples: {labels.sum().item()}", flush=True)
            print(f"Average positive probability: {probs[labels == 1].mean():.4f}", flush=True)
            print(f"Average negative probability: {probs[labels == 0].mean():.4f}", flush=True)
            print(f"Individual losses - Positive: {pos_loss.mean():.4f}, Negative: {neg_loss.mean():.4f}", flush=True)
            return total_loss

    def run_inference_and_log_metrics(self, model, test_dataset, device, mlflow_run, logging=True):
        """Run inference on test dataset and log metrics."""
        model.eval()
        all_predictions = []
        all_labels = []
        results = []  # List to store detailed results
        
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                patient_id, study_yr = test_dataset.scan_list[idx]
                patient_tensor, slice_positions, true_label = test_dataset[idx]
                
                # Move tensors to device and add batch dimension
                patient_tensor = patient_tensor.unsqueeze(0).to(device)  # Add batch dimension
                slice_positions = slice_positions.unsqueeze(0).to(device)  # Add batch dimension
                
                # Get model prediction
                output = model(patient_tensor, slice_positions)
                prediction = torch.sigmoid(output).item()
                
                # Store results
                results.append({
                    'patient_id': patient_id,
                    'study_yr': study_yr,
                    'true_label': true_label.item(),
                    'prediction': prediction
                })
                
                # Store for metrics calculation
                all_predictions.append(prediction)
                all_labels.append(true_label.item())
                
                print(f"Patient {patient_id}, Study Year {study_yr}:")
                print(f"True Label: {true_label.item():.2f}, Prediction: {prediction:.2f}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        results_path = 'inference_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Convert predictions to binary using 0.5 threshold
        binary_predictions = (all_predictions > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, binary_predictions)
        precision = precision_score(all_labels, binary_predictions)
        recall = recall_score(all_labels, binary_predictions)
        f1 = f1_score(all_labels, binary_predictions)
        auc = roc_auc_score(all_labels, all_predictions)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, binary_predictions)
        
        # Calculate confusion matrix metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Create a figure with two subplots side by side
        plt.figure(figsize=(16, 6))
        
        # Plot confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add confusion matrix metrics as text
        plt.text(0.5, -0.3, f'Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\nPPV: {ppv:.3f}\nNPV: {npv:.3f}',
                 horizontalalignment='center', transform=plt.gca().transAxes)
        
        # Plot ROC curve
        plt.subplot(1, 2, 2)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save combined plot
        plot_path = 'model_evaluation.png'
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'ppv': ppv,
            'npv': npv
        }
        
        if logging:
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(results_path)
            mlflow.log_artifact(plot_path)
        
        print("\nTest Metrics Summary:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"PPV: {ppv:.4f}")
        print(f"NPV: {npv:.4f}")
        
        return metrics


# def run_cross_validation(model, dataset, n_splits=5, batch_size=4, device='cuda'):
#     """Run k-fold cross validation."""
#     from sklearn.model_selection import KFold
    
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     fold_metrics = []
    
#     for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
#         print(f"\nStarting fold {fold + 1}/{n_splits}", flush=True)
        
#         # Create data loaders for this fold
#         train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
#         val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
#         train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, collate_fn=collate_fn)
#         val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, collate_fn=collate_fn)
        
#         # Train model for this fold
#         model.train()
#         optimizer = optim.Adam(model.parameters(), lr=0.00001)
#         criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
        
#         for epoch in range(10):  # Train for 10 epochs per fold
#             train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
#             val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
            
#             print(f"Fold {fold + 1}, Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}", flush=True)
        
#         # Evaluate on validation set
#         val_metrics = validate_epoch(model, val_loader, criterion, device)[1]
#         fold_metrics.append(val_metrics)
        
#         # Log fold metrics
#         mlflow.log_metrics({
#             f'fold_{fold+1}_accuracy': val_metrics['accuracy'],
#             f'fold_{fold+1}_auc': val_metrics['auc'],
#             f'fold_{fold+1}_f1': val_metrics['f1']
#         })
    
#     # Calculate average metrics across folds
#     avg_metrics = {}
#     for metric in fold_metrics[0].keys():
#         if metric != 'confusion_matrix':
#             values = [m[metric] for m in fold_metrics]
#             avg_metrics[f'avg_{metric}'] = np.mean(values)
#             avg_metrics[f'std_{metric}'] = np.std(values)
    
#     # Log average metrics
#     mlflow.log_metrics(avg_metrics)
    
#     return avg_metrics


# def train_epoch(model, train_loader, optimizer, criterion, device):
#     """Train for one epoch."""
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         inputs = batch['images'].to(device)
#         positions = batch['positions'].to(device)
#         labels = batch['labels'].float().to(device)
#         attention_masks = batch['attention_mask'].to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs, positions, attention_masks)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     return total_loss / len(train_loader)


# def validate_epoch(model, val_loader, criterion, device):
#     """Validate for one epoch."""
#     model.eval()
#     total_loss = 0
#     all_predictions = []
#     all_probabilities = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch in val_loader:
#             inputs = batch['images'].to(device)
#             positions = batch['positions'].to(device)
#             labels = batch['labels'].float().to(device)
#             attention_masks = batch['attention_mask'].to(device)
            
#             outputs = model(inputs, positions, attention_masks)
#             loss = criterion(outputs, labels)
            
#             total_loss += loss.item()
#             all_predictions.append(outputs.cpu().numpy())
#             all_probabilities.append(torch.sigmoid(outputs).cpu().numpy())
#             all_labels.append(labels.cpu().numpy())
    
#     # Concatenate all predictions and labels
#     all_predictions = np.concatenate(all_predictions)
#     all_probabilities = np.concatenate(all_probabilities)
#     all_labels = np.concatenate(all_labels)
    
#     # Calculate metrics
#     metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
    
#     return total_loss / len(val_loader), metrics


    def retrain_model(self, run_id, model, num_epochs, learning_rate, patience, dataset, device, start_epoch=0):
        """Retrain a loaded model with new parameters."""
        # Move model to device
        model = model.to(device)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, min_lr=1e-6)
        
        # Setup loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Setup dataloader
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=self.collate_fn, shuffle=True)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        scaler = torch.cuda.amp.GradScaler()
        
        # Start MLflow run for retraining
        with mlflow.start_run(run_name=f"retrain_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log retraining parameters
            mlflow.log_params({
                'retrain_epochs': num_epochs,
                'retrain_learning_rate': learning_rate,
                'retrain_patience': patience,
                'original_run_id': run_id,
                'start_epoch': start_epoch
            })
            
            start_time = datetime.datetime.now()
            
            for epoch in range(start_epoch, start_epoch + num_epochs):
                model.train()
                total_loss = 0
                
                print(f"\nStarting Epoch {epoch+1}/{start_epoch + num_epochs}", flush=True)
                print("Training phase:", flush=True)
                
                for batch_idx, batch in enumerate(dataloader):
                    try:
                        inputs = batch['images'].to(device)
                        positions = batch['positions'].to(device)
                        labels = batch['labels'].float().to(device)
                        attention_masks = batch['attention_mask'].to(device)
                        
                        optimizer.zero_grad()
                        
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs, positions, attention_masks)
                            # Add gradient clipping to prevent extreme values
                            outputs = torch.clamp(outputs, min=-10, max=10)
                            print(f'Outputs: {outputs}', flush=True)
                            # Check for saturation
                            if batch_idx == 0:  # Check first batch of each epoch
                                self.check_loss_saturation(outputs, labels)
                            loss = criterion(outputs, labels)
                            
                            # Check for NaN loss
                            if torch.isnan(loss):
                                print(f"Warning: NaN loss detected in batch {batch_idx}", flush=True)
                                print(f"Outputs range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]", flush=True)
                                print(f"Labels range: [{labels.min().item():.2f}, {labels.max().item():.2f}]", flush=True)
                                # Skip this batch
                                continue
                        
                        # Add gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        batch_loss = loss.item()
                        if not np.isnan(batch_loss):  # Only add non-NaN losses
                            total_loss += batch_loss
                        
                        # Log batch metrics
                        if not np.isnan(batch_loss):
                            mlflow.log_metric('retrain_batch_loss', batch_loss, 
                                            step=epoch * len(dataloader) + batch_idx)
                        
                    except RuntimeError as e:
                        print(f"Error in batch {batch_idx}: {e}", flush=True)
                        if "out of memory" in str(e):
                            print(f"OOM in batch {batch_idx}. Exiting program...", flush=True)
                            import sys
                            sys.exit()
                        else:
                            raise e
                
                avg_loss = total_loss / len(dataloader)
                
                # Log epoch metrics
                mlflow.log_metrics({
                    'retrain_epoch_loss': avg_loss,
                    'retrain_learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1
                }, step=epoch)
                
                print(f"\nRetraining Epoch {epoch+1}/{start_epoch + num_epochs} Summary:", flush=True)
                print(f"Average Loss: {avg_loss:.4f}", flush=True)
                
                scheduler.step(avg_loss)
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    mlflow.pytorch.log_model(
                        model,
                        "retrain_best_model",
                        registered_model_name="DinoVisionTransformerCancerPredictor"
                    )
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print("\nEarly stopping triggered!", flush=True)
                    break
            
            end_time = datetime.datetime.now()
            training_duration = str(end_time - start_time)
            
            # Log final model
            mlflow.pytorch.log_model(
                model,
                "retrain_final_model",
                registered_model_name="DinoVisionTransformerCancerPredictor"
            )
            
            # Log final metrics
            mlflow.log_metrics({
                'retrain_final_loss': avg_loss,
                'retrain_duration': training_duration,
                'final_epoch': epoch + 1
            })
            
            return model, avg_loss, training_duration, epoch + 1
        
    @staticmethod
    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_last_epoch(self, run_id):
        """Get the last epoch number from the MLflow run."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Get the final epoch from metrics
        metrics = run.data.metrics
        if 'final_epoch' in metrics:
            return int(metrics['final_epoch'])
        elif 'epoch' in metrics:
            return int(metrics['epoch'])
        else:
            print("Warning: Could not find epoch information in MLflow run. Starting from epoch 0.")
            return 0

    @staticmethod
    def get_training_params(self, run_id):
        """Get training parameters from the original MLflow run."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        
        # Extract relevant parameters
        training_params = {
            'batch_size': int(params.get('batch_size', 10)),
            'learning_rate': float(params.get('learning_rate', 0.0001)),
            'num_epochs': int(params.get('num_epochs', 100)),
            'patience': int(params.get('early_stopping_patience', 5)),
            'train_size': float(params.get('train_split', 0.8)),
            'seed': int(params.get('seed', 42)),
            'check_slice_thickness': params.get('check_slice_thickness', 'false').lower() == 'true'
        }
        
        return training_params

    @staticmethod
    def get_model_config(self, run_id):
        """Get model configuration from the original MLflow run."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Get the config file from artifacts
        config_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
