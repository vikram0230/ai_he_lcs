import os

from matplotlib import pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
from torch import nn, optim
from tqdm import tqdm
import seaborn as sns
from torch.utils.data import DataLoader


# Create data loader with collate function to handle variable number of slices
def collate_fn(batch):
    print("\nBatch contents:", flush=True)
    # for i, (img, pos, label) in enumerate(batch):
    #     print(f"Item {i}: Image shape: {img.shape}, Positions shape: {pos.shape}", flush=True)
    
    max_reconstructions = max(x[0].size(0) for x in batch)
    max_slices = max(x[0].size(1) for x in batch)
    print(f"Max reconstructions: {max_reconstructions}", flush=True)
    print(f"Max slices: {max_slices}", flush=True)
    
    images = []
    positions = []
    labels = []
    attention_masks = []
    
    for img, pos, label in batch:
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

    print("\nFinal batch shapes:", flush=True)
    print(f"Images: {torch.stack(images).shape}", flush=True)
    print(f"Positions: {torch.stack(positions).shape}", flush=True)
    print(f"Labels: {torch.stack(labels).shape}", flush=True)
    
    return {
        'images': torch.stack(images),           # [batch, max_recon, max_slices, C, H, W]
        'positions': torch.stack(positions),     # [batch, max_slices]
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_masks)  # [batch, max_recon, max_slices]
    }


def save_checkpoint(model, optimizer, epoch, loss, path):
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


def load_checkpoint(model, optimizer, path):
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


def setup_mlflow():
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create a new experiment
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


def log_model_config(model, dataset, optimizer, scheduler, criterion, num_epochs, training_duration, final_loss, early_stopping_patience, batch_size):
    """Log model configuration to MLflow."""
    
    # Dynamically extract predictor architecture
    predictor_layers = []
    for layer in model.predictor:
        layer_info = {
            'type': layer.__class__.__name__,
        }
        # Add layer-specific parameters
        if isinstance(layer, nn.Dropout):
            layer_info['p'] = layer.p
        elif isinstance(layer, nn.Linear):
            layer_info['in_features'] = layer.in_features
            layer_info['out_features'] = layer.out_features
        predictor_layers.append(layer_info)
    
    config = {
        'model_name': model.__class__.__name__,
        'base_model': 'dinov2_vits14',
        'feature_dim': model.transformer.embed_dim,
        'num_transformer_layers': len(model.transformer.blocks),
        'num_slice_processor_layers': model.slice_processor.num_layers,
        'num_heads': model.slice_processor.layers[0].self_attn.num_heads,
        'dropout_rate': model.slice_processor.layers[0].dropout.p,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'scheduler_patience': scheduler.patience,
        'scheduler_factor': scheduler.factor,
        'scheduler_min_lr': scheduler.min_lrs[0],
        'early_stopping_patience': early_stopping_patience,
        'num_epochs': num_epochs,
        'dataset_size': len(dataset),
        'image_size': '224x224',
        'normalization_mean': str([0.485, 0.456, 0.406]),
        'normalization_std': str([0.229, 0.224, 0.225]),
        'min_slices': 50,
        'num_outputs': 2,
        'loss_function': criterion.__class__.__name__,
        'use_amp': True,
        'use_gradient_checkpointing': False,
        'device': str(next(model.parameters()).device),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'predictor_architecture': {
            'layers': predictor_layers,
            'total_parameters': sum(p.numel() for p in model.predictor.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
        }
    }
    
    # Log all parameters
    mlflow.log_params(config)
    
    # Convert training duration to seconds
    if isinstance(training_duration, str):
        # Parse the duration string (format: "HH:MM:SS.microseconds")
        h, m, s = training_duration.split(':')
        duration_seconds = float(h) * 3600 + float(m) * 60 + float(s)
    else:
        duration_seconds = float(training_duration)
    
    # Log initial metrics (only training duration)
    mlflow.log_metrics({
        'training_duration_seconds': duration_seconds
    })
    
    # If final_loss is provided, log it separately
    if final_loss is not None:
        mlflow.log_metrics({
            'final_loss': final_loss
        })
    
    return config  # Return the config for potential reuse


def load_model_from_mlflow(run_id, model_name="best_model"):
    """Load a model from MLflow."""
    # Load the model
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Get the run details
    run = mlflow.get_run(run_id)
    
    print(f"Loaded model from run: {run_id}", flush=True)
    print(f"Model parameters: {run.data.params}", flush=True)
    print(f"Model metrics: {run.data.metrics}", flush=True)
    
    return model


def get_slurm_job_id():
    """Get SLURM job ID from environment variable."""
    return os.environ.get('SLURM_JOB_ID', 'local')


def ensure_log_dir():
    """Create logs directory with SLURM job ID if it doesn't exist."""
    slurm_id = get_slurm_job_id()
    log_dir = f'logs/predictions_{slurm_id}'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various classification metrics with confidence intervals."""
    metrics = {}
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
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

def check_loss_saturation(outputs, labels):
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

def run_inference_and_log_metrics(model, test_dataset, device, mlflow_run):
    """Run inference on test dataset and log metrics to MLflow."""
    print("\nRunning inference on test dataset...", flush=True)
    
    # Create log directory
    log_dir = ensure_log_dir()
    print(f"Storing outputs in: {log_dir}", flush=True)
    
    # Create test dataloader
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
    
    # Perform inference
    model.eval()
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
    
    # Save predictions and true labels to CSV
    results_df = pd.DataFrame({
        'raw_predictions': all_predictions.flatten(),
        'probabilities': all_probabilities.flatten(),
        'true_labels': all_true_labels.flatten(),
        'predicted_class': (all_probabilities > 0.5).flatten().astype(int)
    })
    
    # Save to log directory
    predictions_path = os.path.join(log_dir, 'model_predictions.csv')
    results_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to '{predictions_path}'", flush=True)
    
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
    print(f"\nMetrics saved to '{metrics_path}'", flush=True)
    
    # Log metrics CSV to MLflow
    mlflow.log_artifact(metrics_path)
    
    print("\nAll outputs have been saved to:", log_dir, flush=True)
    print("\nAll metrics have been logged to MLflow in the same run", flush=True)
    
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


# def retrain_model(run_id, model, num_epochs, learning_rate, patience, dataset, device, start_epoch=0):
#     """Retrain a loaded model with new parameters."""
#     # Move model to device
#     model = model.to(device)
    
#     # Setup optimizer and scheduler
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, min_lr=1e-6)
    
#     # Setup loss function
#     criterion = nn.BCEWithLogitsLoss()
    
#     # Setup dataloader
#     dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    
#     # Training loop
#     best_loss = float('inf')
#     patience_counter = 0
#     scaler = torch.cuda.amp.GradScaler()
    
#     # Start MLflow run for retraining
#     with mlflow.start_run(run_name=f"retrain_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
#         # Log retraining parameters
#         mlflow.log_params({
#             'retrain_epochs': num_epochs,
#             'retrain_learning_rate': learning_rate,
#             'retrain_patience': patience,
#             'original_run_id': run_id,
#             'start_epoch': start_epoch
#         })
        
#         start_time = datetime.datetime.now()
        
#         for epoch in range(start_epoch, start_epoch + num_epochs):
#             model.train()
#             total_loss = 0
            
#             print(f"\nStarting Epoch {epoch+1}/{start_epoch + num_epochs}", flush=True)
#             print("Training phase:", flush=True)
            
#             for batch_idx, batch in enumerate(dataloader):
#                 try:
#                     inputs = batch['images'].to(device)
#                     positions = batch['positions'].to(device)
#                     labels = batch['labels'].float().to(device)
#                     attention_masks = batch['attention_mask'].to(device)
                    
#                     optimizer.zero_grad()
                    
#                     with torch.cuda.amp.autocast():
#                         outputs = model(inputs, positions, attention_masks)
#                         # Add gradient clipping to prevent extreme values
#                         outputs = torch.clamp(outputs, min=-10, max=10)
#                         print(f'Outputs: {outputs}', flush=True)
#                         # Check for saturation
#                         if batch_idx == 0:  # Check first batch of each epoch
#                             check_loss_saturation(outputs, labels)
#                         loss = criterion(outputs, labels)
                        
#                         # Check for NaN loss
#                         if torch.isnan(loss):
#                             print(f"Warning: NaN loss detected in batch {batch_idx}", flush=True)
#                             print(f"Outputs range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]", flush=True)
#                             print(f"Labels range: [{labels.min().item():.2f}, {labels.max().item():.2f}]", flush=True)
#                             # Skip this batch
#                             continue
                    
#                     # Add gradient clipping
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()
                    
#                     batch_loss = loss.item()
#                     if not np.isnan(batch_loss):  # Only add non-NaN losses
#                         total_loss += batch_loss
                    
#                     # Log batch metrics
#                     if not np.isnan(batch_loss):
#                         mlflow.log_metric('retrain_batch_loss', batch_loss, 
#                                         step=epoch * len(dataloader) + batch_idx)
                    
#                 except RuntimeError as e:
#                     print(f"Error in batch {batch_idx}: {e}", flush=True)
#                     if "out of memory" in str(e):
#                         print(f"OOM in batch {batch_idx}. Exiting program...", flush=True)
#                         import sys
#                         sys.exit()
#                     else:
#                         raise e
            
#             avg_loss = total_loss / len(dataloader)
            
#             # Log epoch metrics
#             mlflow.log_metrics({
#                 'retrain_epoch_loss': avg_loss,
#                 'retrain_learning_rate': optimizer.param_groups[0]['lr'],
#                 'epoch': epoch + 1
#             }, step=epoch)
            
#             print(f"\nRetraining Epoch {epoch+1}/{start_epoch + num_epochs} Summary:", flush=True)
#             print(f"Average Loss: {avg_loss:.4f}", flush=True)
            
#             scheduler.step(avg_loss)
            
#             # Save best model
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 patience_counter = 0
#                 mlflow.pytorch.log_model(
#                     model,
#                     "retrain_best_model",
#                     registered_model_name="DinoVisionTransformerCancerPredictor"
#                 )
#             else:
#                 patience_counter += 1
            
#             if patience_counter >= patience:
#                 print("\nEarly stopping triggered!", flush=True)
#                 break
        
#         end_time = datetime.datetime.now()
#         training_duration = str(end_time - start_time)
        
#         # Log final model
#         mlflow.pytorch.log_model(
#             model,
#             "retrain_final_model",
#             registered_model_name="DinoVisionTransformerCancerPredictor"
#         )
        
#         # Log final metrics
#         mlflow.log_metrics({
#             'retrain_final_loss': avg_loss,
#             'retrain_duration': training_duration,
#             'final_epoch': epoch + 1
#         })
        
#         return model, avg_loss, training_duration, epoch + 1
