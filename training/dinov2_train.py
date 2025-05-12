import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from classifier import DinoVisionTransformerCancerPredictor
from dataset_loader import PatientDicomDataset
import csv
import datetime
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import random


# Create data loader with collate function to handle variable number of slices
def collate_fn(batch):
    print("\nBatch contents:")
    # for i, (img, pos, label) in enumerate(batch):
    #     print(f"Item {i}: Image shape: {img.shape}, Positions shape: {pos.shape}")
    
    max_reconstructions = max(x[0].size(0) for x in batch)
    max_slices = max(x[0].size(1) for x in batch)
    print(f"Max reconstructions: {max_reconstructions}")
    print(f"Max slices: {max_slices}")
    
    images = []
    positions = []
    labels = []
    attention_masks = []
    
    for img, pos, label in batch:
        print(f"Processing item: img shape {img.shape}, pos shape {pos.shape}")

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

    print("\nFinal batch shapes:")
    print(f"Images: {torch.stack(images).shape}")
    print(f"Positions: {torch.stack(positions).shape}")
    print(f"Labels: {torch.stack(labels).shape}")
    
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
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint and restore training state."""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss:.4f}")
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
        'learning_rate': optimizer.param_groups[0]['lr'],
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'scheduler_patience': scheduler.patience,
        'scheduler_factor': scheduler.factor,
        'min_lr': scheduler.min_lrs[0],
        'early_stopping_patience': early_stopping_patience,
        'num_epochs': num_epochs,
        'pos_weight_1yr': criterion.pos_weight[0].item(),
        'pos_weight_5yr': criterion.pos_weight[1].item(),
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
    
    # Log final metrics
    mlflow.log_metrics({
        'final_loss': final_loss,
        'training_duration_seconds': duration_seconds
    })

def load_model_from_mlflow(run_id, model_name="best_model"):
    """Load a model from MLflow."""
    # Load the model
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Get the run details
    run = mlflow.get_run(run_id)
    
    print(f"Loaded model from run: {run_id}")
    print(f"Model parameters: {run.data.params}")
    print(f"Model metrics: {run.data.metrics}")
    
    return model

def retrain_model(run_id, model, num_epochs, learning_rate, patience, dataset, device, start_epoch=0):
    """Retrain a loaded model with new parameters."""
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, min_lr=1e-6)
    
    # Setup loss function
    pos_weight = torch.tensor([112.83])  # Only year 1 weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    # Setup dataloader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    scaler = torch.amp.GradScaler()
    
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
            
            print(f"\nStarting Epoch {epoch+1}/{start_epoch + num_epochs}")
            print("Training phase:")
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    inputs = batch['images'].to(device)
                    positions = batch['positions'].to(device)
                    labels = batch['labels'].float().to(device)
                    attention_masks = batch['attention_mask'].to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(inputs, positions, attention_masks)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    batch_loss = loss.item()
                    total_loss += batch_loss
                    
                    # Log batch metrics
                    mlflow.log_metric('retrain_batch_loss', batch_loss, 
                                    step=epoch * len(dataloader) + batch_idx)
                    
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    if "out of memory" in str(e):
                        print(f"OOM in batch {batch_idx}. Exiting program...")
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
            
            print(f"\nRetraining Epoch {epoch+1}/{start_epoch + num_epochs} Summary:")
            print(f"Average Loss: {avg_loss:.4f}")
            
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
                print("\nEarly stopping triggered!")
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

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various classification metrics."""
    metrics = {}
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics for year 1 prediction
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
    metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    metrics['confusion_matrix'] = cm
    
    return metrics

def run_inference_and_log_metrics(model, test_dataset, device, mlflow_run):
    """Run inference on test dataset and log metrics to MLflow."""
    print("\nRunning inference on test dataset...")
    
    # Create test dataloader
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
    
    # Plot and log confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Year 1 Cancer Prediction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save confusion matrix plot to MLflow
    confusion_matrix_path = 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    plt.close()
    mlflow.log_artifact(confusion_matrix_path)
    
    # Print metrics
    print("\nTest Results:")
    print("-" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC-ROC: {metrics['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    return metrics

def main():
    # Setup MLflow
    experiment_id = setup_mlflow()
    
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Start MLflow run for new training
    with mlflow.start_run(run_name=f"frozen_dinov2_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}") as mlflow_run:
        # Set up data transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load datasets
        full_dataset = PatientDicomDataset(
            root_dir='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_train_data',
            labels_file='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_actual.csv',
            transform=transform,
            patients_count=200
        )
        
        # Load test dataset
        test_dataset = PatientDicomDataset(
            root_dir='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_test_data',
            labels_file='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_actual.csv',
            transform=transform,
            patients_count=50
        )
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        batch_size = 10
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

        # Initialize the model and wrap with DataParallel
        model = DinoVisionTransformerCancerPredictor()
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set up loss function and optimizer
        learning_rate = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(f"Optimizer: {optimizer}")
        print(f"Learning rate: {learning_rate}")

        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, min_lr=1e-6)

        # Add early stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        # Training loop
        num_epochs = 100
        pos_weight = torch.tensor([112.83])  # Only year 1 weight
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        
        # Use automatic mixed precision
        scaler = torch.amp.GradScaler()

        # Add diagnostic function
        def check_loss_saturation(outputs, labels):
            with torch.no_grad():
                # Get probabilities
                probs = torch.sigmoid(outputs)
                # Calculate individual losses
                pos_loss = -labels * torch.log(probs + 1e-10) * pos_weight.to(device)
                neg_loss = -(1 - labels) * torch.log(1 - probs + 1e-10)
                total_loss = (pos_loss + neg_loss).mean()
                
                # Print diagnostics
                print("\nLoss Saturation Diagnostics:")
                print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
                print(f"Probability range: [{probs.min():.2f}, {probs.max():.2f}]")
                print(f"Positive samples: {labels.sum().item()}")
                print(f"Average positive probability: {probs[labels == 1].mean():.4f}")
                print(f"Average negative probability: {probs[labels == 0].mean():.4f}")
                print(f"Individual losses - Positive: {pos_loss.mean():.4f}, Negative: {neg_loss.mean():.4f}")
                return total_loss
        
        print(f"\nStarting training with:")
        print(f"Batch size: {batch_size}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Device: {device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print("-" * 50)

        start_time = datetime.datetime.now()
        final_loss = None

        # Log training parameters
        mlflow.log_params({
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'early_stopping_patience': patience,
            'train_size': 0.8,
            'seed': seed
        })

        # Log model configuration before training starts
        print("\nLogging model configuration...")
        log_model_config(
            model=model.module if isinstance(model, nn.DataParallel) else model,
            dataset=full_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=num_epochs,
            training_duration="0:00:00",  # Will be updated after training
            final_loss=None,  # Will be updated after training
            early_stopping_patience=patience,
            batch_size=batch_size
        )
        print("Model configuration logged successfully")

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_batch_count = 0
            
            print(f"\nStarting Epoch {epoch+1}/{num_epochs}")
            print("Training phase:")
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    inputs = batch['images'].to(device)
                    positions = batch['positions'].to(device)
                    labels = batch['labels'].float().to(device)
                    attention_masks = batch['attention_mask'].to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(inputs, positions, attention_masks)
                        # Add gradient clipping to prevent extreme values
                        outputs = torch.clamp(outputs, min=-10, max=10)
                        # Check for saturation
                        if batch_idx == 0:  # Check first batch of each epoch
                            check_loss_saturation(outputs, labels)
                        loss = criterion(outputs, labels)
                    
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    batch_loss = loss.item()
                    train_loss += batch_loss
                    train_batch_count += 1
                    
                    # Log batch metrics
                    mlflow.log_metric('train_batch_loss', batch_loss, step=epoch * len(train_loader) + batch_idx)
                    
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    if "out of memory" in str(e):
                        print(f"OOM in batch {batch_idx}. Exiting program...")
                        import sys
                        sys.exit()
                    else:
                        raise e
                
                print(f"Batch {batch_idx+1} Loss: {batch_loss:.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            print("\nValidation phase:")
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    try:
                        inputs = batch['images'].to(device)
                        positions = batch['positions'].to(device)
                        labels = batch['labels'].float().to(device)
                        attention_masks = batch['attention_mask'].to(device)
                        
                        with torch.amp.autocast(device_type=device.type):
                            outputs = model(inputs, positions, attention_masks)
                            loss = criterion(outputs, labels)
                        
                        batch_loss = loss.item()
                        val_loss += batch_loss
                        val_batch_count += 1
                        
                        # Log batch metrics
                        mlflow.log_metric('val_batch_loss', batch_loss, step=epoch * len(val_loader) + batch_idx)
                        
                    except RuntimeError as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        if "out of memory" in str(e):
                            print(f"OOM in validation batch {batch_idx}. Exiting program...")
                            import sys
                            sys.exit()
                        else:
                            raise e
                    
                    print(f"Validation Batch {batch_idx+1} Loss: {batch_loss:.4f}")
            
            avg_val_loss = val_loss / len(val_loader)
            final_loss = avg_val_loss
            
            # Log epoch metrics
            mlflow.log_metrics({
                'train_epoch_loss': avg_train_loss,
                'val_epoch_loss': avg_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch + 1
            }, step=epoch)
            
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"Average Training Loss: {avg_train_loss:.4f}")
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Total training batches: {train_batch_count}")
            print(f"Total validation batches: {val_batch_count}")
            print("-" * 50)

            # Add learning rate scheduler step based on validation loss
            scheduler.step(avg_val_loss)

            # Save best model if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                
                # Log best model to MLflow
                mlflow.pytorch.log_model(
                    model.module if isinstance(model, nn.DataParallel) else model,
                    "best_model",
                    registered_model_name="DinoVisionTransformerCancerPredictor"
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

        end_time = datetime.datetime.now()
        training_duration = str(end_time - start_time)

        print("\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Log final model to MLflow
        mlflow.pytorch.log_model(
            model.module if isinstance(model, nn.DataParallel) else model,
            "final_model",
            registered_model_name="DinoVisionTransformerCancerPredictor"
        )
        
        # Update model configuration with final metrics
        log_model_config(
            model=model.module if isinstance(model, nn.DataParallel) else model,
            dataset=full_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=num_epochs,
            training_duration=training_duration,
            final_loss=final_loss,
            early_stopping_patience=patience,
            batch_size=batch_size
        )
        
        print("Model configuration updated with final metrics")
        
        print("Model Specifications:")
        print(f"Epochs: {num_epochs}")
        print(f"Loss Function: {criterion}")
        print(f"Optimizer: {optimizer}")
        print(f"Learning Rate Scheduler: {scheduler}")
        print(f"Training Duration: {training_duration}")
        print(f"Final Validation Loss: {final_loss:.4f}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")

        # After training is complete, run inference and log metrics
        print("\nRunning final inference and logging metrics...")
        test_metrics = run_inference_and_log_metrics(
            model=model.module if isinstance(model, nn.DataParallel) else model,
            test_dataset=test_dataset,
            device=device,
            mlflow_run=mlflow_run
        )
        
        print("\nTraining and evaluation complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("\nTest Metrics Summary:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"AUC-ROC: {test_metrics['auc']:.4f}")

if __name__ == "__main__":
    main()