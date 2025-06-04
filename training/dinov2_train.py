import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from classifier import DinoVisionTransformerCancerPredictor
from dataset_loader import PatientDicomDataset
import datetime
import mlflow
import mlflow.pytorch
import numpy as np
import random
import pandas as pd
import yaml
import os
import shutil

from training.helper_functions import check_loss_saturation, collate_fn, create_transforms, ensure_log_dir, log_dataset_info, log_model_config, run_inference_and_log_metrics, setup_mlflow


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

def main():
    # Set CUDA_VISIBLE_DEVICES to use the MIG instance
    os.environ['CUDA_VISIBLE_DEVICES'] = '0.3'  # Use MIG instance 3
    
    # Load configuration
    config_path = 'training/config.yaml'
    config = load_config(config_path)
    
    # Setup MLflow
    experiment_id = setup_mlflow(config['mlflow']['experiment_name'])
    print(f"\nExperiment ID: {experiment_id}", flush=True)
    
    # Set random seed for reproducibility
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Initialize CUDA device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Please run on a machine with GPU support.")
    
    # Get number of CUDA devices
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices detected: {num_devices}", flush=True)
    
    if num_devices == 0:
        # Try to force CUDA initialization
        torch.cuda.init()
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices after forced initialization: {num_devices}", flush=True)
        
        if num_devices == 0:
            # Try to reset CUDA device
            torch.cuda.empty_cache()
            torch.cuda.reset_device()
            num_devices = torch.cuda.device_count()
            print(f"Number of CUDA devices after reset: {num_devices}", flush=True)
            
            if num_devices == 0:
                raise RuntimeError("No CUDA devices found. Please ensure GPU is properly configured.")
    
    # Use the first available device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    # Print detailed device information
    print(f"Using GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"CUDA Version: {torch.version.cuda}", flush=True)
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Current CUDA device: {torch.cuda.current_device()}", flush=True)
    print(f"Device capability: {torch.cuda.get_device_capability(device)}", flush=True)
    print(f"Device memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB", flush=True)
    print(f"Device memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB", flush=True)
    
    # Test CUDA functionality
    test_tensor = torch.zeros(1).to(device)
    print(f"Test tensor device: {test_tensor.device}", flush=True)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Start MLflow run for new training
    with mlflow.start_run(run_name=f"frozen_dinov2_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}") as mlflow_run:
        print(f"\nMLflow run started with ID: {mlflow_run.info.run_id}", flush=True)
        
        # Log config file as artifact
        print("\nLogging config file as artifact...", flush=True)
        mlflow.log_artifact(config_path, "config")
        print("Config file logged successfully", flush=True)
    
        # Set up data transformations
        train_transform = create_transforms(config['transforms']['train'])
        test_transform = create_transforms(config['transforms']['test'])

        # Load datasets
        print("\nLoading train and val datasets...", flush=True)
        full_dataset = PatientDicomDataset(
            config=config,
            is_train=True,
            transform=train_transform
        )
        
        # Load test dataset
        print('\nLoading test dataset...', flush=True)
        test_dataset = PatientDicomDataset(
            config=config,
            is_train=False,
            transform=test_transform
        )
        
        # Split dataset into train and validation
        train_size = int(config['data']['train_val_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Log dataset information
        print("\nLogging dataset information...", flush=True)
        log_dataset_info(train_dataset, 'train', mlflow_run)
        log_dataset_info(val_dataset, 'val', mlflow_run)
        log_dataset_info(test_dataset, 'test', mlflow_run)
        print("Dataset information logged successfully", flush=True)
        
        batch_size = config['training']['batch_size']
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

        # Initialize the model and wrap with DataParallel
        print("\nInitializing model...", flush=True)
        model = DinoVisionTransformerCancerPredictor(config)
        print(f"Model device before to(device): {next(model.parameters()).device}", flush=True)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!", flush=True)
            model = nn.DataParallel(model)
        
        model.to(device)
        print(f"Model device after to(device): {next(model.parameters()).device}", flush=True)

        # Set up loss function and optimizer
        learning_rate = config['training']['learning_rate']
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=config['training']['optimizer']['weight_decay']
        )
        print(f"Optimizer: {optimizer}", flush=True)
        print(f"Learning rate: {learning_rate}", flush=True)

        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=config['training']['scheduler']['patience'],
            factor=config['training']['scheduler']['factor'],
            min_lr=config['training']['scheduler']['min_lr']
        )

        # Add early stopping
        best_val_loss = float('inf')
        patience = config['training']['early_stopping']['patience']
        patience_counter = 0

        # Training loop
        num_epochs = config['training']['num_epochs']
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"\nStarting training with:", flush=True)
        print(f"Batch size: {batch_size}", flush=True)
        print(f"Number of epochs: {num_epochs}", flush=True)
        print(f"Device: {device}", flush=True)
        print(f"Training samples: {len(train_dataset)}", flush=True)
        print(f"Validation samples: {len(val_dataset)}", flush=True)
        print("-" * 50, flush=True)

        start_time = datetime.datetime.now()
        final_loss = None

        # Log training parameters
        mlflow.log_params({
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'early_stopping_patience': patience,
            'train_split': config['data']['train_val_split'],
            'seed': seed,
            'check_slice_thickness': config['data']['check_slice_thickness'],
            'precision': 'float32'  # Log that we're using float32
        })

        # Log model configuration before training starts
        print("\nLogging model configuration...", flush=True)
        log_model_config(
            model=model.module if isinstance(model, nn.DataParallel) else model,
            base_model=config['model']['dinov2_version'],
            dataset=full_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=num_epochs,
            training_duration="0:00:00",  # Will be updated after training
            early_stopping_patience=patience,
            batch_size=batch_size,
            check_slice_thickness=config['data']['check_slice_thickness']
        )
        print("Model configuration logged successfully", flush=True)

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_batch_count = 0
            
            print(f"\nStarting Epoch {epoch+1}/{num_epochs}", flush=True)
            print("Training phase:", flush=True)
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    inputs = batch['images'].to(device)
                    positions = batch['positions'].to(device)
                    labels = batch['labels'].float().to(device)
                    attention_masks = batch['attention_mask'].to(device)
                    
                    optimizer.zero_grad()
                    
                    # Remove mixed precision context
                    outputs = model(inputs, positions, attention_masks)
                    outputs = torch.clamp(outputs, min=-10, max=10)
                    # Check for saturation
                    if batch_idx == 0:  # Check first batch of each epoch
                        check_loss_saturation(outputs, labels)
                    loss = criterion(outputs, labels)
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss detected in batch {batch_idx}", flush=True)
                        print(f"Outputs range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]", flush=True)
                        print(f"Labels range: [{labels.min().item():.2f}, {labels.max().item():.2f}]", flush=True)
                        # Skip this batch
                        continue
                    
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=config['training']['gradient_clipping']['max_norm']
                    )
                    
                    # Remove scaler usage
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item()
                    if not np.isnan(batch_loss):  # Only add non-NaN losses
                        train_loss += batch_loss
                        train_batch_count += 1
                    
                    # Log batch metrics
                    if not np.isnan(batch_loss):
                        mlflow.log_metric('train_batch_loss', batch_loss, step=epoch * len(train_loader) + batch_idx)
                    
                    # Clear cache after each batch
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {e}", flush=True)
                    if "out of memory" in str(e):
                        print(f"OOM in batch {batch_idx}. Exiting program...", flush=True)
                        import sys
                        sys.exit()
                    else:
                        raise e
                
                if not np.isnan(batch_loss):
                    print(f"Batch {batch_idx+1} Loss: {batch_loss:.4f}", flush=True)
            
            # Calculate average loss only from valid batches
            avg_train_loss = train_loss / train_batch_count if train_batch_count > 0 else float('inf')
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            print("\nValidation phase:", flush=True)
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    try:
                        inputs = batch['images'].to(device)
                        positions = batch['positions'].to(device)
                        labels = batch['labels'].float().to(device)
                        attention_masks = batch['attention_mask'].to(device)
                        
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs, positions, attention_masks)
                            outputs = torch.clamp(outputs, min=-10, max=10)
                            loss = criterion(outputs, labels)
                            
                            # Check for NaN loss
                            if torch.isnan(loss):
                                print(f"Warning: NaN loss detected in validation batch {batch_idx}", flush=True)
                                print(f"Outputs range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]", flush=True)
                                print(f"Labels range: [{labels.min().item():.2f}, {labels.max().item():.2f}]", flush=True)
                                continue
                        
                        batch_loss = loss.item()
                        if not np.isnan(batch_loss):  # Only add non-NaN losses
                            val_loss += batch_loss
                            val_batch_count += 1
                        
                        # Log batch metrics
                        if not np.isnan(batch_loss):
                            mlflow.log_metric('val_batch_loss', batch_loss, step=epoch * len(val_loader) + batch_idx)
                        
                    except RuntimeError as e:
                        print(f"Error in validation batch {batch_idx}: {e}", flush=True)
                        if "out of memory" in str(e):
                            print(f"OOM in validation batch {batch_idx}. Exiting program...", flush=True)
                            import sys
                            sys.exit()
                        else:
                            raise e
                    
                    if not np.isnan(batch_loss):
                        print(f"Validation Batch {batch_idx+1} Loss: {batch_loss:.4f}", flush=True)
            
            # Calculate average validation loss only from valid batches
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            
            # Log epoch metrics only if losses are valid
            if not np.isnan(avg_train_loss) and not np.isnan(avg_val_loss):
                mlflow.log_metrics({
                    'train_epoch_loss': avg_train_loss,
                    'val_epoch_loss': avg_val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1
                }, step=epoch)
            
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:", flush=True)
            print(f"Average Training Loss: {avg_train_loss:.4f}", flush=True)
            print(f"Average Validation Loss: {avg_val_loss:.4f}", flush=True)
            print(f"Total training batches: {train_batch_count}", flush=True)
            print(f"Total validation batches: {val_batch_count}", flush=True)
            print("-" * 50, flush=True)

            # Add learning rate scheduler step based on validation loss
            if not np.isnan(avg_val_loss):
                scheduler.step(avg_val_loss)

            # Save best model if validation loss improved
            if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"New best model saved with validation loss: {best_val_loss:.4f}", flush=True)
                
                # Log best model to MLflow
                mlflow.pytorch.log_model(
                    model.module if isinstance(model, nn.DataParallel) else model,
                    "best_model",
                    registered_model_name=config['mlflow']['model_name']
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("\nEarly stopping triggered!", flush=True)
                break

        end_time = datetime.datetime.now()
        training_duration = str(end_time - start_time)

        print("\nTraining complete!", flush=True)
        print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
        
        # Log final model to MLflow
        mlflow.pytorch.log_model(
            model.module if isinstance(model, nn.DataParallel) else model,
            "final_model",
            registered_model_name=config['mlflow']['model_name']
        )
        
        # Update model configuration with final metrics
        log_model_config(
            model=model.module if isinstance(model, nn.DataParallel) else model,
            base_model=config['model']['dinov2_version'],
            dataset=full_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=num_epochs,
            training_duration=training_duration,
            early_stopping_patience=patience,
            batch_size=batch_size
        )
        
        print("Model configuration updated with final metrics", flush=True)
        
        print("Model Specifications:", flush=True)
        print(f"Epochs: {num_epochs}", flush=True)
        print(f"Loss Function: {criterion}", flush=True)
        print(f"Optimizer: {optimizer}", flush=True)
        print(f"Learning Rate Scheduler: {scheduler}", flush=True)
        print(f"Training Duration: {training_duration}", flush=True)
        print(f"Best Validation Loss: {best_val_loss:.4f}", flush=True)

        # After training is complete, run inference and log metrics
        print("\nRunning final inference and logging metrics...", flush=True)
        test_metrics = run_inference_and_log_metrics(
            model=model.module if isinstance(model, nn.DataParallel) else model,
            test_dataset=test_dataset,
            device=device,
            mlflow_run=mlflow_run
        )
        
        print("\nTraining and evaluation complete!", flush=True)
        print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
        print("\nTest Metrics Summary:", flush=True)
        print(f"Accuracy: {test_metrics['accuracy']:.4f}", flush=True)
        print(f"AUC-ROC: {test_metrics['auc']:.4f}", flush=True)

if __name__ == "__main__":
    main()