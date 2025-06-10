import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from training.dinov2_classifier import DinoVisionTransformerCancerPredictor
from training.rad_dino_classifier import RadDinoClassifier
from training.dataset_loader import PatientDicomDataset
import datetime
import mlflow
import mlflow.pytorch
import numpy as np
import random
import yaml
import os

from training.helper_functions import check_loss_saturation, collate_fn, create_transforms, ensure_log_dir, log_dataset_info, log_model_config, run_inference_and_log_metrics, setup_mlflow

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model(config):
    """Initialize the appropriate model based on config."""
    model_type = config['model'].get('type', 'dinov2')  # Default to DINOv2 if not specified
    
    # Set model name based on type
    if model_type == 'dinov2':
        config['model']['name'] = 'DinoVisionTransformerCancerPredictor'
        config['mlflow']['model_name'] = 'DinoVisionTransformerCancerPredictor'
        print("\nInitializing DINOv2 model...", flush=True)
        return DinoVisionTransformerCancerPredictor(config)
    elif model_type == 'rad_dino':
        config['model']['name'] = 'RadDinoClassifier'
        config['mlflow']['model_name'] = 'RadDinoClassifier'
        config['model']['dinov2_version'] = 'vitb14'
        print("\nInitializing RAD-DINO model...", flush=True)
        return RadDinoClassifier(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types are: 'dinov2', 'rad_dino'")

def print_memory_stats(prefix=""):
    """Print detailed memory statistics"""
    print(f"\n{prefix} Memory Statistics:", flush=True)
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB", flush=True)
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB", flush=True)
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB", flush=True)
    print(f"Max Cached: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB", flush=True)
    print(torch.cuda.memory_summary(), flush=True)

def check_memory_leak(prev_allocated, prev_cached, threshold_mb=100):
    """Check if memory usage has increased significantly"""
    current_allocated = torch.cuda.memory_allocated() / 1024**2
    current_cached = torch.cuda.memory_reserved() / 1024**2
    
    allocated_increase = current_allocated - prev_allocated
    cached_increase = current_cached - prev_cached
    
    if allocated_increase > threshold_mb or cached_increase > threshold_mb:
        print(f"\nWARNING: Potential memory leak detected!", flush=True)
        print(f"Allocated memory increased by: {allocated_increase:.2f} MB", flush=True)
        print(f"Cached memory increased by: {cached_increase:.2f} MB", flush=True)
        return True
    return False

def main():
    # Set PyTorch memory allocator settings to handle fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
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
    
    # Verify CUDA device is properly set
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Use the appropriate device
    device = torch.device("cuda:0")
    
    # Print detailed device information
    try:
        print(f"Using GPU: {torch.cuda.get_device_name(device)}", flush=True)
        print(f"CUDA Version: {torch.version.cuda}", flush=True)
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"Current CUDA device: {torch.cuda.current_device()}", flush=True)
        print(f"Device capability: {torch.cuda.get_device_capability(device)}", flush=True)
        print(f"Device memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB", flush=True)
        print(f"Device memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB", flush=True)
    except Exception as e:
        print(f"Warning: Could not print device information: {e}", flush=True)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Start MLflow run for new training
    model_type = config['model'].get('type', 'dinov2')
    mlflow_run = mlflow.start_run(run_name=f"{model_type}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    print('Loading test dataset...', flush=True)
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
    
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize the model
    print("\nInitializing model...", flush=True)
    model = get_model(config)
    model = model.to(device)
    
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
        'precision': 'float32'
    })

    # Log model configuration before training starts
    print("\nLogging model configuration...", flush=True)
    log_model_config(
        model=model,
        base_model=config['model']['dinov2_version'],
        dataset=full_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=num_epochs,
        training_duration="0:00:00",
        early_stopping_patience=patience,
        batch_size=batch_size,
        check_slice_thickness=config['data']['check_slice_thickness']
    )
    print("Model configuration logged successfully", flush=True)

    # Initialize memory tracking
    initial_allocated = torch.cuda.memory_allocated() / 1024**2
    initial_cached = torch.cuda.memory_reserved() / 1024**2
    print("\nInitial Memory State:", flush=True)
    print(f"Initial Allocated: {initial_allocated:.2f} MB", flush=True)
    print(f"Initial Cached: {initial_cached:.2f} MB", flush=True)

    for epoch in range(num_epochs):
        # Track memory at start of epoch
        epoch_start_allocated = torch.cuda.memory_allocated() / 1024**2
        epoch_start_cached = torch.cuda.memory_reserved() / 1024**2
        
        # Training phase
        model.train()
        train_loss = 0
        train_batch_count = 0
        
        print(f"\nStarting Epoch {epoch+1}/{num_epochs}", flush=True)
        print("Training phase:", flush=True)
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Track memory before batch
                batch_start_allocated = torch.cuda.memory_allocated() / 1024**2
                batch_start_cached = torch.cuda.memory_reserved() / 1024**2
                
                inputs = batch['images'].to(device)
                positions = batch['positions'].to(device)
                labels = batch['labels'].float().to(device)
                attention_masks = batch['attention_mask'].to(device)
                
                clinical_features = batch.get('clinical_features')
                if clinical_features is not None:
                    clinical_features = clinical_features.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with or without clinical features
                if clinical_features is not None:
                    outputs = model(inputs, positions, attention_masks, clinical_features)
                else:
                    outputs = model(inputs, positions, attention_masks)
                
                outputs = torch.clamp(outputs, min=-10, max=10)
                
                if batch_idx == 0:
                    check_loss_saturation(outputs, labels)
                
                loss = criterion(outputs, labels)
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected in batch {batch_idx}", flush=True)
                    print(f"Outputs range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]", flush=True)
                    print(f"Labels range: [{labels.min().item():.2f}, {labels.max().item():.2f}]", flush=True)
                    continue
                
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config['training']['gradient_clipping']['max_norm']
                )
                
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
                if not np.isnan(batch_loss):
                    train_loss += batch_loss
                    train_batch_count += 1
                
                if not np.isnan(batch_loss):
                    mlflow.log_metric('train_batch_loss', batch_loss, step=epoch * len(train_loader) + batch_idx)
                    print(f"Batch {batch_idx+1} Loss: {batch_loss:.4f}", flush=True)
                
                # Check for memory leaks after batch
                if check_memory_leak(batch_start_allocated, batch_start_cached):
                    print_memory_stats("After batch processing")
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}", flush=True)
                if "out of memory" in str(e):
                    print(f"OOM in batch {batch_idx}. Memory state:", flush=True)
                    print_memory_stats("OOM Error")
                    import sys
                    sys.exit()
                else:
                    raise e
        
        # Calculate average loss only from valid batches
        avg_train_loss = train_loss / train_batch_count if train_batch_count > 0 else float('inf')
        
        # Check for memory leaks after epoch
        if check_memory_leak(epoch_start_allocated, epoch_start_cached):
            print_memory_stats("After epoch completion")
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        print("\nValidation phase:", flush=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # Track memory before validation batch
                    val_batch_start_allocated = torch.cuda.memory_allocated() / 1024**2
                    val_batch_start_cached = torch.cuda.memory_reserved() / 1024**2
                    
                    inputs = batch['images'].to(device)
                    positions = batch['positions'].to(device)
                    labels = batch['labels'].float().to(device)
                    attention_masks = batch['attention_mask'].to(device)
                    
                    clinical_features = batch.get('clinical_features')
                    if clinical_features is not None:
                        clinical_features = clinical_features.to(device)
                    
                    # Forward pass with or without clinical features
                    if clinical_features is not None:
                        outputs = model(inputs, positions, attention_masks, clinical_features)
                    else:
                        outputs = model(inputs, positions, attention_masks)
                    
                    outputs = torch.clamp(outputs, min=-10, max=10)
                    loss = criterion(outputs, labels)
                    
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss detected in validation batch {batch_idx}", flush=True)
                        print(f"Outputs range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]", flush=True)
                        print(f"Labels range: [{labels.min().item():.2f}, {labels.max().item():.2f}]", flush=True)
                        continue
                    
                    batch_loss = loss.item()
                    if not np.isnan(batch_loss):
                        val_loss += batch_loss
                        val_batch_count += 1
                    
                    if not np.isnan(batch_loss):
                        mlflow.log_metric('val_batch_loss', batch_loss, step=epoch * len(val_loader) + batch_idx)
                        print(f"Validation Batch {batch_idx+1} Loss: {batch_loss:.4f}", flush=True)
                    
                    # Check for memory leaks after validation batch
                    if check_memory_leak(val_batch_start_allocated, val_batch_start_cached):
                        print_memory_stats("After validation batch")
                    
                    # Clear memory
                    del inputs, positions, labels, attention_masks, outputs, loss
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    print(f"Error in validation batch {batch_idx}: {e}", flush=True)
                    if "out of memory" in str(e):
                        print(f"OOM in validation batch {batch_idx}. Memory state:", flush=True)
                        print_memory_stats("OOM Error")
                        import sys
                        sys.exit()
                    else:
                        raise e
        
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
                model,
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
        model,
        "final_model",
        registered_model_name=config['mlflow']['model_name']
    )
    
    # Update model configuration with final metrics
    log_model_config(
        model=model,
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
        model=model,
        test_dataset=test_dataset,
        device=device,
        mlflow_run=mlflow_run
    )

if __name__ == "__main__":
    main()