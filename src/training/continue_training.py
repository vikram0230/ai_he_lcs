import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import numpy as np
import random
from src.models.dinov2_classifier import DinoVisionTransformerCancerPredictor
from src.data.dataset_loader import PatientDicomDataset
import mlflow
import mlflow.pytorch
from dinov2_train import retrain_model, load_model_from_mlflow, setup_mlflow, collate_fn, log_model_config

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_last_epoch(run_id):
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

def get_training_params(run_id):
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
        'train_size': float(params.get('train_size', 0.8)),
        'seed': int(params.get('seed', 42))
    }
    
    return training_params

def main():
    # Setup MLflow
    experiment_id = setup_mlflow()
    
    # Load the best model from MLflow
    run_id = "YOUR_RUN_ID"  # Get this from MLflow UI
    print(f"Loading model from run ID: {run_id}")
    
    # Get training parameters from original run
    params = get_training_params(run_id)
    print("\nOriginal training parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # Set random seed for reproducibility
    set_seed(params['seed'])
    print(f"\nSet random seed to {params['seed']}")
    
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
        patient_scan_count=200
    )
    
    # Split dataset into train and validation using the same ratio as original training
    train_size = int(params['train_size'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(params['seed']))
    
    print(f"\nDataset sizes:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get the last epoch from the previous training
    start_epoch = get_last_epoch(run_id)
    print(f"Continuing from epoch {start_epoch}")
    
    model = load_model_from_mlflow(run_id, model_name="best_model")
    
    # Setup optimizer and scheduler for logging
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([112.83]).to(device))
    
    # Log model configuration before training
    print("\nLogging model configuration...")
    log_model_config(
        model=model,
        dataset=full_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=params['num_epochs'],
        training_duration="0:00:00",  # Will be updated after training
        final_loss=None,  # Will be updated after training
        early_stopping_patience=params['patience'],
        batch_size=params['batch_size']
    )
    print("Model configuration logged successfully")
    
    # Continue training with parameters from original run
    print("\nStarting retraining with parameters:")
    print(f"Starting from epoch: {start_epoch}")
    print(f"Number of epochs: {params['num_epochs']}")
    print(f"Learning rate: {params['learning_rate']}")
    print(f"Patience: {params['patience']}")
    print(f"Batch size: {params['batch_size']}")
    
    retrained_model, final_loss, duration, final_epoch = retrain_model(
        run_id=run_id,
        model=model,
        num_epochs=params['num_epochs'],
        learning_rate=params['learning_rate'],
        patience=params['patience'],
        dataset=train_dataset,  # Use train_dataset instead of full_dataset
        device=device,
        start_epoch=start_epoch
    )
    
    print("\nRetraining complete!")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Training duration: {duration}")
    print(f"Final epoch: {final_epoch}")

if __name__ == "__main__":
    main() 