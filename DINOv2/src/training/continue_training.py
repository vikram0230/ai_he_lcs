import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torch.utils.data import random_split
from torch import nn
import numpy as np
import random
from src.data.dataset_loader import PatientDicomDataset
import mlflow
import yaml
from utils.helpers import HelperUtils


def main():
    # Initialize helper utils
    helper = HelperUtils()
    
    # Parse command line arguments
    args = helper.parse_args(mode='continue')
    
    # Get the run ID from command line arguments
    run_id = args.run_id
    if not run_id:
        raise ValueError("Please provide a run_id to continue training from")
    
    print(f"Loading model from run ID: {run_id}")
    
    # Get model configuration from original run
    config = helper.get_model_config(run_id)
    print("\nLoaded configuration from original run")
    
    # Get training parameters from original run
    params = helper.get_training_params(run_id)
    print("\nOriginal training parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # Set random seed for reproducibility
    helper.set_seed(params['seed'])
    print(f"\nSet random seed to {params['seed']}")
    
    # Set up data transformations using helper
    train_transform = helper.create_transforms(config['transforms']['train'])
    test_transform = helper.create_transforms(config['transforms']['test'])

    # Load datasets
    print("\nLoading train and val datasets...")
    full_dataset = PatientDicomDataset(
        config=config,
        is_train=True,
        transform=train_transform
    )
    
    # Split dataset into train and validation using the same ratio as original training
    train_size = int(params['train_size'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(params['seed'])
    )
    
    print(f"\nDataset sizes:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model and its state using helper
    model = helper.load_model_from_mlflow(run_id, model_name="final_model")
    model = model.to(device)
    
    # Setup optimizer with the same parameters
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=params['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # Setup scheduler with the same parameters
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        patience=config['training']['scheduler']['patience'],
        factor=config['training']['scheduler']['factor'],
        min_lr=config['training']['scheduler']['min_lr']
    )
    
    # Setup loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Setup MLflow for the continuation run using helper
    experiment_id = helper.setup_mlflow(config['mlflow']['experiment_name'])
    mlflow_run = mlflow.start_run(run_name=f"continuation_{run_id}")
    
    # Initialize start_epoch and best_loss
    start_epoch = 0
    best_loss = float('inf')
    
    # Check if we should resume from a checkpoint
    if args.resume and args.checkpoint:
        start_epoch, best_loss = helper.resume_training(model, optimizer, args.checkpoint)
    else:
        # Get the last epoch from the previous training
        start_epoch = helper.get_last_epoch(run_id)
        print(f"Continuing from epoch {start_epoch}")
    
    # Log model configuration before training using helper
    print("\nLogging model configuration...")
    helper.log_model_config(
        model=model,
        base_model=config['model']['dinov2_version'],
        dataset=full_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=params['patience'],
        batch_size=params['batch_size'],
        check_slice_thickness=params['check_slice_thickness']
    )
    print("Model configuration logged successfully")
    
    # Continue training with parameters from original run
    print("\nStarting retraining with parameters:")
    print(f"Starting from epoch: {start_epoch}")
    print(f"Number of epochs: {config['training']['num_epochs']}")
    print(f"Learning rate: {params['learning_rate']}")
    print(f"Patience: {params['patience']}")
    print(f"Batch size: {params['batch_size']}")
    
    retrained_model, final_loss, duration, final_epoch = helper.retrain_model(
        run_id=run_id,
        model=model,
        num_epochs=config['training']['num_epochs'],
        learning_rate=params['learning_rate'],
        patience=params['patience'],
        dataset=train_dataset,
        device=device,
        start_epoch=start_epoch
    )
    
    print("\nRetraining complete!")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Training duration: {duration}")
    print(f"Final epoch: {final_epoch}")

if __name__ == "__main__":
    main() 