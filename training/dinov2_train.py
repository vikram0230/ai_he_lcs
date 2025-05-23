import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from classifier import DinoVisionTransformerCancerPredictor
from dataset_loader import PatientDicomDataset
import datetime
import mlflow
import mlflow.pytorch
import numpy as np
import random

from training.config import check_loss_saturation, collate_fn, ensure_log_dir, log_model_config, run_inference_and_log_metrics, setup_mlflow


def main():
    # Setup MLflow
    experiment_id = setup_mlflow()
    print(f"Experiment ID: {experiment_id}", flush=True)
    
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
        print(f"MLflow run started with ID: {mlflow_run.info.run_id}", flush=True)
        
        # Set up data transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
        
        # Load test dataset
        test_dataset = PatientDicomDataset(
            root_dir='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_test_data',
            labels_file='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_actual.csv',
            transform=transform,
            patient_scan_count=50
        )
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        batch_size = 1
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

        # Initialize the model and wrap with DataParallel
        model = DinoVisionTransformerCancerPredictor()
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!", flush=True)
            model = nn.DataParallel(model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set up loss function and optimizer
        learning_rate = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(f"Optimizer: {optimizer}", flush=True)
        print(f"Learning rate: {learning_rate}", flush=True)

        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, min_lr=1e-7)

        # Add early stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        # Training loop
        num_epochs = 30
        criterion = nn.BCEWithLogitsLoss()
        
        # Use automatic mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
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
            'train_size': 0.8,
            'seed': seed
        })

        # Log model configuration before training starts
        print("\nLogging model configuration...", flush=True)
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
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs, positions, attention_masks)
                        print(f"Outputs shape: {outputs.shape}", flush=True)
                        print(f"Outputs: {outputs}", flush=True)
                        print(f"Probability: {torch.sigmoid(outputs)}", flush=True)
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    batch_loss = loss.item()
                    if not np.isnan(batch_loss):  # Only add non-NaN losses
                        train_loss += batch_loss
                        train_batch_count += 1
                    
                    # Log batch metrics
                    if not np.isnan(batch_loss):
                        mlflow.log_metric('train_batch_loss', batch_loss, step=epoch * len(train_loader) + batch_idx)
                    
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
                            print(f"Outputs shape: {outputs.shape}", flush=True)
                            print(f"Outputs: {outputs}", flush=True)
                            print(f"Probability: {torch.sigmoid(outputs)}", flush=True)
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
                    registered_model_name="DinoVisionTransformerCancerPredictor"
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
        
        print("Model configuration updated with final metrics", flush=True)
        
        print("Model Specifications:", flush=True)
        print(f"Epochs: {num_epochs}", flush=True)
        print(f"Loss Function: {criterion}", flush=True)
        print(f"Optimizer: {optimizer}", flush=True)
        print(f"Learning Rate Scheduler: {scheduler}", flush=True)
        print(f"Training Duration: {training_duration}", flush=True)
        print(f"Final Validation Loss: {final_loss:.4f}", flush=True)
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