import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from classifier import DinoVisionTransformerCancerPredictor
from dataset_loader import PatientDicomDataset


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

def main():
    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = PatientDicomDataset(
        root_dir='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_cancer_imaging_archive',
        labels_file='/home/vhari/dom_ameen_chi_link/common/SENTINL0/dinov2/nlst_actual.csv',
        transform=transform
    )
    
    batch_size = 2

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize the model
    model = DinoVisionTransformerCancerPredictor()

    # Set up loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Add early stopping
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Add checkpointing
    def save_checkpoint(model, optimizer, epoch, loss, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

    # Training loop
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Enable gradient checkpointing in the transformer
    # model.transformer.gradient_checkpointing_enable()
    
    # Use automatic mixed precision
    scaler = torch.amp.GradScaler()

    print(f"\nStarting training with:")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Device: {device}")
    print("-" * 50)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        print(f"\nStarting Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                print(f"Batch {batch_idx+1} shape: {batch['images'].shape}")
                inputs = batch['images'].to(device)
                positions = batch['positions'].to(device)
                labels = batch['labels'].float().to(device)
                attention_masks = batch['attention_mask'].to(device)
                
                optimizer.zero_grad()
                
                # Use AMP
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs, positions, attention_masks)
                    print(f"Outputs shape: {outputs.shape}")
                    loss = criterion(outputs, labels)
                
                # Scale loss and backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                batch_loss = loss.item()
                total_loss += batch_loss
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                if "out of memory" in str(e):
                    print(f"OOM in batch {batch_idx}. Clearing cache and skipping...")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    continue
                raise e
            
            batch_count += 1
            
            # Print batch loss every 10 batches
            # if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1} Loss: {batch_loss:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Total batches processed: {batch_count}")
        print("-" * 50)

        # Add learning rate scheduler step
        scheduler.step(avg_loss)

        # Add early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("\nEarly stopping triggered!")
            break

    print("\nTraining complete!")

    # Save the model
    torch.save(model.state_dict(), 'model/dinov2_cancer_predictor.pth') 
    print(f"Model saved as: dinov2_cancer_predictor.pth")
    
if __name__ == "__main__":
    main()