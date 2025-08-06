"""
Fast training script for Tacotron2 with smaller dataset and batch size
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from tacotron2.configs.config import Config
from tacotron2.models.tacotron2 import Tacotron2, HParams
from tacotron2.data.dataset import LJSpeechDataset, TextMelCollate
from tacotron2.training.loss import Tacotron2Loss

def create_fast_dataloaders(config, max_samples=1000, train_split=0.9):
    """Create dataloaders with limited samples for fast training"""
    # Load dataset
    dataset = LJSpeechDataset(config.METADATA_FILE, config.MAX_TEXT_LENGTH)
    
    # Limit dataset size for fast training
    print(f"Original dataset size: {len(dataset.data)}")
    dataset.data = dataset.data[:max_samples]  # Use only first max_samples
    print(f"Using {len(dataset.data)} samples for fast training")
    
    # Split dataset
    dataset_size = len(dataset.data)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create collate function
    collate_fn = TextMelCollate(config.N_FRAMES_PER_STEP)
    
    # Create dataloaders with smaller batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,  # Now 8 instead of 32
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"Created fast dataloaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print(f"Batches per epoch: {len(train_loader)} (batch size: {config.BATCH_SIZE})")
    return train_loader, val_loader

def train_model_fast():
    """Fast training function with progress tracking"""
    # Initialize config
    config = Config()
    config.create_dirs()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    hparams = HParams()
    model = Tacotron2(hparams).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize loss and optimizer
    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Create fast dataloaders (only 1000 samples)
    train_loader, val_loader = create_fast_dataloaders(config, max_samples=1000)
    
    # Training loop
    print(f"\nStarting fast training...")
    print(f"Epochs: {config.N_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Expected time per epoch: ~{len(train_loader) * 20 / 60:.1f} minutes")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config.N_EPOCHS):
        # Training
        model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.N_EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
            text_padded = text_padded.to(device)
            input_lengths = input_lengths.to(device)
            mel_padded = mel_padded.to(device)
            gate_padded = gate_padded.to(device)
            output_lengths = output_lengths.to(device)
            
            # Forward pass
            inputs = (text_padded, input_lengths, mel_padded, torch.max(input_lengths).item(), output_lengths)
            targets = (mel_padded, gate_padded)
            
            try:
                model_output = model(inputs)
                total_loss, mel_loss, gate_loss = criterion(model_output, targets)
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_THRESH)
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'Mel': f"{mel_loss.item():.4f}",
                    'Gate': f"{gate_loss.item():.4f}"
                })

                # Save checkpoint every 20 batches for safety
                if (batch_idx + 1) % 20 == 0:
                    safety_checkpoint = {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_losses': train_losses,
                        'current_loss': total_loss.item()
                    }
                    torch.save(safety_checkpoint, os.path.join(config.CHECKPOINT_DIR, 'safety_checkpoint.pth'))
                    print(f"\n💾 Safety checkpoint saved at epoch {epoch+1}, batch {batch_idx+1}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate average epoch loss
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        train_losses.append(avg_train_loss)
        
        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == config.N_EPOCHS - 1:
            model.eval()
            val_epoch_losses = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 10:  # Only validate on first 10 batches for speed
                        break
                        
                    text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
                    text_padded = text_padded.to(device)
                    input_lengths = input_lengths.to(device)
                    mel_padded = mel_padded.to(device)
                    gate_padded = gate_padded.to(device)
                    output_lengths = output_lengths.to(device)
                    
                    inputs = (text_padded, input_lengths, mel_padded, torch.max(input_lengths).item(), output_lengths)
                    targets = (mel_padded, gate_padded)
                    
                    try:
                        model_output = model(inputs)
                        total_loss, mel_loss, gate_loss = criterion(model_output, targets)
                        val_epoch_losses.append(total_loss.item())
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        continue
            
            avg_val_loss = np.mean(val_epoch_losses) if val_epoch_losses else float('inf')
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1:2d} - Train Loss: {avg_train_loss:7.4f}, Val Loss: {avg_val_loss:7.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
                torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, 'best_model_fast.pth'))
                print(f"💾 Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1:2d} - Train Loss: {avg_train_loss:7.4f}")
    
    # Save final model
    final_checkpoint = {
        'epoch': config.N_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(final_checkpoint, os.path.join(config.CHECKPOINT_DIR, 'final_model_fast.pth'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(range(0, len(train_losses), 5), val_losses, 'o-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-20:], label='Train Loss (Last 20)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Recent Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress_fast.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Fast training completed!")
    print(f"📊 Final train loss: {train_losses[-1]:.4f}")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Models saved in: {config.CHECKPOINT_DIR}")
    print(f"📈 Training plot saved as: training_progress_fast.png")

if __name__ == '__main__':
    train_model_fast()
