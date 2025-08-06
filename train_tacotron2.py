"""
Simple training script for Tacotron2
Run this script from the project root directory
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

def create_simple_dataloaders(config, train_split=0.9):
    """Create simple dataloaders"""
    # Load dataset
    dataset = LJSpeechDataset(config.METADATA_FILE, config.MAX_TEXT_LENGTH)
    
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create collate function
    collate_fn = TextMelCollate(config.N_FRAMES_PER_STEP)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
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
    
    print(f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    return train_loader, val_loader

def train_model():
    """Main training function"""
    # Initialize config
    config = Config()
    config.create_dirs()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    hparams = HParams()
    model = Tacotron2(hparams).to(device)
    
    # Initialize loss and optimizer
    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Create dataloaders
    train_loader, val_loader = create_simple_dataloaders(config)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.N_EPOCHS):
        # Training
        model.train()
        train_losses = []
        
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
                
                train_losses.append(total_loss.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'Mel': f"{mel_loss.item():.4f}",
                    'Gate': f"{gate_loss.item():.4f}"
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
            
            # Break after a few batches for testing
            if batch_idx >= 10:  # Only train on first 10 batches for testing
                break
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 5:  # Only validate on first 5 batches
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
                        val_losses.append(total_loss.item())
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {e}")
                        continue
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
            
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))
    
    print("Training completed!")

if __name__ == '__main__':
    train_model()
