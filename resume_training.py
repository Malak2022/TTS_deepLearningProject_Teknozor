"""
Resume training from checkpoint
"""

import os
import sys
import torch
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tacotron2.configs.config import Config

def find_latest_checkpoint():
    """Find the most recent checkpoint"""
    config = Config()
    checkpoint_dir = config.CHECKPOINT_DIR
    
    # Look for different types of checkpoints
    checkpoint_patterns = [
        'safety_checkpoint.pth',
        'best_model_fast.pth',
        'best_model.pth',
        'checkpoint_epoch_*.pth'
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        checkpoints.extend(files)
    
    if not checkpoints:
        print("No checkpoints found!")
        return None
    
    # Get the most recent checkpoint
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def resume_training_from_checkpoint(checkpoint_path):
    """Resume training from a specific checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Resuming training from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint contents:")
    for key in checkpoint.keys():
        if key in ['train_losses', 'val_losses']:
            print(f"  {key}: {len(checkpoint[key])} values")
        else:
            print(f"  {key}: {checkpoint[key] if not key.endswith('_state_dict') else 'state_dict loaded'}")
    
    # Import training function and modify it to resume
    from train_tacotron2_fast import train_model_fast
    
    # You would modify the training function to load from this checkpoint
    print("\nTo resume training, modify train_tacotron2_fast.py to load this checkpoint")
    print("Or run the training script - it will automatically find and load the latest checkpoint")

def main():
    """Main function"""
    print("=== Resume Training Tool ===\n")
    
    # Find latest checkpoint
    latest_checkpoint = find_latest_checkpoint()
    
    if latest_checkpoint:
        print(f"Latest checkpoint found: {latest_checkpoint}")
        resume_training_from_checkpoint(latest_checkpoint)
    else:
        print("No checkpoints found. Start training from scratch.")

if __name__ == '__main__':
    main()
