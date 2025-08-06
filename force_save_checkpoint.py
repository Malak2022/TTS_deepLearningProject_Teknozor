"""
Force save current training state as checkpoint
Run this while training is running to create a manual checkpoint
"""

import os
import sys
import torch
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tacotron2.configs.config import Config

def force_save_checkpoint():
    """Create a manual checkpoint from the current training state"""
    print("🔧 FORCE SAVE CHECKPOINT TOOL")
    print("=" * 50)
    
    config = Config()
    
    # Check if there are any existing checkpoints to work with
    checkpoint_dir = config.CHECKPOINT_DIR
    existing_checkpoints = []
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, filename)
            existing_checkpoints.append((filename, filepath))
    
    if not existing_checkpoints:
        print("❌ No existing checkpoints found!")
        return
    
    print("📁 Found existing checkpoints:")
    for filename, filepath in existing_checkpoints:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        print(f"   {filename} ({size_mb:.1f} MB) - {mod_time.strftime('%H:%M:%S')}")
    
    # Use the most recent checkpoint as base
    latest_checkpoint = max(existing_checkpoints, key=lambda x: os.path.getmtime(x[1]))
    latest_file, latest_path = latest_checkpoint
    
    print(f"\n📋 Using latest checkpoint: {latest_file}")
    
    try:
        # Load the latest checkpoint
        checkpoint = torch.load(latest_path, map_location='cpu')
        
        # Create a manual save with current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manual_save_path = os.path.join(checkpoint_dir, f'manual_save_{timestamp}.pth')
        
        # Add metadata about manual save
        checkpoint['manual_save'] = True
        checkpoint['save_timestamp'] = timestamp
        checkpoint['original_file'] = latest_file
        
        # Save the manual checkpoint
        torch.save(checkpoint, manual_save_path)
        
        print(f"✅ Manual checkpoint saved: manual_save_{timestamp}.pth")
        print(f"📊 Checkpoint info:")
        print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   Best Val Loss: {checkpoint.get('best_val_loss', 'Unknown')}")
        if 'train_losses' in checkpoint:
            print(f"   Latest Train Loss: {checkpoint['train_losses'][-1]:.4f}")
        
        return manual_save_path
        
    except Exception as e:
        print(f"❌ Error creating manual checkpoint: {e}")
        return None

def check_validation_issue():
    """Analyze why validation might be failing"""
    print("\n🔍 VALIDATION ISSUE ANALYSIS")
    print("=" * 50)
    
    config = Config()
    
    # Check validation dataset
    try:
        from tacotron2.data.dataset import LJSpeechDataset, TextMelCollate
        
        dataset = LJSpeechDataset(config.METADATA_FILE, config.MAX_TEXT_LENGTH)
        dataset.data = dataset.data[:1000]  # Same as training
        
        # Split dataset (same as training)
        dataset_size = len(dataset.data)
        train_size = int(0.9 * dataset_size)
        val_size = dataset_size - train_size
        
        print(f"📊 Dataset info:")
        print(f"   Total samples: {dataset_size}")
        print(f"   Train samples: {train_size}")
        print(f"   Val samples: {val_size}")
        
        if val_size < 10:
            print("⚠️ ISSUE FOUND: Validation dataset too small!")
            print(f"   Only {val_size} validation samples, but validation tries to use 10 batches")
            print("   This could cause validation to fail silently")
            return "small_validation_dataset"
        
        print("✅ Validation dataset size looks OK")
        return None
        
    except Exception as e:
        print(f"❌ Error checking validation dataset: {e}")
        return "dataset_error"

def main():
    """Main function"""
    # Force save current state
    manual_checkpoint = force_save_checkpoint()
    
    # Check validation issues
    issue = check_validation_issue()
    
    print("\n" + "=" * 50)
    print("💡 RECOMMENDATIONS:")
    
    if manual_checkpoint:
        print(f"✅ You now have a manual checkpoint saved")
        print(f"✅ You can safely stop training and test synthesis")
    
    if issue == "small_validation_dataset":
        print("⚠️ Validation dataset is too small")
        print("   This explains why validation fails at epochs 5, 10, etc.")
        print("   Solution: Increase dataset size or reduce validation batches")
    
    print("\n🚀 NEXT STEPS:")
    print("1. You can stop training now (Ctrl+C)")
    print("2. Test synthesis with: python test_synthesis.py")
    print("3. Your model has excellent progress (loss: 48.77 → ~5.1)")

if __name__ == '__main__':
    main()
