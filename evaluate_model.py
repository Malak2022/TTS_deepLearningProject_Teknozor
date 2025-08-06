"""
Evaluate trained Tacotron2 model
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tacotron2.configs.config import Config
from tacotron2.models.tacotron2 import Tacotron2, HParams

def evaluate_training_progress():
    """Evaluate training progress from checkpoints"""
    print("=== TRAINING EVALUATION ===\n")
    
    config = Config()
    checkpoint_dir = config.CHECKPOINT_DIR
    
    # Check available checkpoints
    checkpoints = []
    checkpoint_files = ['best_model_fast.pth', 'final_model_fast.pth', 'best_model.pth']
    
    for filename in checkpoint_files:
        filepath = os.path.join(checkpoint_dir, filename)
        if os.path.exists(filepath):
            checkpoints.append((filename, filepath))
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return None
    
    print("📁 Available checkpoints:")
    for name, path in checkpoints:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✅ {name} ({size_mb:.1f} MB)")
    
    # Load the best checkpoint
    best_checkpoint_path = checkpoints[0][1]  # Use first available
    print(f"\n📊 Loading checkpoint: {checkpoints[0][0]}")
    
    try:
        checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\n=== TRAINING RESULTS ===")
        print(f"🏁 Completed Epochs: {checkpoint.get('epoch', 'Unknown')}")
        print(f"🎯 Best Validation Loss: {checkpoint.get('best_val_loss', 'Unknown'):.4f}")
        
        # Analyze training progress
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            print(f"📈 Training Progress:")
            print(f"   Start Loss: {train_losses[0]:.4f}")
            print(f"   Final Loss: {train_losses[-1]:.4f}")
            print(f"   Improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
            
            # Plot training curve
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss', color='blue')
            if 'val_losses' in checkpoint and checkpoint['val_losses']:
                val_losses = checkpoint['val_losses']
                val_epochs = list(range(0, len(train_losses), len(train_losses) // len(val_losses)))[:len(val_losses)]
                plt.plot(val_epochs, val_losses, 'ro-', label='Validation Loss', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            # Show recent progress (last 20 epochs)
            recent_losses = train_losses[-20:] if len(train_losses) > 20 else train_losses
            plt.plot(range(len(train_losses) - len(recent_losses), len(train_losses)), recent_losses, 'g-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Recent Training Progress')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('training_evaluation.png', dpi=150, bbox_inches='tight')
            print(f"📊 Training plot saved: training_evaluation.png")
            plt.show()
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def check_model_quality(checkpoint):
    """Assess model quality based on loss values"""
    print("\n=== MODEL QUALITY ASSESSMENT ===")
    
    if not checkpoint:
        print("❌ No checkpoint to evaluate")
        return
    
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"🎯 Validation Loss: {best_val_loss:.4f}")
    
    if best_val_loss < 2.0:
        quality = "🌟 EXCELLENT"
        recommendation = "Ready for high-quality synthesis!"
    elif best_val_loss < 5.0:
        quality = "✅ GOOD"
        recommendation = "Should produce decent quality speech"
    elif best_val_loss < 10.0:
        quality = "⚠️ FAIR"
        recommendation = "May need more training for better quality"
    else:
        quality = "❌ POOR"
        recommendation = "Needs significantly more training"
    
    print(f"📊 Quality Assessment: {quality}")
    print(f"💡 Recommendation: {recommendation}")

def main():
    """Main evaluation function"""
    print("🔍 TACOTRON2 MODEL EVALUATION")
    print("=" * 50)
    
    # Evaluate training progress
    checkpoint = evaluate_training_progress()
    
    # Assess model quality
    check_model_quality(checkpoint)
    
    print("\n" + "=" * 50)
    print("🚀 NEXT STEPS:")
    print("1. 🎤 Test synthesis: python test_synthesis.py")
    print("2. 🔊 Generate samples: python synthesize_text.py")
    print("3. 📊 Compare with original audio")
    print("4. 🎯 Fine-tune if needed")

if __name__ == '__main__':
    main()
