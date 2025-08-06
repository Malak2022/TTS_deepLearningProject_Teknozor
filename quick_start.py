"""
Quick start script for Tacotron2
This script provides an interactive way to train and test the model
"""

import os
import sys
import subprocess

def run_setup():
    """Run the setup verification"""
    print("Running setup verification...")
    result = subprocess.run([sys.executable, "setup_tacotron2.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode == 0

def run_training(epochs=10):
    """Run training with specified epochs"""
    print(f"Starting training for {epochs} epochs...")
    
    # Modify the training script to use specified epochs
    training_code = f"""
import sys
sys.path.append('tacotron2')

from tacotron2.configs.config import Config
config = Config()
config.N_EPOCHS = {epochs}  # Set custom epochs

# Run training
exec(open('train_tacotron2.py').read())
"""
    
    with open('temp_train.py', 'w') as f:
        f.write(training_code)
    
    try:
        result = subprocess.run([sys.executable, "temp_train.py"], capture_output=False, text=True)
        return result.returncode == 0
    finally:
        if os.path.exists('temp_train.py'):
            os.remove('temp_train.py')

def run_synthesis():
    """Run synthesis"""
    print("Running text synthesis...")
    result = subprocess.run([sys.executable, "synthesize_text.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    return result.returncode == 0

def main():
    """Main interactive function"""
    print("=== Tacotron2 Quick Start ===\n")
    
    while True:
        print("\nOptions:")
        print("1. Run setup verification")
        print("2. Start training (quick test - 5 epochs)")
        print("3. Start training (short - 20 epochs)")
        print("4. Start training (full - 100 epochs)")
        print("5. Run synthesis (requires trained model)")
        print("6. Open TensorBoard")
        print("7. Check project structure")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            run_setup()
            
        elif choice == '2':
            if run_training(5):
                print("Quick training completed!")
            else:
                print("Training failed. Check the output above.")
                
        elif choice == '3':
            if run_training(20):
                print("Short training completed!")
            else:
                print("Training failed. Check the output above.")
                
        elif choice == '4':
            if run_training(100):
                print("Full training completed!")
            else:
                print("Training failed. Check the output above.")
                
        elif choice == '5':
            checkpoint_path = "tacotron2/checkpoints/best_model.pth"
            if os.path.exists(checkpoint_path):
                run_synthesis()
            else:
                print("No trained model found. Please train the model first.")
                
        elif choice == '6':
            print("Starting TensorBoard...")
            print("Open your browser and go to: http://localhost:6006")
            try:
                subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", "tacotron2/logs"])
                print("TensorBoard started in background.")
            except Exception as e:
                print(f"Error starting TensorBoard: {e}")
                print("You can manually run: tensorboard --logdir tacotron2/logs")
                
        elif choice == '7':
            print("\nProject Structure:")
            print("├── tacotron2/")
            print("│   ├── configs/config.py")
            print("│   ├── models/")
            print("│   │   ├── tacotron2.py")
            print("│   │   ├── layers.py")
            print("│   │   └── decoder.py")
            print("│   ├── data/dataset.py")
            print("│   ├── training/")
            print("│   │   ├── train.py")
            print("│   │   └── loss.py")
            print("│   ├── inference/synthesize.py")
            print("│   └── utils/")
            print("│       ├── text.py")
            print("│       └── audio.py")
            print("├── train_tacotron2.py")
            print("├── synthesize_text.py")
            print("├── setup_tacotron2.py")
            print("└── README_Tacotron2.md")
            
            # Check if key files exist
            key_files = [
                "tacotron2/models/tacotron2.py",
                "tacotron2/data/dataset.py",
                "tacotron2/training/train.py",
                "train_tacotron2.py",
                "synthesize_text.py"
            ]
            
            print("\nFile Status:")
            for file in key_files:
                status = "✓" if os.path.exists(file) else "✗"
                print(f"{status} {file}")
                
        elif choice == '8':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1-8.")

if __name__ == '__main__':
    main()
