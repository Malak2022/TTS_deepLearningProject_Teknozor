"""
Setup script for Tacotron2 project
This script will verify your setup and create necessary directories
"""

import os
import sys
import json
import torch
import numpy as np

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'torch', 'numpy', 'librosa', 'matplotlib', 
        'tqdm', 'scipy', 'soundfile'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\nPlease install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data():
    """Check if preprocessed data exists"""
    data_path = r"D:\Project\data\LJSpeech-1.1\preprocessed"
    metadata_file = os.path.join(data_path, "metadata.json")
    
    if not os.path.exists(data_path):
        print(f"✗ Preprocessed data directory not found: {data_path}")
        return False
    
    if not os.path.exists(metadata_file):
        print(f"✗ Metadata file not found: {metadata_file}")
        return False
    
    # Check metadata content
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"✓ Found {len(metadata)} preprocessed samples")
        
        # Check a few sample files
        sample_count = min(5, len(metadata))
        for i in range(sample_count):
            item = metadata[i]
            mel_path = item['mel_path']
            tokens_path = item['tokens_path']
            
            if not os.path.exists(mel_path):
                print(f"✗ Mel file missing: {mel_path}")
                return False
            
            if not os.path.exists(tokens_path):
                print(f"✗ Tokens file missing: {tokens_path}")
                return False
        
        print(f"✓ Sample files verified")
        return True
        
    except Exception as e:
        print(f"✗ Error reading metadata: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "tacotron2/checkpoints",
        "tacotron2/logs",
        "outputs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_model_loading():
    """Test if the model can be loaded"""
    try:
        sys.path.append('tacotron2')
        from tacotron2.models.tacotron2 import Tacotron2, HParams
        
        hparams = HParams()
        model = Tacotron2(hparams)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_data_loading():
    """Test if data can be loaded"""
    try:
        sys.path.append('tacotron2')
        from tacotron2.data.dataset import LJSpeechDataset
        from tacotron2.configs.config import Config
        
        config = Config()
        dataset = LJSpeechDataset(config.METADATA_FILE, config.MAX_TEXT_LENGTH)
        
        # Test loading a sample
        text, mel = dataset[0]
        
        print(f"✓ Data loading successful")
        print(f"✓ Sample text shape: {text.shape}")
        print(f"✓ Sample mel shape: {mel.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def main():
    """Main setup function"""
    print("=== Tacotron2 Setup Verification ===\n")
    
    # Check requirements
    print("1. Checking requirements...")
    if not check_requirements():
        print("Please install missing packages and run setup again.")
        return
    
    print("\n2. Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA not available, will use CPU (training will be slow)")
    
    # Check data
    print("\n3. Checking preprocessed data...")
    if not check_data():
        print("Please run the preprocessing notebook first.")
        return
    
    # Create directories
    print("\n4. Creating directories...")
    create_directories()
    
    # Test model loading
    print("\n5. Testing model loading...")
    if not test_model_loading():
        print("Model loading failed. Check the installation.")
        return
    
    # Test data loading
    print("\n6. Testing data loading...")
    if not test_data_loading():
        print("Data loading failed. Check the preprocessed data.")
        return
    
    print("\n=== Setup Complete! ===")
    print("\nYou can now:")
    print("1. Start training: python train_tacotron2.py")
    print("2. After training, synthesize speech: python synthesize_text.py")
    print("3. Monitor training: tensorboard --logdir tacotron2/logs")
    
    print("\nTips:")
    print("- Training will take several hours depending on your hardware")
    print("- Monitor the loss curves to ensure the model is learning")
    print("- Check attention alignments to verify proper convergence")
    print("- Start with a small number of epochs for testing")

if __name__ == '__main__':
    main()
