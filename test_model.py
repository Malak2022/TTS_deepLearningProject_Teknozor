"""
Test script to debug model issues
"""

import os
import sys
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tacotron2.configs.config import Config
from tacotron2.models.tacotron2 import Tacotron2, HParams
from tacotron2.data.dataset import LJSpeechDataset, TextMelCollate
from tacotron2.utils.text import text_to_sequence, symbols

def test_text_processing():
    """Test text processing"""
    print("=== Testing Text Processing ===")
    
    test_text = "Hello world"
    sequence = text_to_sequence(test_text)
    print(f"Text: '{test_text}'")
    print(f"Sequence: {sequence}")
    print(f"Sequence length: {len(sequence)}")
    print(f"Max symbol ID: {max(sequence)}")
    print(f"Total symbols: {len(symbols)}")
    print()

def test_data_loading():
    """Test data loading"""
    print("=== Testing Data Loading ===")
    
    config = Config()
    dataset = LJSpeechDataset(config.METADATA_FILE, config.MAX_TEXT_LENGTH)
    
    # Test a single sample
    text, mel = dataset[0]
    print(f"Text shape: {text.shape}")
    print(f"Text content: {text}")
    print(f"Mel shape: {mel.shape}")
    print(f"Max text ID: {torch.max(text).item()}")
    print()
    
    return dataset

def test_collate_function():
    """Test collate function"""
    print("=== Testing Collate Function ===")
    
    config = Config()
    dataset = LJSpeechDataset(config.METADATA_FILE, config.MAX_TEXT_LENGTH)
    
    # Create a small batch
    batch = [dataset[i] for i in range(3)]
    collate_fn = TextMelCollate(config.N_FRAMES_PER_STEP)
    
    try:
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = collate_fn(batch)
        print(f"Batch size: {len(batch)}")
        print(f"Text padded shape: {text_padded.shape}")
        print(f"Input lengths: {input_lengths}")
        print(f"Mel padded shape: {mel_padded.shape}")
        print(f"Gate padded shape: {gate_padded.shape}")
        print(f"Output lengths: {output_lengths}")
        print("Collate function works!")
        print()
        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
    except Exception as e:
        print(f"Error in collate function: {e}")
        return None

def test_model_creation():
    """Test model creation"""
    print("=== Testing Model Creation ===")
    
    try:
        hparams = HParams()
        print(f"Model expects {hparams.n_symbols} symbols")
        
        model = Tacotron2(hparams)
        print(f"Model created successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        print()
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

def test_model_forward():
    """Test model forward pass"""
    print("=== Testing Model Forward Pass ===")
    
    # Get model and data
    model = test_model_creation()
    if model is None:
        return
    
    batch_data = test_collate_function()
    if batch_data is None:
        return
    
    text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch_data
    
    try:
        model.eval()
        with torch.no_grad():
            inputs = (text_padded, input_lengths, mel_padded, torch.max(input_lengths).item(), output_lengths)
            
            print(f"Input text shape: {text_padded.shape}")
            print(f"Input lengths: {input_lengths}")
            print(f"Max input length: {torch.max(input_lengths).item()}")
            print(f"Mel shape: {mel_padded.shape}")
            print(f"Output lengths: {output_lengths}")
            
            # Try forward pass
            outputs = model(inputs)
            print("Forward pass successful!")
            print(f"Output shapes: {[o.shape if hasattr(o, 'shape') else type(o) for o in outputs]}")
            
    except Exception as e:
        print(f"Error in forward pass: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("=== Tacotron2 Model Testing ===\n")
    
    # Test each component
    test_text_processing()
    test_data_loading()
    test_collate_function()
    test_model_creation()
    test_model_forward()
    
    print("=== Testing Complete ===")

if __name__ == '__main__':
    main()
