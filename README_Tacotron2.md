# Tacotron2 Text-to-Speech Implementation

This is a complete implementation of Tacotron2 for text-to-speech synthesis using the LJSpeech dataset that you've already preprocessed.

## Project Structure

```
tacotron2/
├── configs/
│   └── config.py          # Configuration settings
├── models/
│   ├── __init__.py
│   ├── tacotron2.py       # Main Tacotron2 model
│   ├── layers.py          # Model layers (encoder, attention, etc.)
│   └── decoder.py         # Decoder implementation
├── data/
│   ├── __init__.py
│   └── dataset.py         # Dataset and data loading utilities
├── training/
│   ├── __init__.py
│   ├── loss.py            # Loss functions
│   └── train.py           # Training script
├── inference/
│   ├── __init__.py
│   └── synthesize.py      # Inference and synthesis
└── utils/
    ├── __init__.py
    ├── text.py            # Text processing utilities
    └── audio.py           # Audio processing utilities

train_tacotron2.py         # Simple training script
synthesize_text.py         # Simple synthesis script
```

## Features

- **Complete Tacotron2 Architecture**: Encoder, Decoder with Attention, and Postnet
- **Location-Sensitive Attention**: Improved attention mechanism for better alignment
- **Comprehensive Loss Functions**: Mel spectrogram loss and gate loss
- **Data Loading**: Custom dataset class for your preprocessed LJSpeech data
- **Training Pipeline**: Full training loop with checkpointing and validation
- **Inference Pipeline**: Text-to-speech synthesis with visualization
- **Audio Processing**: Griffin-Lim vocoder for audio generation

## Requirements

Make sure you have the following packages installed (they should already be in your requirements.txt):

```
torch
torchaudio
numpy
pandas
librosa
matplotlib
tqdm
scipy
soundfile
tensorboard
```

## Quick Start

### 1. Training the Model

Run the simple training script:

```bash
python train_tacotron2.py
```

This will:
- Load your preprocessed LJSpeech data
- Initialize the Tacotron2 model
- Start training with progress bars
- Save checkpoints in `tacotron2/checkpoints/`
- Create logs in `tacotron2/logs/`

### 2. Synthesizing Speech

After training, synthesize speech from text:

```bash
python synthesize_text.py
```

This will:
- Load the trained model
- Synthesize example texts
- Save audio files in `outputs/`
- Generate mel spectrogram and attention plots

## Configuration

Edit `tacotron2/configs/config.py` to modify:

- **Data paths**: Update paths to your preprocessed data
- **Model parameters**: Adjust model architecture
- **Training settings**: Batch size, learning rate, etc.
- **Audio parameters**: Sample rate, mel channels, etc.

## Model Architecture

### Encoder
- Character embedding layer
- 3 convolutional layers with batch normalization
- Bidirectional LSTM

### Decoder
- Prenet with dropout
- Attention RNN
- Location-sensitive attention mechanism
- Decoder RNN
- Linear projection to mel spectrogram
- Gate prediction for end-of-sequence

### Postnet
- 5 convolutional layers with batch normalization
- Residual connection to improve mel spectrogram quality

## Training Details

- **Loss Function**: Combination of mel spectrogram L1 loss and binary cross-entropy gate loss
- **Optimizer**: Adam with learning rate scheduling
- **Gradient Clipping**: Prevents gradient explosion
- **Checkpointing**: Automatic saving of best models
- **Validation**: Regular validation with early stopping

## Data Format

The model expects your preprocessed data structure:
- `metadata.json`: Contains file paths and metadata
- `*_mel.npy`: Mel spectrogram files
- `*_tokens.npy`: Tokenized text files

This matches the format from your preprocessing notebook.

## Advanced Usage

### Custom Training

For more control over training, use the advanced training script:

```python
from tacotron2.training.train import Trainer
from tacotron2.configs.config import Config

config = Config()
trainer = Trainer(config)
trainer.train()
```

### Custom Synthesis

For custom synthesis with more options:

```python
from tacotron2.inference.synthesize import Tacotron2Synthesizer

synthesizer = Tacotron2Synthesizer('path/to/checkpoint.pth')
audio, mel, alignments = synthesizer.synthesize("Your text here", "output.wav")
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir tacotron2/logs
```

This will show:
- Training and validation losses
- Learning rate schedules
- Model gradients and weights

## Tips for Better Results

1. **Data Quality**: Ensure your preprocessed data is clean and consistent
2. **Training Time**: Tacotron2 requires significant training time (several hours to days)
3. **Batch Size**: Adjust based on your GPU memory
4. **Learning Rate**: Start with 1e-3 and reduce if training is unstable
5. **Attention**: Monitor attention alignments to ensure proper learning

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in config
2. **Poor Attention**: Increase training time or adjust attention parameters
3. **Audio Quality**: The Griffin-Lim vocoder has limitations; consider using WaveGlow or HiFi-GAN for better quality
4. **Training Instability**: Reduce learning rate or increase gradient clipping

### Model Not Learning

- Check data loading and preprocessing
- Verify text tokenization matches preprocessing
- Monitor loss curves for convergence
- Ensure sufficient training data

## Next Steps

1. **Vocoder Integration**: Replace Griffin-Lim with neural vocoders like WaveGlow or HiFi-GAN
2. **Multi-Speaker**: Extend to multi-speaker synthesis
3. **Fine-tuning**: Fine-tune on specific voices or domains
4. **Optimization**: Implement mixed precision training for faster training

## File Descriptions

- `train_tacotron2.py`: Simple training script for quick start
- `synthesize_text.py`: Simple synthesis script with examples
- `tacotron2/models/tacotron2.py`: Main model implementation
- `tacotron2/training/train.py`: Advanced training with full features
- `tacotron2/data/dataset.py`: Data loading and preprocessing
- `tacotron2/configs/config.py`: All configuration parameters

This implementation provides a solid foundation for text-to-speech synthesis with Tacotron2. The modular design allows for easy customization and extension.
