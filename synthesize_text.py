"""
Simple text-to-speech synthesis script
Run this script from the project root directory
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

# Add tacotron2 to path
sys.path.append('tacotron2')

from tacotron2.configs.config import Config
from tacotron2.models.tacotron2 import Tacotron2, HParams
from tacotron2.utils.text import text_to_sequence, clean_text

class SimpleSynthesizer:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.config = Config()
        
        # Initialize model
        self.hparams = HParams()
        self.model = Tacotron2(self.hparams).to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        print("Synthesizer initialized!")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded!")
    
    def text_to_mel(self, text):
        """Convert text to mel spectrogram"""
        # Clean and convert text
        cleaned_text = clean_text(text)
        sequence = text_to_sequence(cleaned_text)
        
        # Convert to tensor
        text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(text_tensor)
        
        mel_spectrogram = mel_outputs_postnet[0].cpu().numpy()
        alignments = alignments[0].cpu().numpy()
        
        return mel_spectrogram, alignments
    
    def mel_to_audio(self, mel_spectrogram):
        """Convert mel spectrogram to audio using Griffin-Lim"""
        # Denormalize mel spectrogram
        mel_spectrogram = np.exp(mel_spectrogram)
        
        # Create mel filter bank
        mel_basis = librosa.filters.mel(
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            n_mels=self.config.N_MELS,
            fmin=0,
            fmax=self.config.SAMPLE_RATE // 2
        )
        
        # Convert mel to linear spectrogram
        linear_spectrogram = np.dot(np.linalg.pinv(mel_basis), mel_spectrogram)
        
        # Apply Griffin-Lim
        audio = librosa.griffinlim(
            linear_spectrogram,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
            n_iter=60
        )
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def synthesize(self, text, output_path=None):
        """Complete synthesis pipeline"""
        print(f"Synthesizing: '{text}'")
        
        # Generate mel spectrogram
        mel_spectrogram, alignments = self.text_to_mel(text)
        
        # Convert to audio
        audio = self.mel_to_audio(mel_spectrogram)
        
        # Save audio
        if output_path:
            sf.write(output_path, audio, self.config.SAMPLE_RATE)
            print(f"Audio saved to: {output_path}")
        
        # Save visualizations
        if output_path:
            base_name = output_path.replace('.wav', '')
            self.plot_mel_spectrogram(mel_spectrogram, f"{base_name}_mel.png")
            self.plot_alignment(alignments, f"{base_name}_alignment.png")
        
        return audio, mel_spectrogram, alignments
    
    def plot_mel_spectrogram(self, mel_spectrogram, save_path):
        """Plot mel spectrogram"""
        plt.figure(figsize=(12, 6))
        plt.imshow(mel_spectrogram, aspect='auto', origin='lower', interpolation='none')
        plt.colorbar()
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Mel spectrogram saved to: {save_path}")
    
    def plot_alignment(self, alignments, save_path):
        """Plot attention alignment"""
        plt.figure(figsize=(12, 6))
        plt.imshow(alignments.T, aspect='auto', origin='lower', interpolation='none')
        plt.colorbar()
        plt.title('Attention Alignment')
        plt.xlabel('Decoder Steps')
        plt.ylabel('Encoder Steps')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Attention alignment saved to: {save_path}")

def main():
    # Example usage
    checkpoint_path = "tacotron2/checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train_tacotron2.py")
        return
    
    # Initialize synthesizer
    synthesizer = SimpleSynthesizer(checkpoint_path)
    
    # Example texts to synthesize
    texts = [
        "Hello, this is a test of the Tacotron2 text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of artificial intelligence."
    ]
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Synthesize each text
    for i, text in enumerate(texts):
        output_path = f"outputs/synthesis_{i:03d}.wav"
        try:
            synthesizer.synthesize(text, output_path)
            print(f"Completed synthesis {i+1}/{len(texts)}")
        except Exception as e:
            print(f"Error synthesizing text {i}: {e}")

if __name__ == '__main__':
    main()
