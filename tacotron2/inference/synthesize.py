"""
Text-to-Speech synthesis using trained Tacotron2 model
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy.io.wavfile import write
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.tacotron2 import Tacotron2, HParams
from utils.text import text_to_sequence, clean_text
from utils.audio import mel_spectrogram

class Tacotron2Synthesizer:
    def __init__(self, checkpoint_path, config=None):
        """
        Initialize Tacotron2 synthesizer
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config: Configuration object (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load config
        if config is None:
            self.config = Config()
        else:
            self.config = config
        
        # Initialize model
        self.hparams = HParams()
        self.model = Tacotron2(self.hparams).to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print("Tacotron2 synthesizer initialized successfully!")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully!")
    
    def text_to_mel(self, text):
        """
        Convert text to mel spectrogram
        
        Args:
            text: Input text string
            
        Returns:
            mel_spectrogram: Generated mel spectrogram
            alignments: Attention alignments
        """
        # Clean and convert text to sequence
        cleaned_text = clean_text(text)
        sequence = text_to_sequence(cleaned_text)
        
        # Convert to tensor
        text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(text_tensor)
        
        # Use post-processed mel spectrogram
        mel_spectrogram = mel_outputs_postnet[0].cpu().numpy()
        alignments = alignments[0].cpu().numpy()
        
        return mel_spectrogram, alignments
    
    def mel_to_audio(self, mel_spectrogram, output_path=None):
        """
        Convert mel spectrogram to audio using Griffin-Lim algorithm
        
        Args:
            mel_spectrogram: Mel spectrogram to convert
            output_path: Path to save audio file (optional)
            
        Returns:
            audio: Generated audio waveform
        """
        # Convert mel spectrogram to linear spectrogram (simplified approach)
        # Note: For better quality, you should use a vocoder like WaveGlow or HiFi-GAN
        
        # Denormalize mel spectrogram
        mel_spectrogram = np.exp(mel_spectrogram)
        
        # Convert mel to linear spectrogram using inverse mel filter bank
        n_fft = self.config.N_FFT
        hop_length = self.config.HOP_LENGTH
        win_length = self.config.WIN_LENGTH
        
        # Create mel filter bank
        mel_basis = librosa.filters.mel(
            sr=self.config.SAMPLE_RATE,
            n_fft=n_fft,
            n_mels=self.config.N_MELS,
            fmin=0,
            fmax=self.config.SAMPLE_RATE // 2
        )
        
        # Pseudo-inverse to convert mel to linear
        linear_spectrogram = np.dot(np.linalg.pinv(mel_basis), mel_spectrogram)
        
        # Apply Griffin-Lim algorithm
        audio = librosa.griffinlim(
            linear_spectrogram,
            hop_length=hop_length,
            win_length=win_length,
            n_iter=60
        )
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save audio if path provided
        if output_path:
            sf.write(output_path, audio, self.config.SAMPLE_RATE)
            print(f"Audio saved to: {output_path}")
        
        return audio
    
    def synthesize(self, text, output_path=None, save_mel=False, save_alignment=False):
        """
        Complete text-to-speech synthesis
        
        Args:
            text: Input text string
            output_path: Path to save audio file (optional)
            save_mel: Whether to save mel spectrogram plot
            save_alignment: Whether to save attention alignment plot
            
        Returns:
            audio: Generated audio waveform
            mel_spectrogram: Generated mel spectrogram
            alignments: Attention alignments
        """
        print(f"Synthesizing: '{text}'")
        
        # Generate mel spectrogram
        mel_spectrogram, alignments = self.text_to_mel(text)
        
        # Convert to audio
        audio = self.mel_to_audio(mel_spectrogram, output_path)
        
        # Save visualizations if requested
        if save_mel or save_alignment:
            base_name = output_path.replace('.wav', '') if output_path else 'synthesis'
            
            if save_mel:
                self.plot_mel_spectrogram(mel_spectrogram, f"{base_name}_mel.png")
            
            if save_alignment:
                self.plot_alignment(alignments, f"{base_name}_alignment.png")
        
        return audio, mel_spectrogram, alignments
    
    def plot_mel_spectrogram(self, mel_spectrogram, save_path):
        """Plot and save mel spectrogram"""
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
        """Plot and save attention alignment"""
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
    
    def batch_synthesize(self, texts, output_dir):
        """
        Synthesize multiple texts
        
        Args:
            texts: List of text strings
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"synthesis_{i:03d}.wav")
            try:
                self.synthesize(text, output_path, save_mel=True, save_alignment=True)
                print(f"Completed synthesis {i+1}/{len(texts)}")
            except Exception as e:
                print(f"Error synthesizing text {i}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Tacotron2 Text-to-Speech Synthesis')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--text', type=str, help='Text to synthesize')
    parser.add_argument('--text_file', type=str, help='File containing texts to synthesize')
    parser.add_argument('--output', type=str, default='output.wav', help='Output audio file path')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for batch synthesis')
    parser.add_argument('--save_mel', action='store_true', help='Save mel spectrogram plot')
    parser.add_argument('--save_alignment', action='store_true', help='Save attention alignment plot')
    
    args = parser.parse_args()
    
    # Initialize synthesizer
    synthesizer = Tacotron2Synthesizer(args.checkpoint)
    
    if args.text:
        # Single text synthesis
        synthesizer.synthesize(
            args.text, 
            args.output, 
            save_mel=args.save_mel, 
            save_alignment=args.save_alignment
        )
    elif args.text_file:
        # Batch synthesis from file
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        synthesizer.batch_synthesize(texts, args.output_dir)
    else:
        # Interactive mode
        print("Interactive synthesis mode. Type 'quit' to exit.")
        while True:
            text = input("Enter text to synthesize: ")
            if text.lower() == 'quit':
                break
            
            output_path = f"interactive_synthesis_{int(time.time())}.wav"
            synthesizer.synthesize(text, output_path, save_mel=True, save_alignment=True)


if __name__ == '__main__':
    import time
    main()
