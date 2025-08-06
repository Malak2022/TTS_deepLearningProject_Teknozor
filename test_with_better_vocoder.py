"""
Test synthesis with better vocoder options
"""

import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tacotron2.configs.config import Config
from tacotron2.models.tacotron2 import Tacotron2, HParams
from tacotron2.utils.text import text_to_sequence, clean_text

class ImprovedTTS:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model_fast.pth')
        
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Model not found: {checkpoint_path}")
        
        print(f"📁 Loading model from: {os.path.basename(checkpoint_path)}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Initialize model
        hparams = HParams()
        self.model = Tacotron2(hparams).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded! Loss: {checkpoint.get('best_val_loss', 'Unknown')}")
    
    def synthesize(self, text):
        """Synthesize speech from text"""
        print(f"🎤 Synthesizing: '{text}'")
        
        # Clean and convert text
        cleaned_text = clean_text(text)
        sequence = text_to_sequence(cleaned_text)
        
        if not sequence:
            raise ValueError("Failed to convert text to sequence")
        
        print(f"📝 Text sequence length: {len(sequence)}")
        
        # Convert to tensor
        text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(text_tensor)
            mel_spectrogram = mel_outputs_postnet[0].cpu().numpy()
            
            print(f"✅ Generated mel spectrogram: {mel_spectrogram.shape}")
            print(f"📊 Mel range: {np.min(mel_spectrogram):.3f} to {np.max(mel_spectrogram):.3f}")
            
            return mel_spectrogram, alignments[0].cpu().numpy()
    
    def mel_to_audio_simple(self, mel_spectrogram):
        """Simple mel to audio conversion with basic settings"""
        print("🔄 Converting mel to audio (simple method)...")
        
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
        
        # Ensure positive values
        linear_spectrogram = np.maximum(linear_spectrogram, 0.001)
        
        # Apply Griffin-Lim with basic settings
        audio = librosa.griffinlim(
            linear_spectrogram,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
            n_iter=32,  # Fewer iterations for speed
            length=None
        )
        
        # Simple normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def mel_to_audio_advanced(self, mel_spectrogram):
        """Advanced mel to audio conversion"""
        print("🔄 Converting mel to audio (advanced method)...")
        
        # Better denormalization
        mel_spectrogram = np.exp(mel_spectrogram)
        
        # Create mel filter bank with better parameters
        mel_basis = librosa.filters.mel(
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            n_mels=self.config.N_MELS,
            fmin=80,
            fmax=7600,
            norm='slaney'
        )
        
        # Convert mel to linear spectrogram
        linear_spectrogram = np.dot(np.linalg.pinv(mel_basis), mel_spectrogram)
        
        # Ensure positive values and add small epsilon
        linear_spectrogram = np.maximum(linear_spectrogram, 1e-10)
        
        # Apply advanced Griffin-Lim
        audio = librosa.griffinlim(
            linear_spectrogram,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
            n_iter=60,
            momentum=0.99,
            init='random',
            random_state=42
        )
        
        # Post-processing
        # Remove DC component
        audio = audio - np.mean(audio)
        
        # Apply gentle high-pass filter
        from scipy.signal import butter, filtfilt
        nyquist = self.config.SAMPLE_RATE / 2
        low_cutoff = 50 / nyquist
        b, a = butter(3, low_cutoff, btype='high')
        audio = filtfilt(b, a, audio)
        
        # Normalize with headroom
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
    
    def save_visualization(self, mel_spectrogram, alignments, output_path):
        """Save mel spectrogram and alignment visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot mel spectrogram
        im1 = ax1.imshow(mel_spectrogram, aspect='auto', origin='lower', interpolation='none')
        ax1.set_title('Mel Spectrogram')
        ax1.set_ylabel('Mel Frequency')
        plt.colorbar(im1, ax=ax1)
        
        # Plot alignment
        im2 = ax2.imshow(alignments.T, aspect='auto', origin='lower', interpolation='none')
        ax2.set_title('Attention Alignment')
        ax2.set_xlabel('Decoder Steps')
        ax2.set_ylabel('Encoder Steps')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {output_path}")

def main():
    print("🎤 IMPROVED TTS TESTING")
    print("=" * 40)
    
    # Initialize TTS
    tts = ImprovedTTS()
    
    # Test texts (start with very simple)
    test_texts = [
        "Hello.",
        "Hi there.",
        "How are you?",
        "This is a test."
    ]
    
    os.makedirs("improved_tests", exist_ok=True)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}/{len(test_texts)}")
        print("-" * 30)
        
        try:
            # Synthesize
            mel_spectrogram, alignments = tts.synthesize(text)
            
            # Try both vocoder methods
            print("\n🔄 Testing Simple Vocoder...")
            audio_simple = tts.mel_to_audio_simple(mel_spectrogram)
            
            print("🔄 Testing Advanced Vocoder...")
            audio_advanced = tts.mel_to_audio_advanced(mel_spectrogram)
            
            # Save both versions
            simple_file = f"improved_tests/test_{i:02d}_simple.wav"
            advanced_file = f"improved_tests/test_{i:02d}_advanced.wav"
            viz_file = f"improved_tests/test_{i:02d}_visualization.png"
            
            sf.write(simple_file, audio_simple, tts.config.SAMPLE_RATE)
            sf.write(advanced_file, audio_advanced, tts.config.SAMPLE_RATE)
            
            # Save visualization
            tts.save_visualization(mel_spectrogram, alignments, viz_file)
            
            # Analysis
            print(f"\n📊 Analysis for '{text}':")
            print(f"   Mel shape: {mel_spectrogram.shape}")
            print(f"   Mel range: {np.min(mel_spectrogram):.3f} to {np.max(mel_spectrogram):.3f}")
            print(f"   Simple audio: {len(audio_simple)/tts.config.SAMPLE_RATE:.2f}s, range: {np.min(audio_simple):.3f} to {np.max(audio_simple):.3f}")
            print(f"   Advanced audio: {len(audio_advanced)/tts.config.SAMPLE_RATE:.2f}s, range: {np.min(audio_advanced):.3f} to {np.max(audio_advanced):.3f}")
            
            # Check if audio has meaningful content
            simple_energy = np.mean(audio_simple**2)
            advanced_energy = np.mean(audio_advanced**2)
            
            print(f"   Simple energy: {simple_energy:.6f}")
            print(f"   Advanced energy: {advanced_energy:.6f}")
            
            if simple_energy > 1e-6:
                print("   ✅ Simple audio has good energy")
            else:
                print("   ⚠️ Simple audio energy is very low")
            
            if advanced_energy > 1e-6:
                print("   ✅ Advanced audio has good energy")
            else:
                print("   ⚠️ Advanced audio energy is very low")
            
            print(f"   🔊 Files saved: {simple_file}, {advanced_file}")
            
        except Exception as e:
            print(f"❌ Error with '{text}': {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 Testing complete! Check the 'improved_tests' folder for results.")
    print("💡 Try listening to both simple and advanced versions to compare quality.")

if __name__ == '__main__':
    main()
