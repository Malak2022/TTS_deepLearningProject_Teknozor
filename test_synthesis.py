"""
Test speech synthesis with trained Tacotron2 model
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from datetime import datetime
import scipy.signal

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tacotron2.configs.config import Config
from tacotron2.models.tacotron2 import Tacotron2, HParams
from tacotron2.utils.text import text_to_sequence, clean_text

class TacotronTester:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_best_model()
    
    def load_best_model(self):
        """Load the best trained model"""
        checkpoint_files = [
            'best_model_fast.pth',
            'final_model_fast.pth', 
            'best_model.pth'
        ]
        
        checkpoint_path = None
        for filename in checkpoint_files:
            path = os.path.join(self.config.CHECKPOINT_DIR, filename)
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            print("❌ No trained model found!")
            print("Please train the model first using train_tacotron2_fast.py")
            return
        
        print(f"📁 Loading model from: {os.path.basename(checkpoint_path)}")
        
        # Load checkpoint (fix for newer PyTorch versions)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Initialize model
        hparams = HParams()
        self.model = Tacotron2(hparams).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Training loss: {checkpoint.get('best_val_loss', 'Unknown')}")
    
    def synthesize_text(self, text, output_path=None):
        """Synthesize speech from text"""
        if self.model is None:
            print("❌ No model loaded!")
            return None, None, None
        
        print(f"🎤 Synthesizing: '{text}'")
        
        # Clean and convert text
        cleaned_text = clean_text(text)
        sequence = text_to_sequence(cleaned_text)
        
        if not sequence:
            print("❌ Failed to convert text to sequence")
            return None, None, None
        
        # Convert to tensor
        text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            try:
                mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(text_tensor)
                
                # Use post-processed mel spectrogram
                mel_spectrogram = mel_outputs_postnet[0].cpu().numpy()
                alignments = alignments[0].cpu().numpy()
                gate_outputs = gate_outputs[0].cpu().numpy()
                
                print(f"✅ Generated mel spectrogram: {mel_spectrogram.shape}")
                
                # Convert to audio using Griffin-Lim
                audio = self.mel_to_audio(mel_spectrogram)
                
                # Save audio if path provided
                if output_path:
                    sf.write(output_path, audio, self.config.SAMPLE_RATE)
                    print(f"🔊 Audio saved: {output_path}")
                
                return audio, mel_spectrogram, alignments
                
            except Exception as e:
                print(f"❌ Synthesis error: {e}")
                return None, None, None
    
    def mel_to_audio(self, mel_spectrogram):
        """Convert mel spectrogram to audio using improved Griffin-Lim"""
        # Denormalize mel spectrogram (improved)
        mel_spectrogram = np.exp(mel_spectrogram)

        # Create mel filter bank with better parameters
        mel_basis = librosa.filters.mel(
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            n_mels=self.config.N_MELS,
            fmin=50,  # Better low frequency cutoff
            fmax=7600  # Better high frequency cutoff
        )

        # Convert mel to linear spectrogram with pseudo-inverse
        linear_spectrogram = np.dot(np.linalg.pinv(mel_basis), mel_spectrogram)

        # Ensure positive values
        linear_spectrogram = np.maximum(linear_spectrogram, 0.01)

        # Apply improved Griffin-Lim algorithm
        audio = librosa.griffinlim(
            linear_spectrogram,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
            n_iter=100,  # More iterations for better quality
            length=None,
            pad_mode='reflect',
            momentum=0.99,  # Better momentum
            init='random',  # Better initialization
            random_state=42
        )

        # Apply high-pass filter to remove low-frequency noise
        from scipy.signal import butter, filtfilt
        nyquist = self.config.SAMPLE_RATE / 2
        low_cutoff = 80 / nyquist
        b, a = butter(5, low_cutoff, btype='high')
        audio = filtfilt(b, a, audio)

        # Normalize audio with better scaling
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom

        return audio
    
    def plot_synthesis_results(self, text, mel_spectrogram, alignments, output_prefix):
        """Plot synthesis results"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot mel spectrogram
        axes[0].imshow(mel_spectrogram, aspect='auto', origin='lower', interpolation='none')
        axes[0].set_title(f'Generated Mel Spectrogram: "{text}"')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Mel Frequency Bins')
        
        # Plot attention alignment
        axes[1].imshow(alignments.T, aspect='auto', origin='lower', interpolation='none')
        axes[1].set_title('Attention Alignment')
        axes[1].set_xlabel('Decoder Steps')
        axes[1].set_ylabel('Encoder Steps')
        
        plt.tight_layout()
        plot_path = f"{output_prefix}_synthesis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Plots saved: {plot_path}")
    
    def test_multiple_texts(self):
        """Test synthesis with multiple example texts"""
        test_texts = [
            "Hello, this is a test of the Tacotron2 text to speech system.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "How are you doing today?",
            "This model was trained on the LJ Speech dataset."
        ]
        
        print("🎯 TESTING MULTIPLE TEXTS")
        print("=" * 50)
        
        os.makedirs("synthesis_tests", exist_ok=True)
        
        for i, text in enumerate(test_texts):
            print(f"\n📝 Test {i+1}/{len(test_texts)}")
            
            output_path = f"synthesis_tests/test_{i+1:02d}.wav"
            audio, mel, alignments = self.synthesize_text(text, output_path)
            
            if audio is not None:
                # Save plots
                plot_prefix = f"synthesis_tests/test_{i+1:02d}"
                self.plot_synthesis_results(text, mel, alignments, plot_prefix)
                
                # Audio statistics
                duration = len(audio) / self.config.SAMPLE_RATE
                print(f"⏱️ Duration: {duration:.2f} seconds")
                print(f"🔊 Audio range: {np.min(audio):.3f} to {np.max(audio):.3f}")
            else:
                print("❌ Synthesis failed")
        
        print(f"\n✅ All tests completed! Check 'synthesis_tests' folder")

def main():
    """Main testing function"""
    print("🎤 TACOTRON2 SYNTHESIS TESTING")
    print("=" * 50)
    
    # Initialize tester
    tester = TacotronTester()
    
    if tester.model is None:
        return
    
    # Test multiple texts
    tester.test_multiple_texts()
    
    print("\n" + "=" * 50)
    print("🎉 SYNTHESIS TESTING COMPLETE!")
    print("\n📁 Generated files:")
    print("   🔊 Audio files: synthesis_tests/test_*.wav")
    print("   📊 Plots: synthesis_tests/test_*_synthesis.png")
    print("\n💡 Next steps:")
    print("   1. Listen to the generated audio files")
    print("   2. Check attention alignments (should be diagonal)")
    print("   3. Compare quality with original LJSpeech samples")
    print("   4. If quality is poor, consider more training")

if __name__ == '__main__':
    main()
