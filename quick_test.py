"""
Quick synthesis test with short text
"""

import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
import scipy.signal

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tacotron2.configs.config import Config
from tacotron2.models.tacotron2 import Tacotron2, HParams
from tacotron2.utils.text import text_to_sequence, clean_text

class QuickTTS:
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
        
        # Convert to tensor
        text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(text_tensor)
            mel_spectrogram = mel_outputs_postnet[0].cpu().numpy()
            
            print(f"✅ Generated mel spectrogram: {mel_spectrogram.shape}")
            
            # Convert to audio
            audio = self.mel_to_audio(mel_spectrogram)
            
            return audio, mel_spectrogram
    
    def mel_to_audio(self, mel_spectrogram):
        """Convert mel spectrogram to audio using improved Griffin-Lim"""
        # Denormalize mel spectrogram
        mel_spectrogram = np.exp(mel_spectrogram)
        
        # Create mel filter bank
        mel_basis = librosa.filters.mel(
            sr=self.config.SAMPLE_RATE,
            n_fft=self.config.N_FFT,
            n_mels=self.config.N_MELS,
            fmin=50,
            fmax=7600
        )
        
        # Convert mel to linear spectrogram
        linear_spectrogram = np.dot(np.linalg.pinv(mel_basis), mel_spectrogram)
        linear_spectrogram = np.maximum(linear_spectrogram, 0.01)
        
        # Apply Griffin-Lim algorithm
        audio = librosa.griffinlim(
            linear_spectrogram,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
            n_iter=100,
            momentum=0.99,
            init='random',
            random_state=42
        )
        
        # Apply high-pass filter
        nyquist = self.config.SAMPLE_RATE / 2
        low_cutoff = 80 / nyquist
        b, a = scipy.signal.butter(5, low_cutoff, btype='high')
        audio = scipy.signal.filtfilt(b, a, audio)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        return audio

def main():
    print("🎤 QUICK TTS TEST")
    print("=" * 30)
    
    # Initialize TTS
    tts = QuickTTS()
    
    # Test with short text
    test_text = "Hello world."
    
    try:
        # Synthesize
        audio, mel = tts.synthesize(test_text)
        
        # Save audio
        output_file = "quick_test.wav"
        sf.write(output_file, audio, tts.config.SAMPLE_RATE)
        
        print(f"🔊 Audio saved: {output_file}")
        print(f"📊 Audio length: {len(audio) / tts.config.SAMPLE_RATE:.2f} seconds")
        print(f"📈 Audio range: {np.min(audio):.3f} to {np.max(audio):.3f}")
        
        # Check if audio has content
        if np.max(np.abs(audio)) > 0.01:
            print("✅ Audio has good amplitude")
        else:
            print("⚠️ Audio amplitude is very low")
        
        print(f"\n🎯 Test complete! Listen to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
