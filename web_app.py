"""
Flask Web Application for Tacotron2 Text-to-Speech
"""

import os
import sys
import io
import base64
import torch
import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tacotron2.configs.config import Config
from tacotron2.models.tacotron2 import Tacotron2, HParams
from tacotron2.utils.text import text_to_sequence, clean_text
import librosa

app = Flask(__name__)

class TTSService:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained Tacotron2 model"""
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
            raise Exception("No trained model found! Please train the model first.")
        
        print(f"Loading model from: {os.path.basename(checkpoint_path)}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        hparams = HParams()
        self.model = Tacotron2(hparams).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully! Training loss: {checkpoint.get('best_val_loss', 'Unknown')}")
    
    def synthesize(self, text):
        """Synthesize speech from text"""
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        
        if len(text) > 500:
            raise ValueError("Text too long (max 500 characters)")
        
        # Clean and convert text
        cleaned_text = clean_text(text)
        sequence = text_to_sequence(cleaned_text)
        
        if not sequence:
            raise ValueError("Failed to convert text to sequence")
        
        # Convert to tensor
        text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(text_tensor)
            
            # Use post-processed mel spectrogram
            mel_spectrogram = mel_outputs_postnet[0].cpu().numpy()
            
            # Convert to audio using Griffin-Lim
            audio = self.mel_to_audio(mel_spectrogram)
            
            return audio
    
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
        
        # Apply Griffin-Lim algorithm
        audio = librosa.griffinlim(
            linear_spectrogram,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
            n_iter=60
        )
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        return audio

# Initialize TTS service
try:
    tts_service = TTSService()
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False
    tts_service = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_loaded=MODEL_LOADED)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Synthesize speech from text"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Synthesize audio
        audio = tts_service.synthesize(text)
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, tts_service.config.SAMPLE_RATE)
            
            # Read file as bytes
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
        
        # Encode audio as base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'audio': audio_base64,
            'text': text,
            'duration': len(audio) / tts_service.config.SAMPLE_RATE
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'device': str(tts_service.device) if tts_service else 'none'
    })

if __name__ == '__main__':
    print("🎤 TACOTRON2 WEB APPLICATION")
    print("=" * 50)
    
    if MODEL_LOADED:
        print("✅ Model loaded successfully!")
        print(f"🔧 Device: {tts_service.device}")
        print("🌐 Starting web server...")
        print("📱 Open your browser to: http://localhost:5000")
    else:
        print("❌ Model not loaded! Please train the model first.")
        print("🌐 Starting web server anyway (will show error page)...")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
