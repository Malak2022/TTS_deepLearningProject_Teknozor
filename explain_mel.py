"""
Explain Mel Spectrograms with visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_mel_explanation():
    """Create visual explanation of mel spectrograms"""
    
    # Create a simple synthetic audio signal
    sample_rate = 22050
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a signal with multiple frequencies (like speech)
    frequencies = [440, 880, 1320]  # A4, A5, E6 notes
    audio = np.zeros_like(t)
    
    for i, freq in enumerate(frequencies):
        # Each frequency appears at different times (like phonemes in speech)
        start_time = i * 0.6
        end_time = start_time + 0.8
        mask = (t >= start_time) & (t <= end_time)
        audio[mask] += 0.3 * np.sin(2 * np.pi * freq * t[mask])
    
    # Add some noise (like natural speech)
    audio += 0.05 * np.random.randn(len(audio))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate,
        n_mels=80,  # Same as Tacotron2
        hop_length=256,
        win_length=1024,
        n_fft=1024
    )
    
    # Convert to log scale (like in training)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Original audio waveform
    axes[0].plot(t, audio)
    axes[0].set_title('1. Original Audio Waveform (like human speech)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Mel spectrogram
    img = librosa.display.specshow(
        mel_spec_db, 
        sr=sample_rate,
        hop_length=256,
        x_axis='time', 
        y_axis='mel',
        ax=axes[1],
        cmap='viridis'
    )
    axes[1].set_title('2. Mel Spectrogram (what Tacotron2 learns to generate)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Mel Frequency')
    plt.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # 3. Training process explanation
    axes[2].text(0.05, 0.8, 'TACOTRON2 TRAINING PROCESS:', fontsize=14, fontweight='bold', transform=axes[2].transAxes)
    axes[2].text(0.05, 0.65, '1. Input: "Hello world" (text)', fontsize=12, transform=axes[2].transAxes)
    axes[2].text(0.05, 0.55, '2. Target: Mel spectrogram (from real audio)', fontsize=12, transform=axes[2].transAxes)
    axes[2].text(0.05, 0.45, '3. Model generates: Predicted mel spectrogram', fontsize=12, transform=axes[2].transAxes)
    axes[2].text(0.05, 0.35, '4. Mel Loss: Difference between target and predicted', fontsize=12, transform=axes[2].transAxes, color='red')
    axes[2].text(0.05, 0.25, '5. Training: Minimize this difference', fontsize=12, transform=axes[2].transAxes)
    
    axes[2].text(0.05, 0.1, f'YOUR PROGRESS: Mel Loss went from 48.16 → 6.08 (87% better!)', 
                fontsize=12, fontweight='bold', color='green', transform=axes[2].transAxes)
    
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mel_spectrogram_explanation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=== MEL SPECTROGRAM EXPLANATION ===")
    print()
    print("🎵 WHAT IS A MEL SPECTROGRAM?")
    print("- Visual representation of audio frequencies over time")
    print("- Like a 'picture' of sound that shows pitch and timing")
    print("- Used in speech processing because it matches human hearing")
    print()
    print("🤖 HOW TACOTRON2 USES MEL SPECTROGRAMS:")
    print("1. Input: Text ('Hello world')")
    print("2. Output: Mel spectrogram (visual representation of speech)")
    print("3. Vocoder: Converts mel spectrogram back to audio")
    print()
    print("📊 YOUR TRAINING METRICS:")
    print("- Mel Loss: How well the model predicts mel spectrograms")
    print("- Gate Loss: How well it knows when to stop speaking")
    print("- Total Loss: Combined score")
    print()
    print("🎯 YOUR PROGRESS:")
    print("- Started: Mel Loss = 48.16 (very poor)")
    print("- Current: Mel Loss = 6.08 (very good!)")
    print("- Improvement: 87% better!")
    print()
    print("✅ WHAT THIS MEANS:")
    print("- Your model is learning to 'see' speech patterns")
    print("- Lower mel loss = more natural sounding speech")
    print("- You're on track for good quality text-to-speech!")

if __name__ == '__main__':
    try:
        create_mel_explanation()
    except ImportError as e:
        print(f"Missing library: {e}")
        print("Install with: pip install librosa matplotlib")
        
        # Provide text explanation instead
        print("\n=== MEL SPECTROGRAM EXPLANATION (Text Version) ===")
        print()
        print("🎵 MEL SPECTROGRAM = 'Picture of Sound'")
        print("   - X-axis: Time (when sounds happen)")
        print("   - Y-axis: Frequency (pitch - high/low sounds)")
        print("   - Colors: Intensity (loud/quiet)")
        print()
        print("🤖 IN YOUR TRAINING:")
        print("   - Model learns to convert text → mel spectrogram")
        print("   - Mel Loss = how accurate these 'sound pictures' are")
        print("   - Lower loss = better quality speech")
        print()
        print("📊 YOUR AMAZING PROGRESS:")
        print("   - Started: Mel=48.16 (terrible quality)")
        print("   - Current: Mel=6.08 (good quality!)")
        print("   - Improvement: 87% better!")
        print()
        print("🎯 WHAT TO EXPECT:")
        print("   - Mel < 10: Decent speech quality")
        print("   - Mel < 5: Good speech quality")
        print("   - Mel < 2: Excellent speech quality")
