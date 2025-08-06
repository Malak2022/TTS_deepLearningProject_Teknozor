"""
Complete guide for next steps after Tacotron2 training
"""

import os
import sys

def check_training_status():
    """Check if training is complete and what models are available"""
    print("🔍 CHECKING TRAINING STATUS")
    print("=" * 50)
    
    checkpoint_dir = "tacotron2/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("❌ No checkpoint directory found!")
        return False
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(checkpoint_dir, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            checkpoints.append((filename, size_mb))
    
    if not checkpoints:
        print("❌ No trained models found!")
        print("Please complete training first.")
        return False
    
    print("✅ Found trained models:")
    for filename, size_mb in checkpoints:
        print(f"   📁 {filename} ({size_mb:.1f} MB)")
    
    return True

def main():
    """Main guide function"""
    print("🚀 TACOTRON2 POST-TRAINING WORKFLOW")
    print("=" * 60)
    
    # Check if training is complete
    if not check_training_status():
        print("\n⚠️ Complete training first before proceeding!")
        return
    
    print("\n🎯 COMPLETE NEXT STEPS WORKFLOW:")
    print("=" * 60)
    
    steps = [
        {
            "step": 1,
            "title": "📊 Evaluate Training Results",
            "command": "python evaluate_model.py",
            "description": [
                "• Check training progress and loss curves",
                "• Assess model quality based on validation loss",
                "• Generate training progress plots",
                "• Determine if more training is needed"
            ]
        },
        {
            "step": 2,
            "title": "🎤 Test Speech Synthesis",
            "command": "python test_synthesis.py",
            "description": [
                "• Generate speech from multiple test texts",
                "• Create mel spectrograms and attention plots",
                "• Save audio files for quality assessment",
                "• Check attention alignment (should be diagonal)"
            ]
        },
        {
            "step": 3,
            "title": "🔊 Interactive Synthesis",
            "command": "python synthesize_text.py",
            "description": [
                "• Synthesize custom text inputs",
                "• Generate high-quality audio samples",
                "• Create visualizations of synthesis process",
                "• Test with your own sentences"
            ]
        },
        {
            "step": 4,
            "title": "📈 Quality Assessment",
            "command": "Manual evaluation",
            "description": [
                "• Listen to generated audio samples",
                "• Compare with original LJSpeech recordings",
                "• Check for naturalness and intelligibility",
                "• Evaluate pronunciation accuracy"
            ]
        },
        {
            "step": 5,
            "title": "🎯 Model Optimization (Optional)",
            "command": "Various options",
            "description": [
                "• Fine-tune with more epochs if quality is poor",
                "• Adjust hyperparameters for better results",
                "• Train on additional data for robustness",
                "• Implement neural vocoder for better audio quality"
            ]
        },
        {
            "step": 6,
            "title": "🚀 Production Deployment",
            "command": "Implementation dependent",
            "description": [
                "• Create API for text-to-speech service",
                "• Optimize model for inference speed",
                "• Implement real-time synthesis",
                "• Deploy to web/mobile applications"
            ]
        }
    ]
    
    for step_info in steps:
        print(f"\n{step_info['step']}. {step_info['title']}")
        print(f"   Command: {step_info['command']}")
        for desc in step_info['description']:
            print(f"   {desc}")
    
    print("\n" + "=" * 60)
    print("🎉 EXPECTED OUTCOMES:")
    print("=" * 60)
    
    outcomes = {
        "🔊 Audio Quality": [
            "• Intelligible speech (you can understand the words)",
            "• Natural prosody (rhythm and intonation)",
            "• Clear pronunciation of most phonemes",
            "• Minimal artifacts or distortions"
        ],
        "📊 Technical Metrics": [
            "• Validation loss < 5.0 (good quality)",
            "• Validation loss < 2.0 (excellent quality)",
            "• Diagonal attention alignments",
            "• Stable gate predictions"
        ],
        "🎯 Performance": [
            "• Synthesis time: ~1-5 seconds per sentence",
            "• Memory usage: ~1-2 GB for inference",
            "• Model size: ~100-200 MB",
            "• Compatible with CPU and GPU inference"
        ]
    }
    
    for category, items in outcomes.items():
        print(f"\n{category}:")
        for item in items:
            print(f"   {item}")
    
    print("\n" + "=" * 60)
    print("🚨 TROUBLESHOOTING:")
    print("=" * 60)
    
    troubleshooting = {
        "Poor Audio Quality": [
            "• Train for more epochs (100+ recommended)",
            "• Check attention alignments (should be diagonal)",
            "• Verify data preprocessing quality",
            "• Consider using neural vocoder (WaveGlow/HiFi-GAN)"
        ],
        "Robotic/Unnatural Speech": [
            "• Increase training data diversity",
            "• Adjust learning rate and training schedule",
            "• Fine-tune on target domain data",
            "• Implement better text preprocessing"
        ],
        "Synthesis Errors": [
            "• Check vocabulary coverage",
            "• Verify text cleaning and normalization",
            "• Ensure model convergence",
            "• Debug attention mechanism"
        ]
    }
    
    for problem, solutions in troubleshooting.items():
        print(f"\n❌ {problem}:")
        for solution in solutions:
            print(f"   {solution}")
    
    print("\n" + "=" * 60)
    print("🎊 CONGRATULATIONS!")
    print("You've successfully implemented and trained Tacotron2!")
    print("Start with Step 1 above to evaluate your results.")
    print("=" * 60)

if __name__ == '__main__':
    main()
