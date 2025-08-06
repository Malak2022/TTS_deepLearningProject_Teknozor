"""
Tacotron2 Configuration File
"""

import os

class Config:
    # Data paths
    DATA_PATH = r"D:\Project\data\LJSpeech-1.1"
    PREPROCESSED_PATH = r"D:\Project\data\LJSpeech-1.1\preprocessed"
    METADATA_FILE = os.path.join(PREPROCESSED_PATH, "metadata.json")
    
    # Model checkpoints
    CHECKPOINT_DIR = r"D:\Project\tacotron2\checkpoints"
    LOG_DIR = r"D:\Project\tacotron2\logs"
    
    # Audio parameters (should match preprocessing)
    SAMPLE_RATE = 22050
    N_MELS = 80
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_FFT = 1024
    
    # Text processing
    MAX_TEXT_LENGTH = 200
    VOCAB_SIZE = 43  # Based on your character set: a-z, 0-9, punctuation, space, pad, eos
    
    # Model architecture
    # Encoder
    ENCODER_EMBEDDING_DIM = 512
    ENCODER_N_CONVOLUTIONS = 3
    ENCODER_KERNEL_SIZE = 5
    
    # Decoder
    N_FRAMES_PER_STEP = 1  # Number of frames to predict per step
    DECODER_RNN_DIM = 1024
    PRENET_DIM = 256
    MAX_DECODER_STEPS = 1000
    GATE_THRESHOLD = 0.5
    P_ATTENTION_DROPOUT = 0.1
    P_DECODER_DROPOUT = 0.1
    
    # Attention
    ATTENTION_RNN_DIM = 1024
    ATTENTION_DIM = 128
    
    # Location Layer
    ATTENTION_LOCATION_N_FILTERS = 32
    ATTENTION_LOCATION_KERNEL_SIZE = 31
    
    # Postnet
    POSTNET_EMBEDDING_DIM = 512
    POSTNET_KERNEL_SIZE = 5
    POSTNET_N_CONVOLUTIONS = 5
    
    # Training
    BATCH_SIZE = 8  # Reduced from 32 for faster training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-6
    GRAD_CLIP_THRESH = 1.0
    N_EPOCHS = 50  # Reduced from 500 for testing
    
    # Loss weights
    MEL_LOSS_WEIGHT = 1.0
    GATE_LOSS_WEIGHT = 1.0
    
    # Validation
    VAL_SPLIT = 0.1
    VAL_BATCH_SIZE = 16
    
    # Inference
    INFERENCE_BATCH_SIZE = 1
    
    # Device
    DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    # Logging
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 1000
    VALIDATE_INTERVAL = 500
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
