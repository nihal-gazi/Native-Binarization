import os
import torch

# Image Dimensions (Grayscale)
IMAGE_SIZE = 32  # Default to 32x32 for speed and stability
CHANNELS = 1     # MUST be grayscale as per SGD instructions

# Training Parameters
BATCH_SIZE = 128
NUM_STEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Epoch Limits
EPOCHS_TINY = 300
EPOCHS_ABLATION = 100  # Dynamic 100-epoch training run

# Model Capacity Config
CHANNELS_LIST = [128, 256, 512]

# Learning Rates
LR_FP16 = 1e-3
LR_W1A16 = 1e-3

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "dataset-samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# Ensure directories exist
for path in [DATA_DIR, SAMPLE_DIR, OUTPUT_DIR, CHECKPOINT_DIR]:
    os.makedirs(path, exist_ok=True)
