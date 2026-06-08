import os
import torch
from torch.utils.data import DataLoader
from .config import DEVICE, EPOCHS_ABLATION, BATCH_SIZE, CHECKPOINT_DIR
from .data_loader import get_celeba_dataset
from .models.unet import ResUNet_W1A16
from .samplers.schedule import DiffusionSchedule
from .trainers.base import train_model
from .logger import logger

def train_w1a16_optimized():
    logger.info("=== Starting Optimized W1A16 Native Training (AdamW, LR=2e-4) ===")
    
    # 1. Load dataset (10,000 real grayscale CelebA images)
    dataset = get_celeba_dataset(truncate_size=10000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    # Initialize schedule
    schedule = DiffusionSchedule(device=DEVICE)
    
    # 2. Train W1A16 Native with AdamW and lower LR
    w1a16_model = ResUNet_W1A16().to(DEVICE)
    w1a16_opt = torch.optim.AdamW(w1a16_model.parameters(), lr=2e-4, weight_decay=1e-2)
    
    train_model(w1a16_model, loader, w1a16_opt, schedule, EPOCHS_ABLATION, DEVICE, model_name="W1A16_Native")
    
    # Save the updated checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "w1a16_native.pth")
    torch.save(w1a16_model.state_dict(), ckpt_path)
    logger.info(f"Optimized W1A16 Native checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    train_w1a16_optimized()
