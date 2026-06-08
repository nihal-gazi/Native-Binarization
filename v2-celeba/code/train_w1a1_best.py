import os
import torch
from torch.utils.data import DataLoader
from .config import DEVICE, EPOCHS_ABLATION, BATCH_SIZE, CHECKPOINT_DIR
from .data_loader import get_celeba_dataset
from .models.unet import ResUNet_W1A1
from .samplers.schedule import DiffusionSchedule
from .trainers.base import train_model
from .logger import logger

def train_w1a1_best():
    logger.info("=== Restoring Best W1A1 Native Training (AdamW, LR=1e-4) ===")
    
    # 1. Load dataset (10,000 real grayscale CelebA images)
    dataset = get_celeba_dataset(truncate_size=10000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    # Initialize schedule
    schedule = DiffusionSchedule(device=DEVICE)
    
    # 2. Train W1A1 Native with AdamW and 1e-4 LR
    w1a1_model = ResUNet_W1A1().to(DEVICE)
    w1a1_opt = torch.optim.AdamW(w1a1_model.parameters(), lr=1e-4)
    
    train_model(w1a1_model, loader, w1a1_opt, schedule, EPOCHS_ABLATION, DEVICE, model_name="W1A1_Native")
    
    # Save the updated checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "w1a1_native.pth")
    torch.save(w1a1_model.state_dict(), ckpt_path)
    logger.info(f"Best W1A1 Native checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    train_w1a1_best()
