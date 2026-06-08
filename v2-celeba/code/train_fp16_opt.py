import os
import torch
from torch.utils.data import DataLoader
from .config import DEVICE, EPOCHS_ABLATION, BATCH_SIZE, CHECKPOINT_DIR
from .data_loader import get_celeba_dataset
from .models.unet import ResUNet_FP16
from .samplers.schedule import DiffusionSchedule
from .trainers.base import train_model
from .logger import logger

def train_fp16_optimized():
    logger.info("=== Starting Optimized FP16 Baseline Training (AdamW, LR=2e-4) ===")
    
    # 1. Load dataset (10,000 real grayscale CelebA images)
    dataset = get_celeba_dataset(truncate_size=10000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    # Initialize schedule
    schedule = DiffusionSchedule(device=DEVICE)
    
    # 2. Train FP16 Baseline with AdamW and lower LR
    fp16_model = ResUNet_FP16().to(DEVICE)
    fp16_opt = torch.optim.AdamW(fp16_model.parameters(), lr=2e-4, weight_decay=1e-2)
    
    train_model(fp16_model, loader, fp16_opt, schedule, EPOCHS_ABLATION, DEVICE, model_name="FP16_Baseline")
    
    # Save the updated checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "fp16_baseline.pth")
    torch.save(fp16_model.state_dict(), ckpt_path)
    logger.info(f"Optimized FP16 Baseline checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    train_fp16_optimized()
