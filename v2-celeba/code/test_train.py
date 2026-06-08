import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .config import DEVICE, EPOCHS_TINY, LR_W1A16, OUTPUT_DIR, CHECKPOINT_DIR
from .data_loader import get_celeba_dataset
from .models.unet import ResUNet_W1A16
from .samplers.schedule import DiffusionSchedule
from .samplers.ddpm import sample_ddpm
from .trainers.base import train_model
from .logger import logger

def run_test_train():
    logger.info("=== Phase 2: Starting Tiny Test Train ===")
    
    # 1. Load micro-batch of exactly 100 images
    dataset = get_celeba_dataset(truncate_size=100)
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    # 2. Build model and schedule
    model = ResUNet_W1A16().to(DEVICE)
    schedule = DiffusionSchedule(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_W1A16)
    
    # 3. Save original target batch for reference
    batch, _ = next(iter(loader))
    orig_path = os.path.join(OUTPUT_DIR, "tiny_original.png")
    save_image((batch + 1.0) / 2.0, orig_path, nrow=10)
    logger.info(f"Saved original micro-batch images to {orig_path}")
    
    # 4. Train/Overfit the W1A16 model
    train_model(model, loader, optimizer, schedule, EPOCHS_TINY, DEVICE, model_name="W1A16_Tiny")
    
    # 5. Generate and save final samples
    logger.info("Generating samples from the overfit model...")
    samples = sample_ddpm(model, schedule, batch_size=16, device=DEVICE)
    sample_path = os.path.join(OUTPUT_DIR, "tiny_overfit_samples.png")
    save_image(samples, sample_path, nrow=4)
    logger.info(f"Saved generated overfit samples to {sample_path}")
    
    # 6. Save checkpoint
    chk_path = os.path.join(CHECKPOINT_DIR, "w1a16_tiny_overfit.pth")
    torch.save(model.state_dict(), chk_path)
    logger.info(f"Saved checkpoint to {chk_path}")
    logger.info("=== Phase 2 Completed successfully ===")

if __name__ == "__main__":
    run_test_train()
