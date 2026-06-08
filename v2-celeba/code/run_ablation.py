import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .config import DEVICE, EPOCHS_ABLATION, BATCH_SIZE, LR_FP16, LR_W1A16, CHECKPOINT_DIR, OUTPUT_DIR
from .data_loader import get_celeba_dataset
from .models.unet import ResUNet_FP16, ResUNet_W1A16, ResUNet_W1A1
from .samplers.schedule import DiffusionSchedule
from .samplers.ddpm import sample_ddpm
from .trainers.base import train_model
from .logger import logger

def run_ablation():
    logger.info("=== Phase 3: Starting Academic Ablation Study (Grayscale) ===")
    
    # Clear previous training log
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_log.txt")
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
        except Exception as e:
            logger.warning(f"Could not remove old training log: {e}")
            
    # 1. Load dataset (10,000 real grayscale CelebA images)
    dataset = get_celeba_dataset(truncate_size=10000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    # Initialize schedule
    schedule = DiffusionSchedule(device=DEVICE)
    
    # 2. Train FP16 Baseline
    logger.info("--- Model 1/3: FP16 Baseline ---")
    fp16_model = ResUNet_FP16().to(DEVICE)
    fp16_opt = torch.optim.Adam(fp16_model.parameters(), lr=LR_FP16)
    train_model(fp16_model, loader, fp16_opt, schedule, EPOCHS_ABLATION, DEVICE, model_name="FP16_Baseline")
    torch.save(fp16_model.state_dict(), os.path.join(CHECKPOINT_DIR, "fp16_baseline.pth"))
    
    # 3. Train W1A16 Native (binarized weights during training)
    logger.info("--- Model 2/3: W1A16 Native ---")
    w1a16_model = ResUNet_W1A16().to(DEVICE)
    w1a16_opt = torch.optim.Adam(w1a16_model.parameters(), lr=LR_W1A16)
    train_model(w1a16_model, loader, w1a16_opt, schedule, EPOCHS_ABLATION, DEVICE, model_name="W1A16_Native")
    torch.save(w1a16_model.state_dict(), os.path.join(CHECKPOINT_DIR, "w1a16_native.pth"))
    
    # 4. Train W1A1 Native (binarized weights and activations during training)
    logger.info("--- Model 3/3: W1A1 Native ---")
    w1a1_model = ResUNet_W1A1().to(DEVICE)
    # W1A1 uses smaller LR (AdamW) for BNN optimization stability
    w1a1_opt = torch.optim.AdamW(w1a1_model.parameters(), lr=1e-4)
    train_model(w1a1_model, loader, w1a1_opt, schedule, EPOCHS_ABLATION, DEVICE, model_name="W1A1_Native")
    torch.save(w1a1_model.state_dict(), os.path.join(CHECKPOINT_DIR, "w1a1_native.pth"))
    
    # 5. Generate and save visual comparison grids (16 images each)
    logger.info("Generating final samples for visual comparison...")
    
    fp16_samples = sample_ddpm(fp16_model, schedule, batch_size=16, device=DEVICE)
    save_image(fp16_samples, os.path.join(OUTPUT_DIR, "ablation_fp16.png"), nrow=4)
    
    w1a16_samples = sample_ddpm(w1a16_model, schedule, batch_size=16, device=DEVICE)
    save_image(w1a16_samples, os.path.join(OUTPUT_DIR, "ablation_w1a16_native.png"), nrow=4)
    
    w1a1_samples = sample_ddpm(w1a1_model, schedule, batch_size=16, device=DEVICE)
    save_image(w1a1_samples, os.path.join(OUTPUT_DIR, "ablation_w1a1_native.png"), nrow=4)
    
    logger.info("Grayscale ablation visual grids saved successfully to outputs directory.")
    logger.info("=== Phase 3 Completed ===")

if __name__ == "__main__":
    run_ablation()
