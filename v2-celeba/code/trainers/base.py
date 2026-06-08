import time
import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from ..logger import logger, log_vram
from ..config import BASE_DIR, CHECKPOINT_DIR

def train_one_epoch(model, loader, optimizer, schedule, scaler, device):
    """Executes a single training epoch with mixed-precision (AMP)."""
    model.train()
    total_loss = 0.0
    mse = nn.MSELoss()
    
    for batch_idx, (x0, _) in enumerate(loader):
        x0 = x0.to(device)
        t = torch.randint(0, schedule.timesteps, (x0.size(0),), device=device).long()
        noise = torch.randn_like(x0)
        
        # Forward process coefficients
        sqrt_alphas_cumprod_t = schedule.extract(schedule.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = schedule.extract(
            schedule.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )
        
        # Add noise to image
        noisy_x = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        optimizer.zero_grad()
        
        # Use mixed-precision forward pass
        with autocast(device_type="cuda" if "cuda" in device else "cpu"):
            noise_pred = model(noisy_x, t)
            loss = mse(noise_pred, noise)
            
        # Scaled backprop
        if "cuda" in device:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        
    return total_loss / len(loader)

def train_model(model, loader, optimizer, schedule, epochs, device, model_name="model"):
    """Manages the full multi-epoch training lifecycle and records resource stats."""
    logger.info(f"Starting training for {model_name} on {device} for {epochs} epochs...")
    scaler = GradScaler(enabled=("cuda" in device))
    log_vram(f"{model_name} Pre-Train")
    
    epoch_times = []
    log_path = os.path.join(BASE_DIR, "training_log.txt")
    
    # Initialize log file
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== Training Session Started for {model_name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        lf.flush()
        
    for epoch in range(epochs):
        start_time = time.time()
        
        try:
            epoch_loss = train_one_epoch(model, loader, optimizer, schedule, scaler, device)
            elapsed = time.time() - start_time
            epoch_times.append(elapsed)
            
            # Compute rolling average of last 5 epochs
            avg_time = sum(epoch_times[-5:]) / len(epoch_times[-5:])
            remaining_epochs = epochs - (epoch + 1)
            eta_sec = remaining_epochs * avg_time
            
            if eta_sec >= 3600:
                eta_str = f"{int(eta_sec // 3600)} hr {int((eta_sec % 3600) // 60)} mins"
            elif eta_sec >= 60:
                eta_str = f"{int(eta_sec // 60)} mins"
            else:
                eta_str = f"{int(eta_sec)}s"
                
            log_msg = f"[Epoch {epoch+1:03d}/{epochs:03d}] Loss: {epoch_loss:.5f} | Time/Epoch: {elapsed:.1f}s | ETA: {eta_str}"
            logger.info(f"[{model_name}] {log_msg}")
            
            # Real-time log flushing to file
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{model_name}] {log_msg}\n")
                lf.flush()
                
            # Periodically log memory profiles
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                log_vram(f"{model_name} Epoch {epoch+1}")
                
            # Intermediate checkpoints every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                ckpt_name = f"{model_name.lower()}_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, ckpt_name))
                logger.info(f"Saved checkpoint: {ckpt_name}")
                
            # Proactively clear CUDA memory cache
            if "cuda" in device:
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.critical(f"FATAL EXCEPTION in training loop at epoch {epoch+1}: {e}", exc_info=True)
            raise e
            
    logger.info(f"Finished training {model_name}.")
    log_vram(f"{model_name} Post-Train")
