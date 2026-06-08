"""
Save all PTQ checkpoints from the FP16 baseline.
No training required — PTQ just copies FP16 weights into
the binarized architecture, which applies sign() on-the-fly.
"""
import os
import torch
from .config import DEVICE, CHECKPOINT_DIR
from .models.unet import ResUNet_FP16, ResUNet_W1A16, ResUNet_W1A1
from .trainers.ptq import convert_fp16_to_w1a16
from .logger import logger


def save_all_ptq():
    fp16_ckpt = os.path.join(CHECKPOINT_DIR, "fp16_baseline.pth")
    assert os.path.exists(fp16_ckpt), f"FP16 baseline not found at {fp16_ckpt}"

    # Load FP16 source
    fp16_model = ResUNet_FP16().to(DEVICE)
    fp16_model.load_state_dict(torch.load(fp16_ckpt, map_location=DEVICE))
    logger.info(f"Loaded FP16 baseline from {fp16_ckpt}")

    # --- W1A16 PTQ ---
    w1a16_ptq = ResUNet_W1A16().to(DEVICE)
    w1a16_ptq = convert_fp16_to_w1a16(fp16_model, w1a16_ptq)
    w1a16_path = os.path.join(CHECKPOINT_DIR, "w1a16_ptq.pth")
    torch.save(w1a16_ptq.state_dict(), w1a16_path)
    size_kb = os.path.getsize(w1a16_path) / 1024
    logger.info(f"Saved W1A16 PTQ checkpoint: {w1a16_path} ({size_kb:.1f} KB)")

    # --- W1A1 PTQ ---
    w1a1_ptq = ResUNet_W1A1().to(DEVICE)
    # Same conversion logic — copy FP16 weights into the W1A1 container
    fp16_state = fp16_model.state_dict()
    w1a1_state = w1a1_ptq.state_dict()
    converted = 0
    for key in w1a1_state.keys():
        if key in fp16_state:
            w1a1_state[key].copy_(fp16_state[key].float())
            converted += 1
        else:
            logger.warning(f"Key '{key}' not found in FP16 state dict.")
    w1a1_ptq.load_state_dict(w1a1_state)
    w1a1_path = os.path.join(CHECKPOINT_DIR, "w1a1_ptq.pth")
    torch.save(w1a1_ptq.state_dict(), w1a1_path)
    size_kb = os.path.getsize(w1a1_path) / 1024
    logger.info(f"Saved W1A1 PTQ checkpoint: {w1a1_path} ({size_kb:.1f} KB)")
    logger.info(f"Converted {converted} parameter tensors for W1A1 PTQ.")

    logger.info("=== All PTQ checkpoints saved successfully. No retraining needed. ===")


if __name__ == "__main__":
    save_all_ptq()
