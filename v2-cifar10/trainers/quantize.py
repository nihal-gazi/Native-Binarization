"""
Post-Training Quantization (PTQ): convert a trained FP16 model to W1A16 or W1A1.

The key idea: copy all weights from a trained FP16 model into a binary model.
At inference time, the BitConv2d layers will binarize the weights on-the-fly.
No retraining is performed — this is pure weight transfer.

Usage:
    python -m trainers.quantize --source checkpoints/fp16/fp16_best.pth --target w1a16
    python -m trainers.quantize --source checkpoints/fp16/fp16_best.pth --target w1a1
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model


def quantize_model(source_path, target_variant, save_path=None):
    """
    Convert FP16 checkpoint to a binary variant via weight copying.

    Args:
        source_path:    Path to trained FP16 checkpoint (.pth)
        target_variant: 'w1a16' or 'w1a1'
        save_path:      Where to save the quantized model

    Returns:
        The quantized model (on CPU)
    """
    # Load source checkpoint
    ckpt = torch.load(source_path, map_location='cpu', weights_only=True)
    source_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    # Build source and target models
    model_fp16 = build_model('fp16')
    model_fp16.load_state_dict(source_sd)

    model_target = build_model(target_variant)
    target_sd = model_target.state_dict()

    # Copy matching weights
    copied, skipped = 0, 0
    for key in target_sd:
        if key in source_sd and target_sd[key].shape == source_sd[key].shape:
            target_sd[key] = source_sd[key].clone()
            copied += 1
        else:
            skipped += 1
            if key in source_sd:
                print(f"  Shape mismatch: {key} "
                      f"src={source_sd[key].shape} target={target_sd[key].shape}")
            else:
                print(f"  Missing in source: {key}")

    model_target.load_state_dict(target_sd)
    print(f"PTQ: FP16 -> {target_variant.upper()} | Copied: {copied} | Skipped: {skipped}")

    # Save
    if save_path is None:
        save_dir = f'./checkpoints/{target_variant}_ptq'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{target_variant}_ptq.pth')

    torch.save({
        'model_state_dict': model_target.state_dict(),
        'variant': target_variant,
        'source': source_path,
        'method': 'ptq',
    }, save_path)
    print(f"Saved quantized model to: {save_path}")

    return model_target


def main():
    parser = argparse.ArgumentParser(description='Post-Training Quantization')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to trained FP16 checkpoint')
    parser.add_argument('--target', type=str, required=True,
                        choices=['w1a16', 'w1a1'],
                        help='Target binary variant')
    parser.add_argument('--save', type=str, default=None,
                        help='Output path for quantized model')
    args = parser.parse_args()

    quantize_model(args.source, args.target, args.save)


if __name__ == '__main__':
    main()
