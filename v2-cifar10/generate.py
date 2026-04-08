"""
Generate and save sample grids from trained CIFAR-10 diffusion models.

Supports both DDPM and DDIM sampling with configurable step counts.

Usage:
    python generate.py --checkpoint checkpoints/fp16/fp16_best.pth \
                       --variant fp16 --sampler ddim --steps 50 --n 64
"""

import os
import sys
import argparse

import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from samplers import DiffusionSchedule, ddpm_sample, ddim_sample


def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = build_model(variant=args.variant).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    schedule = DiffusionSchedule(timesteps=1000, device=device)

    print(f"Generating {args.n} samples with {args.sampler.upper()} "
          f"({args.steps} steps)...")

    # Generate in batches
    all_samples = []
    remaining = args.n
    while remaining > 0:
        batch_n = min(args.batch_size, remaining)
        if args.sampler == 'ddim':
            batch = ddim_sample(model, schedule, batch_n, num_steps=args.steps,
                                device=device)
        else:
            batch = ddpm_sample(model, schedule, batch_n, device=device)
        all_samples.append(batch.cpu())
        remaining -= batch_n

    samples = torch.cat(all_samples, dim=0)
    samples = (samples + 1) / 2  # [-1,1] -> [0,1]

    # Save grid
    os.makedirs(args.output_dir, exist_ok=True)
    nrow = int(args.n ** 0.5)
    filename = f'{args.variant}_{args.sampler}_{args.steps}steps_{args.n}samples.png'
    path = os.path.join(args.output_dir, filename)
    save_image(samples, path, nrow=nrow, padding=2)
    print(f"Saved: {path}")

    # Also save individual images if requested
    if args.save_individual:
        ind_dir = os.path.join(args.output_dir, 'individual',
                               f'{args.variant}_{args.sampler}_{args.steps}')
        os.makedirs(ind_dir, exist_ok=True)
        for i, img in enumerate(samples):
            save_image(img, os.path.join(ind_dir, f'{i:05d}.png'))
        print(f"Individual images saved to: {ind_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate CIFAR-10 samples')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True,
                        choices=['fp16', 'w1a16', 'w1a1'])
    parser.add_argument('--sampler', type=str, default='ddim',
                        choices=['ddpm', 'ddim'])
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--n', type=int, default=64,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./assets')
    parser.add_argument('--save_individual', action='store_true',
                        help='Also save each image individually')
    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
