"""
Unified CIFAR-10 DDPM trainer for all model variants.

Usage:
    python -m trainers.train --variant fp16 --epochs 200 --lr 1e-4
    python -m trainers.train --variant w1a16 --epochs 300 --lr 1e-4
    python -m trainers.train --variant w1a1  --epochs 300 --lr 5e-5

Training objective is always DDPM (epsilon prediction).
DDIM is only used at inference time — the training loss is identical.
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from samplers import DiffusionSchedule, ddim_sample
from torchvision.utils import save_image


def get_cifar10_loader(data_dir='./data', batch_size=128, num_workers=2):
    """CIFAR-10 dataloader normalized to [-1, 1]."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),  # [0,1] -> [-1,1]
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    model = build_model(variant=args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResUNet_{args.variant.upper()} | Params: {n_params:,}")

    # Noise schedule
    schedule = DiffusionSchedule(timesteps=1000, device=device)

    # Data
    loader = get_cifar10_loader(data_dir=args.data_dir, batch_size=args.batch_size)
    print(f"CIFAR-10 loaded: {len(loader.dataset)} images, batch_size={args.batch_size}")

    # Optimizer + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    loss_fn = nn.MSELoss()

    # Learning rate scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpoint directory
    ckpt_dir = args.ckpt_dir or f'./checkpoints/{args.variant}'
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs('./assets', exist_ok=True)

    # Training loop
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for images, _ in loader:
            images = images.to(device)
            B = images.shape[0]

            # Sample random timesteps
            t = torch.randint(0, schedule.T, (B,), device=device)

            # Sample noise and create noisy images
            noise = torch.randn_like(images)
            x_noisy = schedule.q_sample(images, t, noise)

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                noise_pred = model(x_noisy, t)
                loss = loss_fn(noise_pred, noise)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.6f} | "
              f"LR: {lr_now:.2e} | Time: {elapsed:.1f}s")

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0 or avg_loss < best_loss:
            path = os.path.join(ckpt_dir, f'{args.variant}_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'variant': args.variant,
            }, path)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(ckpt_dir, f'{args.variant}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'variant': args.variant,
                }, best_path)
                print(f"  -> Best model saved (loss={avg_loss:.6f})")

        # Generate samples every N epochs
        if epoch % args.sample_every == 0:
            model.eval()
            samples = ddim_sample(model, schedule, n_samples=16, num_steps=50,
                                  device=device)
            samples = (samples + 1) / 2  # [-1,1] -> [0,1]
            save_image(samples, f'./assets/{args.variant}_epoch{epoch}.png',
                       nrow=4, padding=2)
            print(f"  -> Samples saved to assets/{args.variant}_epoch{epoch}.png")

    # Save final model
    final_path = os.path.join(ckpt_dir, f'{args.variant}_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'loss': avg_loss,
        'variant': args.variant,
    }, final_path)
    print(f"Training complete. Final model: {final_path}")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Binary DDPM Trainer')
    parser.add_argument('--variant', type=str, default='fp16',
                        choices=['fp16', 'w1a16', 'w1a1'],
                        help='Model variant to train')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=25,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=25,
                        help='Generate sample grid every N epochs')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
