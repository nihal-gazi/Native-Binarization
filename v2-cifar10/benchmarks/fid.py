"""
FID (Fréchet Inception Distance) evaluation for CIFAR-10 diffusion models.

Computes FID between generated samples and CIFAR-10 test set using
InceptionV3 features.

Usage:
    python -m benchmarks.fid --checkpoint checkpoints/fp16/fp16_best.pth \
                              --variant fp16 --sampler ddim --steps 50
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from samplers import DiffusionSchedule, ddpm_sample, ddim_sample


class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 with final FC replaced by Identity for feature extraction."""

    def __init__(self):
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        # Inception expects 299x299 RGB in [-1, 1]
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear',
                                       align_corners=False)
        return self.model(x)


def compute_statistics(features):
    """Compute mean and covariance of feature vectors."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2):
    """
    Fréchet distance between two multivariate Gaussians.

    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


@torch.no_grad()
def extract_features(extractor, dataloader, device, max_samples=10000):
    """Extract InceptionV3 features from a dataloader."""
    all_features = []
    n = 0
    for images, _ in dataloader:
        if n >= max_samples:
            break
        images = images.to(device)
        feats = extractor(images).cpu().numpy()
        all_features.append(feats)
        n += feats.shape[0]
    return np.concatenate(all_features, axis=0)[:max_samples]


@torch.no_grad()
def extract_features_from_tensor(extractor, images, device, batch_size=64):
    """Extract InceptionV3 features from a tensor of images."""
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        feats = extractor(batch).cpu().numpy()
        all_features.append(feats)
    return np.concatenate(all_features, axis=0)


def evaluate_fid(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = build_model(variant=args.variant).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded {args.variant.upper()} from {args.checkpoint}")

    # Schedule
    schedule = DiffusionSchedule(timesteps=1000, device=device)

    # Inception feature extractor
    extractor = InceptionFeatureExtractor().to(device)

    # --- Real features (CIFAR-10 test set) ---
    print("Extracting real image features...")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    test_set = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                 transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    real_features = extract_features(extractor, test_loader, device,
                                      max_samples=args.n_real)
    mu_real, sigma_real = compute_statistics(real_features)
    print(f"  Real: {real_features.shape[0]} samples, feature dim={real_features.shape[1]}")

    # --- Generated features ---
    print(f"Generating {args.n_gen} samples with {args.sampler} ({args.steps} steps)...")
    gen_images = []
    remaining = args.n_gen
    gen_batch = min(args.gen_batch, remaining)
    while remaining > 0:
        n = min(gen_batch, remaining)
        if args.sampler == 'ddim':
            batch = ddim_sample(model, schedule, n, num_steps=args.steps,
                                device=device)
        else:
            batch = ddpm_sample(model, schedule, n, device=device)
        gen_images.append(batch.cpu())
        remaining -= n
        print(f"  Generated {args.n_gen - remaining}/{args.n_gen}")

    gen_images = torch.cat(gen_images, dim=0)
    gen_features = extract_features_from_tensor(extractor, gen_images, device)
    mu_gen, sigma_gen = compute_statistics(gen_features)
    print(f"  Generated: {gen_features.shape[0]} samples")

    # --- Compute FID ---
    fid = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"\n{'='*50}")
    print(f"FID ({args.variant.upper()}, {args.sampler.upper()}, {args.steps} steps): {fid:.2f}")
    print(f"{'='*50}")

    return fid


def main():
    parser = argparse.ArgumentParser(description='FID Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True,
                        choices=['fp16', 'w1a16', 'w1a1'])
    parser.add_argument('--sampler', type=str, default='ddim',
                        choices=['ddpm', 'ddim'])
    parser.add_argument('--steps', type=int, default=50,
                        help='DDIM steps (ignored for DDPM)')
    parser.add_argument('--n_gen', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--n_real', type=int, default=10000,
                        help='Number of real samples for statistics')
    parser.add_argument('--gen_batch', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    evaluate_fid(args)


if __name__ == '__main__':
    main()
