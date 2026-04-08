"""
DDPM sampler — standard 1000-step stochastic reverse diffusion.

Implements Algorithm 2 from Ho et al. (2020):
  x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ā_t) * ε_θ(x_t, t)) + √β_t * z
"""

import torch
from .schedule import DiffusionSchedule


@torch.no_grad()
def ddpm_sample(model, schedule, n_samples, image_shape=(3, 32, 32), device='cuda'):
    """
    Generate samples using DDPM reverse process.

    Args:
        model:       Trained noise-prediction model ε_θ(x_t, t)
        schedule:    DiffusionSchedule instance
        n_samples:   Number of images to generate
        image_shape: (C, H, W) tuple
        device:      Target device

    Returns:
        Tensor of shape (n_samples, C, H, W) in [-1, 1] range
    """
    model.eval()
    x = torch.randn(n_samples, *image_shape, device=device)

    for i in reversed(range(schedule.T)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)

        predicted_noise = model(x, t)

        alpha = schedule.alphas[i]
        alpha_cumprod = schedule.alphas_cumprod[i]
        beta = schedule.betas[i]

        # DDPM reverse step
        coeff = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
        x = (1 / torch.sqrt(alpha)) * (x - coeff * predicted_noise)

        # Add noise for all steps except t=0
        if i > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta) * noise

    return x.clamp(-1, 1)
