"""
DDIM sampler — deterministic accelerated sampling.

Implements Song et al. (2020) "Denoising Diffusion Implicit Models":
  x_{t-1} = √ā_{t-1} * x̂_0 + √(1 - ā_{t-1} - σ²) * ε_θ(x_t, t) + σ * z

With eta=0 (deterministic), σ=0 and no noise is added:
  x_{t-1} = √ā_{t-1} * x̂_0 + √(1 - ā_{t-1}) * ε_θ(x_t, t)

where x̂_0 = (x_t - √(1-ā_t) * ε_θ) / √ā_t  (predicted clean image)

Key advantage: uses a subsequence of timesteps (e.g. 50 out of 1000),
enabling 20x speedup with minimal quality loss. Critical for binary models
where error accumulates over many steps.
"""

import torch
import numpy as np
from .schedule import DiffusionSchedule


def make_ddim_timesteps(num_ddim_steps, ddpm_steps=1000, skip_type='uniform'):
    """
    Create a subsequence of timesteps for DDIM sampling.

    Args:
        num_ddim_steps: Number of DDIM steps (e.g. 50, 100, 200)
        ddpm_steps:     Total DDPM timesteps (typically 1000)
        skip_type:      'uniform' for evenly spaced, 'quad' for quadratic spacing

    Returns:
        1D numpy array of timestep indices in ascending order
    """
    if skip_type == 'uniform':
        step = ddpm_steps // num_ddim_steps
        timesteps = np.arange(0, ddpm_steps, step)
    elif skip_type == 'quad':
        timesteps = (np.linspace(0, np.sqrt(ddpm_steps * 0.8), num_ddim_steps) ** 2).astype(int)
    else:
        raise ValueError(f"Unknown skip_type: {skip_type}")
    return timesteps


@torch.no_grad()
def ddim_sample(model, schedule, n_samples, num_steps=50, eta=0.0,
                image_shape=(3, 32, 32), device='cuda'):
    """
    Generate samples using DDIM reverse process.

    Args:
        model:       Trained noise-prediction model ε_θ(x_t, t)
        schedule:    DiffusionSchedule instance
        n_samples:   Number of images to generate
        num_steps:   Number of DDIM sampling steps (50, 100, 200, etc.)
        eta:         Stochasticity parameter. 0 = deterministic (pure DDIM),
                     1 = equivalent to DDPM
        image_shape: (C, H, W) tuple
        device:      Target device

    Returns:
        Tensor of shape (n_samples, C, H, W) in [-1, 1] range
    """
    model.eval()

    # Build timestep subsequence
    ddim_timesteps = make_ddim_timesteps(num_steps, schedule.T)

    # Start from pure noise
    x = torch.randn(n_samples, *image_shape, device=device)

    # Iterate in reverse over the DDIM timestep subsequence
    for i in reversed(range(len(ddim_timesteps))):
        t_cur = ddim_timesteps[i]
        t_batch = torch.full((n_samples,), t_cur, device=device, dtype=torch.long)

        # Current and previous alpha_cumprod
        alpha_cumprod_t = schedule.alphas_cumprod[t_cur]
        alpha_cumprod_prev = (schedule.alphas_cumprod[ddim_timesteps[i - 1]]
                              if i > 0 else torch.tensor(1.0, device=device))

        # Predict noise
        eps_pred = model(x, t_batch)

        # Predict x_0 from x_t and predicted noise
        x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * eps_pred) / torch.sqrt(alpha_cumprod_t)
        x0_pred = x0_pred.clamp(-1, 1)  # Clip for stability

        # Compute sigma (stochasticity)
        sigma = (eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
                                   * (1 - alpha_cumprod_t / alpha_cumprod_prev)))

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma ** 2) * eps_pred

        # Combine
        x = torch.sqrt(alpha_cumprod_prev) * x0_pred + dir_xt

        # Add noise if eta > 0 and not the final step
        if eta > 0 and i > 0:
            x = x + sigma * torch.randn_like(x)

    return x.clamp(-1, 1)
