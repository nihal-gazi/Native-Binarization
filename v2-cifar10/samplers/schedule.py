"""
Diffusion noise schedule utilities.

Provides the linear beta schedule and all derived quantities needed
by both DDPM and DDIM samplers.
"""

import torch


def linear_beta_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule:
    """
    Pre-computes and stores all schedule quantities on a given device.

    Attributes:
        T:                  Number of timesteps
        betas:              (T,) noise schedule
        alphas:             (T,) = 1 - betas
        alphas_cumprod:     (T,) cumulative product of alphas = ā_t
        sqrt_alphas_cumprod:        √ā_t
        sqrt_one_minus_alphas_cumprod:  √(1 - ā_t)
        sqrt_recip_alphas:  1/√α_t  (used in DDPM reverse step)
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = timesteps
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

        # Derived quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # For DDPM posterior variance
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device),
                                          self.alphas_cumprod[:-1]])
        self.posterior_variance = (self.betas * (1.0 - alphas_cumprod_prev)
                                   / (1.0 - self.alphas_cumprod))

    def to(self, device):
        """Move all tensors to the given device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: add noise to x_0 at timestep t.

        q(x_t | x_0) = √ā_t * x_0 + √(1-ā_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_a = self.sqrt_alphas_cumprod[t]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t]
        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        while sqrt_a.dim() < x_0.dim():
            sqrt_a = sqrt_a.unsqueeze(-1)
            sqrt_1ma = sqrt_1ma.unsqueeze(-1)
        return sqrt_a * x_0 + sqrt_1ma * noise
