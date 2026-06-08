import torch
import torch.nn.functional as F
from ..config import NUM_STEPS

class DiffusionSchedule:
    """Precomputes DDPM noise schedule parameters and maps them to device."""
    def __init__(self, timesteps=NUM_STEPS, device="cuda"):
        self.device = device
        self.timesteps = timesteps

        # Linear noise schedule
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precomputed values for forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Precomputed values for backward process p_theta(x_{t-1} | x_t)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def extract(self, a, t, x_shape):
        """Extract coefficients at timestep t and reshape to broadcast."""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
