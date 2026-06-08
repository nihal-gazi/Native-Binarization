import torch
from tqdm import tqdm
from ..config import IMAGE_SIZE, CHANNELS, NUM_STEPS

@torch.no_grad()
def sample_ddpm(model, schedule, batch_size=16, device="cuda", seed=None):
    """
    DDPM sampling loop starting from pure Gaussian noise.
    Returns generated tensor normalized to [0, 1].
    """
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
        
    # 1. Start from Gaussian Noise
    img = torch.randn((batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=device)
    
    # 2. Sequential denoising loop
    for i in tqdm(reversed(range(0, NUM_STEPS)), desc="DDPM Denoising", total=NUM_STEPS, leave=False):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model(img, t)
        
        # Extract schedule variables
        betas_t = schedule.extract(schedule.betas, t, img.shape)
        sqrt_one_minus_alphas_cumprod_t = schedule.extract(
            schedule.sqrt_one_minus_alphas_cumprod, t, img.shape
        )
        sqrt_recip_alphas_t = schedule.extract(schedule.sqrt_recip_alphas, t, img.shape)
        
        # Calculate reverse mean
        model_mean = sqrt_recip_alphas_t * (
            img - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        
        if i == 0:
            img = model_mean
        else:
            posterior_variance_t = schedule.extract(schedule.posterior_variance, t, img.shape)
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance_t) * noise
            
    # Normalize images from [-1, 1] to [0, 1] and clamp
    img = (img + 1.0) / 2.0
    return img.clamp(0.0, 1.0)
