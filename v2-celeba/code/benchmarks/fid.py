import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights

@torch.no_grad()
def get_features(images, model, device, batch_size=32):
    model.eval()
    features_list = []
    # Upsample to Inception's expected 299x299 input size
    upsampler = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False)
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        if batch.size(1) == 1:
            batch = batch.repeat(1, 3, 1, 1)  # Expand grayscale to 3 channels
        batch = upsampler(batch)
        feats = model(batch)
        features_list.append(feats.cpu().numpy())
        
    return np.concatenate(features_list, axis=0)

def compute_fid(real_imgs, gen_imgs, device="cuda"):
    """Computes Frechet Inception Distance between real and generated image tensors."""
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=True).to(device)
    model.fc = nn.Identity()  # Extract features from pool3 layer
    
    # Calculate stats
    real_feats = get_features(real_imgs, model, device)
    gen_feats = get_features(gen_imgs, model, device)
    
    mu_real, sigma_real = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_feats, axis=0), np.cov(gen_feats, rowvar=False)
    
    # Fréchet Distance formula
    ssdiff = np.sum((mu_real - mu_gen) ** 2.0)
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return float(fid)
