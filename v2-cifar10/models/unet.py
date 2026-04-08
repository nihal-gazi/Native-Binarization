"""
Binary ResUNet architectures for CIFAR-10 diffusion.

Three variants with identical topology but different precision:
- ResUNet_FP16:  Full-precision baseline (FP32/16 weights + activations)
- ResUNet_W1A16: Binary weights, full-precision activations (BitConv2d_Std)
- ResUNet_W1A1:  Fully binary weights + activations (BitConv2d_BNN + BinaryTanh)

Architecture: 3-channel input, widths [128, 256, 512], 3 downsample levels
Image path: 32x32 -> 16x16 -> 8x8 -> 4x4 (bottleneck) -> 8x8 -> 16x16 -> 32x32

Boundary convention: first conv (in) and last conv (out) are always FP16.
"""

import torch
import torch.nn as nn

from .layers import (
    SinusoidalPositionEmbeddings,
    BitConv2d_Std,
    BitConv2d_BNN,
    BinaryTanh,
)


# ---------------------------------------------------------------------------
# Time embedding MLP (shared by all variants)
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """Sinusoidal -> Linear -> Act -> Linear time embedding."""

    def __init__(self, sinusoidal_dim=64, hidden_dim=256, act=nn.SiLU):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPositionEmbeddings(sinusoidal_dim),
            nn.Linear(sinusoidal_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        return self.net(t)


# ===================================================================
# FP16 VARIANT
# ===================================================================

class ResBlock_FP16(nn.Module):
    """Pre-activation residual block: BN -> SiLU -> Conv (full precision)."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t)[(...,) + (None,) * 2]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class ResUNet_FP16(nn.Module):
    """Full-precision ResUNet for CIFAR-10 (32x32 RGB)."""

    def __init__(self, in_channels=3, base_ch=128, ch_mult=(1, 2, 4), time_dim=256):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]  # [128, 256, 512]

        # Time embedding
        self.time_embed = TimeEmbedding(sinusoidal_dim=64, hidden_dim=time_dim)

        # Encoder (boundary: first conv is always FP)
        self.conv_in = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down1 = ResBlock_FP16(chs[0], chs[0], time_dim)
        self.pool1 = nn.MaxPool2d(2)                           # 32 -> 16

        self.down2 = ResBlock_FP16(chs[0], chs[1], time_dim)
        self.pool2 = nn.MaxPool2d(2)                           # 16 -> 8

        self.down3 = ResBlock_FP16(chs[1], chs[2], time_dim)
        self.pool3 = nn.MaxPool2d(2)                           # 8 -> 4

        # Bottleneck
        self.bottleneck = ResBlock_FP16(chs[2], chs[2], time_dim)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')  # 4 -> 8
        self.dec3 = ResBlock_FP16(chs[2] + chs[2], chs[1], time_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')  # 8 -> 16
        self.dec2 = ResBlock_FP16(chs[1] + chs[1], chs[0], time_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')  # 16 -> 32
        self.dec1 = ResBlock_FP16(chs[0] + chs[0], chs[0], time_dim)

        # Output (boundary: last conv is always FP)
        self.conv_out = nn.Conv2d(chs[0], in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        # Encoder
        x0 = self.conv_in(x)                           # (B, 128, 32, 32)
        d1 = self.down1(x0, t_emb)                     # (B, 128, 32, 32)
        d2 = self.down2(self.pool1(d1), t_emb)          # (B, 256, 16, 16)
        d3 = self.down3(self.pool2(d2), t_emb)          # (B, 512,  8,  8)

        # Bottleneck
        b = self.bottleneck(self.pool3(d3), t_emb)      # (B, 512,  4,  4)

        # Decoder with skip connections
        h = self.dec3(torch.cat([self.up3(b), d3], dim=1), t_emb)   # (B, 256, 8, 8)
        h = self.dec2(torch.cat([self.up2(h), d2], dim=1), t_emb)   # (B, 128, 16, 16)
        h = self.dec1(torch.cat([self.up1(h), d1], dim=1), t_emb)   # (B, 128, 32, 32)

        return self.conv_out(h)


# ===================================================================
# W1A16 VARIANT (binary weights, full-precision activations)
# ===================================================================

class ResBlock_W1A16(nn.Module):
    """Pre-activation residual block with BitConv2d_Std (mean-centered)."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = BitConv2d_Std(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = BitConv2d_Std(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t)[(...,) + (None,) * 2]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class ResUNet_W1A16(nn.Module):
    """W1A16 ResUNet: binary weights, FP activations. Boundary layers stay FP."""

    def __init__(self, in_channels=3, base_ch=128, ch_mult=(1, 2, 4), time_dim=256):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]

        self.time_embed = TimeEmbedding(sinusoidal_dim=64, hidden_dim=time_dim)

        # Boundary: first conv FP
        self.conv_in = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down1 = ResBlock_W1A16(chs[0], chs[0], time_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ResBlock_W1A16(chs[0], chs[1], time_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = ResBlock_W1A16(chs[1], chs[2], time_dim)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ResBlock_W1A16(chs[2], chs[2], time_dim)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = ResBlock_W1A16(chs[2] + chs[2], chs[1], time_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ResBlock_W1A16(chs[1] + chs[1], chs[0], time_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ResBlock_W1A16(chs[0] + chs[0], chs[0], time_dim)

        # Boundary: last conv FP
        self.conv_out = nn.Conv2d(chs[0], in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        x0 = self.conv_in(x)
        d1 = self.down1(x0, t_emb)
        d2 = self.down2(self.pool1(d1), t_emb)
        d3 = self.down3(self.pool2(d2), t_emb)

        b = self.bottleneck(self.pool3(d3), t_emb)

        h = self.dec3(torch.cat([self.up3(b), d3], dim=1), t_emb)
        h = self.dec2(torch.cat([self.up2(h), d2], dim=1), t_emb)
        h = self.dec1(torch.cat([self.up1(h), d1], dim=1), t_emb)

        return self.conv_out(h)


# ===================================================================
# W1A1 VARIANT (fully binary weights + activations)
# ===================================================================

class ResBlock_W1A1(nn.Module):
    """Pre-activation residual block: BN -> BinaryTanh -> BitConv2d_BNN."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.act1 = BinaryTanh()
        self.conv1 = BitConv2d_BNN(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act2 = BinaryTanh()
        self.conv2 = BitConv2d_BNN(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t)[(...,) + (None,) * 2]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class ResUNet_W1A1(nn.Module):
    """W1A1 ResUNet: fully binary. Boundary layers stay FP."""

    def __init__(self, in_channels=3, base_ch=128, ch_mult=(1, 2, 4), time_dim=256):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]

        # W1A1 uses GELU in time embedding (matches v1 convention)
        self.time_embed = TimeEmbedding(sinusoidal_dim=64, hidden_dim=time_dim, act=nn.GELU)

        # Boundary: first conv FP
        self.conv_in = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        self.down1 = ResBlock_W1A1(chs[0], chs[0], time_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ResBlock_W1A1(chs[0], chs[1], time_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = ResBlock_W1A1(chs[1], chs[2], time_dim)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ResBlock_W1A1(chs[2], chs[2], time_dim)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec3 = ResBlock_W1A1(chs[2] + chs[2], chs[1], time_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ResBlock_W1A1(chs[1] + chs[1], chs[0], time_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ResBlock_W1A1(chs[0] + chs[0], chs[0], time_dim)

        # Boundary: last conv FP
        self.conv_out = nn.Conv2d(chs[0], in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        x0 = self.conv_in(x)
        d1 = self.down1(x0, t_emb)
        d2 = self.down2(self.pool1(d1), t_emb)
        d3 = self.down3(self.pool2(d2), t_emb)

        b = self.bottleneck(self.pool3(d3), t_emb)

        h = self.dec3(torch.cat([self.up3(b), d3], dim=1), t_emb)
        h = self.dec2(torch.cat([self.up2(h), d2], dim=1), t_emb)
        h = self.dec1(torch.cat([self.up1(h), d1], dim=1), t_emb)

        return self.conv_out(h)


# ===================================================================
# Model factory
# ===================================================================

def build_model(variant='fp16', **kwargs):
    """
    Factory function to build a model by variant name.

    Args:
        variant: One of 'fp16', 'w1a16', 'w1a1'
        **kwargs: Passed to the model constructor
    """
    models = {
        'fp16': ResUNet_FP16,
        'w1a16': ResUNet_W1A16,
        'w1a1': ResUNet_W1A1,
    }
    if variant not in models:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(models.keys())}")
    return models[variant](**kwargs)
