"""
Binary neural network layers for native binarization of diffusion models.

Implements:
- BitConv2d_Std: Mean-centered weight binarization (W1A16)
- BitConv2d_BNN: Standard BNN binarization (W1A1)
- BinaryActivation: Sign activation with clipped STE gradient
- BinaryTanh: Module wrapper for BinaryActivation
- SinusoidalPositionEmbeddings: Timestep encoding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embeddings
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# W1A16: Mean-centered weight binarization with STE
# ---------------------------------------------------------------------------

class BitConv2d_Std(nn.Conv2d):
    """
    Binary convolution with mean-centered weight binarization.

    Weights are centered per-filter (subtract channel mean), then binarized
    via sign() and scaled by L1 norm. The straight-through estimator (STE)
    passes gradients through the non-differentiable sign operation.

    Used for W1A16 (binary weights, full-precision activations).
    """

    def forward(self, x):
        w = self.weight
        # Per-filter centering: subtract mean across spatial + input channel dims
        w_centered = w - w.mean(dim=(1, 2, 3), keepdim=True)
        # Channel-wise scale factor (L1 norm)
        scale = w_centered.abs().mean(dim=(1, 2, 3), keepdim=True)
        # Binarize with STE: forward uses sign*scale, backward uses real weights
        w_bin = torch.sign(w_centered) * scale
        w_final = (w_bin - w).detach() + w  # STE trick
        return F.conv2d(x, w_final, self.bias, self.stride, self.padding)


# ---------------------------------------------------------------------------
# W1A1: Standard BNN binarization (no centering)
# ---------------------------------------------------------------------------

class BitConv2d_BNN(nn.Conv2d):
    """
    Binary convolution without centering (standard BNN approach).

    Weights are binarized directly via sign() and scaled by L1 norm.
    No per-filter centering is applied, matching traditional BNN literature.

    Used for W1A1 (fully binary weights and activations).
    """

    def forward(self, x):
        w = self.weight
        alpha = w.abs().mean(dim=(1, 2, 3), keepdim=True)
        w_bin = (w.sign() * alpha - w).detach() + w  # STE
        return F.conv2d(x, w_bin, self.bias, self.stride, self.padding)


class BinaryActivation(torch.autograd.Function):
    """
    Binary activation: sign() in forward, clipped STE in backward.

    Forward: output = sign(input), producing {-1, +1}
    Backward: gradient is passed through where |input| <= 1, zeroed elsewhere
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0  # Clip gradient outside [-1, 1]
        return grad_input


class BinaryTanh(nn.Module):
    """Module wrapper for BinaryActivation."""

    def forward(self, x):
        return BinaryActivation.apply(x)
