import torch
import torch.nn as nn
from .binarization import BitConv2d_Std, BitConv2d_BNN, BinaryTanh_BNN

class ResBlock16(nn.Module):
    """Pre-activation ResBlock for FP16 (Full Precision) layers."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.act1(self.bn1(x))
        h = self.conv1(h)
        time_emb = self.time_mlp(t)[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.act2(self.bn2(h))
        h = self.conv2(h)
        return h + self.skip(x)

class ResBlock1Bit(nn.Module):
    """Pre-activation ResBlock with 1-bit weights (BitConv2d_Std) and FP16 activations."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = BitConv2d_Std(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = BitConv2d_Std(out_ch, out_ch, 3, padding=1)
        self.skip = BitConv2d_Std(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.act1(self.bn1(x))
        h = self.conv1(h)
        time_emb = self.time_mlp(t)[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.act2(self.bn2(h))
        h = self.conv2(h)
        return h + self.skip(x)

class ResBlockBNN(nn.Module):
    """Pre-activation ResBlock with 1-bit weights (BitConv2d_BNN) and 1-bit activations (BinaryTanh_BNN)."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = BinaryTanh_BNN()
        self.conv1 = BitConv2d_BNN(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = BinaryTanh_BNN()
        self.conv2 = BitConv2d_BNN(out_ch, out_ch, 3, padding=1)
        self.skip = BitConv2d_BNN(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.act1(self.bn1(x))
        h = self.conv1(h)
        time_emb = self.time_mlp(t)[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.act2(self.bn2(h))
        h = self.conv2(h)
        return h + self.skip(x)
