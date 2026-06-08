import torch
import torch.nn as nn
from ..config import CHANNELS, CHANNELS_LIST
from .embed import SinusoidalPositionEmbeddings
from .blocks import ResBlock16, ResBlock1Bit, ResBlockBNN

class ResUNet_FP16(nn.Module):
    """Residual UNet in Full Precision (FP16/FP32 baseline)."""
    def __init__(self, channels=CHANNELS_LIST):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(32), nn.Linear(32, 32), nn.ReLU())
        self.conv0 = nn.Conv2d(CHANNELS, channels[0], 3, padding=1)
        self.down1 = ResBlock16(channels[0], channels[1], 32)
        self.down2 = ResBlock16(channels[1], channels[2], 32)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = ResBlock16(channels[2] + channels[1], channels[1], 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = ResBlock16(channels[1] + channels[0], channels[0], 32)
        self.output = nn.Conv2d(channels[0], CHANNELS, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x0 = self.conv0(x)
        x1 = self.pool(x0)
        x1 = self.down1(x1, t_emb)
        x2 = self.pool(x1)
        x2 = self.down2(x2, t_emb)
        
        x_up1 = self.up1(x2)
        x_up1 = torch.cat([x_up1, x1], dim=1)
        x_up1 = self.up_conv1(x_up1, t_emb)
        
        x_up2 = self.up2(x_up1)
        x_up2 = torch.cat([x_up2, x0], dim=1)
        x_up2 = self.up_conv2(x_up2, t_emb)
        return self.output(x_up2)

class ResUNet_W1A16(nn.Module):
    """Residual UNet with 1-Bit weights inside ResBlocks. Boundary convs are FP."""
    def __init__(self, channels=CHANNELS_LIST):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(32), nn.Linear(32, 32), nn.ReLU())
        self.conv0 = nn.Conv2d(CHANNELS, channels[0], 3, padding=1)
        self.down1 = ResBlock1Bit(channels[0], channels[1], 32)
        self.down2 = ResBlock1Bit(channels[1], channels[2], 32)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = ResBlock1Bit(channels[2] + channels[1], channels[1], 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = ResBlock1Bit(channels[1] + channels[0], channels[0], 32)
        self.output = nn.Conv2d(channels[0], CHANNELS, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x0 = self.conv0(x)
        x1 = self.pool(x0)
        x1 = self.down1(x1, t_emb)
        x2 = self.pool(x1)
        x2 = self.down2(x2, t_emb)
        
        x_up1 = self.up1(x2)
        x_up1 = torch.cat([x_up1, x1], dim=1)
        x_up1 = self.up_conv1(x_up1, t_emb)
        
        x_up2 = self.up2(x_up1)
        x_up2 = torch.cat([x_up2, x0], dim=1)
        x_up2 = self.up_conv2(x_up2, t_emb)
        return self.output(x_up2)

class ResUNet_W1A1(nn.Module):
    """Strict BNN Residual UNet: 1-Bit weights & 1-Bit activations inside ResBlocks. Boundary convs are FP."""
    def __init__(self, channels=CHANNELS_LIST):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(32), nn.Linear(32, 32), nn.ReLU())
        self.conv0 = nn.Conv2d(CHANNELS, channels[0], 3, padding=1)
        self.down1 = ResBlockBNN(channels[0], channels[1], 32)
        self.down2 = ResBlockBNN(channels[1], channels[2], 32)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = ResBlockBNN(channels[2] + channels[1], channels[1], 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = ResBlockBNN(channels[1] + channels[0], channels[0], 32)
        self.output = nn.Conv2d(channels[0], CHANNELS, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x0 = self.conv0(x)
        x1 = self.pool(x0)
        x1 = self.down1(x1, t_emb)
        x2 = self.pool(x1)
        x2 = self.down2(x2, t_emb)
        
        x_up1 = self.up1(x2)
        x_up1 = torch.cat([x_up1, x1], dim=1)
        x_up1 = self.up_conv1(x_up1, t_emb)
        
        x_up2 = self.up2(x_up1)
        x_up2 = torch.cat([x_up2, x0], dim=1)
        x_up2 = self.up_conv2(x_up2, t_emb)
        return self.output(x_up2)
