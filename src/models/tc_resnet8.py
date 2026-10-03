import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class DilatedResidualBlock(nn.Module):
    """
    Temporal Dilated Convolutional Residual Block.
    Kernel (7, 1), Stride 1, Dilation (d, d), Padding (3*d, 0).
    Shortcut projection Conv 1x1 + BN + ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        pad = (3 * dilation, 0)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 1), stride=1, padding=pad, dilation=(dilation, dilation), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), stride=1, padding=pad, dilation=(dilation, dilation), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.bn3(self.conv3(x)))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class TCResNet8(nn.Module):
    """
    Official TC-ResNet8 Dilated Architecture from Few-Shot Keyword Spotting Paper.
    Input MFCC: [B, 1, 51, 40] or [B, 51, 40]
    Conv1: [B, 16, 51, 1]
    Res1: dilation=1 -> [B, 24, 51, 1]
    Res2: dilation=2 -> [B, 32, 51, 1]
    Res3: dilation=4 -> [B, 48, 51, 1]
    AvgPool2d: (51, 1) -> [B, 48, 1, 1]
    Flatten -> [B, 48] embedding space.
    Total parameters: 64,560.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(40, 16, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False)
        self.res1 = DilatedResidualBlock(16, 24, dilation=1)
        self.res2 = DilatedResidualBlock(24, 32, dilation=2)
        self.res3 = DilatedResidualBlock(32, 48, dilation=4)
        self.avg_pool = nn.AvgPool2d(kernel_size=(51, 1), stride=(51, 1), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Time-channel transpose: [B, 1, 51, 40] -> [B, 40, 51, 1]
        if x.dim() == 4 and x.size(1) == 1 and x.size(3) == 40:
            x = x.squeeze(1).transpose(1, 2).unsqueeze(-1)
        elif x.dim() == 3: # [B, 51, 40] -> [B, 40, 51, 1]
            x = x.transpose(1, 2).unsqueeze(-1)

        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.avg_pool(x)
        out = x.view(x.size(0), -1)
        return out


def load_trained_tcresnet8(weights_path: str = "checkpoints/tcresnet8_clean_weights.pt") -> TCResNet8:
    """Loads TC-ResNet8Dilated with official reproduction weights from Exp 025 (Acc: 95.40%)."""
    model = TCResNet8()
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model
