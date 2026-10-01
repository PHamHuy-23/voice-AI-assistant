import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List


class TemporalDilatedConvBlock(nn.Module):
    """
    Residual Block with Temporal Dilated Convolutions.
    Dilation is applied along the temporal dimension (W/T) to expand the receptive field
    across phoneme sequences without exponentially increasing parameter counts.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        # Conv 1: Dilated along time axis
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, dilation),
            dilation=(1, dilation),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv 2: Standard refinement
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1),
            dilation=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class TDResNetEmbeddingNet(nn.Module):
    """
    TD-ResNet: Temporal Dilated Residual Network for Speech Feature Embedding.
    Maps an input Mel-Spectrogram x -> fixed-dimensional vector f_theta(x) in R^D.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        bb_cfg = config.get("backbone", {})
        in_channels = bb_cfg.get("in_channels", 1)
        base_channels = bb_cfg.get("base_channels", 32)
        num_blocks = bb_cfg.get("num_blocks", [2, 2, 2])
        dilation_rates = bb_cfg.get("dilation_rates", [1, 2, 4])
        dropout = bb_cfg.get("dropout", 0.1)
        self.embedding_dim = bb_cfg.get("embedding_dim", 128)

        # Stem Conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Residual Stages with Temporal Dilations
        layers: List[nn.Module] = []
        curr_channels = base_channels
        for stage_idx, blocks in enumerate(num_blocks):
            out_channels = base_channels * (2**stage_idx)
            stride = 2 if stage_idx > 0 else 1
            dilation = dilation_rates[stage_idx] if stage_idx < len(dilation_rates) else 1

            # First block of stage (downsamples if stage_idx > 0)
            layers.append(
                TemporalDilatedConvBlock(
                    curr_channels,
                    out_channels,
                    stride=stride,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            # Subsequent blocks in stage
            for _ in range(1, blocks):
                layers.append(
                    TemporalDilatedConvBlock(
                        out_channels,
                        out_channels,
                        stride=1,
                        dilation=dilation,
                        dropout=dropout,
                    )
                )
            curr_channels = out_channels

        self.stages = nn.Sequential(*layers)

        # Global Spatio-Temporal Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding Projection Head
        self.head = nn.Linear(curr_channels, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, F, T) -> Mel-spectrogram tensor
        Output: (B, embedding_dim) -> L2-normalized embedding vector f_theta(x)
        """
        feat = self.stem(x)
        feat = self.stages(feat)
        pooled = self.global_pool(feat).flatten(1)
        embedding = self.head(pooled)
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return normalized_embedding
