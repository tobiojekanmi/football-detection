import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FootballNetConfig:
    """
    Configuration for the FootballNet CNN model.
    """

    # Input image shape as (Channels, Height, Width)
    input_shape: Tuple[int, int, int] = (3, 1080, 1920)

    # Number of channels produced by the first convolution layer
    out_channels: int = 32

    # Number of intermediate residual blocks
    num_res_blocks: int = 3

    # Dimension of the latent representation (MLP hidden size)
    latent_dim: int = 64

    # Model output dimension: predicted bounding box (xmin, ymin, xmax, ymax)
    output_dim: int = 4


class ResidualBlock(nn.Module):
    """
    Simple Residual Block:
    Conv -> ReLU -> LayerNorm -> +skip -> Pool
    """

    def __init__(self, channels, kernel_size=3, pool_kernel=2):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_kernel, stride=pool_kernel)

    def forward(self, x):
        skip = x

        out = self.conv(x)
        out = self.relu(out)
        out = F.layer_norm(out, out.shape[1:])

        out = out + skip
        out = self.pool(out)

        return out


class FootballNet(nn.Module):
    """
    Simple CNN model:
    Base -> Residual blocks -> GlobalAvgPool -> MLP head
    """

    def __init__(self, config: FootballNetConfig):
        super().__init__()
        self.config = config

        in_channels = config.input_shape[0]
        out_channels = config.out_channels

        # -------- Base (no residuals) --------
        self.base = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )  # total downsample ×4 (2 for stride, 2 for pooling)

        # -------- Middle: residual blocks --------
        blocks = []
        for _ in range(config.num_res_blocks):
            blocks.append(ResidualBlock(out_channels))

        self.res_blocks = nn.Sequential(*blocks)

        # -------- Global pooling --------
        self.global_pool = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # -------- MLP Head --------
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, config.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.latent_dim, config.output_dim),
        )

    def forward(self, x):
        x = self.base(x)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        return self.head(x)
