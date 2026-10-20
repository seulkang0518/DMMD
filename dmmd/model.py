import math

import torch
from torch import nn
from torch.nn import functional as F


def normalization(channels):
    return nn.GroupNorm(math.gcd(channels, 8), channels)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(dim // 2) / (dim // 2 - 1)
        )
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        angles = t[:, None] * 1000 * self.frequencies[None, :]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, time_dim, dropout):
        super().__init__()
        self.norm1 = normalization(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(time_dim, 2 * cout)
        self.norm2 = normalization(cout)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x, time):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(time)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None])
        h = h + shift[:, :, None, None]
        return (self.skip(x) + self.conv2(self.dropout(F.silu(h)))) / math.sqrt(2)


class FeatureHead(nn.Module):
    def __init__(self, channels, output_dim):
        super().__init__()
        self.norm = normalization(channels)
        self.projection = nn.Linear(channels, output_dim)

    def forward(self, h):
        h = F.silu(self.norm(h)).mean(dim=(2, 3))
        return self.projection(h)


class FeatureUNet(nn.Module):
    """Compact, time-conditioned U-Net, not a claim of exact paper architecture.

    Four pooled feature heads: before downsampling, after downsampling,
    before upsampling (after middle blocks), and the final full-resolution map.
    No batch normalization, self-attention, or mixed precision: the training
    objective needs reliable second derivatives through image-input gradients.
    """

    def __init__(
        self, base_channels=64, channel_mult=(1, 2, 2, 2),
        time_dim=128, head_dim=32, dropout=0.2,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_mult]
        self.output_dim = 4 * head_dim
        self.time_embedding = TimeEmbedding(time_dim)
        self.stem = nn.Conv2d(3, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, c in enumerate(channels):
            cin = channels[0] if i == 0 else channels[i - 1]
            self.encoder.append(ResidualBlock(cin, c, time_dim, dropout))
            if i < len(channels) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        self.middle = nn.ModuleList([
            ResidualBlock(channels[-1], channels[-1], time_dim, dropout),
            ResidualBlock(channels[-1], channels[-1], time_dim, dropout),
        ])
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up.append(nn.Conv2d(channels[i], channels[i - 1], 3, padding=1))
            self.decoder.append(
                ResidualBlock(2 * channels[i - 1], channels[i - 1], time_dim, dropout)
            )
        self.heads = nn.ModuleList(
            FeatureHead(c, head_dim)
            for c in (channels[0], channels[-1], channels[-1], channels[0])
        )

    def forward(self, x, t):
        if x.ndim != 4 or x.shape[1:] != (3, 32, 32):
            raise ValueError("FeatureUNet expects [batch, 3, 32, 32]")
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if t.numel() == 1:
            t = t.reshape(1).expand(x.shape[0])
        if t.shape != (x.shape[0],):
            raise ValueError("Time must be scalar or one value per image")
        time = self.time_embedding(t)
        h = self.stem(x)
        skips = []
        for i, block in enumerate(self.encoder):
            h = block(h, time)
            skips.append(h)
            if i < len(self.down):
                h = self.down[i](h)
        before_down, after_down = skips[0], h
        for block in self.middle:
            h = block(h, time)
        before_up = h
        for i, (up, block) in enumerate(zip(self.up, self.decoder)):
            skip = skips[-i - 2]
            h = up(F.interpolate(h, size=skip.shape[-2:], mode="nearest"))
            h = block(torch.cat((h, skip), dim=1), time)
        maps = (before_down, after_down, before_up, h)
        return torch.cat([head(value) for head, value in zip(self.heads, maps)], dim=1)
