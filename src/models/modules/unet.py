"""Small conditional UNet for MNIST DDPM."""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        gn_groups = min(8, in_ch)
        self.norm1 = nn.GroupNorm(gn_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        gn_groups_out = min(8, out_ch)
        self.norm2 = nn.GroupNorm(gn_groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = h + self.time_proj(nn.functional.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_classes: int = 10,
        base_channels: int = 64,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        self.label_emb = nn.Embedding(n_classes, time_emb_dim)

        ch = base_channels
        self.down1 = ResidualBlock(in_channels, ch, time_emb_dim)
        self.down2 = ResidualBlock(ch, ch * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(ch * 2, ch * 2, time_emb_dim)

        self.up1 = ResidualBlock(ch * 2 + ch * 2, ch, time_emb_dim)
        self.up2 = ResidualBlock(ch + ch, in_channels, time_emb_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t) + self.label_emb(labels)

        # Down
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(d2), t_emb)

        # Up
        u1 = nn.functional.interpolate(b, scale_factor=2, mode="nearest")
        u1 = self.up1(torch.cat((u1, d2), dim=1), t_emb)
        u2 = nn.functional.interpolate(u1, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat((u2, d1), dim=1), t_emb)

        return u2
