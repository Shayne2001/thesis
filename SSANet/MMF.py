import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SPE(nn.Module):
    def __init__(self, gate_channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(1, gate_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channels, hidden, kernel_size=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(hidden, gate_channels, kernel_size=1, padding=0, bias=True),
        )
        self.conv2x1 = nn.Conv2d(gate_channels, gate_channels, kernel_size=(2, 1), groups=gate_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_pool = self.mlp(avg_pool)
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = self.mlp(max_pool)
        cat = torch.cat([max_pool, avg_pool], dim=2)
        spectral_vector = self.conv2x1(cat)
        return torch.sigmoid(spectral_vector)


class SPA(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError('kernel size must be 3 or 7')
        padding = 3 if kernel_size == 7 else 1
        self.convlayer = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        spatial_vector = self.convlayer(cat)
        return torch.sigmoid(spatial_vector)


class SSCA(nn.Module):
    def __init__(self, gate_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.spe = SPE(gate_channels, reduction_ratio)
        self.spa = SPA(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_weight = self.spa(x)
        spectral_weight = self.spe(x)
        return torch.sigmoid(spatial_weight + spectral_weight)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x)


class S_MSA(nn.Module):
    """Spectral-wise MSA from CTCSN.

    输入输出均为 [b, h, w, c]（channel last），内部对每个 head 做 L2 normalize，计算 K^T Q。
    """

    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (head d) -> b head n d', head=self.num_heads),
            (q_inp, k_inp, v_inp),
        )

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return out_c + out_p


class MS_FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()

        self.conv1x1_1 = nn.Conv2d(dim, dim * mult, 1, 1, bias=False)
        self.conv1x1_2 = nn.Conv2d(dim * mult, dim, 1, 1, bias=False)

        split_dim = dim * mult // 4
        self.dw_1 = nn.Conv2d(split_dim, split_dim, kernel_size=1, bias=False, groups=split_dim)
        self.dw_3 = nn.Conv2d(split_dim, split_dim, kernel_size=3, padding=1, bias=False, groups=split_dim)
        self.dw_5 = nn.Conv2d(split_dim, split_dim, kernel_size=5, padding=2, bias=False, groups=split_dim)
        self.dw_7 = nn.Conv2d(split_dim, split_dim, kernel_size=7, padding=3, bias=False, groups=split_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1_1(x.permute(0, 3, 1, 2))
        split_size = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, split_size, dim=1)

        out1 = self.dw_1(x1)
        out2 = self.dw_3(x2)
        out3 = self.dw_5(x3)
        out4 = self.dw_7(x4)

        out = torch.cat([out1, out2, out3, out4], dim=1) + x
        out = self.conv1x1_2(out)
        return out.permute(0, 2, 3, 1)


class SpectralT_Block(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        S_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, MS_FFN(dim=dim)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class SimpleMAFE(nn.Module):
    """更简洁的 CNN 分支：两层残差卷积 + SSCA 注意力。"""

    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True),
        )
        self.ssca = SSCA(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x) + x
        att = self.ssca(out)
        return out * att


def _pick_heads_and_dim_head(dim: int) -> tuple[int, int]:
    # 让 dim_head * heads = dim（便于重排），同时尽量保持 head_dim 不要过大。
    if dim >= 32:
        head_dim = 16
        heads = max(1, dim // head_dim)
        head_dim = dim // heads
        return heads, head_dim

    # 小通道数时，固定 2 heads
    heads = 2
    head_dim = max(1, dim // heads)
    return heads, head_dim


class CTMBlock(nn.Module):
    """CTM Block（CNN-Transformer Mixture）

    复用 CTCSN 的“split -> CNN branch + Spectral Transformer branch -> concat -> 1x1 -> residual”核心范式，
    并用更轻量的 SimpleMAFE 作为 CNN 分支。

    输入/输出: [B, 2*dim, H, W]
    """

    def __init__(self, dim: int, num_trans_blocks: int = 1):
        super().__init__()
        heads, head_dim = _pick_heads_and_dim_head(dim)

        self.conv_dim = dim
        self.trans_dim = dim

        self.trans_block = SpectralT_Block(self.trans_dim, head_dim, heads, num_blocks=num_trans_blocks)
        self.in_proj = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.out_proj = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv_block = SimpleMAFE(self.conv_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_x, trans_x = torch.split(self.in_proj(x), [self.conv_dim, self.trans_dim], dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x)
        res = self.out_proj(torch.cat((conv_x, trans_x), dim=1))
        return x + res


class MMF(nn.Module):
    """SSANet 的 MMF 接口兼容层：内部替换为 CTM Block。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError(f'CTM/MMF requires even out_channels, got {out_channels}')

        self.in_proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.ctm = CTMBlock(dim=out_channels // 2, num_trans_blocks=1)
        self.spe = SPE(out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x4: torch.Tensor, last: bool = False):
        x = self.in_proj(x1)
        out = self.ctm(x)
        c_a = self.spe(out)
        return out, c_a
