import logging
import math
from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger("STORM")


def modulate(x, shift=None, scale=None):
    if shift is None and scale is None:
        return x
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim=None, out_dim=None, act_layer=nn.GELU):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    _logged = False

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_norm: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
        is_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads !=0, got {dim} and {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.is_cross_attn = is_cross_attn

        self.fused_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if self.fused_attn and not Attention._logged:
            Attention._logged = True
            logger.info(f"[Attention]: Using {torch.__version__} Fused Attention")

        if is_cross_attn:
            self.c_q = nn.Linear(dim, dim)  # context to q
            self.c_kv = nn.Linear(dim, dim * 2)  # context to kv
        else:
            self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, data: Tensor = None) -> Tensor:
        bs, n_ctx, C = x.shape
        if self.is_cross_attn:
            assert data is not None, "data should not be None for cross attn"
            q = self.c_q(x)
            kv = self.c_kv(data)
            _, n_data, _ = kv.shape
            q = q.view(bs, n_ctx, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = kv.view(bs, n_data, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(dim=0)
        else:
            qkv = (
                self.qkv(x)
                .reshape(bs, n_ctx, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(dim=0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(bs, n_ctx, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    _logged = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_cross_attn = use_cross_attn
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            is_cross_attn=use_cross_attn,
        )
        self.data_norm = norm_layer(dim) if self.use_cross_attn else None
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x: Tensor, data: Tensor = None) -> Tensor:
        if self.use_cross_attn:
            x = x + self.attn(self.norm1(x), self.data_norm(data))
        else:
            x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.grad_checkpointing = grad_checkpointing
        logger.info(f"[Transformer]: grad_checkpointing={grad_checkpointing}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)
        return x


######## Conv Layers ########
class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels
        assert channels % 8 == 0, "channels must be divisible by 8"
        self.gn = GroupNorm(channels)
        self.attention = Attention(dim=channels, num_heads=8)

    def forward(self, x):
        h = self.gn(x)
        h = rearrange(h, "b c h w -> b (h w) c")
        h = self.attention(h)
        h = rearrange(h, "b (h w) c -> b c h w", h=x.shape[-2], w=x.shape[-1])
        return h + x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def pos_enc(x, min_deg=0, max_deg=10, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = 2 ** torch.arange(min_deg, max_deg).float()
    scales = scales.to(x.device)
    shape = x.shape[:-1] + (-1,)
    scaled_x = torch.reshape((x[..., None, :] * scales[:, None]), shape)
    # Note that we're not using safe_sin, unlike IPE.
    four_feat = torch.sin(torch.concat([scaled_x, scaled_x + 0.5 * np.pi], dim=-1))
    if append_identity:
        return torch.concat([x] + [four_feat], dim=-1)
    else:
        return four_feat


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, num_extra_tokens=1):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = [grid_size, grid_size]
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and num_extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([num_extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def resample_abs_pos_embed(
    posemb,
    new_size: List[int],
    old_size: Optional[List[int]] = None,
    n_prefix_tokens: int = 1,
    interpolation: str = "bicubic",  # bicubic is better.
    antialias: bool = True,  # antialias is important.
    verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + n_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - n_prefix_tokens))
        old_size = [hw, hw]

    if n_prefix_tokens:
        posemb_prefix, posemb = (
            posemb[:, :n_prefix_tokens],
            posemb[:, n_prefix_tokens:],
        )
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = torch.nn.functional.interpolate(
        posemb, size=new_size, mode=interpolation, antialias=antialias
    )
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    if not torch.jit.is_scripting() and verbose:
        logger.info(f"Resized position embedding: {old_size} to {new_size}.")

    return posemb
