# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from .embedders import PatchEmbed
from .layers import Transformer, get_2d_sincos_pos_embed, resample_abs_pos_embed


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        qk_norm: bool = False,
        pos_embed_requires_grad: bool = True,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            output_fmt="NHWC",
        )
        self.num_patches = self.patch_embed.num_patches
        self.img_size = self.patch_embed.img_size

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02,
            requires_grad=pos_embed_requires_grad,
        )
        self.pos_embed_requires_grad = pos_embed_requires_grad
        self.transformer = Transformer(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
            grad_checkpointing=grad_checkpointing,
        )
        self.norm = norm_layer(embed_dim)
        self.init_weights()

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)
        if not self.pos_embed_requires_grad:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], self.patch_embed.grid_size
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compared to timm's implementation, this handles non-square images.
        """
        B, H, W, C = x.shape
        pos_embed = resample_abs_pos_embed(
            posemb=self.pos_embed,
            new_size=(H, W),
            old_size=self.patch_embed.grid_size,
            n_prefix_tokens=0,
        )
        x = x.view(B, -1, C) + pos_embed
        return x

    def unpatchify(self, x, hw=None, channel_first=True, patch_size=None) -> torch.Tensor:
        hw = hw or self.img_size
        imgs = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.patch_size if patch_size is None else patch_size,
            p2=self.patch_size if patch_size is None else patch_size,
            h=hw[0] // (self.patch_size if patch_size is None else patch_size),
            w=hw[1] // (self.patch_size if patch_size is None else patch_size),
        )
        if not channel_first:
            imgs = rearrange(imgs, "b c h w -> b h w c")
        return imgs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x


def ViT_S_16(**kwargs):
    return VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)


def ViT_B_16(**kwargs):
    return VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)


def ViT_L_16(**kwargs):
    return VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)


def ViT_H_14(**kwargs):
    return VisionTransformer(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)


ViT_models = {
    "ViT-B/16": ViT_B_16,
    "ViT-L/16": ViT_L_16,
    "ViT-H/14": ViT_H_14,
}
