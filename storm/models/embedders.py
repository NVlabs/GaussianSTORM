# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import math
from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from storm.utils.misc import to_2tuple


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        output_fmt: Literal["NCHW", "NHWC", "NLC", "NCL"] = "NCHW",
        bias: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.output_fmt = output_fmt
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(
                img_size[1] / self.patch_size[1]
            )
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        # defaut output format is NCHW
        x = self.proj(x)
        if self.output_fmt == "NHWC":
            x = rearrange(x, "B C H W -> B H W C")
        elif self.output_fmt == "NLC":
            x = rearrange(x, "B C H W -> B (H W) C")
        elif self.output_fmt == "NCL":
            x = rearrange(x, "B C H W -> B C (H W)")
        if self.norm:
            x = self.norm(x)
        return x


class PluckerEmbedder(nn.Module):
    """
    Convert rays to plucker embedding
    """

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 1,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])

        x, y = torch.meshgrid(
            torch.arange(self.grid_size[1]),
            torch.arange(self.grid_size[0]),
            indexing="xy",
        )
        x = x.float().reshape(1, -1) + 0.5
        y = y.float().reshape(1, -1) + 0.5
        self.register_buffer("x", x)
        self.register_buffer("y", y)

    def forward(
        self,
        intrinsics: Tensor,
        camtoworlds: Tensor,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Tensor:
        assert intrinsics.shape[-2:] == (3, 3), "intrinsics should be (B, 3, 3)"
        assert camtoworlds.shape[-2:] == (4, 4), "camtoworlds should be (B, 4, 4)"
        intrinsics_shape = intrinsics.shape
        intrinsics = intrinsics.reshape(-1, 3, 3)
        camtoworlds = camtoworlds.reshape(-1, 4, 4)
        if image_size is not None:
            image_size = to_2tuple(image_size)
        else:
            image_size = self.img_size
        if patch_size is not None:
            patch_size = to_2tuple(patch_size)
        else:
            patch_size = self.patch_size

        grid_size = tuple([s // p for s, p in zip(image_size, patch_size)])

        if grid_size != self.grid_size:
            grid_size = tuple([s // p for s, p in zip(image_size, patch_size)])
            x, y = torch.meshgrid(
                torch.arange(grid_size[1]),
                torch.arange(grid_size[0]),
                indexing="xy",
            )
            x = x.float().reshape(1, -1) + 0.5
            y = y.float().reshape(1, -1) + 0.5
            x = x.to(intrinsics.device)
            y = y.to(intrinsics.device)
            intrinsics = intrinsics.clone()
            # intrinsics should be scaled to the grid size
            intrinsics[..., 0, 0] = intrinsics[..., 0, 0] / patch_size[1]
            intrinsics[..., 0, 2] = intrinsics[..., 0, 2] / patch_size[1]
            intrinsics[..., 1, 1] = intrinsics[..., 1, 1] / patch_size[0]
            intrinsics[..., 1, 2] = intrinsics[..., 1, 2] / patch_size[0]
        else:
            x, y = self.x, self.y

        x = x.repeat(intrinsics.size(0), 1)
        y = y.repeat(intrinsics.size(0), 1)
        camera_dirs = torch.nn.functional.pad(
            torch.stack(
                [
                    (x - intrinsics[:, 0, 2][..., None] + 0.5) / intrinsics[:, 0, 0][..., None],
                    (y - intrinsics[:, 1, 2][..., None] + 0.5) / intrinsics[:, 1, 1][..., None],
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )
        directions = torch.sum(camera_dirs[:, :, None, :] * camtoworlds[:, None, :3, :3], dim=-1)
        origins = torch.broadcast_to(camtoworlds[:, :3, -1].unsqueeze(1), directions.shape)
        direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
        viewdirs = directions / (direction_norm + 1e-8)
        cross_prod = torch.cross(origins, viewdirs, dim=-1)
        plucker = torch.cat((cross_prod, viewdirs), dim=-1)
        origins = rearrange(origins, "b (h w) c -> b h w c", h=grid_size[0])
        viewdirs = rearrange(viewdirs, "b (h w) c -> b h w c", h=grid_size[0])
        directions = rearrange(directions, "b (h w) c -> b h w c", h=grid_size[0])
        plucker = rearrange(plucker, "b (h w) c -> b h w c", h=grid_size[0])
        return {
            "origins": origins.view(*intrinsics_shape[:-2], *grid_size, 3),
            "viewdirs": viewdirs.view(*intrinsics_shape[:-2], *grid_size, 3),
            "dirs": directions.view(*intrinsics_shape[:-2], *grid_size, 3),
            "plucker": plucker.view(*intrinsics_shape[:-2], *grid_size, 6),
        }


class TimestepEmbedder(nn.Module):
    """
    From DiT
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class NeRFPosEmbedder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: int = 0,
        max_freq_exp: int = 8,
        include_input: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies)
        self.register_buffer("freqs", freqs)

    def get_out_dim(self) -> int:
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(self, in_tensor: Tensor) -> Tensor:
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        scaled_inputs = scaled_in_tensor[..., None] * self.freqs
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)
        encoded_inputs = torch.sin(
            torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)
        )
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs
