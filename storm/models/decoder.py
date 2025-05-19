import logging
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .layers import GroupNorm, NonLocalBlock, ResidualBlock, Swish, UpSampleBlock, modulate

logger = logging.getLogger("STORM")


def check_results(result_dict) -> bool:
    assert "rgb_key" in result_dict, "rgb_key not found in result_dict"
    assert "depth_key" in result_dict, "depth_key not found in result_dict"
    assert "alpha_key" in result_dict, "alpha_key not found in result_dict"
    assert "flow_key" in result_dict, "flow_key not found in result_dict"
    assert "decoder_depth_key" in result_dict, "decoder_depth_key not found in result_dict"
    assert "decoder_alpha_key" in result_dict, "decoder_alpha_key not found in result_dict"
    assert "decoder_flow_key" in result_dict, "decoder_flow_key not found in result_dict"
    return True


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        channels=[512, 256, 256, 128, 128],
        num_res_blocks=3,
    ):
        super(Decoder, self).__init__()

        in_channels = channels[0]
        layers = [
            nn.Conv2d(latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != 0:
                layers.append(UpSampleBlock(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        # rgb-d
        layers.append(nn.Conv2d(in_channels, 4, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, render_results):
        x = render_results["rendered_image"]
        b, t, v, h, w, c = x.shape
        x = rearrange(x, "b t v h w c -> (b t v) c h w")
        x = self.model(x)
        x = rearrange(x, "(b t v) c h w -> b t v h w c", b=b, t=t, v=v)
        decoder_dict = {}
        decoder_dict["decoded_image"] = x[..., :3]
        decoder_dict["decoded_depth"] = x[..., 3]
        decoder_dict["rgb_key"] = "decoded_image"
        decoder_dict["decoder_depth_key"] = "decoded_depth"
        render_results.update(decoder_dict)
        if not check_results(render_results):
            raise ValueError("Invalid result dict")
        return render_results


class ConvDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        out_channels=3,
        channels=[512, 256, 256, 128, 128],
        num_res_blocks=3,
        grad_checkpointing=False,
    ):
        super(ConvDecoder, self).__init__()
        in_channels = channels[0]
        self.input_projection_layer = nn.Conv2d(latent_dim, in_channels, 3, 1, 1)
        layers = [
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for i in range(len(channels)):
            out_chans = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_chans))
                in_channels = out_chans
            if i != 0:
                layers.append(UpSampleBlock(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        # rgb-d
        layers.append(nn.Conv2d(in_channels, 4, 3, 1, 1))
        self.layers = nn.ModuleList(layers)
        self.out_channels = out_channels
        self.grad_checkpointing = grad_checkpointing
        logger.info(f"ConvDecoder: grad_checkpointing: {grad_checkpointing}")
        self.mask_token = nn.Parameter(torch.randn(channels[0]) * 0.02)

    def forward(self, render_results) -> Dict[str, Tensor]:
        x, opacity = render_results["rendered_image"], render_results["rendered_alpha"]
        b, t, v, h, w, c = x.shape
        x = rearrange(x, "b t v h w c -> (b t v) c h w")
        opacity = rearrange(opacity, "b t v h w -> (b t v) h w")
        x = self.input_projection_layer(x)
        mask_token = repeat(
            self.mask_token,
            "d -> b d h w",
            b=x.shape[0],
            h=x.shape[-2],
            w=x.shape[-1],
        )
        x = x * opacity.unsqueeze(1) + mask_token * (1 - opacity.unsqueeze(1))
        # chunk to avoid OOM
        for layer in self.layers:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        x = rearrange(x, "(b t v) c h w -> b t v h w c", b=b, t=t, v=v)
        decoder_dict = {}
        if self.out_channels == 3:
            decoder_dict["decoded_image"] = x[..., :3]
            decoder_dict["rgb_key"] = "decoded_image"
        elif self.out_channels == 4:
            decoder_dict["decoded_image"] = x[..., :3]
            decoder_dict["decoded_depth"] = x[..., 3]
            decoder_dict["rgb_key"] = "decoded_image"
            decoder_dict["decoder_depth_key"] = "decoded_depth"
        else:
            # TODO
            raise ValueError("Invalid out_channels")
        render_results.update(decoder_dict)
        if not check_results(render_results):
            raise ValueError("Invalid result dict")
        return render_results


class ModulatedLinearLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, condition_channels=768, out_channels=3):
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_channels, 2 * hidden_channels, bias=True)
        )
        self.condition_mapping = nn.Linear(condition_channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, c):
        x = self.linear(x)
        c = self.condition_mapping(c.squeeze(1))
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x_shape = x.shape
        x = modulate(self.norm(x.reshape(x_shape[0], -1, x.shape[-1])), shift, scale)
        x = self.output(x)
        x = x.reshape(*x_shape[:-1], -1)
        return x


class ModulatedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        condition_channels=768,
        out_channels=3,
        num_layers=1,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_channels, 2 * hidden_channels, bias=True)
        )
        self.condition_mapping = nn.Linear(condition_channels, hidden_channels)
        output_layers = []
        for i in range(num_layers):
            if i < num_layers - 1:
                output_layers.append(nn.Linear(hidden_channels, hidden_channels))
                output_layers.append(nn.SiLU())
            else:
                output_layers.append(nn.Linear(hidden_channels, out_channels))
        self.output = nn.Sequential(*output_layers)

    def forward(self, x, c):
        x = self.linear(x)
        c = self.condition_mapping(c.squeeze(1))
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x_shape = x.shape
        x = modulate(self.norm(x.reshape(x_shape[0], -1, x.shape[-1])), shift, scale)
        x = self.output(x)
        x = x.reshape(*x_shape[:-1], -1)
        return x


class DummyDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(DummyDecoder, self).__init__()

    def forward(self, render_results):
        if not check_results(render_results):
            raise ValueError("Invalid result dict")
        return render_results
