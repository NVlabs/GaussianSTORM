import torch
import torch.nn as nn
from einops import rearrange, repeat
from gsplat.rendering import rasterization
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .decoder import ConvDecoder, DummyDecoder, ModulatedLinearLayer
from .embedders import PluckerEmbedder, TimestepEmbedder
from .layers import LayerNorm2d, Mlp
from .vit import VisionTransformer as ViT


class STORM(ViT):
    def __init__(
        self,
        img_size=224,
        in_chans=9,
        gs_dim=3,
        decoder_type="dummy",
        near=0.2,
        far=400,
        scale_offset=-2.3,
        opacity_offset=-2.0,
        num_cams=3,  # to ablate
        max_scale=0.5,
        disable_pos_embed=False,
        use_sky_token=True,
        use_affine_token=True,
        num_motion_tokens=32,
        tau=0.5,
        projected_motion_dim=32,
        # ViT parameters
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        grad_checkpointing=True,
        use_latest_gsplat=False,
        sigmoid_rgb=True,
        **kwargs,
    ):
        super(STORM, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            grad_checkpointing=grad_checkpointing,
        )
        # basic attributes
        self.disable_pos_embed = disable_pos_embed
        self.gs_dim = gs_dim
        self.out_channels = gs_dim + 9
        self.num_cams = num_cams
        self.grad_checkpointing = grad_checkpointing
        self.use_latest_gsplat = use_latest_gsplat

        # ------- STORM v.s. Latent-STORM -------
        self.decoder_type = decoder_type
        self.decoder_upsample_ratio = decoder_upsample_ratio = self.patch_size

        # ------- motion predictor -------
        self.num_motion_tokens = num_motion_tokens
        self.tau = tau
        num_velocity_channels = 3

        # ------- embedders -------
        self.plucker_embedder = PluckerEmbedder(img_size=img_size)
        self.time_embedder = TimestepEmbedder(embed_dim)

        # ------- auxiliary tokens -------
        self.use_sky_token = use_sky_token
        self.use_affine_token = use_affine_token

        if self.use_sky_token:
            self.sky_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.sky_head = ModulatedLinearLayer(
                3,
                hidden_channels=512,
                condition_channels=embed_dim,
                out_channels=self.gs_dim,
            )

        if self.use_affine_token:
            self.affine_token = nn.Parameter(torch.randn(1, self.num_cams, embed_dim) * 0.02)
            self.affine_linear = nn.Linear(embed_dim, self.gs_dim * (self.gs_dim + 1))

        # ------- gs predictor and mask decoder -------
        if decoder_type == "dummy":
            self.gs_pred = nn.Linear(embed_dim, decoder_upsample_ratio**2 * self.out_channels)
            self.decoder = DummyDecoder()
            self.unpatch_size = decoder_upsample_ratio

            if self.decoder_upsample_ratio == 8:
                # used for upscaling the low-resolution image features to the pixel-resolution
                # very handcrafted and never tuned
                self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
                    LayerNorm2d(512),
                    nn.GELU(),
                    nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                    LayerNorm2d(256),
                    nn.GELU(),
                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    nn.GELU(),
                )
            elif self.decoder_upsample_ratio == 16:
                # used for upscaling the low-resolution image features to the pixel-resolution
                # very handcrafted and never tuned
                self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
                    LayerNorm2d(512),
                    nn.GELU(),
                    nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                    LayerNorm2d(256),
                    nn.GELU(),
                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    nn.GELU(),
                    nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                    LayerNorm2d(128),
                    nn.GELU(),
                )

        elif decoder_type == "conv":
            self.gs_pred = nn.Linear(embed_dim, self.out_channels)
            # latent-STORM decoder
            self.decoder = ConvDecoder(
                latent_dim=self.gs_dim,
                out_channels=4,  # 3 for RGB, 1 for depth
                num_res_blocks=3,
                channels=[512, 256, 256, 128],  # 8 times upsample
                grad_checkpointing=grad_checkpointing,
            )
            self.unpatch_size = 1
            # upscaling the low-resolution image features to the pixel-resolution
            # the "pixel" resolution here is essentially the feature map resolution
            # which is 1/patch_size of the image resolution
            self.output_upscaling = nn.Sequential(
                nn.Conv2d(embed_dim, 512, kernel_size=1),
                LayerNorm2d(512),
                nn.GELU(),
                nn.Conv2d(512, 256, kernel_size=1),
                LayerNorm2d(256),
                nn.GELU(),
                nn.Conv2d(256, 128, kernel_size=1),
                LayerNorm2d(128),
                nn.GELU(),
            )
        # ------- activation functions for gs parameters -------
        self.max_scale = nn.Parameter(torch.tensor([float(max_scale)]), requires_grad=False)
        self.scale_act_fn = lambda x: torch.minimum(torch.exp(x + scale_offset), self.max_scale)
        self.opacity_act_fn = lambda x: torch.sigmoid(x + opacity_offset)
        self.depth_act_fn = lambda x: near + torch.sigmoid(x) * (far - near)
        self.rgb_act_fn = lambda x: torch.sigmoid(x) * 2 - 1 if sigmoid_rgb else x
        self.near, self.far = near, far

        # ------- motion predictor -------
        self.motion_key_head = Mlp(128, 256, projected_motion_dim)
        if self.num_motion_tokens > 0:
            self.motion_tokens = nn.Parameter(torch.randn(1, num_motion_tokens, embed_dim) * 0.02)
            self.motion_query_heads = nn.ModuleList(
                [
                    Mlp(embed_dim, embed_dim, projected_motion_dim)
                    for _ in range(self.num_motion_tokens)
                ]
            )
            self.motion_basis_decoder = Mlp(embed_dim, 256, num_velocity_channels)
        else:
            self.motion_tokens = None
            self.motion_basis_decoder = Mlp(projected_motion_dim, 256, num_velocity_channels)

        self.init_weights()
        if disable_pos_embed:  # remove the default pos_embed in vit
            del self.pos_embed
            self.pos_embed = None

    def _pos_embed(self, x: Tensor) -> Tensor:
        if not self.disable_pos_embed:
            return super()._pos_embed(x)
        return rearrange(x, "b h w c -> b (h w) c")

    def _time_embed(self, x: Tensor, time: Tensor, num_views=1) -> Tensor:
        if time.ndim == 3:
            b, t, v = time.shape
            time_embedding = (
                self.time_embedder(time.flatten())  # (bt, c)
                .view(b, t, v, -1)  # (b, t, v, c)
                .view(-1, 1, self.embed_dim)  # (btv, 1, c)
                .repeat(1, x.shape[1], 1)  # (btv, n, c)
            )
        else:
            time_embedding = (
                self.time_embedder(time.flatten())  # (bt, c)
                .view(time.shape[0], time.shape[1], 1, -1)  # (b, t, 1, c)
                .repeat(1, 1, num_views, 1)  # (b, t, v, c)
                .view(-1, 1, self.embed_dim)  # (btv, 1, c)
                .repeat(1, x.shape[1], 1)  # (btv, n, c)
            )
        return x + time_embedding

    def forward_decoder(self, render_results):
        render_results["rgb_key"] = "rendered_image"
        render_results["depth_key"] = "rendered_depth"
        render_results["alpha_key"] = "rendered_alpha"
        render_results["flow_key"] = "rendered_flow"
        render_results["decoder_depth_key"] = None
        render_results["decoder_alpha_key"] = None
        render_results["decoder_flow_key"] = None
        render_results = self.decoder(render_results)
        decoded_depth_key = render_results["decoder_depth_key"]
        if decoded_depth_key is not None:
            decoded_depth = self.depth_act_fn(render_results[decoded_depth_key])
            render_results[decoded_depth_key] = decoded_depth
        return render_results

    def forward_features(self, x, plucker_embeds, time):
        b, t, v, c, h, w = x.size()
        x = rearrange(x, "b t v c h w -> (b t v) c h w")
        plucker_embeds = rearrange(plucker_embeds, "b t v h w c-> (b t v) c h w")
        x = torch.cat([x, plucker_embeds], dim=1)
        x = self.patch_embed(x)  # (b t v) h w c2
        x = self._pos_embed(x)  # (b t v) (h w) c2
        x = self._time_embed(x, time, num_views=v)
        x = rearrange(x, "(b t v) hw c -> b (t v hw) c", t=t, v=v)
        if self.num_motion_tokens > 0:
            motion_tokens = repeat(self.motion_tokens, "1 k d -> b k d", b=x.shape[0])
            x = torch.cat([motion_tokens, x], dim=-2)
        if self.use_affine_token:
            affine_token = repeat(self.affine_token, "1 k d -> b k d", b=b)
            x = torch.cat([affine_token, x], dim=-2)
        if self.use_sky_token:
            sky_token = repeat(self.sky_token, "1 1 d -> b 1 d", b=x.shape[0])
            x = torch.cat([sky_token, x], dim=-2)
        x = self.transformer(x)
        x = self.norm(x)
        return x

    def forward_motion_predictor(self, x, motion_tokens=None, gs_params=None):
        b, t, v, h, w, _ = gs_params["means"].shape
        img_embeds = self.unpatchify(
            rearrange(x, "b (t v hw) c -> (b t v) hw c", t=t, v=v),
            hw=(h // self.unpatch_size, w // self.unpatch_size),
            patch_size=1,
        )
        if self.grad_checkpointing:
            img_embeds = checkpoint(self.output_upscaling, img_embeds)
        else:
            img_embeds = self.output_upscaling(img_embeds)
        img_embeds = rearrange(img_embeds, "(b t v) c h w -> b t v h w c", t=t, v=v)
        img_keys = self.motion_key_head(img_embeds)

        if self.num_motion_tokens > 0:
            hyper_in_list = []
            for i in range(self.num_motion_tokens):
                hyper_in = self.motion_query_heads[i](motion_tokens[:, i])
                hyper_in_list.append(hyper_in)
            motion_token_queries = torch.stack(hyper_in_list, dim=1)
            motion_bases = self.motion_basis_decoder(motion_tokens)
            dot_product_similarity = torch.einsum(
                "b k c, b t v h w c -> b t v h w k",
                motion_token_queries,
                img_keys,
            )
            motion_weights = torch.softmax(dot_product_similarity / self.tau, dim=-1)
            forward_flow = torch.einsum(
                "b t v h w k, b k c -> b t v h w c", motion_weights, motion_bases
            )
            gs_params["motion_weights"] = motion_weights
            gs_params["motion_bases"] = motion_bases
        else:
            # if there's no motion token, directly predict the velocity from the upsampled image features
            forward_flow = self.motion_basis_decoder(img_keys)

        gs_params["forward_flow"] = forward_flow
        return {k: v for k, v in gs_params.items() if v is not None}

    def forward_gs_predictor(self, x, origins, directions):
        b, t, v, h, w, _ = origins.shape
        x = rearrange(x, "b (t v hw) c -> (b t v) hw c", t=t, v=v)
        gs_params = self.gs_pred(x)
        gs_params = self.unpatchify(gs_params, hw=(h, w), patch_size=self.unpatch_size)
        gs_params = rearrange(gs_params, "(b t v) c h w -> b t v h w c", t=t, v=v)
        depth, scales, quats, opacitys, colors = gs_params.split([1, 3, 4, 1, self.gs_dim], dim=-1)
        scales = self.scale_act_fn(scales)
        opacitys = self.opacity_act_fn(opacitys)
        depths = self.depth_act_fn(depth)
        colors = self.rgb_act_fn(colors)
        means = origins + directions * depths
        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacitys.squeeze(-1),
            "colors": colors,
            "depths": depths.squeeze(-1),
        }

    def forward_renderer(self, gs_params, data_dict, render_motion_seg=True, radius_clip=0.0):
        b, t, v, h, w, _ = gs_params["means"].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["target_camtoworlds"].shape[1:3]
        means = rearrange(gs_params["means"], "b t v h w c -> b (t v h w) c")
        scales = rearrange(gs_params["scales"], "b t v h w c -> b (t v h w) c")
        quats = rearrange(gs_params["quats"], "b t v h w c -> b (t v h w) c")
        opacities = rearrange(gs_params["opacities"], "b t v h w -> b (t v h w)")
        colors = rearrange(gs_params["colors"], "b t v h w c -> b (t v h w) c")
        forward_v = rearrange(gs_params["forward_flow"], "b t v h w c -> b (t v h w) c")

        means_batched = means.repeat_interleave(tgt_t, dim=0)
        scales_batched = scales.repeat_interleave(tgt_t, dim=0)
        quats_batched = quats.repeat_interleave(tgt_t, dim=0)
        opacities_batched = opacities.repeat_interleave(tgt_t, dim=0)
        color_batched = colors.repeat_interleave(tgt_t, dim=0)
        forward_v_batched = forward_v.repeat_interleave(tgt_t, dim=0)

        ctx_time = data_dict["context_time"] * data_dict["timespan"]
        tgt_time = data_dict["target_time"] * data_dict["timespan"]
        if tgt_time.ndim == 3:
            tdiff_forward = tgt_time.unsqueeze(2) - ctx_time.unsqueeze(1)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t * v, 1)
            tdiff_forward_batched = tdiff_forward.repeat_interleave(h * w, dim=1)
        else:
            tdiff_forward = tgt_time.unsqueeze(-1) - ctx_time.unsqueeze(-2)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t, 1)
            tdiff_forward_batched = tdiff_forward.repeat_interleave(v * h * w, dim=1)
        forward_translation = forward_v_batched * tdiff_forward_batched
        means_batched = means_batched + forward_translation

        if not self.training:  # mask out some noisy flow
            forward_v[forward_v.norm(dim=-1) < 1.0] = 0.0
            forward_v_batched = forward_v.repeat_interleave(tgt_t, dim=0)

        if not self.training and self.num_motion_tokens > 0 and render_motion_seg:
            # render the motion segmentation map
            motion_weights = rearrange(gs_params["motion_weights"], "b t v h w k -> b (t v h w) k")
            weights_batched = motion_weights.repeat_interleave(tgt_t, dim=0)
            colors_batched = torch.cat([color_batched, forward_v_batched, weights_batched], dim=-1)
        else:
            colors_batched = torch.cat([color_batched, forward_v_batched], dim=-1)

        camtoworlds_batched = data_dict["target_camtoworlds"].view(b * tgt_t, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        Ks_batched = data_dict["target_intrinsics"].view(b * tgt_t, -1, 3, 3)

        motion_seg = None
        if self.use_latest_gsplat:
            means_batched = means_batched.float()
            quats_batched = quats_batched.float()
            scales_batched = scales_batched.float()
            opacities_batched = opacities_batched.float()
            colors_batched = colors_batched.float()
            viewmats_batched = viewmats_batched.float()
            Ks_batched = Ks_batched.float()

            if not self.training:
                rendered_colors, rendered_alphas, rendered_flow, motion_seg = [], [], [], []
                rendered_depths = []
                with torch.autocast("cuda", enabled=False):
                    for bid in range(means_batched.size(0)):
                        renderings, alpha, _ = rasterization(
                            means=means_batched[bid],
                            quats=quats_batched[bid],
                            scales=scales_batched[bid],
                            opacities=opacities_batched[bid],
                            colors=colors_batched[bid],
                            viewmats=viewmats_batched[bid],
                            Ks=Ks_batched[bid],
                            width=data_dict["width"],
                            height=data_dict["height"],
                            render_mode="RGB+ED",
                            near_plane=self.near,
                            far_plane=self.far,
                            packed=False,
                            radius_clip=radius_clip,
                        )
                        color, forward_flow, weights, depth = renderings.split(
                            [self.gs_dim, 3, self.num_motion_tokens, 1], dim=-1
                        )
                        rendered_colors.append(color)
                        rendered_alphas.append(alpha)
                        rendered_flow.append(forward_flow)
                        motion_seg.append(weights)
                        rendered_depths.append(depth)
                color = torch.stack(rendered_colors, dim=0)
                rendered_alpha = torch.stack(rendered_alphas, dim=0)
                forward_flow = torch.stack(rendered_flow, dim=0)
                depth = torch.stack(rendered_depths, dim=0)
                motion_seg = torch.stack(motion_seg, dim=0)
                if motion_seg.numel() > 0:
                    motion_seg = motion_seg.reshape(b, tgt_t, v, h, w, -1).argmax(dim=-1)
                else:
                    motion_seg = None
            else:
                rendered_colors, rendered_alphas, rendered_flow, rendered_depths = [], [], [], []
                with torch.autocast("cuda", enabled=False):
                    for bid in range(means_batched.size(0)):
                        renderings, alpha, _ = rasterization(
                            means=means_batched[bid],
                            quats=quats_batched[bid],
                            scales=scales_batched[bid],
                            opacities=opacities_batched[bid],
                            colors=colors_batched[bid],
                            viewmats=viewmats_batched[bid],
                            Ks=Ks_batched[bid],
                            width=data_dict["width"],
                            height=data_dict["height"],
                            render_mode="RGB+ED",
                            near_plane=self.near,
                            far_plane=self.far,
                            packed=False,
                            radius_clip=radius_clip,
                        )
                        color, forward_flow, depth = renderings.split([self.gs_dim, 3, 1], dim=-1)
                        rendered_colors.append(color)
                        rendered_alphas.append(alpha)
                        rendered_flow.append(forward_flow)
                        rendered_depths.append(depth)
                color = torch.stack(rendered_colors, dim=0)
                rendered_alpha = torch.stack(rendered_alphas, dim=0)
                forward_flow = torch.stack(rendered_flow, dim=0)
                depth = torch.stack(rendered_depths, dim=0)

        else:
            if not self.training:
                with torch.autocast("cuda", enabled=False):
                    rendered_color, rendered_alpha, _ = rasterization(
                        means=means_batched.float(),
                        quats=quats_batched.float(),
                        scales=scales_batched.float(),
                        opacities=opacities_batched.float(),
                        colors=(
                            colors_batched[..., : -self.num_motion_tokens].float()
                            if self.num_motion_tokens > 0 and render_motion_seg
                            else colors_batched.float()
                        ),
                        viewmats=viewmats_batched,
                        Ks=Ks_batched,
                        width=tgt_w,
                        height=tgt_h,
                        render_mode="RGB+ED",
                        near_plane=self.near,
                        far_plane=self.far,
                        packed=False,
                        radius_clip=radius_clip,
                    )
                    color, forward_flow, depth = rendered_color.split([self.gs_dim, 3, 1], dim=-1)
                    if self.num_motion_tokens > 0 and render_motion_seg:
                        chunksize = 32
                        assignment_map = []
                        rendered_colors = colors_batched[..., -self.num_motion_tokens :]
                        for i in range(0, self.num_motion_tokens, chunksize):
                            weights, _, _ = rasterization(
                                means=means_batched.float(),
                                quats=quats_batched.float(),
                                scales=scales_batched.float(),
                                opacities=opacities_batched.float(),
                                colors=rendered_colors[..., i : i + chunksize],
                                viewmats=viewmats_batched,
                                Ks=Ks_batched,
                                width=tgt_w,
                                height=tgt_h,
                                render_mode="RGB+ED",
                                near_plane=self.near,
                                far_plane=self.far,
                                packed=False,
                                radius_clip=radius_clip,
                            )
                            weights = weights.split([weights.size(-1) - 1, 1], dim=-1)[0]
                            assignment_map.append(weights)
                        motion_seg = torch.cat(assignment_map, dim=-1)
                        motion_seg = motion_seg.reshape(b, tgt_t, tgt_v, tgt_h, tgt_w, -1).argmax(
                            dim=-1
                        )
            else:
                with torch.autocast("cuda", enabled=False):
                    rendered_color, rendered_alpha, _ = rasterization(
                        means=means_batched.float(),
                        quats=quats_batched.float(),
                        scales=scales_batched.float(),
                        opacities=opacities_batched.float(),
                        colors=colors_batched.float(),
                        viewmats=viewmats_batched,
                        Ks=Ks_batched,
                        width=tgt_w,
                        height=tgt_h,
                        render_mode="RGB+ED",
                        near_plane=self.near,
                        far_plane=self.far,
                        packed=False,
                        radius_clip=radius_clip,
                    )
                color, forward_flow, depth = rendered_color.split([self.gs_dim, 3, 1], dim=-1)
        output_dict = {
            "rendered_image": color.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "rendered_depth": depth.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_alpha": rendered_alpha.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_flow": forward_flow.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "means_batched": means_batched,
        }
        if motion_seg is not None:
            output_dict["rendered_motion_seg"] = motion_seg.squeeze(-1)
        return output_dict

    def get_ray_dict(self, data_dict):
        ray_dict = self.plucker_embedder(
            data_dict["context_intrinsics"],
            data_dict["context_camtoworlds"],
            image_size=data_dict["context_image"].shape[-2:],
        )
        if self.decoder_type != "dummy":
            feat_ray_dict = self.plucker_embedder(
                data_dict["context_intrinsics"],
                data_dict["context_camtoworlds"],
                image_size=data_dict["context_image"].shape[-2:],
                patch_size=self.patch_size,
            )
            ray_dict["origins"] = feat_ray_dict["origins"]
            ray_dict["dirs"] = feat_ray_dict["dirs"]

            tgt_intrinsics = data_dict["target_intrinsics"]
            tgt_intrinsics[..., 0, 0] = tgt_intrinsics[..., 0, 0] / self.patch_size
            tgt_intrinsics[..., 1, 1] = tgt_intrinsics[..., 1, 1] / self.patch_size
            tgt_intrinsics[..., 0, 2] = tgt_intrinsics[..., 0, 2] / self.patch_size
            tgt_intrinsics[..., 1, 2] = tgt_intrinsics[..., 1, 2] / self.patch_size
            data_dict["target_intrinsics"] = tgt_intrinsics
            data_dict["width"] //= self.patch_size
            data_dict["height"] //= self.patch_size
        return data_dict, ray_dict

    def forward(self, data_dict):
        x = data_dict["context_image"]
        b, t, v, c, h, w = x.size()
        data_dict, ray_dict = self.get_ray_dict(data_dict)
        x = self.forward_features(x, ray_dict["plucker"], data_dict["context_time"])

        sky_token, affine_tokens, motion_tokens = None, None, None
        if self.use_sky_token:
            sky_token = x[:, :1]
            x = x[:, 1:]

        if self.use_affine_token:
            affine_tokens = x[:, : self.num_cams]
            x = x[:, self.num_cams :]

        if self.num_motion_tokens > 0:
            motion_tokens = x[:, : self.num_motion_tokens]
            x = x[:, self.num_motion_tokens :]

        gs_params = self.forward_gs_predictor(x, ray_dict["origins"], ray_dict["dirs"])
        gs_params = self.forward_motion_predictor(x, motion_tokens, gs_params)
        # sometimes the number of views is too large, so we split the rendering into chunks
        step = 20
        if data_dict["target_camtoworlds"].shape[1] <= step:
            render_results = self.forward_renderer(gs_params, data_dict)
        else:
            chunk_data_dict = data_dict.copy()
            for chunk_start in range(0, data_dict["target_camtoworlds"].shape[1], step):
                chunk_end = min(chunk_start + step, data_dict["target_camtoworlds"].shape[1])
                chunk_data_dict["target_camtoworlds"] = data_dict["target_camtoworlds"][
                    :, chunk_start:chunk_end
                ]
                chunk_data_dict["target_intrinsics"] = data_dict["target_intrinsics"][
                    :, chunk_start:chunk_end
                ]
                chunk_data_dict["target_time"] = data_dict["target_time"][:, chunk_start:chunk_end]
                chunk_render_results = self.forward_renderer(gs_params, chunk_data_dict)
                if chunk_start == 0:
                    render_results = chunk_render_results
                else:
                    for k, v in chunk_render_results.items():
                        render_results[k] = torch.cat([render_results[k], v], dim=1)
        images, opacities = render_results["rendered_image"], render_results["rendered_alpha"]
        if self.use_sky_token:
            target_ray_dict = self.plucker_embedder(
                data_dict["target_intrinsics"],
                data_dict["target_camtoworlds"],
                image_size=(data_dict["height"], data_dict["width"]),
            )
            if data_dict["target_camtoworlds"].shape[1] <= step:
                sky = self.sky_head(target_ray_dict["dirs"], sky_token)
                images = images + (1 - opacities[..., None]) * sky
            else:
                for chunk_start in range(0, data_dict["target_camtoworlds"].shape[1], step):
                    dirs = target_ray_dict["dirs"][:, chunk_start : chunk_start + step]
                    chunk_sky = self.sky_head(dirs, sky_token)
                    images[:, chunk_start : chunk_start + step] += (
                        1 - opacities[:, chunk_start : chunk_start + step][..., None]
                    ) * chunk_sky
            gs_params["sky_token"] = sky_token

        if self.use_affine_token:
            affine = self.affine_linear(affine_tokens)  # b v (gs_dim * (gs_dim + 1))
            affine = rearrange(affine, "b v (p q) -> b v p q", p=self.gs_dim)
            images = torch.einsum("b t v h w p, b v p q -> b t v h w p", images, affine)
            gs_params["affine"] = affine
        render_results["rendered_image"] = images
        render_results = self.forward_decoder(render_results)
        return {
            "ray_dict": ray_dict,
            "gs_params": gs_params,
            "render_results": render_results,
        }

    def from_gs_params_to_output(self, gs_params, target_dict, num_cams=1):
        render_results = self.forward_renderer(
            gs_params, target_dict, render_motion_seg=False, radius_clip=4.0
        )
        rendered_images = render_results["rendered_image"]
        if self.use_sky_token:
            sky_token = gs_params["sky_token"]
            target_ray_dict = self.plucker_embedder(
                target_dict["target_intrinsics"],
                target_dict["target_camtoworlds"],
                image_size=(target_dict["height"], target_dict["width"]),
            )
            sky = self.sky_head(target_ray_dict["dirs"], sky_token)
            rendered_opacities = render_results["rendered_alpha"]
            rendered_images = rendered_images + (1 - rendered_opacities[..., None]) * sky

        if self.use_affine_token:
            if num_cams == 1:
                affine = gs_params["affine"].mean(dim=1)
                rendered_images = torch.einsum(
                    "b t v h w p, b p q -> b t v h w p", rendered_images, affine
                )
            else:
                affine = gs_params["affine"]
                rendered_images = torch.einsum(
                    "b t v h w p, b v p q -> b t v h w p", rendered_images, affine
                )
        render_results["rendered_image"] = rendered_images
        render_results = self.forward_decoder(render_results)
        return {"render_results": render_results}

    def get_gs_params(self, data_dict):
        x = data_dict["context_image"]
        data_dict, ray_dict = self.get_ray_dict(data_dict)
        x = self.forward_features(x, ray_dict["plucker"], data_dict["context_time"])

        sky_token, affine_tokens, motion_tokens = None, None, None
        if self.use_sky_token:
            sky_token = x[:, :1]
            x = x[:, 1:]

        if self.use_affine_token:
            affine_tokens = x[:, : self.num_cams]
            x = x[:, self.num_cams :]

        if self.num_motion_tokens > 0:
            motion_tokens = x[:, : self.num_motion_tokens]
            x = x[:, self.num_motion_tokens :]

        gs_params = self.forward_gs_predictor(x, ray_dict["origins"], ray_dict["dirs"])
        gs_params = self.forward_motion_predictor(x, motion_tokens, gs_params)
        if self.use_sky_token:
            gs_params["sky_token"] = sky_token

        if self.use_affine_token:
            affine = self.affine_linear(affine_tokens)  # b v (gs_dim * (gs_dim + 1))
            affine = rearrange(affine, "b v (p q) -> b v p q", p=self.gs_dim)
            gs_params["affine"] = affine
        return gs_params


def STORM_B_8(**kwargs):
    return STORM(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)


def STORM_L_8(**kwargs):
    return STORM(patch_size=8, embed_dim=1024, depth=24, num_heads=16, **kwargs)


def STORM_B_16(**kwargs):
    return STORM(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)


def STORM_L_16(**kwargs):
    return STORM(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)


def STORM_XL_8(**kwargs):
    return STORM(patch_size=8, embed_dim=1152, depth=28, num_heads=16, **kwargs)


def STORM_H_8(**kwargs):
    return STORM(patch_size=8, embed_dim=1280, depth=32, num_heads=16, **kwargs)


def STORM_H_16(**kwargs):
    return STORM(patch_size=16, embed_dim=1280, depth=32, num_heads=16, **kwargs)


STORM_models = {
    "STORM-B/8": STORM_B_8,
    "STORM-L/8": STORM_L_8,
    "STORM-XL/8": STORM_XL_8,
    "STORM-H/8": STORM_H_8,
    "STORM-B/16": STORM_B_16,
    "STORM-L/16": STORM_L_16,
    "STORM-H/16": STORM_H_16,
}
