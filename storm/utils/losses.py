import torch
import torch.nn.functional as F
from einops import rearrange

from storm.dataset.constants import MEAN, STD


def compute_depth_loss(pred_depth, gt_depth, max_depth=None):
    pred_depth = pred_depth.squeeze()
    gt_depth = gt_depth.squeeze()
    if pred_depth.shape != gt_depth.shape:
        # resize pred_depth to match gt depth size
        try:
            b, v, h, w = pred_depth.shape
            gt_h, gt_w = gt_depth.shape[-2:]
            pred_depth = F.interpolate(
                rearrange(pred_depth, "b v h w -> (b v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            pred_depth = rearrange(pred_depth, "(b v) 1 h w -> b v h w", b=b, v=v)
        except:
            b, t, v, h, w = pred_depth.shape
            gt_h, gt_w = gt_depth.shape[-2:]
            pred_depth = F.interpolate(
                rearrange(pred_depth, "b t v h w -> (b t v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            pred_depth = rearrange(pred_depth, "(b t v) 1 h w -> b t v h w", b=b, t=t, v=v)
    valid_mask = gt_depth > 0.01
    if max_depth is None:
        max_depth = gt_depth.max()
    pred_depth = pred_depth[valid_mask] / max_depth
    gt_depth = gt_depth[valid_mask] / max_depth
    return F.l1_loss(pred_depth, gt_depth)


def compute_sky_depth_loss(pred_depth, gt_sky_mask, sky_depth: float = 1e3, flow=None):
    pred_depth = pred_depth.squeeze()
    gt_sky_mask = gt_sky_mask.squeeze()
    gt_h, gt_w = gt_sky_mask.shape[-2:]
    if pred_depth.shape != gt_sky_mask.shape:
        # resize pred_depth to match gt depth size
        b, t, v, h, w = pred_depth.shape
        pred_depth = F.interpolate(
            rearrange(pred_depth, "b t v h w -> (b t v) 1 h w"),
            size=(gt_h, gt_w),
            mode="bilinear",
            align_corners=False,
        )
        pred_depth = rearrange(pred_depth, "(b t v) 1 h w -> b t v h w", b=b, t=t, v=v)
    if flow is not None and (flow.shape[-3] != gt_h or flow.shape[-2] != gt_w):
        flow = F.interpolate(
            rearrange(flow, "b t v h w c -> (b t v) c h w"),
            size=(gt_h, gt_w),
            mode="bilinear",
            align_corners=False,
        )
        flow = rearrange(flow, "(b t v) c h w -> b t v h w c", b=b, t=t, v=v)
        # penalize flow in sky region
        sky_flow = flow[gt_sky_mask > 0]
        sky_flow_reg_loss = F.mse_loss(sky_flow, torch.zeros_like(sky_flow))
    else:
        sky_flow_reg_loss = torch.tensor(0.0).to(pred_depth.device)

    sky_region = gt_sky_mask > 0
    pred_depth = pred_depth[sky_region]
    return (
        F.mse_loss(pred_depth / sky_depth, torch.ones_like(pred_depth)) * 0.01,
        sky_flow_reg_loss,
    )


def compute_loss(output_dict, target_dict, args=None, lpips_loss=None):
    gs_params, pred_dict = output_dict["gs_params"], output_dict["render_results"]
    device = pred_dict[pred_dict["rgb_key"]].device
    mean, std = torch.tensor(MEAN).to(device), torch.tensor(STD).to(device)
    pred_rgb = pred_dict[pred_dict["rgb_key"]] * std + mean
    target_rgb = rearrange(target_dict["target_image"], "b t v c h w -> b t v h w c") * std + mean

    if lpips_loss is not None:
        loss_dict = lpips_loss(pred_rgb, target_rgb)
    else:
        rgb_loss = F.mse_loss(pred_rgb, target_rgb)
        loss_dict = {"rgb_loss": rgb_loss}

    if args.enable_depth_loss and "target_depth" in target_dict:
        pred_depth, target_depth = pred_dict[pred_dict["depth_key"]], target_dict["target_depth"]
        depth_loss = compute_depth_loss(pred_depth, target_depth)
        loss_dict["depth_loss"] = depth_loss

        if pred_dict["decoder_depth_key"] is not None:
            pred_decoder_depth = pred_dict[pred_dict["decoder_depth_key"]]
            decoded_depth_loss = compute_depth_loss(pred_decoder_depth, target_depth)
            loss_dict["decoded_depth_loss"] = decoded_depth_loss
            if (
                args.enable_sky_depth_loss or args.enable_sky_opacity_loss
            ) and "target_sky_masks" in target_dict:
                sky_decoded_depth_loss, _ = compute_sky_depth_loss(
                    pred_decoder_depth,
                    target_dict["target_sky_masks"],
                    sky_depth=args.sky_depth,
                )
                loss_dict["sky_decodede_depth_loss"] = sky_decoded_depth_loss

    if args.enable_flow_reg_loss and pred_dict["flow_key"] is not None:
        pred_flow = gs_params["forward_flow"]
        zero_flow = torch.zeros_like(gs_params["forward_flow"]).to(device)
        forward_flow_reg = F.mse_loss(pred_flow, zero_flow, reduction="none")
        loss_dict["flow_reg_loss"] = args.flow_reg_coeff * forward_flow_reg.mean()

    if args.enable_sky_depth_loss and "target_sky_masks" in target_dict:
        # real gaussian depth
        sky_depth_loss, sky_flow_reg_loss = compute_sky_depth_loss(
            pred_dict[pred_dict["depth_key"]],
            target_dict["target_sky_masks"],
            sky_depth=args.sky_depth,
            flow=(pred_dict[pred_dict["flow_key"]] if pred_dict["flow_key"] is not None else None),
        )
        loss_dict["sky_depth_loss"] = sky_depth_loss
        loss_dict["sky_flow_reg_loss"] = sky_flow_reg_loss
        loss_dict["opacity_loss"] = 0.01 * F.mse_loss(
            pred_dict[pred_dict["alpha_key"]],
            torch.ones_like(pred_dict[pred_dict["alpha_key"]]),
        )
        if pred_dict["decoder_depth_key"] is not None:
            (sky_decoded_depth_loss, sky_decoded_flow_reg_loss,) = compute_sky_depth_loss(
                pred_dict[pred_dict["decoder_depth_key"]],
                target_dict["target_sky_masks"],
                sky_depth=args.sky_depth,
                flow=(
                    pred_dict[pred_dict["decoder_flow_key"]]
                    if pred_dict["decoder_flow_key"] is not None
                    else None
                ),
            )
            loss_dict["sky_decodede_depth_loss"] = sky_decoded_depth_loss
            loss_dict["sky_decoded_flow_reg_loss"] = sky_decoded_flow_reg_loss

    elif args.enable_sky_opacity_loss and "target_sky_masks" in target_dict:
        opacity = pred_dict[pred_dict["alpha_key"]].squeeze(-1)
        b, t, v, h, w = opacity.shape
        gt_h, gt_w = target_dict["target_sky_masks"].shape[-2:]
        if h != gt_h or w != gt_w:
            opacity = F.interpolate(
                rearrange(opacity, "b t v h w -> (b t v) 1 h w"),
                size=(gt_h, gt_w),
                mode="bilinear",
                align_corners=False,
            )
            opacity = rearrange(opacity, "(b t v) 1 h w -> b t v h w", b=b, t=t, v=v)
        sky_opacity_loss = F.l1_loss(opacity, 1 - target_dict["target_sky_masks"])
        loss_dict["sky_opacity_loss"] = sky_opacity_loss * args.sky_opacity_loss_coeff

    return loss_dict


def compute_scene_flow_metrics(pred, labels):
    """
    Computes the scene flow metrics between the predicted and target scene flow values.
    # modified from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/0e4f403c73cb3fcd5503294a7c461926a4cdd1ad/utils.py#L12

    Args:
        pred (Tensor): predicted scene flow values
        labels (Tensor): target scene flow values
    Returns:
        dict: scene flow metrics
    """
    l2_norm = torch.sqrt(torch.sum((pred - labels) ** 2, -1)).cpu()
    # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, -1)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: Acc_5
    error_lt_5 = torch.BoolTensor((l2_norm < 0.05))
    relative_err_lt_5 = torch.BoolTensor((relative_err < 0.05))
    acc3d_strict = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: Acc_10
    error_lt_10 = torch.BoolTensor((l2_norm < 0.1))
    relative_err_lt_10 = torch.BoolTensor((relative_err < 0.1))
    acc3d_relax = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    l2_norm_gt_3 = torch.BoolTensor(l2_norm > 0.3)
    relative_err_gt_10 = torch.BoolTensor(relative_err > 0.1)
    outlier = torch.mean((l2_norm_gt_3 | relative_err_gt_10).float()).item()

    # NOTE: angle error
    unit_label = labels / (labels.norm(dim=-1, keepdim=True) + 1e-7)
    unit_pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-7)

    # it doesn't make sense to compute angle error on zero vectors
    # we use a threshold of 0.1 to avoid noisy gt flow
    non_zero_flow_mask = labels_norm > 0.1
    # Apply the mask to filter out zero vectors
    unit_label = unit_label[non_zero_flow_mask]
    unit_pred = unit_pred[non_zero_flow_mask]
    # Initialize angle_error
    angle_error = 0.0
    # Check if there are any valid vectors to compute the angle error
    if unit_label.numel() > 0:
        eps = 1e-7
        # Compute the dot product and clamp its values to avoid numerical issues with acos
        dot_product = (unit_label * unit_pred).sum(dim=-1).clamp(min=-1 + eps, max=1 - eps)

        # Optionally, handle any remaining NaNs in the dot product
        dot_product = torch.nan_to_num(dot_product, nan=0.0)

        # Compute the angle error in radians and take the mean
        angle_error = torch.acos(dot_product).mean().item()

    torch.cuda.empty_cache()
    return {
        "EPE3D": EPE3D,
        "acc3d_strict": acc3d_strict,
        "acc3d_relax": acc3d_relax,
        "outlier": outlier,
        "angle_error": angle_error,
    }
