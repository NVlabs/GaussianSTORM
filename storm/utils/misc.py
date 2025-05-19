import collections.abc
import datetime
import logging
import math
import os
import random
from collections import OrderedDict
from glob import glob
from itertools import repeat

import numpy as np
import torch
from torch import inf

logger = logging.getLogger("STORM")


def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _ntuple(n):
    """
    Creates a parser that converts an input to a tuple of length n.

    Args:
        n (int): Length of the tuple.

    Returns:
        Callable: A function that parses the input into a tuple of length n.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def cleanup_checkpoints(ckpt_dir, keep_num=1):
    """
    Clean up old checkpoints, keeping only the latest 'keep_num' checkpoints.

    Args:
        ckpt_dir (str): Directory containing the checkpoints.
        keep_num (int): Number of recent checkpoints to keep.
    """
    ckpts = glob(f"{ckpt_dir}/*.pth")
    ckpts = [ckpt for ckpt in ckpts if "latest" not in ckpt and "best" not in ckpt]
    ckpts = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Remove older checkpoints
    for ckpt in ckpts[:-keep_num]:
        os.remove(ckpt)
        logger.info(f"Removed checkpoint: {ckpt}")

    # Create or update latest symlink
    if ckpts:
        latest_symlink = f"{ckpt_dir}/latest.pth"
        try:
            os.remove(latest_symlink)
        except FileNotFoundError:
            pass
        os.symlink(os.path.abspath(ckpts[-1]), latest_symlink)
        logger.info(f"Created symlink: {latest_symlink} -> {ckpts[-1]}")


def load_model(args, model_without_ddp, optimizer=None, loss_scaler=None):
    """
    Load model, optimizer, and loss scaler states from a checkpoint.

    Args:
        args: Arguments containing checkpoint paths and loading configurations.
        model_without_ddp (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer for loading states.
        loss_scaler (torch.cuda.amp.GradScaler, optional): Loss scaler for AMP.

    Returns:
        int: Visualization slice ID if available.
    """
    vis_slice_id, checkpoint_loaded = 0, False
    if args.resume_from or args.auto_resume:
        if not args.resume_from:
            # Checkpoint not provided, auto-resume from the latest checkpoint
            checkpoints = [ckpt for ckpt in glob(f"{args.ckpt_dir}/*.pth") if "latest" not in ckpt]
            checkpoints = sorted(checkpoints, key=os.path.getmtime)
            if len(checkpoints) > 0:
                # Resume from the latest checkpoint
                args.resume_from = checkpoints[-1]

        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"[Model-resume] Resuming from: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
            msg = model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
            logger.info(f"[Model-resume] Loaded model: {msg}")
            checkpoint_loaded = True
            if "optimizer" in checkpoint and "latest_step" in checkpoint and optimizer is not None:
                msg = optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info(f"[Model-resume] Loaded optimizer: {msg}")
                args.start_iteration = checkpoint["latest_step"] + 1
                if "loss_scaler" in checkpoint and loss_scaler is not None:
                    msg = loss_scaler.load_state_dict(checkpoint["loss_scaler"])
                    logger.info(f"[Model-resume] Loaded loss_scaler: {msg}")
                if "vis_slice_id" in checkpoint:
                    vis_slice_id = checkpoint["vis_slice_id"] + 1
            if "latest_step" in checkpoint:
                args.prev_num_iterations = checkpoint["latest_step"]
                args.start_iteration = checkpoint["latest_step"] + 1

            if "total_elapsed_time" in checkpoint:
                args.total_elapsed_time = float(checkpoint["total_elapsed_time"])
                elapsed_time_str = str(datetime.timedelta(seconds=int(args.total_elapsed_time)))
                logger.info(f"Loaded elapsed_time: {elapsed_time_str}")
            del checkpoint

    if not checkpoint_loaded and args.load_from and os.path.exists(args.load_from):
        # args.resume_from has the highest priority. If it's not found, try args.load_from
        # this is useful for loading a model without optimizer and scheduler states
        # or for loading a pre-trained model for initialization, fine-tuning, or evaluation.
        logger.info(f"Loading checkpoint from: {args.load_from}")
        checkpoint = torch.load(args.load_from)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        try:
            msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
            checkpoint_loaded = True
            logger.info(f"[Model-init] Loaded model: {msg}")
        except Exception as e:
            logger.error(e)
            logger.info(f"[Model-init] Loading model from {args.load_from} failed. Error: {e}")
            model_state_dict = model_without_ddp.state_dict()
            # Create a new OrderedDict that will only contain matching parameter shapes
            filtered_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        logger.info(
                            f"Skipping parameter due to shape mismatch: {k} "
                            f"({v.shape} vs {model_state_dict[k].shape})"
                        )
                else:
                    logger.info(f"Skipping unexpected key: {k}")

            # Load the filtered state dict into the model (strict=False to allow missing keys)
            msg = model_without_ddp.load_state_dict(filtered_dict, strict=False)
            logger.info(f"Load status: {msg}")
        del checkpoint

    if not checkpoint_loaded:
        logger.info(f"Training from scratch. No checkpoint found.")
    return vis_slice_id


def adjust_learning_rate(optimizer, iteration, args):
    """
    Adjust the learning rate using a cosine decay schedule with warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to update learning rate.
        iteration (int): Current training iteration.
        args: Arguments defining the learning rate schedule.

    Returns:
        float: Updated learning rate.
    """
    if iteration < args.warmup_iters:
        lr = args.lr * iteration / args.warmup_iters
    else:
        if args.lr_sched == "constant":
            lr = args.lr
        elif args.lr_sched == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (iteration - args.warmup_iters)
                    / (args.num_iterations - args.warmup_iters)
                )
            )
        else:
            raise ValueError(f"Unknown lr_sched: {args.lr_sched}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

    return lr


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_grad_norm_(parameters, norm_type=2.0):
    """
    Compute gradient norm for a set of parameters.

    Args:
        parameters (Iterable): Parameters to compute gradients for.
        norm_type (float): Norm type for gradient computation.

    Returns:
        torch.Tensor: Gradient norm.
    """
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    """
    A wrapper for torch.cuda.amp.GradScaler with gradient norm tracking.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
    """

    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(
        self,
        loss,
        optimizer,
        parameters,
        clip_grad=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        norm = None
        if update_grad:
            self._scaler.unscale_(optimizer)
            if clip_grad is not None and clip_grad > 0.0:
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        return norm

    def state_dict(self):
        """Save state dictionary for the scaler."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dictionary for the scaler."""
        self._scaler.load_state_dict(state_dict)
