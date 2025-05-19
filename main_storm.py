import argparse
import copy
import datetime
import json
import logging
import math
import os
import sys
import time

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import torch.distributed
import torch.utils.data

# STORM imports
import storm.models as models
import storm.utils.distributed as distributed
import storm.utils.misc as misc
from engine_storm import evaluate, evaluate_flow, visualize
from storm.dataset.constants import DATASET_DICT
from storm.dataset.data_utils import prepare_inputs_and_targets
from storm.dataset.samplers import InfiniteSampler, NoPaddingDistributedSampler
from storm.dataset.storm_dataset import STORMDataset, STORMDatasetEval
from storm.utils.logging import MetricLogger, WandbLogger, setup_logging
from storm.utils.losses import compute_loss
from storm.utils.lpips_loss import RGBLpipsLoss
from storm.utils.misc import NativeScalerWithGradNormCount as NativeScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True


def get_args_parser():
    parser = argparse.ArgumentParser("STORM training", add_help=False)

    # =============== Model parameters ================= #
    parser.add_argument("--model", default="STORM-B/8", type=str)
    parser.add_argument("--num_context_timesteps", default=4, type=int)
    parser.add_argument("--num_target_timesteps", default=4, type=int)
    parser.add_argument("--gs_dim", default=3, type=int, help="Number of gs dimensions")
    parser.add_argument("--use_sky_token", action="store_true")
    parser.add_argument("--use_affine_token", action="store_true")
    parser.add_argument("--use_latest_gsplat", action="store_true")
    parser.add_argument(
        "--decoder_type",
        type=str,
        choices=["dummy", "conv"],
        default="dummy",
        help="STORM or LatentSTORM",
    )
    parser.add_argument("--num_motion_tokens", default=16, type=int, help="Number of motion tokens")

    # =============== Losses =============== #
    parser.add_argument("--enable_depth_loss", action="store_true")

    # Option 1: push the sky depth to a fixed value
    parser.add_argument("--enable_sky_depth_loss", action="store_true")
    parser.add_argument("--sky_depth", type=float, default=300.0)
    # Option 2: make sky gaussians transparent and use a sky token to represent sky
    parser.add_argument("--enable_sky_opacity_loss", action="store_true")
    parser.add_argument("--sky_opacity_loss_coeff", type=float, default=0.1)

    # flow regularization loss
    parser.add_argument("--enable_flow_reg_loss", action="store_true")
    parser.add_argument("--flow_reg_coeff", type=float, default=0.005)

    # perceptual loss
    parser.add_argument("--enable_perceptual_loss", action="store_true")
    parser.add_argument("--perceptual_weight", default=0.05, type=float, help="LPIPS weight")
    parser.add_argument("--perceptual_loss_start_iter", default=5000, type=int)

    # ============= Optimizer and LR parameters ============= #
    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=8e-4, help="base learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_sched", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_iters", type=int, default=5000, help="iters to warmup LR")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=3.0, help="Gradient clip")
    parser.add_argument("--disable_grad_checkpointing", action="store_true")

    parser.add_argument("--start_iteration", default=0, type=int, help="start iteration")
    parser.add_argument("--num_iterations", default=200_000, type=int, help="num of iterations")
    parser.add_argument("--resume_from", default=None, help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--load_from", type=str, default=None)

    # ============= Dataset parameters ============= #
    parser.add_argument("--data_root", default="./data/STORM2", type=str, help="dataset path")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--input_size", default=(160, 240), type=int, nargs=2)
    parser.add_argument("--num_max_cameras", type=int, default=3)
    parser.add_argument("--timespan", type=float, default=2.0)
    parser.add_argument("--load_ground", action="store_true")
    parser.add_argument("--load_depth", action="store_true")
    parser.add_argument("--load_flow", action="store_true")
    parser.add_argument("--dataset", default="waymo", type=str, choices=DATASET_DICT.keys())
    parser.add_argument("--subset_ratio", default=1.0, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--skip_sky_mask", action="store_true", help="skip sky mask loading")
    # ============= Logging ============= #
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--num_vis_samples", type=int, default=1)
    parser.add_argument("--log_every_n_iters", type=int, default=50)
    parser.add_argument("--vis_every_n_iters", type=int, default=5000)
    parser.add_argument("--ckpt_every_n_iters", type=int, default=5000)
    parser.add_argument("--eval_every_n_iters", type=int, default=50000)
    parser.add_argument("--total_elapsed_time", type=float, default=0.0, help="total time elapsed")
    parser.add_argument("--keep_n_ckpts", default=1, type=int)

    # ============= Miscellaneous ============= #
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--visualization_only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    # ============= WandB ============= #
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--project", default="debug", type=str)
    parser.add_argument("--entity", default="YOUR_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--overwrite_wandb", action="store_true")

    return parser


def main(args):
    # Prepare distributed training
    distributed.enable(overwrite=True)

    global logger
    args.exp_name = args.model.replace("/", "-") if args.exp_name is None else args.exp_name
    log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    video_dir = os.path.join(log_dir, "videos")
    args.log_dir, args.ckpt_dir, args.video_dir = log_dir, checkpoint_dir, video_dir

    device = torch.device(args.device)
    world_size, global_rank = distributed.get_world_size(), distributed.get_global_rank()
    seed = args.seed + global_rank
    misc.fix_random_seeds(seed)

    log_writer = None
    if global_rank == 0:
        [os.makedirs(d, exist_ok=True) for d in [log_dir, checkpoint_dir, video_dir]]
        if args.enable_wandb:
            run_id_path, run_id = os.path.join(log_dir, "wandb_run_id.txt"), None
            if os.path.exists(run_id_path) and not args.overwrite_wandb:
                with open(run_id_path, "r") as f:
                    run_id = f.readlines()[-1].strip()
            log_writer = WandbLogger(args=args, resume="must", id=run_id)
            if run_id is None:
                with open(run_id_path, "a") as f:
                    f.write(log_writer.run_id + "\n")

    # set up logging
    setup_logging(output=log_dir, level=logging.INFO)
    logger = logging.getLogger("STORM")
    logger.info(f"hostname: {os.uname().nodename}\n")
    logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info(f"Logging to {log_dir}")
    logger.info(json.dumps(args.__dict__, indent=4, sort_keys=True))
    if global_rank == 0:
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)

    dataset_meta = DATASET_DICT[args.dataset]
    train_annotation = dataset_meta["annotation_txt_file_train"]
    val_annotation = dataset_meta["annotation_txt_file_val"]
    if train_annotation is not None:
        if args.dataset == "nuscenes":
            train_annotation = f"data/dataset_scene_list/nuscenes_train.txt"
        else:
            train_annotation = f"{args.data_root}/{train_annotation}"
    if val_annotation is not None:
        if args.dataset == "nuscenes":
            val_annotation = f"data/dataset_scene_list/nuscenes_val.txt"
        else:
            val_annotation = f"{args.data_root}/{val_annotation}"
        if not os.path.exists(val_annotation):
            val_annotation = None

    dataset_train = STORMDataset(
        data_root=args.data_root,
        annotation_txt_file_list=train_annotation,
        target_size=args.input_size,
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
        skip_sky_mask=args.skip_sky_mask,
    )
    sampler_train = InfiniteSampler(sample_count=len(dataset_train), shuffle=True, seed=seed)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )

    if val_annotation is not None:
        dataset_val = STORMDataset(
            data_root=args.data_root,
            annotation_txt_file_list=val_annotation,
            target_size=args.input_size,
            num_context_timesteps=args.num_context_timesteps,
            num_target_timesteps=args.num_target_timesteps,
            timespan=args.timespan,
            num_max_cams=args.num_max_cameras,
            load_depth=args.load_depth,
            load_flow=args.load_flow,
            skip_sky_mask=args.skip_sky_mask,
        )
        dataset_eval = STORMDatasetEval(
            data_root=args.data_root,
            annotation_txt_file_list=val_annotation,
            target_size=args.input_size,
            num_context_timesteps=args.num_context_timesteps,
            num_target_timesteps=args.num_target_timesteps,
            timespan=args.timespan,
            num_max_cams=args.num_max_cameras,
            load_depth=args.load_depth,
            load_flow=args.load_flow,
            load_dynamic_mask=True,
            load_ground_label=args.load_ground,
            skip_sky_mask=args.skip_sky_mask,
        )
        dataset_eval_flow = STORMDatasetEval(
            data_root=args.data_root,
            annotation_txt_file_list=val_annotation,
            target_size=args.input_size,
            num_context_timesteps=args.num_context_timesteps,
            num_target_timesteps=args.num_target_timesteps,
            timespan=args.timespan,
            num_max_cams=args.num_max_cameras,
            load_depth=args.load_depth,
            load_flow=args.load_flow,
            load_dynamic_mask=False,
            load_ground_label=args.load_ground,
            return_context_as_target=True,
            skip_sky_mask=args.skip_sky_mask,
        )
        sampler = NoPaddingDistributedSampler(
            dataset_eval,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
        )
        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            pin_memory=False,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )
        data_loader_eval_flow = torch.utils.data.DataLoader(
            dataset_eval_flow,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            pin_memory=False,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )
    else:
        dataset_val = None
        dataset_eval = None
        dataset_eval_flow = None
        data_loader_eval = None
        data_loader_eval_flow = None

    logger.info(f"Dataset: {args.dataset}, train: {train_annotation}, val: {val_annotation}")
    logger.info(f"Dataset contains {len(dataset_train):,} sequences using {train_annotation}.")

    if args.model in models.STORM_models:
        model = models.STORM_models[args.model](
            img_size=args.input_size,
            gs_dim=args.gs_dim,
            decoder_type=args.decoder_type,
            grad_checkpointing=not args.disable_grad_checkpointing,
            use_sky_token=args.use_sky_token,
            use_affine_token=args.use_affine_token,
            num_motion_tokens=args.num_motion_tokens,
            use_latest_gsplat=args.use_latest_gsplat,
        )
    else:
        raise ValueError(f"Invalid model name: {args.model}")

    logger.info(f"Model = {str(model)}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    model.to(device)
    model_without_ddp = model

    if distributed.is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module
    global_batch_size = args.batch_size * world_size
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * global_batch_size / 256
    logger.info("Global batch size: %d" % global_batch_size)
    logger.info(f"Base lr: {args.lr * 256 / global_batch_size:.2e}, Actual lr: {args.lr:.2e}")

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    logger.info(f"Optimizer = {optimizer}")
    logger.info(f"Loss Scaler = {loss_scaler}")

    # Load checkpoint or resume training
    logger.info(f"Original start Iteration: {args.start_iteration}")
    vis_slice_id = misc.load_model(args, model_without_ddp, optimizer, loss_scaler)
    logger.info(f"New start iteration {args.start_iteration}")

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {num_trainable_params / 1e6:.2f}M")
    logger.info(f"Training with {world_size} GPUs")

    data_iter_step = args.start_iteration
    if log_writer is not None:
        log_writer.set_step(data_iter_step)

    if args.evaluate:
        eval_result = evaluate(data_loader_eval, model_without_ddp, args)
        if log_writer is not None and eval_result is not None:
            eval_result = {f"eval/{k}": v for k, v in eval_result.items()}
            log_writer.update(eval_result)
        if args.dataset == "waymo":
            if args.decoder_type != "conv":
                flow_eval_result = evaluate_flow(data_loader_eval_flow, model_without_ddp, args)
                if log_writer is not None and flow_eval_result is not None:
                    flow_eval_result = {f"eval/{k}": v for k, v in flow_eval_result.items()}
                    log_writer.update(flow_eval_result)
        logger.info("Evaluation done, exiting.")
        exit()

    valid_slice_id = copy.deepcopy(vis_slice_id)
    if dataset_val is not None and valid_slice_id >= len(dataset_val):
        valid_slice_id = 0
    for _ in range(args.num_vis_samples):
        vis_slice_id, valid_slice_id = visualize(
            args=args,
            model=model_without_ddp,
            dset_train=dataset_train,
            step=data_iter_step,
            train_vis_id=vis_slice_id,
            device=device,
            dset_val=dataset_val,
            val_vis_id=valid_slice_id,
        )
    if args.visualization_only:
        logger.info("Visualization done, exiting.")
        exit()

    rgb_and_lpips_loss = RGBLpipsLoss(
        perceptual_weight=args.perceptual_weight,
        enable_perceptual_loss=args.enable_perceptual_loss,
    ).to(device)
    # will turn on perceptual loss after a certain number of iterations
    rgb_and_lpips_loss.set_perceptual_loss(False)

    logger.info(f"Starting training from iteration {args.start_iteration} to {args.num_iterations}")
    metrics_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    start_time = time.time()
    num_tokens_printed = False

    for data_dict in metric_logger.log_every(
        data_loader_train,
        print_freq=args.log_every_n_iters,
        header="Training",
        n_iterations=args.num_iterations,
        start_iteration=args.start_iteration,
    ):
        if data_iter_step > args.num_iterations:
            break
        if log_writer is not None:
            log_writer.set_step(data_iter_step)
        if args.enable_perceptual_loss and data_iter_step >= args.perceptual_loss_start_iter:
            rgb_and_lpips_loss.set_perceptual_loss(True)

        model.train()
        misc.adjust_learning_rate(optimizer, data_iter_step, args)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            input_dict, target_dict = prepare_inputs_and_targets(data_dict, device)
            pred_dict = model(input_dict)
            loss_dict = compute_loss(pred_dict, target_dict, args, rgb_and_lpips_loss)

        loss_value = sum(loss for k, loss in loss_dict.items() if "loss" in k)

        if not math.isfinite(loss_value):
            logger.info("NaN detected")
            raise AssertionError

        grad_norm = loss_scaler(
            loss_value,
            optimizer,
            parameters=model.parameters(),
            clip_grad=args.grad_clip,
        )
        optimizer.zero_grad()
        torch.cuda.synchronize()
        if world_size > 1:
            [torch.distributed.all_reduce(v) for v in loss_dict.values()]
        loss_dict_reduced = {k: v.item() / world_size for k, v in loss_dict.items()}
        total_loss_reduced = sum(loss for k, loss in loss_dict_reduced.items() if "loss" in k)
        lr = optimizer.param_groups[0]["lr"]
        psnr = -10 * np.log10(loss_dict_reduced["rgb_loss"])
        metric_logger.update(lr=lr, psnr=psnr, loss=total_loss_reduced, **loss_dict_reduced)
        metric_logger.update(grad_norm=grad_norm)

        if "num_tokens" in pred_dict and not num_tokens_printed:
            logger.info(f"num_tokens: {pred_dict['num_tokens']}")
            num_tokens_printed = True

        if log_writer is not None:
            log_writer.update(
                {
                    "psnr": psnr,
                    "loss": total_loss_reduced,
                    **loss_dict_reduced,
                    "lr": lr,
                    "grad_norm": grad_norm,
                }
            )
            log_writer.set_step()

        if (data_iter_step + 1) % args.ckpt_every_n_iters == 0:
            if distributed.is_main_process():
                elapsed_t = time.time() - start_time + args.total_elapsed_time
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss_scaler": loss_scaler.state_dict(),
                    "latest_step": data_iter_step,
                    "vis_slice_id": vis_slice_id,
                    "args": args,
                    "total_elapsed_time": elapsed_t,
                }
                checkpoint_path = os.path.join(args.ckpt_dir, f"ckpt_{data_iter_step:06d}.pth")
                torch.save(checkpoint, checkpoint_path)
                misc.cleanup_checkpoints(args.ckpt_dir, keep_num=args.keep_n_ckpts)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            torch.distributed.barrier()
            torch.cuda.empty_cache()

        if (data_iter_step + 1) % args.vis_every_n_iters == 0:
            for _ in range(args.num_vis_samples):
                vis_slice_id, valid_slice_id = visualize(
                    args=args,
                    model=model_without_ddp,
                    dset_train=dataset_train,
                    step=data_iter_step,
                    train_vis_id=vis_slice_id,
                    device=device,
                    dset_avl=dataset_val,
                    val_vis_id=valid_slice_id,
                )
            torch.distributed.barrier()
            torch.cuda.empty_cache()

        if (data_iter_step + 1) % args.eval_every_n_iters == 0 and (
            data_iter_step + 1
        ) != args.num_iterations:
            eval_result = evaluate(data_loader_eval, model_without_ddp, args, f"{data_iter_step}")
            if log_writer is not None and eval_result is not None:
                log_writer.update({f"eval/{k}": v for k, v in eval_result.items()})
            if args.decoder_type != "conv" and args.dataset == "waymo":
                flow_eval_result = evaluate_flow(
                    data_loader_eval_flow,
                    model_without_ddp,
                    args,
                    name_str=f"{data_iter_step}",
                )
                if log_writer is not None and flow_eval_result is not None:
                    log_writer.update({f"eval/{k}": v for k, v in flow_eval_result.items()})
            torch.distributed.barrier()
            torch.cuda.empty_cache()

        data_iter_step += 1

    metric_logger.synchronize_between_processes()

    total_time = time.time() - start_time + args.total_elapsed_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    eval_result = evaluate(data_loader_eval, model_without_ddp, args)
    if log_writer is not None and eval_result is not None:
        log_writer.update({f"eval/{k}": v for k, v in eval_result.items()})
    if args.decoder_type != "conv" and args.dataset == "waymo":
        flow_eval_result = evaluate_flow(data_loader_eval_flow, model_without_ddp, args)
        if log_writer is not None and flow_eval_result is not None:
            log_writer.update({f"eval/{k}": v for k, v in flow_eval_result.items()})
    logger.info("Done!")



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
