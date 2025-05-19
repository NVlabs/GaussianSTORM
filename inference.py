import argparse
import copy
import datetime
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import storm.models as models
import storm.utils.misc as misc
from storm.dataset.constants import DATASET_DICT
from storm.dataset.data_utils import to_batch_tensor
from storm.dataset.storm_dataset import SingleSequenceDataset
from storm.utils.logging import setup_logging
from storm.visualization.video_maker import make_video

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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

    # ============= Checkpoint parameters ============= #
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None, help="resume from checkpoint")
    parser.add_argument("--load_from", type=str, default=None)

    # ============= Dataset parameters ============= #
    parser.add_argument("--data_root", default="./data/STORM2", type=str, help="dataset path")
    parser.add_argument("--overwrite_train_ctx_view_with", default=None, type=int)
    parser.add_argument("--overwrite_train_tgt_view_with", default=None, type=int)
    parser.add_argument("--overwrite_test_ctx_view_with", default=None, type=int)
    parser.add_argument("--input_size", default=(160, 240), type=int, nargs=2)
    parser.add_argument("--num_max_cameras", type=int, default=3)
    parser.add_argument("--timespan", type=float, default=2.0)
    parser.add_argument("--load_ground", action="store_true")
    parser.add_argument("--load_depth", action="store_true")
    parser.add_argument("--load_flow", action="store_true")
    parser.add_argument("--dataset", default="waymo", type=str, choices=DATASET_DICT.keys())
    parser.add_argument("--skip_sky_mask", action="store_true", help="skip sky mask loading")
    # ============= Logging ============= #
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--num_vis_samples", type=int, default=1)

    # ============= Miscellaneous ============= #
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    # ============= WandB ============= #
    parser.add_argument("--project", default="debug", type=str)
    parser.add_argument("--exp_name", default=None, type=str)

    return parser


def main(args):
    global logger
    args.exp_name = args.model.replace("/", "-") if args.exp_name is None else args.exp_name
    log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    video_dir = os.path.join(log_dir, "videos")
    args.log_dir, args.ckpt_dir, args.video_dir = log_dir, checkpoint_dir, video_dir

    device = torch.device(args.device)
    seed = args.seed
    misc.fix_random_seeds(seed)
    cudnn.benchmark = True

    # set up logging
    setup_logging(output=log_dir, level=logging.INFO)
    logger = logging.getLogger("STORM")
    logger.info(f"hostname: {os.uname().nodename}\n")
    logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info(f"Logging to {log_dir}")
    logger.info(json.dumps(args.__dict__, indent=4, sort_keys=True))
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
    num_context_timesteps = dataset_meta["num_context_timesteps"]
    num_target_timesteps = dataset_meta["num_target_timesteps"]
    if args.overwrite_train_ctx_view_with is not None:
        num_context_timesteps = args.overwrite_train_ctx_view_with
    if args.overwrite_test_ctx_view_with is not None:
        num_context_timesteps = args.overwrite_test_ctx_view_with
    if args.overwrite_train_tgt_view_with is not None:
        num_target_timesteps = args.overwrite_train_tgt_view_with
    input_size = dataset_meta["size"]
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"annotation_txt_file_list_train: {train_annotation}")

    if args.model in models.STORM_models:
        model = models.STORM_models[args.model](
            img_size=args.input_size,
            gs_dim=args.gs_dim,
            decoder_type=args.decoder_type,
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
    dataset = SingleSequenceDataset(
        data_root=args.data_root,
        annotation_txt_file_list=train_annotation,
        target_size=input_size,
        num_context_timesteps=num_context_timesteps,
        num_target_timesteps=num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
    )

    logger.info(f"Dataset contains {len(dataset):,} sequences using {train_annotation}.")
    misc.load_model(args, model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {num_trainable_params / 1e6:.2f}M")
    model.eval().cuda()
    logger.info(f"Preparing data... (This may take a while)")
    data_dict_list = dataset.__getitem__(index=0, start_index=0, end_index=60)
    data_dict_list = to_batch_tensor(data_dict_list)
    logger.info(f"Done preparing data.")
    for i in range(len(data_dict_list)):
        data_dict = data_dict_list[i]
        output_name = f"test_{i}.mp4"
        make_video(
            dataset=None,
            model=model,
            device=device,
            output_filename=output_name,
            data_dict=data_dict,
        )
        print(f"Saved video to {output_name}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
