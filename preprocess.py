import argparse
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data converter arg parser")
    parser.add_argument("--data_root", type=str, required=True, help="root path of dataset")
    parser.add_argument("--dataset", type=str, default="waymo", help="dataset name")
    parser.add_argument("--scene_list_file", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="split of the dataset, e.g. training, validation, testing, please specify the split name for different dataset",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="output directory of processed data",
    )
    parser.add_argument(
        "--json_folder_to_save",
        type=str,
        required=True,
        help="to save the json files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of threads to be used",
    )
    # priority: scene_ids > start_idx + num_scenes
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id is given, use start_idx and num_scenes to generate scene_ids",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        "--interpolate_N",
        type=int,
        default=0,
        help="Interpolate to get frames at higher frequency, this is only used for nuscene dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the existing files",
    )
    parser.add_argument(
        "--process_keys",
        nargs="+",
        default=["images", "lidar", "calib", "pose", "ground", "dynamic_masks"],
    )
    args = parser.parse_args()
    if args.dataset != "nuscenes" and args.interpolate_N > 0:
        parser.error("interpolate_N > 0 is only allowed when dataset is 'nuscenes'")
    os.makedirs(args.target_dir, exist_ok=True)

    if args.scene_ids is not None:
        scene_ids = args.scene_ids
    else:
        scene_ids = np.arange(args.start_idx, args.start_idx + args.num_scenes)

    if args.dataset == "nuscenes":
        scene_lists = scene_ids
    else:
        if args.scene_list_file is None:
            raise ValueError("scene_list_file is required for non-nuscenes dataset")
        if not os.path.exists(args.scene_list_file):
            raise ValueError(f"scene_list_file {args.scene_list_file} does not exist")
        scene_lists = open(args.scene_list_file, "r").read().splitlines()
        if np.max(scene_ids) >= len(scene_lists):
            scene_ids = [scene_id for scene_id in scene_ids if scene_id < len(scene_lists)]
        scene_lists = [(i, scene_lists[i]) for i in scene_ids]
        print(f"scene_lists: {scene_lists}")

    if args.dataset == "waymo":
        from preproc.waymo_preprocess import WaymoProcessor

        dataset_processor = WaymoProcessor(
            load_dir=args.data_root,
            save_dir=args.target_dir,
            scene_lists=scene_lists,
            prefix=args.split,
            process_keys=args.process_keys,
            json_folder_to_save=args.json_folder_to_save,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )
    elif args.dataset == "argoverse":
        raise NotImplementedError("ArgoverseProcessor is not implemented yet")
        from preproc.argoverse_preprocess import ArgoVerseProcessor

        scene_ids = [int(scene_id) for scene_id in scene_ids]
        dataset_processor = ArgoVerseProcessor(
            load_dir=args.data_root,
            save_dir=args.target_dir,
            process_keys=args.process_keys,
            scene_lists=scene_lists,
            prefix=args.split,
            num_workers=args.num_workers,
            json_folder_to_save=args.json_folder_to_save,
        )
    elif args.dataset == "nuscenes":
        raise NotImplementedError("NuScenesProcessor is not implemented yet")
        from preproc.nuscenes_preprocess import NuScenesProcessor

        scene_ids = [f"{scene_id:03d}" for scene_id in scene_ids]
        dataset_processor = NuScenesProcessor(
            load_dir=args.data_root,
            save_dir=args.target_dir,
            split=args.split,
            interpolate_N=args.interpolate_N,
            process_keys=args.process_keys,
            scene_lists=scene_ids,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            json_folder_to_save=args.json_folder_to_save,
        )
    else:
        raise ValueError(
            f"Unknown dataset {args.dataset}, please choose from waymo, pandaset, argoverse, nuscenes, kitti, nuplan"
        )

    if args.scene_ids is not None and args.num_workers <= 1:
        for idx in range(len(args.scene_ids)):
            dataset_processor.convert_one(idx)
    else:
        dataset_processor.convert()
