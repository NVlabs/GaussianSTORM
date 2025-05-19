import json
import logging
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import trange

from .constants import DATASET_DICT, DATASETS, MEAN, STD
from .data_utils import resize_depth, resize_flow, to_float_tensor, to_tensor

logger = logging.getLogger("STORM")


class STORMDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        annotation_txt_file_list: Union[str, List[str]],
        target_size: Tuple[int, int] = (160, 240),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[1, 3, 5, 6, 7] = 3,
        timespan: float = 2.0,  # 2.0 seconds
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        load_depth: bool = True,
        load_flow: bool = False,
        load_dynamic_mask: bool = False,
        load_ground_label: bool = False,
        return_context_as_target: bool = False,
        skip_sky_mask: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.target_size = target_size
        self.num_context_timesteps = num_context_timesteps
        self.num_target_timesteps = num_target_timesteps
        self.num_max_cams = num_max_cams
        self.timespan = timespan
        self.load_depth = load_depth
        self.load_flow = load_flow
        self.load_dynamic_mask = load_dynamic_mask
        self.load_ground_label = load_ground_label
        self.skip_sky_mask = skip_sky_mask
        if isinstance(annotation_txt_file_list, str):
            annotation_txt_file_list = [annotation_txt_file_list]
        scene_list = []
        for annotation_txt_file in annotation_txt_file_list:
            with open(annotation_txt_file, "r") as f:
                scene_list += f.readlines()
        annotation_paths = [line.strip() for line in scene_list]
        if subset_indices is not None:
            annotation_paths = [annotation_paths[i] for i in subset_indices]
        self.annotations = []
        for annotation_path in annotation_paths:
            with open(os.path.join(data_root, annotation_path), "r") as f:
                self.annotations.append(json.load(f))
        logger.info(f"Loaded {len(self.annotations)} annotations.")
        self.num_replicas = num_replicas
        if self.num_replicas > 1:
            self.annotations *= self.num_replicas
        self.equispaced = equispaced
        self.return_context_as_target = return_context_as_target
        self.img_transformation = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def get_frame(
        self,
        scene_json: Dict[str, Any],
        frame_idx: int,
        source_frame_idx: int = -1,
    ) -> Dict[str, Any]:
        """Retrieve a single frame from the dataset."""
        normalized_intrinsics = scene_json["normalized_intrinsics"]
        dataset_name = scene_json["dataset"]
        cam_to_world = scene_json["camera_to_world"]

        images, depths, sky_masks, flows = [], [], [], []
        camtoworlds, intrinsics = [], []
        dynamic_masks, ground_masks = [], []

        if source_frame_idx < 0:
            source_frame_idx = frame_idx

        camera_list = DATASET_DICT[dataset_name]["camera_list"][self.num_max_cams]
        ref_camera_name = DATASET_DICT[dataset_name]["ref_camera"]
        world_to_canonical = np.linalg.inv(cam_to_world[ref_camera_name][source_frame_idx])

        for camera in camera_list:
            img_relative_path = scene_json["relative_image_path"][camera][frame_idx]
            if dataset_name in ["waymo", "nuscenes", "argoverse2", "argoverse"]:
                img_relative_path = img_relative_path.replace("images", f"images_4")
                img_relative_path = img_relative_path.replace("sweeps", f"sweeps_4")
                img_relative_path = img_relative_path.replace("samples", f"samples_4")

            # Get RGB
            img_path = os.path.join(self.data_root, "datasets", dataset_name, img_relative_path)
            img = Image.open(img_path).convert("RGB")
            img = self.img_transformation(img)
            images.append(img)

            # Get sky mask
            if dataset_name in ["waymo", "nuscenes", "argoverse2"]:
                # if dataset_name in ["waymo", "nuscenes"]:
                if dataset_name == "nuscenes":
                    sky_path = img_path.replace("samples", "samples_sky_mask")
                    sky_path = sky_path.replace("sweeps", "sweeps_sky_mask")
                elif dataset_name == "waymo":
                    sky_path = img_path.replace("images", "sky_masks")
                elif dataset_name == "argoverse2":
                    sky_path = img_path.replace("images_4", "sky_masks_512")
                sky_path = sky_path.replace("jpg", "png")
                if self.skip_sky_mask:
                    sky = torch.zeros(self.target_size[0], self.target_size[1]).float()
                else:
                    try:
                        new_sky_path = sky_path.replace("STORM2", "STORM_masks")
                        sky = Image.open(new_sky_path).convert("L").resize(self.target_size[::-1])
                    except FileNotFoundError:
                        sky = Image.open(sky_path).convert("L").resize(self.target_size[::-1])
                sky = to_tensor(np.array(sky) > 0).float()
                sky_masks.append(sky)

            # Get dynamic mask, this is for dynamic region evaluation.
            if dataset_name in ["waymo"] and self.load_dynamic_mask:
                dynamic_path = img_path.replace("images_8", "dynamic_masks/all")
                dynamic_path = dynamic_path.replace("images_4", "dynamic_masks/all")
                dynamic_path = dynamic_path.replace("jpg", "png")
                if not os.path.exists(dynamic_path):
                    dynamic_path = dynamic_path.replace("STORM2", "STORM")
                dynamic_mask = Image.open(dynamic_path).convert("L").resize(self.target_size[::-1])
                dynamic_mask = to_tensor(np.array(dynamic_mask) > 0).float()
                dynamic_masks.append(dynamic_mask)

            # Get ground label, this is for flow evaluation, i.e., we use this to exclude the ground lidar points.
            if dataset_name in ["waymo"] and self.load_ground_label:
                ground_path = img_path.replace("images", "ground_label")
                ground_path = ground_path.replace("jpg", "png")
                ground = Image.open(ground_path).convert("L").resize(self.target_size[::-1])
                ground = to_tensor(np.array(ground) > 0).float()
                ground_masks.append(ground)

            camtoworld = (
                DATASETS[dataset_name]["canonical_to_flu"]
                @ world_to_canonical
                @ cam_to_world[camera][frame_idx]
                @ DATASETS[dataset_name]["opencv2dataset"]
            )

            camtoworld = to_tensor(camtoworld)
            camtoworlds.append(camtoworld)

            # intrinsics
            fx, fy, cx, cy = np.array(normalized_intrinsics[camera])
            fx = fx * self.target_size[1]
            fy = fy * self.target_size[0]
            cx = cx * self.target_size[1]
            cy = cy * self.target_size[0]
            intrinsics.append(
                torch.tensor(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ]
                ).float()
            )

            if self.load_depth or self.load_flow:
                if dataset_name == "waymo":
                    depth_path = img_path.replace("images", "depth_flows").replace("jpg", "npy")
                    depth_and_flow = np.load(depth_path)
                    if self.load_depth:
                        depth = depth_and_flow[..., 0]
                        depth = torch.tensor(depth).float()
                        depth = resize_depth(depth, self.target_size)
                        depths.append(depth)
                    if self.load_flow:
                        flow = depth_and_flow[..., 1:]
                        flow = torch.tensor(flow).float()
                        flow = resize_flow(flow, self.target_size)
                        # there must be a better way to do this: rotate the flow to the canonical view
                        flow = (
                            flow
                            @ torch.tensor(
                                (
                                    world_to_canonical
                                    @ cam_to_world[camera][frame_idx]
                                    @ np.linalg.inv(scene_json["camera_to_ego"][camera])
                                )
                            )
                            .float()[:3, :3]
                            .T
                        )
                        flows.append(flow)
                if dataset_name == "nuscenes":
                    depth_path = img_path.replace("samples", "samples_depth")
                    depth_path = depth_path.replace("sweeps", "sweeps_depth")
                    depth_path = depth_path.replace("jpg", "npy")
                    depth = np.load(depth_path)
                    depth = torch.tensor(depth).float()
                    depth = resize_depth(depth, self.target_size)
                    depths.append(depth)

                if dataset_name == "argoverse2":
                    depth_path = img_path.replace("images", "depths")
                    depth_path = depth_path.replace("jpg", "npy")
                    depth = np.load(depth_path)
                    depth = torch.tensor(depth).float()
                    depth = resize_depth(depth, self.target_size)
                    depths.append(depth)

        frame_images = torch.stack(images)
        frame_depths = torch.stack(depths) if len(depths) > 0 else None
        frame_sky_masks = torch.stack(sky_masks) if len(sky_masks) > 0 else None
        frame_flows = torch.stack(flows) if len(flows) > 0 else None
        frame_dynamic_masks = torch.stack(dynamic_masks) if len(dynamic_masks) > 0 else None
        frame_camtoworlds = torch.stack(camtoworlds)
        frame_intrinsics = torch.stack(intrinsics)
        ground_masks = torch.stack(ground_masks) if len(ground_masks) > 0 else None
        data_dict = {
            "image": frame_images,
            "camtoworld": frame_camtoworlds,
            "intrinsics": frame_intrinsics,
            "frame_idx": frame_idx,
            "depth": frame_depths,
            "sky_masks": frame_sky_masks,
            "flow": frame_flows,
            "dynamic_masks": frame_dynamic_masks,
            "ground_masks": ground_masks,
        }
        return {k: v for k, v in data_dict.items() if v is not None}

    def __getitem__(
        self, index: int, context_frame_idx: int = -1, return_all=False
    ) -> Dict[str, Any]:
        try:
            scene_json = self.annotations[index % len(self.annotations)]
            scene_id = scene_json["scene_id"]
            num_timesteps = scene_json["num_timesteps"]
            fps = scene_json["fps"]
            num_max_future_frames = int(self.timespan * fps)
            if num_max_future_frames > num_timesteps:
                num_max_future_frames = int(fps)  # make it 1 seconds
            time_in_seconds = scene_json["normalized_time"]
            if context_frame_idx < 0:
                context_frame_idx = np.random.randint(0, num_timesteps - num_max_future_frames)
            if context_frame_idx + num_max_future_frames >= num_timesteps:
                context_frame_idx = np.random.randint(0, num_timesteps - num_max_future_frames)
            assert (
                context_frame_idx + num_max_future_frames < num_timesteps
            ), f"scene_id: {scene_id}, context_frame_idx: {context_frame_idx}, num_timesteps: {num_timesteps}, num_max_future_frames: {num_max_future_frames}"

            if self.equispaced:
                context_frame_idx = np.arange(
                    context_frame_idx,
                    context_frame_idx + num_max_future_frames,
                    num_max_future_frames // self.num_context_timesteps,
                )
            else:
                context_frame_idx = np.random.choice(
                    np.arange(
                        context_frame_idx,
                        context_frame_idx + num_max_future_frames,
                    ),
                    size=self.num_context_timesteps,
                    replace=False,
                )
                context_frame_idx = sorted(context_frame_idx)

            if return_all:
                # return all frames between context_frame_idx and context_frame_idx + num_max_future_frames
                target_frame_idx = np.arange(
                    context_frame_idx[0],
                    context_frame_idx[0] + num_max_future_frames,
                )
            else:
                # randomly sample "num_target_timesteps" frames
                target_frame_idx = np.random.choice(
                    np.arange(
                        context_frame_idx[0],
                        context_frame_idx[0] + num_max_future_frames,
                    ),
                    self.num_target_timesteps,
                    replace=False,
                )
            target_frame_idx = [min(idx, num_timesteps - 1) for idx in target_frame_idx]
            target_frame_idx = sorted(target_frame_idx)

            # get context
            context_dict_list = []
            for ctx_id in context_frame_idx:
                context_dict = self.get_frame(
                    scene_json=scene_json,
                    frame_idx=ctx_id,
                    source_frame_idx=context_frame_idx[0],
                )
                context_dict["time"] = torch.tensor(
                    [time_in_seconds[ctx_id] - time_in_seconds[context_frame_idx[0]]]
                    * self.num_max_cams
                )
                context_dict_list.append(context_dict)
            if self.return_context_as_target:
                target_frame_idx = context_frame_idx
            target_dict_list = []
            for target_id in target_frame_idx:
                target_dict = self.get_frame(
                    scene_json=scene_json,
                    frame_idx=target_id,
                    source_frame_idx=context_frame_idx[0],
                )
                target_dict["time"] = torch.tensor(
                    [time_in_seconds[target_id] - time_in_seconds[context_frame_idx[0]]]
                    * self.num_max_cams
                )
                target_dict_list.append(target_dict)
            context_dict = default_collate(context_dict_list)
            target_dict = default_collate(target_dict_list)

            for k, v in context_dict.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    context_dict[k] = torch.cat([d for d in v], dim=0)
            for k, v in target_dict.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    target_dict[k] = torch.cat([d for d in v], dim=0)
            sample = {
                "context": context_dict,
                "target": target_dict,
                "scene_id": scene_id,
                "scene_name": scene_json["scene_name"],
                "width": self.target_size[1],
                "height": self.target_size[0],
                "fps": fps,
                "timespan": self.timespan,
            }
            return to_float_tensor(sample)
        except Exception as e:
            logger.info(
                f"Error in scene_id: {scene_id}, context_frame_idx: {context_frame_idx}, scene_name: {scene_json['scene_name']}"
            )
            logger.info(e)
            try:
                return self.__getitem__(index + 1, 0, return_all)
            except Exception as e:
                logger.info(e)
                return self.__getitem__(index + 1)


class STORMDatasetEval(STORMDataset):
    def __init__(
        self,
        data_root: str,
        annotation_txt_file_list: Union[str, List[str]],
        target_size: Tuple[int, int] = (225, 400),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[1, 3, 5, 6, 7] = 3,
        timespan: float = 2.0,  # how many seconds
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        load_depth: bool = True,
        load_flow: bool = False,
        load_dynamic_mask: bool = False,
        load_ground_label: bool = False,
        return_context_as_target: bool = False,
        skip_sky_mask: bool = False,
        scene_id_list: Optional[List[int]] = None,
    ):
        super(STORMDatasetEval, self).__init__(
            data_root=data_root,
            annotation_txt_file_list=annotation_txt_file_list,
            target_size=target_size,
            num_context_timesteps=num_context_timesteps,
            num_target_timesteps=num_target_timesteps,
            num_max_cams=num_max_cams,
            timespan=timespan,
            subset_indices=subset_indices,
            num_replicas=num_replicas,
            equispaced=equispaced,
            load_depth=load_depth,
            load_flow=load_flow,
            load_dynamic_mask=load_dynamic_mask,
            load_ground_label=load_ground_label,
            return_context_as_target=return_context_as_target,
            skip_sky_mask=skip_sky_mask,
        )
        val_sample_list = []
        if scene_id_list is None:
            scene_id_list = list(range(len(self.annotations)))
        for scene_id in scene_id_list:
            for start_id in range(0, 200, 20):
                if scene_id == 63 and start_id == 0:
                    continue
                val_sample_list.append((scene_id, start_id))
        self.val_sample_list = val_sample_list

    def __len__(self) -> int:
        return len(self.val_sample_list)

    def __getitem__(self, index: int):
        return super(STORMDatasetEval, self).__getitem__(
            self.val_sample_list[index][0],
            self.val_sample_list[index][1],
            return_all=True,
        )


class SingleSequenceDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        annotation_txt_file_list: Union[str, List[str]],
        target_size: Tuple[int, int] = (225, 400),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[1, 3, 5, 6, 7] = 3,
        timespan: float = 2.0,  # how many seconds
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        load_depth: bool = True,
        load_flow: bool = False,
        load_dynamic_mask: bool = False,
        load_ground_label: bool = False,
        load_scale="4",
    ):
        super().__init__()
        self.data_root = data_root
        self.target_size = target_size
        self.timespan = timespan
        self.num_context_timesteps = num_context_timesteps
        self.num_target_timesteps = num_target_timesteps
        self.num_max_cams = num_max_cams
        self.load_depth = load_depth
        self.load_flow = load_flow
        self.load_dynamic_mask = load_dynamic_mask
        self.load_ground_label = load_ground_label
        self.load_scale = load_scale
        if isinstance(annotation_txt_file_list, str):
            annotation_txt_file_list = [annotation_txt_file_list]
        scene_list = []
        for annotation_txt_file in annotation_txt_file_list:
            with open(annotation_txt_file, "r") as f:
                scene_list += f.readlines()
        annotation_paths = [line.strip() for line in scene_list]
        if subset_indices is not None:
            annotation_paths = [annotation_paths[i] for i in subset_indices]
        self.annotations = []
        for annotation_path in annotation_paths:
            with open(os.path.join(data_root, annotation_path), "r") as f:
                self.annotations.append(json.load(f))
        logger.info(f"Loaded {len(self.annotations)} annotations.")
        self.num_replicas = num_replicas
        if self.num_replicas > 1:
            self.annotations *= self.num_replicas
        self.equispaced = equispaced
        self.img_transformation = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def get_frame(
        self,
        scene_json: Dict[str, Any],
        frame_idx: int,
        source_frame_idx: int = -1,
    ) -> Dict[str, Any]:
        normalized_intrinsics = scene_json["normalized_intrinsics"]
        dataset_name = scene_json["dataset"]
        cam_to_world = scene_json["camera_to_world"]

        images, depths, sky_masks, flows = [], [], [], []
        camtoworlds, intrinsics = [], []
        dynamic_masks, ground_masks = [], []

        if source_frame_idx < 0:
            source_frame_idx = frame_idx

        camera_list = DATASET_DICT[dataset_name]["camera_list"][self.num_max_cams]
        ref_camera_name = DATASET_DICT[dataset_name]["ref_camera"]
        world_to_canonical = np.linalg.inv(cam_to_world[ref_camera_name][source_frame_idx])

        for camera in camera_list:
            img_relative_path = scene_json["relative_image_path"][camera][frame_idx]
            if dataset_name in ["waymo", "nuscenes", "argoverse2", "argoverse"]:
                if self.load_scale == "4":
                    img_relative_path = img_relative_path.replace("images", f"images_4")
                    img_relative_path = img_relative_path.replace("sweeps", f"sweeps_4")
                    img_relative_path = img_relative_path.replace("samples", f"samples_4")

            # Get RGB
            img_path = os.path.join(self.data_root, "datasets", dataset_name, img_relative_path)
            img = Image.open(img_path).convert("RGB")
            img = self.img_transformation(img)
            images.append(img)

            # Get sky mask
            if dataset_name in ["waymo", "nuscenes", "argoverse2"]:
                # if dataset_name in ["waymo", "nuscenes"]:
                if dataset_name == "nuscenes":
                    sky_path = img_path.replace("samples", "samples_sky_mask")
                    sky_path = sky_path.replace("sweeps", "sweeps_sky_mask")
                elif dataset_name == "waymo":
                    sky_path = img_path.replace("images", "sky_masks")
                elif dataset_name == "argoverse2":
                    sky_path = img_path.replace("images_4", "sky_masks_512")
                sky_path = sky_path.replace("jpg", "png")
                try:
                    new_sky_path = sky_path.replace("STORM2", "STORM_masks")
                    sky = Image.open(new_sky_path).convert("L").resize(self.target_size[::-1])
                except FileNotFoundError:
                    sky = Image.open(sky_path).convert("L").resize(self.target_size[::-1])
                sky = to_tensor(np.array(sky) > 0).float()
                sky_masks.append(sky)

            # Get dynamic mask, this is for dynamic region evaluation.
            if dataset_name in ["waymo"] and self.load_dynamic_mask:
                dynamic_path = dynamic_path.replace("images_4", "dynamic_masks/all")
                dynamic_path = dynamic_path.replace("jpg", "png")
                if not os.path.exists(dynamic_path):
                    dynamic_path = dynamic_path.replace("STORM2", "STORM")
                dynamic_mask = Image.open(dynamic_path).convert("L").resize(self.target_size[::-1])
                dynamic_mask = to_tensor(np.array(dynamic_mask) > 0).float()
                dynamic_masks.append(dynamic_mask)

            # Get ground label, this is for flow evaluation, i.e., we use this to exclude the ground lidar points.
            if dataset_name in ["waymo"] and self.load_ground_label:
                ground_path = img_path.replace("images", "ground_label")
                ground_path = ground_path.replace("jpg", "png")
                ground = Image.open(ground_path).convert("L").resize(self.target_size[::-1])
                ground = to_tensor(np.array(ground) > 0).float()
                ground_masks.append(ground)

            camtoworld = (
                DATASETS[dataset_name]["canonical_to_flu"]
                @ world_to_canonical
                @ cam_to_world[camera][frame_idx]
                @ DATASETS[dataset_name]["opencv2dataset"]
            )

            camtoworld = to_tensor(camtoworld)
            camtoworlds.append(camtoworld)

            # intrinsics
            fx, fy, cx, cy = np.array(normalized_intrinsics[camera])
            fx = fx * self.target_size[1]
            fy = fy * self.target_size[0]
            cx = cx * self.target_size[1]
            cy = cy * self.target_size[0]
            intrinsics.append(
                torch.tensor(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ]
                ).float()
            )

            if self.load_depth or self.load_flow:
                if dataset_name == "waymo":
                    depth_path = img_path.replace("images", "depth_flows").replace("jpg", "npy")
                    depth_and_flow = np.load(depth_path)
                    if self.load_depth:
                        depth = depth_and_flow[..., 0]
                        depth = torch.tensor(depth).float()
                        depth = resize_depth(depth, self.target_size)
                        depths.append(depth)
                    if self.load_flow:
                        flow = depth_and_flow[..., 1:]
                        flow = torch.tensor(flow).float()
                        flow = resize_flow(flow, self.target_size)
                        # there must be a better way to do this: rotate the flow to the canonical view
                        flow = (
                            flow
                            @ torch.tensor(
                                (
                                    world_to_canonical
                                    @ cam_to_world[camera][frame_idx]
                                    @ np.linalg.inv(scene_json["camera_to_ego"][camera])
                                )
                            )
                            .float()[:3, :3]
                            .T
                        )
                        flows.append(flow)
                if dataset_name == "nuscenes":
                    depth_path = img_path.replace("samples", "samples_depth")
                    depth_path = depth_path.replace("sweeps", "sweeps_depth")
                    depth_path = depth_path.replace("jpg", "npy")
                    depth = np.load(depth_path)
                    depth = torch.tensor(depth).float()
                    depth = resize_depth(depth, self.target_size)
                    depths.append(depth)

                if dataset_name == "argoverse2":
                    depth_path = img_path.replace("images", "depths")
                    depth_path = depth_path.replace("jpg", "npy")
                    depth = np.load(depth_path)
                    depth = torch.tensor(depth).float()
                    depth = resize_depth(depth, self.target_size)
                    depths.append(depth)

        frame_images = torch.stack(images)
        frame_depths = torch.stack(depths) if len(depths) > 0 else None
        frame_sky_masks = torch.stack(sky_masks) if len(sky_masks) > 0 else None
        frame_flows = torch.stack(flows) if len(flows) > 0 else None
        frame_dynamic_masks = torch.stack(dynamic_masks) if len(dynamic_masks) > 0 else None
        frame_camtoworlds = torch.stack(camtoworlds)
        frame_intrinsics = torch.stack(intrinsics)
        ground_masks = torch.stack(ground_masks) if len(ground_masks) > 0 else None
        data_dict = {
            "image": frame_images,
            "camtoworld": frame_camtoworlds,
            "intrinsics": frame_intrinsics,
            "frame_idx": frame_idx,
            "depth": frame_depths,
            "sky_masks": frame_sky_masks,
            "flow": frame_flows,
            "dynamic_masks": frame_dynamic_masks,
            "ground_masks": ground_masks,
        }
        return {k: v for k, v in data_dict.items() if v is not None}

    def get_segment(self, index: int, context_frame_idx: int = -1, return_all=False):
        scene_json = self.annotations[index % len(self.annotations)]
        scene_id = scene_json["scene_id"]
        num_timesteps = scene_json["num_timesteps"]
        fps = scene_json["fps"]
        num_max_future_frames = int(self.timespan * fps)
        if num_max_future_frames > num_timesteps:
            num_max_future_frames = int(fps)  # make it 1 seconds
        time_in_seconds = scene_json["normalized_time"]
        if context_frame_idx < 0:
            context_frame_idx = np.random.randint(0, num_timesteps - num_max_future_frames)
        if context_frame_idx + num_max_future_frames >= num_timesteps:
            context_frame_idx = np.random.randint(0, num_timesteps - num_max_future_frames)
        assert (
            context_frame_idx + num_max_future_frames < num_timesteps
        ), f"scene_id: {scene_id}, context_frame_idx: {context_frame_idx}, num_timesteps: {num_timesteps}, num_max_future_frames: {num_max_future_frames}"

        if self.equispaced:
            context_frame_idx = np.arange(
                context_frame_idx,
                context_frame_idx + num_max_future_frames,
                num_max_future_frames // self.num_context_timesteps,
            )
        else:
            context_frame_idx = np.random.choice(
                np.arange(
                    context_frame_idx,
                    context_frame_idx + num_max_future_frames,
                ),
                size=self.num_context_timesteps,
                replace=False,
            )
            context_frame_idx = sorted(context_frame_idx)

        if return_all:
            # return all frames between context_frame_idx and context_frame_idx + num_max_future_frames
            target_frame_idx = np.arange(
                context_frame_idx[0],
                context_frame_idx[0] + num_max_future_frames,
            )
        else:
            # randomly sample "num_target_timesteps" frames
            target_frame_idx = np.random.choice(
                np.arange(
                    context_frame_idx[0],
                    context_frame_idx[0] + num_max_future_frames,
                ),
                self.num_target_timesteps,
                replace=False,
            )
        target_frame_idx = [min(idx, num_timesteps - 1) for idx in target_frame_idx]
        target_frame_idx = sorted(target_frame_idx)

        # get context
        context_dict_list = []
        for ctx_id in context_frame_idx:
            context_dict = self.get_frame(
                scene_json=scene_json,
                frame_idx=ctx_id,
                source_frame_idx=context_frame_idx[0],
            )
            context_dict["time"] = torch.tensor(
                [time_in_seconds[ctx_id] - time_in_seconds[context_frame_idx[0]]]
                * self.num_max_cams
            )
            context_dict_list.append(context_dict)
        target_dict_list = []
        for target_id in target_frame_idx:
            target_dict = self.get_frame(
                scene_json=scene_json,
                frame_idx=target_id,
                source_frame_idx=context_frame_idx[0],
            )
            target_dict["time"] = torch.tensor(
                [time_in_seconds[target_id] - time_in_seconds[context_frame_idx[0]]]
                * self.num_max_cams
            )
            target_dict_list.append(target_dict)
        context_dict = default_collate(context_dict_list)
        target_dict = default_collate(target_dict_list)

        for k, v in context_dict.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                context_dict[k] = torch.cat([d for d in v], dim=0)
        for k, v in target_dict.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                target_dict[k] = torch.cat([d for d in v], dim=0)
        sample = {
            "context": context_dict,
            "target": target_dict,
            "scene_id": scene_id,
            "scene_name": scene_json["scene_name"],
            "width": self.target_size[1],
            "height": self.target_size[0],
            "fps": fps,
            "timespan": self.timespan,
        }
        return to_float_tensor(sample)

    def __getitem__(self, index: int, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        scene_json = self.annotations[index % len(self.annotations)]
        num_timesteps = scene_json["num_timesteps"]
        if end_index < 0:
            end_index = num_timesteps
        segment_data = []
        cam_to_world = scene_json["camera_to_world"]
        ref_camera_name = DATASET_DICT[scene_json["dataset"]]["ref_camera"]
        world_to_canonical = np.linalg.inv(cam_to_world[ref_camera_name][start_index])
        for start_id in trange(start_index, end_index, 20):
            segment_dict = self.get_segment(
                index=index, context_frame_idx=start_id, return_all=True
            )
            # compute the relative transformation from a start_id to the first frame
            current_world = cam_to_world[ref_camera_name][start_id]
            segment_to_ref = world_to_canonical @ current_world
            segment_dict["segment_to_ref"] = to_float_tensor(segment_to_ref)
            segment_data.append(segment_dict)
        return segment_data
