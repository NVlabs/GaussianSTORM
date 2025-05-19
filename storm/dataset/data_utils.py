from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def forward_pose(pose: torch.tensor, transform: torch.tensor, inv=False) -> torch.tensor:
    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    if inv:
        return (torch.transpose(rotation, -1, -2) * (transform - translation).unsqueeze(-2)).sum(-1)
    else:
        return (rotation * transform.unsqueeze(-2)).sum(-1) + translation


def get_path_front_left_lift_then_spiral_forward(
    pose_ref: np.ndarray,  # [N0,4,4], Original ego pose; will generate trajectories along this
    num_frames: int,  # Number of frames / waypoints to generate
    duration_frames: int = 100,  # Number of frames per round / cycle
    # Spiral configs
    up_max: float = 0.3,
    up_min: float = -0.3,
    left_max: float = 2.0,
    left_min: float = -2.0,
    elongation: float = 1.0,
    first_with_forward=False,
    # Frontal direction vector of OpenCV
    forward_vec: np.ndarray = np.array([0.0, 0.0, 1.0]),
    # Frontal direction vector of OpenCV
    up_vec: np.ndarray = np.array([0.0, -1.0, 0.0]),
    # Frontal direction vector of OpenCV
    left_vec: np.ndarray = np.array([-1.0, 0.0, 0.0]),
) -> np.ndarray:
    """
    First lift ego in the front left direction, then do spiral forward
    """
    pose_ref = torch.tensor(pose_ref, dtype=torch.float32, device="cpu")
    n_rots = num_frames / duration_frames
    track_ref = pose_ref[..., :3, 3]
    rot_ref = pose_ref[..., :3, :3]
    # NOTE: Convert from observer's coords dir to world coords dir;
    #       If the pose_ref is the pose of camera, `forward/up/left_vec` can be used as-is;
    #           (since the cameras are already OpenCV cameras if dataio's xxx_dataset.py is correctly implemented)
    #       otherwise, you should specify the coords dir vector of your given pose_ref.
    forward_vecs_ref = (
        forward_pose(pose_ref, torch.tensor(forward_vec, dtype=torch.float32)) - track_ref
    )
    up_vecs_ref = forward_pose(pose_ref, torch.tensor(up_vec, dtype=torch.float32)) - track_ref
    left_vecs_ref = forward_pose(pose_ref, torch.tensor(left_vec, dtype=torch.float32)) - track_ref

    forward_vecs_ref = forward_vecs_ref.numpy()
    up_vecs_ref = up_vecs_ref.numpy()
    left_vecs_ref = left_vecs_ref.numpy()
    track_ref = track_ref.numpy()
    rot_ref = rot_ref.numpy()

    nvs_seqs = []

    verti_radius = (up_max - up_min) / 2.0
    up_offset = (up_max + up_min) / 2.0
    horiz_radius = (left_max - left_min) / 2.0
    left_offset = (left_max + left_min) / 2.0
    assert (verti_radius >= -1e-5) and (horiz_radius >= -1e-5)

    # ----------------------------------------
    # ---- First: lift up & left
    # ----------------------------------------
    first_frames = int(0.1 * duration_frames) + 1
    remain_frames = num_frames - first_frames
    pace = np.linalg.norm(track_ref[0] - track_ref[-1]) * elongation
    forward_1st = ((first_frames / num_frames) * pace) if first_with_forward else 0

    track = (
        track_ref[0]
        + np.linspace(0, verti_radius + up_offset, first_frames)[..., None] * up_vecs_ref[0]
        + np.linspace(0, horiz_radius + left_offset, first_frames)[..., None] * left_vecs_ref[0]
        + np.linspace(0, forward_1st, first_frames)[..., None] * forward_vecs_ref[0]
    )
    pose = np.eye(4)[None, ...].repeat(first_frames, 0)
    pose[:, :3, 3] = track
    pose[:, :3, :3] = rot_ref[0]
    nvs_seqs.append(pose)

    # ----------------------------------------
    # ---- Then:   Sprial forward
    # ----------------------------------------
    w = np.linspace(0, 1, remain_frames)  # [0->1], data key time
    t = np.arange(len(track_ref)) / (
        len(track_ref) - 1
    )  # [0->1], render key time (could be extended)
    track_interp = interpolate.interp1d(t, track_ref, axis=0, fill_value="extrapolate")
    up_vec_interp = interpolate.interp1d(t, up_vecs_ref, axis=0, fill_value="extrapolate")
    left_vec_interp = interpolate.interp1d(t, left_vecs_ref, axis=0, fill_value="extrapolate")

    up_vecs_all = up_vec_interp(w * elongation)
    left_vecs_all = left_vec_interp(w * elongation)
    # ---- Base: left * [1], up * [1]
    track_base_all = (
        track_interp(w * elongation)
        + (up_offset + verti_radius) * up_vecs_all
        + (left_offset + horiz_radius) * left_vecs_all
        + forward_1st * forward_vecs_ref[None, 0]
    )

    # up_vecs_all = np.percentile(up_vecs_ref, w*100, 0)
    # left_vecs_all = np.percentile(left_vecs_ref, w*100, 0)
    # track_base_all = np.percentile(track_ref, w*100, 0) + (up_max+up_offset) * up_vecs_all + (left_max+left_offset) * left_vecs_all

    key_rots = R.from_matrix(rot_ref)
    rot_slerp = Slerp(t, key_rots)
    if elongation > 1:
        mask = (w * elongation) < 1.0
        rot_base_all = np.eye(3)[None, ...].repeat(remain_frames, 0)
        rot_base_all[mask] = rot_slerp(w[mask]).as_matrix()
        rot_base_all[~mask] = rot_ref[-1]
    else:
        rot_base_all = rot_slerp(w).as_matrix()

    rads = np.linspace(0, remain_frames / duration_frames * np.pi * 2.0, remain_frames)
    # ---- Spiral:
    # left: [0, -1, -2, -1, 0] + base: [1] -> [1, 0, -1, 0, 1]
    # up: [0, 1, 0, -1, 0] + base [1] -> [1, 2, 1, 0, 1]
    track = (
        track_base_all
        + (np.cos(rads) - 1)[..., None] * horiz_radius * left_vecs_all
        + (np.sin(rads))[..., None] * verti_radius * up_vecs_all
    )
    pose = np.eye(4)[None, ...].repeat(remain_frames, 0)
    pose[:, :3, 3] = track
    pose[:, :3, :3] = rot_base_all
    nvs_seqs.append(pose)

    render_pose_all = np.concatenate(nvs_seqs, 0)

    return render_pose_all


def to_tensor(x: Union[np.ndarray, List, Tuple]) -> torch.Tensor:
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    return x


def to_float_tensor(d):
    if isinstance(d, dict):
        return {k: to_float_tensor(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_float_tensor(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.float()
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float()
    else:
        return d


def to_batch_tensor(d):
    if isinstance(d, dict):
        return {k: to_batch_tensor(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_batch_tensor(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.unsqueeze(0)
    else:
        return d


def resize_depth(depth, target_size):
    height, width = depth.shape[-2:]
    if (height, width) == target_size:
        return depth
    if len(depth.shape) == 2:
        depth = depth[None, None, ...]
    elif len(depth.shape) == 3:
        depth = depth[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width

    if kernel_size_h > 0 and kernel_size_w > 0:
        depth = F.max_pool2d(
            depth,
            kernel_size=(kernel_size_h, kernel_size_w),
        )
    depth = F.interpolate(depth, size=target_size, mode="nearest")
    return depth.squeeze()


def resize_flow(flow, target_size):
    height, width = flow.shape[-3:-1]
    if (height, width) == target_size:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width
    # flows have direction, so we can't just use max_pool2d.
    # otherwise the direction will be wrong, e.g., max_pool2d([0, -1]) = [0]
    # previous_valid_flow = (torch.norm(flow, p=2, dim=-1) > 0.5).float().sum()
    flow[torch.norm(flow, p=2, dim=-1) < 0.5] = -100000
    if kernel_size_h > 0 and kernel_size_w > 0:
        flow = F.max_pool2d(
            flow.permute(0, 3, 1, 2),
            kernel_size=(kernel_size_h, kernel_size_w),
        )
        flow = F.interpolate(flow, size=target_size, mode="nearest")
    else:
        flow = F.interpolate(flow.permute(0, 3, 1, 2), size=target_size, mode="nearest")
    flow = flow.permute(0, 2, 3, 1)
    flow[torch.norm(flow, p=2, dim=-1) > 1000] = 0
    # new_valid_flow = (torch.norm(flow, p=2, dim=-1) > 0.5).float().sum()
    return flow.squeeze()


def prepare_inputs_and_targets(data_dict, device=torch.device("cuda"), v=3, timespan=2.0):
    assert data_dict["context"]["image"].dim() == 5, "need to be b, tv, c, h, w"
    b, tv, c, h, w = data_dict["context"]["image"].shape
    context_t = tv // v
    target_t = data_dict["target"]["image"].shape[1] // v
    input_dict = {
        "context_image": data_dict["context"]["image"].reshape(b, context_t, v, c, h, w),
        # targets to render
        "context_camtoworlds": data_dict["context"]["camtoworld"].reshape(b, context_t, v, 4, 4),
        "context_intrinsics": data_dict["context"]["intrinsics"].reshape(b, context_t, v, 3, 3),
        "target_camtoworlds": data_dict["target"]["camtoworld"].reshape(b, target_t, v, 4, 4),
        "target_intrinsics": data_dict["target"]["intrinsics"].reshape(b, target_t, v, 3, 3),
    }
    if "depth" in data_dict["context"]:
        depth_h, depth_w = data_dict["context"]["depth"].shape[-2:]
        input_dict["context_depth"] = data_dict["context"]["depth"].reshape(
            b, context_t, v, depth_h, depth_w
        )
    if "flow" in data_dict["context"]:
        flow_h, flow_w = data_dict["context"]["flow"].shape[-3:-1]
        input_dict["context_flow"] = data_dict["context"]["flow"].reshape(
            b, context_t, v, flow_h, flow_w, 3
        )
    if "time" in data_dict["context"]:
        input_dict["context_time"] = (
            data_dict["context"]["time"].reshape(b, context_t, v) / timespan
        )
    if "time" in data_dict["target"]:
        input_dict["target_time"] = data_dict["target"]["time"].reshape(b, target_t, v) / timespan
    if "sky_masks" in data_dict["context"]:
        input_dict["context_sky_masks"] = data_dict["context"]["sky_masks"].reshape(
            b, context_t, v, h, w
        )
    target_dict = {
        "target_image": data_dict["target"]["image"].reshape(b, target_t, v, 3, h, w),
    }

    if "depth" in data_dict["target"]:
        target_dict["target_depth"] = data_dict["target"]["depth"].reshape(
            b, target_t, v, depth_h, depth_w
        )
    if "flow" in data_dict["target"]:
        target_dict["target_flow"] = data_dict["target"]["flow"].reshape(
            b, target_t, v, depth_h, depth_w, 3
        )
        target_dict["context_flow"] = data_dict["context"]["flow"].reshape(b, context_t, v, h, w, 3)
    if "flow" in data_dict["context"]:
        target_dict["context_flow"] = data_dict["context"]["flow"].reshape(
            b, context_t, v, depth_h, depth_w, 3
        )
    if "sky_masks" in data_dict["target"]:
        target_dict["target_sky_masks"] = data_dict["target"]["sky_masks"].reshape(
            b, target_t, v, h, w
        )
    if "sky_masks" in data_dict["context"]:
        target_dict["context_sky_masks"] = data_dict["context"]["sky_masks"].reshape(
            b, context_t, v, h, w
        )
    if "dynamic_masks" in data_dict["target"]:
        target_dict["target_dynamic_masks"] = data_dict["target"]["dynamic_masks"].reshape(
            b, target_t, v, h, w
        )
    if "ground_masks" in data_dict["target"]:
        target_dict["target_ground_masks"] = data_dict["target"]["ground_masks"].reshape(
            b, target_t, v, h, w
        )

    input_dict["context_frame_idx"] = data_dict["context"]["frame_idx"]
    target_dict["target_frame_idx"] = data_dict["target"]["frame_idx"]
    input_dict = {k: v.to(device) for k, v in input_dict.items()}
    target_dict = {k: v.to(device) for k, v in target_dict.items()}
    input_dict["timespan"] = timespan
    input_dict["scene_id"] = data_dict["scene_id"]
    input_dict["scene_name"] = data_dict["scene_name"]
    input_dict["height"], input_dict["width"] = h, w
    return input_dict, target_dict


def prepare_inputs_and_targets_novel_view(data_dict, device=torch.device("cpu")):
    raise NotImplementedError("Legacy code, not tested for a while. But you can use it as a reference.")

    assert data_dict["context"]["image"].dim() == 5, "need to be b, tv, c, h, w"
    b, tv, c, h, w = data_dict["context"]["image"].shape
    v = int(data_dict["context"]["num_views"].flatten()[0].item())
    context_t = tv // v
    target_t = data_dict["target"]["image"].shape[1] // v
    # move the camera to right by 1 meter:
    # freeze time
    # cam_to_worlds = data_dict["target"]["camtoworld"].view(b, target_t, v, 4, 4)
    # cam_to_worlds0 = cam_to_worlds[:, 0:1]
    # cam_to_worlds = cam_to_worlds0.expand(-1, target_t, -1, -1, -1)
    # data_dict["target"]["camtoworld"] = cam_to_worlds.reshape(b, target_t * v, 4, 4)
    ### novel view
    cam_to_worlds = data_dict["target"]["camtoworld"].view(b, target_t, v, 4, 4)
    left_cam_to_worlds = cam_to_worlds[:, :, 0].numpy()
    center_cam_to_worlds = cam_to_worlds[:, :, 1].numpy()
    right_cam_to_worlds = cam_to_worlds[:, :, 2].numpy()
    new_left_cam_to_worlds = []
    new_center_cam_to_worlds = []
    new_right_cam_to_worlds = []
    for bix in range(left_cam_to_worlds.shape[0]):
        new_path = get_path_front_left_lift_then_spiral_forward(
            left_cam_to_worlds[bix],
            num_frames=target_t,
            duration_frames=20,
        )
        new_left_cam_to_worlds.append(torch.from_numpy(new_path))
        new_path = get_path_front_left_lift_then_spiral_forward(
            center_cam_to_worlds[bix],
            num_frames=target_t,
            duration_frames=20,
        )
        new_center_cam_to_worlds.append(torch.from_numpy(new_path))
        new_path = get_path_front_left_lift_then_spiral_forward(
            right_cam_to_worlds[bix],
            num_frames=target_t,
            duration_frames=20,
        )
        new_right_cam_to_worlds.append(torch.from_numpy(new_path))
    new_left_cam_to_worlds = torch.stack(new_left_cam_to_worlds, dim=0)
    new_center_cam_to_worlds = torch.stack(new_center_cam_to_worlds, dim=0)
    new_right_cam_to_worlds = torch.stack(new_right_cam_to_worlds, dim=0)
    new_cam_to_worlds = torch.stack(
        [new_left_cam_to_worlds, new_center_cam_to_worlds, new_right_cam_to_worlds],
        dim=2,
    )
    data_dict["target"]["camtoworld"] = new_cam_to_worlds.reshape(b, target_t * v, 4, 4).to(
        data_dict["target"]["camtoworld"]
    )
    ### novel view
    input_dict = {
        "context_image": data_dict["context"]["image"].reshape(b, context_t, v, c, h, w),
        # targets to render
        "context_camtoworlds": data_dict["context"]["camtoworld"].reshape(b, context_t, v, 4, 4),
        "context_intrinsics": data_dict["context"]["intrinsics"].reshape(b, context_t, v, 3, 3),
        "target_camtoworlds": data_dict["target"]["camtoworld"].reshape(b, target_t, v, 4, 4),
        "target_intrinsics": data_dict["target"]["intrinsics"].reshape(b, target_t, v, 3, 3),
    }
    if "depth" in data_dict["context"]:
        input_dict["context_depth"] = data_dict["context"]["depth"].reshape(b, context_t, v, h, w)
    if "flow" in data_dict["context"]:
        input_dict["context_flow"] = data_dict["context"]["flow"].reshape(b, context_t, v, h, w, 3)
    if "time" in data_dict["context"]:
        input_dict["context_time"] = data_dict["context"]["time"].reshape(b, context_t)
    if "time" in data_dict["target"]:
        input_dict["target_time"] = data_dict["target"]["time"].reshape(b, target_t)
    if "sky_masks" in data_dict["context"]:
        input_dict["context_sky_masks"] = data_dict["context"]["sky_masks"].reshape(
            b, context_t, v, h, w
        )

    target_dict = {
        "target_image": data_dict["target"]["image"].reshape(b, target_t, v, 3, h, w),
    }

    if "depth" in data_dict["target"]:
        target_dict["target_depth"] = data_dict["target"]["depth"].reshape(b, target_t, v, h, w)
    if "depth" in data_dict["context"]:
        target_dict["context_depth"] = data_dict["context"]["depth"].reshape(b, context_t, v, h, w)
    if "flow" in data_dict["target"]:
        target_dict["target_flow"] = data_dict["target"]["flow"].reshape(b, target_t, v, h, w, 3)
    if "sky_masks" in data_dict["target"]:
        target_dict["target_sky_masks"] = data_dict["target"]["sky_masks"].reshape(
            b, target_t, v, h, w
        )
    if "sky_masks" in data_dict["context"]:
        target_dict["context_sky_masks"] = data_dict["context"]["sky_masks"].reshape(
            b, context_t, v, h, w
        )
    if "dynamic_masks" in data_dict["target"]:
        target_dict["target_dynamic_masks"] = data_dict["target"]["dynamic_masks"].reshape(
            b, target_t, v, h, w
        )
    if "ground_masks" in data_dict["target"]:
        target_dict["target_ground_masks"] = data_dict["target"]["ground_masks"].reshape(
            b, target_t, v, h, w
        )

    target_dict["target_frame_idx"] = data_dict["target"]["frame_idx"]
    input_dict = {k: v.to(device) for k, v in input_dict.items()}
    target_dict = {k: v.to(device) for k, v in target_dict.items()}
    try:
        input_dict["height"] = data_dict["height"][0].item()
        input_dict["width"] = data_dict["width"][0].item()
        if "timespan" in data_dict:
            input_dict["timespan"] = data_dict["timespan"].flatten()[0].item()
    except:
        input_dict["height"] = data_dict["height"]
        input_dict["width"] = data_dict["width"]
        if "timespan" in data_dict:
            input_dict["timespan"] = data_dict["timespan"]
    input_dict["context_frame_idx"] = data_dict["context"]["frame_idx"]
    input_dict["scene_id"] = data_dict["scene_id"]
    input_dict["scene_name"] = data_dict["scene_name"]
    return input_dict, target_dict
