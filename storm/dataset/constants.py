import numpy as np

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

opencv2waymo = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

DATASETS = {
    "waymo": {"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)},
    "nuscenes": {"opencv2dataset": np.eye(4), "canonical_to_flu": opencv2waymo},
    "argoverse2": {"opencv2dataset": np.eye(4), "canonical_to_flu": opencv2waymo},
}

DATASET_DICT = {
    "waymo": {
        "size": [160, 240],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": "scene_list/waymo_train.txt",
        "annotation_txt_file_val": "scene_list/waymo_val.txt",
        "camera_list": {
            1: ["0"],
            3: ["1", "0", "2"],
            5: ["3", "1", "0", "2", "4"],
            6: ["3", "1", "0", "2", "4"],  # capped at 5
            7: ["3", "1", "0", "2", "4"],
        },
        "ref_camera": "0",
    },
    "nuscenes": {
        "size": [160, 288],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": "scene_list/nuscenes_train.txt",
        "annotation_txt_file_val": "scene_list/nuscenes_val.txt",
        "camera_list": {
            1: ["CAM_FRONT"],
            3: ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
            5: [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ],
            6: [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ],
            7: [
                "CAM_FRONT_LEFT",
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_BACK_RIGHT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
            ],
        },
        "ref_camera": "CAM_FRONT",
    },
    "argoverse2": {
        "size": [192, 256],
        "temporal": True,
        "num_context_timesteps": 4,
        "num_target_timesteps": 4,
        "annotation_txt_file_train": "scene_list/argoverse2_train.txt",
        "annotation_txt_file_val": "scene_list/argoverse2_val.txt",
        "camera_list": {
            1: ["0"],
            3: ["1", "0", "2"],
            5: ["3", "1", "0", "2", "4"],
            6: ["3", "1", "0", "2", "4"],
            7: ["5", "3", "1", "0", "2", "4", "6"],
        },
        "ref_camera": "0",
    },
    "rel10k": {
        "size": [160, 296],
        "temporal": False,
        "num_context_timesteps": 2,
        "num_target_timesteps": 8,
        "annotation_txt_file_train": "scene_list/rel10k_train.txt",
        "annotation_txt_file_val": "scene_list/rel10k_val.txt",
        "batch_size_scale": 6,
        "camera_list": {
            1: ["0"],
            3: ["0"],
            5: ["0"],
            6: ["0"],
            7: ["0"],
        },
    },
    "dl3dv": {
        "size": [160, 288],
        "temporal": False,
        "num_context_timesteps": 2,
        "num_target_timesteps": 8,
        "annotation_txt_file_train": "scene_list/dl3dv_train.txt",
        "annotation_txt_file_val": "scene_list/dl3dv_val.txt",
        "batch_size_scale": 6,
        "camera_list": {
            1: ["0"],
            3: ["0"],
            5: ["0"],
            6: ["0"],
            7: ["0"],
        },
    },
}
