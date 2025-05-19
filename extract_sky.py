import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.folder import default_loader
from storm.dataset.constants import IMGNET_MEAN, IMGNET_STD
from third_party.depth_anything_v2.dpt import DepthAnythingV2

class ListDataset:
    """Dataset class that loads images from a list of file paths."""
    
    def __init__(
        self,
        data_list: str,
        transform: Optional[transforms.Compose] = None,
        return_path: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_list: Path to text file containing image paths
            transform: Optional transforms to apply to images
            return_path: Whether to return image paths along with images
        """
        self.transform = transform
        self.return_path = return_path
        self.loader = default_loader
        self.samples = self._load_samples(data_list)

    def _load_samples(self, data_list: str) -> list:
        """Load image paths from the data list file."""
        samples = []
        with open(data_list, "r") as f:
            for line in f:
                file_path = line.strip()
                samples.append(file_path)
        return samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """Get an image and optionally its path."""
        img_pth = self.samples[index]
        try:
            img = self.loader(img_pth)
        except Exception as e:
            print(f"Error loading '{img_pth}': {e}")
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:
            img = self.transform(img)
            
        to_return = [img]
        if self.return_path:
            to_return.append(img_pth)
        return tuple(to_return) if len(to_return) > 1 else to_return[0]

    def __len__(self) -> int:
        return len(self.samples)

def get_args_parser() -> argparse.ArgumentParser:
    """Get command line argument parser."""
    parser = argparse.ArgumentParser("extract sky masks", add_help=False)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--file_list", type=str, default="file_list.txt")
    parser.add_argument("--depth_ckpt", type=str, default="ckpts/depth_anything_v2_vitl.pth")

    return parser


def setup_model(device: torch.device, depth_ckpt: str) -> DepthAnythingV2:
    """Initialize and setup the depth estimation model."""
    depthv2 = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    depthv2.load_state_dict(torch.load(depth_ckpt, map_location="cpu"))
    depthv2 = depthv2.eval().to(device)
    for param in depthv2.parameters():
        param.requires_grad = False
    return depthv2

@torch.no_grad()
def get_sky_mask(dataloader: torch.utils.data.DataLoader,  depthv2: DepthAnythingV2) -> None:
    """
    Extract sky masks from images using depth estimation.
    
    Args:
        dataloader: DataLoader containing images
        depthv2: Depth estimation model
    """
    torch.cuda.empty_cache()
    device = next(depthv2.parameters()).device
    
    pbar = tqdm(dataloader, desc=f"Extracting sky masks")
    for samples, paths in pbar:
        samples = samples.to(device)
        # predict depth using the model
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs = depthv2(samples)
        # identify sky regions (depth = 0)
        sky_masks = (outputs == 0).float()
        sky_masks = sky_masks.cpu().numpy()
        # save masks
        for i in range(len(sky_masks)):
            mask = sky_masks[i]
            mask = (mask * 255).astype(np.uint8)
            raise NotImplementedError("please specify the target path by yourself and comment this line")
            
            # e.g:
            tgt_pth = paths[i].replace("images", "sky_masks_518")
            tgt_pth = tgt_pth.replace("jpg", "png")
            
            # ensure directory exists and save mask
            os.makedirs(os.path.dirname(tgt_pth), exist_ok=True)
            Image.fromarray(mask).save(tgt_pth)

def main(args: argparse.Namespace) -> None:
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_transformation = transforms.Compose([
        transforms.Resize([518, 518], interpolation=Image.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
    ])
    # setup dataloaders
    dataset = ListDataset(data_list=args.file_list, transform=img_transformation, return_path=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )
    # setup model
    depthv2 = setup_model(device, args.depth_ckpt)
    get_sky_mask(data_loader, depthv2)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
