<div align="center">

# **STORM**  
### Spatio-Temporal Reconstruction Model for Large-Scale Outdoor Scenes  

[Project Page](https://jiawei-yang.github.io/STORM/) • [arXiv Paper](https://arxiv.org/abs/2501.00602)
</div>

---

## Highlights
* **Fast, feed-forward, and self-supervised** dynamic scene reconstruction from sparse multi-view sequences
* Learns **3D Gaussian** *and* **scene flow** jointly; supports real-time rendering (once Gaussians are generated) and motion segmentation
* Outperforms per‑scene optimization and other generalizable models by **+4 dB PSNR** on dynamic regions while being significantly faster

---

## Installation
> Tested with **CUDA 12.1**, **PyTorch 2.3** and an NVIDIA **A100**  
> Replace the CUDA/PyTorch versions as needed for your environment

```bash
# clone project
git clone https://github.com/NVlabs/GaussianSTORM.git
cd GaussianSTORM

# create conda environment
conda create -n storm python=3.10 -y
conda activate storm

# install python dependencies
pip install -r requirements.txt

# install gsplat (for batch-wise rendering support)
pip install git+https://github.com/nerfstudio-project/gsplat.git@2b0de894232d21e8963179a7bbbd315f27c52c9c
#   └─ if the above fails, drop the commit hash:
#       pip install git+https://github.com/nerfstudio-project/gsplat.git
```
> Note: installing `gsplat` can be machine-dependent and sometimes be tricky. if you encounter issues, please refer to the original `gsplat` repository for troubleshooting.

## Quick Start (Playground)

We provide a tiny subset of Waymo Open Dataset (3 sequences) for quick experimentation:

```bash
# download dataset subset (≈ 600 MB)
gdown 14fapsAGoMCQ5Ky82cg2X6bk-mLQ7fdCF
tar -xf STORM_subset.tar.gz
```

```bash
# run single-GPU inference demo
python inference.py \
    --project storm_playground --exp_name visualization \
    --data_root data/STORM_subset \
    --model STORM-B/8 --num_motion_tokens 16 \
    --use_sky_token --use_affine_token \
    --load_depth --load_flow --load_ground \
    --load_from $CKPT_PTH
```
> `CKPT_PTH` refers to the checkpoint. We cannot share an official checkpoint at this moment. Please refer to the issue page for an unofficial checkpoint.

## Dataset Preparation

### Waymo Dataset
- To prepare the Waymo Open Dataset, please refer to [Waymo Data](docs/WAYMO.md)

### Other datasets
We haven't included instructions for preparing NuScenes and Argoverse2 datasets. We might include these based on the capacity.

## Training

Multi-GPU example that reproduces the paper's STORM-B/8 model:

```bash
# with a global batch size= num_gpus * batch_size = 8 * 4 = 32 (We used 64 global batch size for main experiments)
torchrun --nproc_per_node=8 main_storm.py \
    --project 0504_storm \
    --exp_name 0504_pixel_storm \
    --data_root ../storm2.3/data/STORM2 \ # replace this with your data root.
    --batch_size 4 --num_iterations 100000 --lr_sched constant \
    --model STORM-B/8 --num_motion_tokens 16 \
    --use_sky_token --use_affine_token \
    --load_depth --load_flow --load_ground \
    --enable_depth_loss --enable_flow_reg_loss --flow_reg_coeff 0.005 --enable_sky_opacity_loss \
    --enable_perceptual_loss --perceptual_loss_start_iter 5000 \
    --enable_wandb \
    --auto_resume
```

> **Tips:**
> - Checkpoints and logs are saved at `work_dirs/<project>/<exp_name>`
> - `batch_size` is per-GPU; global batch = batch_size × #GPUs × #nodes
> - For additional arguments, see `main_storm.py`

## Evaluation

```bash
torchrun --nproc_per_node=8 main_storm.py \
    --project 0504_storm \
    --exp_name 0504_pixel_storm \
    --data_root ../storm2.3/data/STORM2 \ # replace this with your data root.
    --batch_size 4 --num_iterations 100000 --lr_sched constant \
    --model STORM-B/8 --num_motion_tokens 16 \
    --use_sky_token --use_affine_token \
    --load_depth --load_flow --load_ground \
    --enable_depth_loss --enable_flow_reg_loss --flow_reg_coeff 0.005 --enable_sky_opacity_loss \
    --enable_perceptual_loss --perceptual_loss_start_iter 5000 \
    --auto_resume \
    --evaluate # this parameter specifies the evaluation mode
```
## TODO
[ ]  Viewers.


## Citation

```bibtex
@inproceedings{yang2025storm,
  title   = {STORM: Spatio-Temporal Reconstruction Model for Large-Scale Outdoor Scenes},
  author  = {Jiawei Yang and Jiahui Huang and Yuxiao Chen and Yan Wang and Boyi Li and Yurong You and Maximilian Igl and Apoorva Sharma and Peter Karkus and Danfei Xu and Boris Ivanovic and Yue Wang and Marco Pavone},
  booktitle = {ICLR},
  year    = {2025}
}
```

## License

This project is licensed under the **NVIDIA License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Our implementation builds upon **gsplat**.
We thank the respective authors for open‑sourcing their excellent work.


