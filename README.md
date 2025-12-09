## RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters
ICCV 2025 Highlight

[Project]([RESCUE Project Page](https://cic.tju.edu.cn/faculty/likun/projects/RESCUE/index.html)) [arXiv]([[2507.20117\] RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters](https://arxiv.org/abs/2507.20117))

## Introduction
This repository will host the code and data for our paper **"RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters"**.  
The code and checkpoints are still being uploaded, and the README will continue to be updated. The full upload is planned to be finalized in December.
We sincerely apologize for the long wait and appreciate your patience.

## Installation

To create the environment, follow the following instructions:

1. Create new conda environment and install dependencies.

```
conda create -n rescue python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym).

3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. Please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. The file structure should look like this:

   ```
   |-- data
       |-- smpl
           |-- SMPL_FEMALE.pkl
           |-- SMPL_NEUTRAL.pkl
           |-- SMPL_MALE.pkl
   ```

## Assets Download

1. Download checkpoints. Download the checkpoint from [ckpt](https://drive.google.com/drive/folders/1g2nT5IaXEzSfERUjEo761l9mwBe9D-_y?usp=drive_link) and place it in the project folder. The file structure should look like this:

   ```
   ├── ckpt
      ├── camdm
      │   └── 100_best.pt
      └── pacer_getup
          └── Humanoid.pth
   ```

2. Download scenes for simulation. Download the scenes for simulation from [scenes](https://drive.google.com/drive/folders/19XEVkd8408nUe93xbkaXrZq5xxJJOQyr?usp=sharing) and place it in the project folder. The file structure should look like this:

   ```
   ├── scenes
       ├── scene1
       │   ├── 1.mtl
       │   ├── 1.obj
       │   └── navmesh_loose.obj
       ├── scene2
       │   ...
       ├── ...
   ```

## Simulation

Run the following script to test.

```
python pacer/run.py \
    --task HumanoidPedestrianTerrain \
    --cfg_env pacer/data/cfg/pacer_getup.yaml \
    --cfg_train pacer/data/cfg/train/rlg/amp_humanoid_smpl_sept_task.yaml \
    --motion_file sample_data/amass_isaac_standing_upright_slim.pkl \
    --ckpt_path ckpt/pacer_getup \
    --test \
    --num_envs 50 \
    --epoch -1 \
    --no_virtual_display \
    --scene_yaml pacer/data/cfg/scene/scene1.yaml
```

Use the `--headless` option when running without a visualization window.

## References

This repository is built on top of the following amazing repositories:

- Main code framework is from: [Pacer]([GitHub - nv-tlabs/pacer: Official implementation of PACER, Pedestrian Animation ControllER, of CVPR 2023 paper: "Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion".](https://github.com/nv-tlabs/pacer)),
- Code of gait converter is adapted from: [CAMDM](https://github.com/AIGAnimation/CAMDM/blob/main/Evaluation/feature_extractor/train_classifier.py),
- Code of finding waypoints is adapted from: [DIMOS](https://github.com/zkf1997/DIMOS)

## Citation
If you find our work useful, please consider citing our paper:
```bibtex
@InProceedings{Liu_2025_ICCV,
    author    = {Liu, Xiaolin and Zhou, Tianyi and Kang, Hongbo and Ma, Jian and Wang, Ziwen and Huang, Jing and Weng, Wenguo and Lai, Yu-Kun and Li, Kun},
    title     = {RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {24955-24964}
}
```
