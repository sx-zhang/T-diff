# Trajectory Diffusion for ObjectGoal Navigation

## Setup
- Clone the repository and move into the top-level directory `cd T-Diff`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate tdiff`
- We provide pre-trained model of [T-Diff](https://drive.google.com/file/d/14FgPEUK3nXsvheg_qT575QGSFoAz8Kdd/view?usp=sharing) and [area_prediction](https://drive.google.com/file/d/113hMyZFT5orwfcFlrX_ESRawbrr6UiT7/view?usp=sharing). For evaluation, you can download them to the directory.
- Download the [t_diff_dataset](https://drive.google.com/file/d/1p5h7wxRwnPZ63cwZK6DWhpJKErhWNuDb/view).
- Download the [semantic maps (gt)](https://drive.google.com/file/d/1lOJlZXWBeCsnPzqgdnvXbEmF2yGxRwY4/view?usp=sharing).  

## Dataset
We use a modified version of the Gibson ObjectNav evaluation setup from [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation).

1. Download the [Gibson ObjectNav dataset](https://utexas.box.com/s/tss7udt3ralioalb6eskj3z3spuvwz7v) to `$T_Diff_ROOT/data/datasets/objectnav/gibson`.
    ```
    cd $T_Diff_ROOT/data/datasets/objectnav
    wget -O gibson_objectnav_episodes.tar.gz https://utexas.box.com/shared/static/tss7udt3ralioalb6eskj3z3spuvwz7v.gz
    tar -xvzf gibson_objectnav_episodes.tar.gz && rm gibson_objectnav_episodes.tar.gz
    ```
2. Download the image segmentation model [[URL](https://utexas.box.com/s/sf4prmup4fsiu6taljnt5ht8unev5ikq)] to `$T_Diff_ROOT/pretrained_models`.
3. To visualize episodes with the semantic map and potential function predictions, add the arguments `--print_images 1 --num_pf_maps 3` in the evaluation script.

The `data` folder should look like this
```python
  data/ 
    ├── datasets/objectnav/gibdon/v1.1
        ├── train/
        │   ├── content/
        │   ├── train_info.pbz2
        │   └── train.json.gz
        ├── val/
        │   ├── content/
        │   ├── val_info.pbz2
        │   └── val.json.gz
    ├── scene_datasets/
        ├── gibson_semantic/
            ├── Allensville_semantic.ply
            ├── Allensville.glb
            ├── Allensville.ids
            ├── Allensville.navmesh
            ├── Allensville.scn
            ├── ...
    ├── semantic_maps/
        ├── gibson/semantic_maps
            ├── semmap_GT_info.json
            ├── Allensville_0.png
            ├── Allensville.h5
            ├── ...
```

<!-- ## Training and Evaluation -->
<!-- ### Train your own trajectory diffusion model  -->

## Evaluation 
`sh experiment_scripts/gibson/eval_tdiff.sh`

## Training
Download the [Gibson Traj dataset](https://drive.google.com/file/d/1p5h7wxRwnPZ63cwZK6DWhpJKErhWNuDb/view?usp=sharing) to `$T_Diff_ROOT/train_traj/data/gibson_traj_32`.

1. Create conda environment. `conda env create -f train_traj/environment.yml`
2. Activate the environment. `conda activate diff_train`
3. `sh $T_Diff_ROOT/train_traj/train.sh`
