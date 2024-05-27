#!/bin/bash

export PYTHONPATH=..T-Diff
export CUDA_VISIBLE_DEVICES=0,1
cd ..T-Diff/semexp

# conda activate tdiff

python eval_tdiff.py \
  --split val \
  --seed 345 \
  --eval 1 \
  --pf_model_path "models_ckpt/area_model.ckpt" \
  --diff_model_path "models_ckpt/diff_model.ckpt" \
  -d ..experiments \
  --num_local_steps 1 \
  --exp_name "debug" \
  --global_downscaling 1 \
  --mask_nearest_locations \
  --pf_masking_opt 'unexplored' \
  --use_nearest_frontier \
  --total_num_scenes "5" \
  --select_diff_step 27 \
  --horizon 32 \