export PYTHONPATH=diffusion_traj
export CUDA_VISIBLE_DEVICES=2
export WANDB_DISABLED=true

conda activate diff

python train.py --config-dir=. --config-name=train_diffusion_traj_gibson.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_train_traj_diff_gibson'
