#!/bin/bash

# GPU
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --gres=gpumem:10g
# GENERAL

#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=1024

#SBATCH -A es_hutter
#SBATCH --time=16:00:00
#SBATCH --job-name="sd-train"
#SBATCH --open-mode=append
#SBATCH --output="/cluster/home/kzaitse/outputs/cond-sd-train/cond-sd-train-%j.txt"


module load StdEnv gcc/8.2.0 cudnn/8.2.1.32 python_gpu/3.10.4 openblas/0.2.20 tree/1.7.0 eth_proxy cuda/11.7.0 nccl/2.11.4-1 zsh/5.8 tmux
export HF_HOME=/cluster/scratch/kzaitse/hf_home
source /cluster/home/kzaitse/venvs/ssdc/bin/activate

cd /cluster/home/kzaitse/rsl_depth_completion/rsl_depth_completion/conditional_diffusion || exit 1
python train_cond_stable_diffusion.py "$@"