#!/bin/bash
#SBATCH --job-name=higher_hrnet_0deconvs
#SBATCH --partition=gpu1a100
#SBATCH --time=15:00:00
#SBATCH --output=logs/higher_hrnet_0deconvs_%j.out
#SBATCH --error=logs/higher_hrnet_0deconvs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mail-user=collin.newman@my.utsa.edu
#SBATCH --mail-type=all

# Load modules
module purge
module load anaconda3/2024.02-1

conda activate higher-hrnet

cd /work/vba875/HigherHRNet-Human-Pose-Estimation

python tools/train.py --cfg /work/vba875/HPEval-Human-Pose-Estimation/src/bottom_up/w32_512_adam_lr1e-3_0deconvs.yaml
