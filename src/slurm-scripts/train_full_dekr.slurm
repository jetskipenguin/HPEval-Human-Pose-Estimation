#!/bin/bash
#SBATCH --job-name=full-dekr
#SBATCH --partition=gpu1a100
#SBATCH --time=10:00:00
#SBATCH --output=logs/full-dekr_%j.out
#SBATCH --error=logs/full-dekr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mail-user=collin.newman@my.utsa.edu
#SBATCH --mail-type=all

# Load modules
module purge
module load anaconda3/2024.10-1
module load cudatoolkit/11.3.1_465.19.01
module load cuda11/cudnn/9.2.0.82

conda activate simple-baselines-for-hpe

cd /work/vba875/DEKR

python tools/train.py --cfg /work/vba875/HPEval-Human-Pose-Estimation/src/bottom_up_dekr/full_dekr.yaml
