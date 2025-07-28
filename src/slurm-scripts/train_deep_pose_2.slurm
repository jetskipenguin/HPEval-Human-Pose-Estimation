#!/bin/bash
#SBATCH --job-name=resnet101-batch32-lr1e-4-deep-pose-train
#SBATCH --partition=gpu1a100
#SBATCH --time=10:00:00
#SBATCH --output=logs/resnet101-batch32-lr1e-4-deep-pose-train_%j.out
#SBATCH --error=logs/resnet101-batch32-lr1e-4-deep-pose-train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-user=collin.newman@my.utsa.edu
#SBATCH --mail-type=all

# Load modules
module purge
module load anaconda3/2024.10-1
module load cudatoolkit/11.3.1_465.19.01
module load cuda11/cudnn/9.2.0.82

conda activate deep-pose

cd /work/vba875/HPEval-Human-Pose-Estimation

python -m src.direct_regression.main --config_path config.ini
