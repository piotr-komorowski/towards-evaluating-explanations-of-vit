#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --job-name=eval_expl

. activate quantus_vit

cd ~/projects/ViT/Transformer-Explainability

python evaluate_explanations.py
