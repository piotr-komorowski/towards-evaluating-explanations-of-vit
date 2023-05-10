#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --job-name=gen_expl

. activate quantus_vit

cd ~/projects/ViT/Transformer-Explainability

python create_explanations.py
