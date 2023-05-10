#!/bin/bash -l
#SBATCH --mem=2G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=vis

. activate quantus_vit

cd ~/projects/ViT/Transformer-Explainability

python create_visualizations.py
