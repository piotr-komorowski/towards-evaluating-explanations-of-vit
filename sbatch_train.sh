#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --job-name=vit_train

. activate train_vit

cd ~/projects/ViT/Transformer-Explainability

python3 train.py --train_dir=data/lung/Train \
        --val_dir=data/lung/Val \
        --test_dir=data/lung/Test \
        --model=base
