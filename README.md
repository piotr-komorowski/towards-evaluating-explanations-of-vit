# PyTorch Implementation of [Towards Evaluating Explanations of Vision Transformers for Medical Imaging]

Create env: `conda env create -f environment.yml`

[Link to download dataset (images only)](https://drive.google.com/file/d/1XrCWP3ICQvurchnJjyVweYy2jHQM0BHO/view?usp=sharing)

[Link to model's checkpoint](https://drive.google.com/file/d/1JZM5ZRncaV3iFX9L6NFT1P0-APyHbBV0/view?usp=sharing)

Use model's checkpoint or train model: `sbatch sbatch_train.sh`

Generate explanation: `sbatch sbatch_gen_expl.sh`

Create visualizations of explanations: `sbatch sbatch_vis.sh`

Evaluate explanations: `sbatch sbatch_eval.sh`
