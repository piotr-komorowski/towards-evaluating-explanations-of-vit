import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import random
import einops
from modules.layers_ours import Linear
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP_base
from baselines.ViT.ViT_explanation_generator import LRP
from utils_explain.visualize import generate_lrp, generate_lime, generate_attn
from utils_explain.preprocess import preprocess_batch, batch_predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

CLS2IDX = {0: 'COVID-19', 1: 'Non-COVID', 2: 'Normal'}

normalize = transforms.Normalize(mean=[0.56, 0.56, 0.56], std=[0.21, 0.21, 0.21])

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf    

pil_transform = get_pil_transform()
preprocess_transform = get_preprocess_transform()

transform = transforms.Compose([
    pil_transform,
    preprocess_transform
])

# Load model
model = vit_LRP_base().to(device)
model.head = Linear(model.head.in_features, 3).to(device)
model.load_state_dict(torch.load('results_model/model_best.pth.tar')['state_dict'])
model.eval()
attribution_generator = LRP(model)


data_dir = "data/lung/Test"
N = 100 # Number of samples to take from each class

# set all seeds from random, numpy, torch, etc. to make sure results are reproducible
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Create a dictionary to store the class labels and their corresponding paths
class_names = ['COVID-19', 'Non-COVID', 'Normal']
class_dict = {
    class_names[0]: os.path.join(data_dir, class_names[0], 'images'),
    class_names[1]: os.path.join(data_dir, class_names[1], 'images'),
    class_names[2]: os.path.join(data_dir, class_names[2], 'images')
}

# Create a dictionary to store the sampled file paths for each class
sampled_paths_dict = {class_label: [] for class_label in class_dict.keys()}

# Loop through each class
for class_label, class_path in class_dict.items():
    random.seed(seed)
    # Get a list of file paths in the class directory
    file_paths = os.listdir(class_path)
    # Randomly sample N file paths from the class directory
    sampled_file_paths = random.sample(file_paths, N)
    # Add the sampled file paths to the dictionary for this class
    for file_path in sampled_file_paths:
        sampled_paths_dict[class_label].append(os.path.join(class_path, file_path))

covid_paths = sampled_paths_dict["COVID-19"]
noncovid_paths = sampled_paths_dict["Non-COVID"]
healthy_paths = sampled_paths_dict["Normal"]


x_batches, y_batches = [], []

print('Preprocessing images...')
for path, class_number in zip([covid_paths,noncovid_paths,healthy_paths],[0,1,2]):
    x_batch,y_batch = preprocess_batch(path, class_number, transform)
    x_batches.append(x_batch)
    y_batches.append(y_batch)

x_batch = torch.cat(x_batches)
y_batch = torch.cat(y_batches)

np_x_batch = x_batch.numpy()
np_y_batch = y_batch.numpy()


print('Generating explanations...')
np_a_batch_lrp = generate_lrp_goodformat(x_batch, attribution_generator=attribution_generator)
np_a_batch_attn_mean = generate_attn(x_batch, model, head_fusion='mean', discard_ratio=0, device=device)
np_a_batch_attn_min = generate_attn(x_batch, model, head_fusion='min', discard_ratio=0, device=device)
np_a_batch_attn_max = generate_attn(x_batch, model, head_fusion='max', discard_ratio=0.99, device=device)
np_a_batch_lime = generate_lime(x_batch, batch_predict=batch_predict)

np.savez('explanations.npz', x_batch=np_x_batch, y_batch=np_y_batch,
    a_batch_lrp=np_a_batch_lrp, a_batch_attn_mean=np_a_batch_attn_mean,\
    a_batch_attn_min=np_a_batch_attn_min, a_batch_attn_max=np_a_batch_attn_max,\
    a_batch_lime=np_a_batch_lime)