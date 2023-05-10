import os
from PIL import Image
import cv2
import numpy as np
import torch

def preprocess_image(image_path):
    image = np.asarray(Image.open(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(image, mode="RGB")
    return image

def preprocess_batch(list_of_xes, cls, transform):
    targets = torch.Tensor(len(list_of_xes)*[cls]).long()
    images = [preprocess_image(x) for x in list_of_xes]
    trans_images = [transform(image) for image in images]
    return torch.stack(trans_images), targets

def batch_predict(images):
    model.eval()
    images = einops.rearrange(images, 'b h w c -> b c h w')
    batch = torch.Tensor(np.stack(images))

    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()
