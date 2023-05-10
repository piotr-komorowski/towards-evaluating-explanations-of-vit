import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import einops
from PIL import Image
from lime import lime_image
from attn_explain.vit_rollout import VITAttentionRollout

def show_mask_on_image(img, mask=None):
    img = (img - img.min()) / (img.max() - img.min())
    if mask is None:
        return img
    mask = mask.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization_lrp(original_image, attribution_generator, with_img=False, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = F.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution) if with_img else transformer_attribution
    vis = einops.repeat(np.array(vis), "h w -> c h w", c=1)

    return vis

def generate_lrp(inputs, model=None, targets=None, attribution_generator=None, device=None):
    if not isinstance(inputs, torch.Tensor): # convert to tensor if not already
        inputs = torch.tensor(inputs)
    vis = np.stack([generate_visualization_lrp(x, attribution_generator, with_img=False) for x in inputs])
    return vis

def generate_visualization_lime(image, explainer, batch_predict, with_img=False):
    image = einops.rearrange(image, "c h w -> h w c")
    explanation = explainer.explain_instance(np.array(image), 
                                            batch_predict, # classification function
                                            top_labels=3, 
                                            hide_color=0, 
                                            num_samples=500) # number of images that will be sent to classification function
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
    return mark_boundaries(temp/255.0, mask) if with_img else einops.repeat(mask, "h w -> c h w", c=1)

def generate_lime(inputs, model=None, targets=None, batch_predict=None, device=None):
    explainer = lime_image.LimeImageExplainer()
    vis = np.stack([generate_visualization_lime(x, explainer, batch_predict, with_img=False) for x in inputs])
    # vis = einops.rearrange(vis,"b h w c -> b c h w")
    return vis


def generate_attn(inputs, model=None, targets=None, head_fusion=None, discard_ratio=None, device=None):
    attention_rollout = VITAttentionRollout(model, head_fusion=head_fusion, 
            discard_ratio=discard_ratio)
    if not isinstance(inputs, torch.Tensor): # convert to tensor if not already
        inputs = torch.tensor(inputs)
    inputs = inputs.to(device)

    vis = np.stack([cv2.resize(attention_rollout(x.unsqueeze(0)), (x.shape[1], x.shape[2]))\
                    for x in inputs])
    attention_rollout.eject()
    vis = einops.repeat(vis, 'b h w -> b c h w', c=1)
    return vis
