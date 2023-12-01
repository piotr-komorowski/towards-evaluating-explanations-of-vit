import os
import torch
import numpy as np
import random
import einops

from modules.layers_ours import Linear
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP_base
from baselines.ViT.ViT_explanation_generator import LRP
from utils_explain.visualize import generate_lrp, generate_lime, generate_attn
from utils_explain.preprocess import batch_predict

import quantus


# Set all seeds from random, numpy, torch, etc. to make sure results are reproducible
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

explanations = np.load('explanations.npz')
x_batch = explanations['x_batch']
y_batch = explanations['y_batch']
a_batch_lrp = explanations['a_batch_lrp']
a_batch_attn_mean = explanations['a_batch_attn_mean']
a_batch_attn_min = explanations['a_batch_attn_min']
a_batch_attn_max = explanations['a_batch_attn_max']
a_batch_lime = explanations['a_batch_lime']

covid_indices, noncovid_indices, healthy_indices = y_batch==0, y_batch==1, y_batch==2

model = vit_LRP_base().to(device)
model.head = Linear(model.head.in_features, 3).cuda()
model.load_state_dict(torch.load('results_model/model_best.pth.tar')['state_dict'])
model.eval()
attribution_generator = LRP(model)

results_explain_txt = os.path.join('results_explain', 'evaluations.txt')
results_explain_f_csv = os.path.join('results_explain', 'evaluations_faithfulness.csv')
results_explain_s_csv = os.path.join('results_explain', 'evaluations_sensitivity.csv')
results_explain_c_csv = os.path.join('results_explain', 'evaluations_complexity.csv')


# Delete the file if it exists
if os.path.exists(results_explain_txt): 
    os.remove(results_explain_txt)
os.makedirs(os.path.dirname(results_explain_txt))

if os.path.exists(results_explain_f_csv):
    os.remove(results_explain_f_csv)

if os.path.exists(results_explain_s_csv):
    os.remove(results_explain_s_csv)

if os.path.exists(results_explain_c_csv):
    os.remove(results_explain_c_csv)


def save_results(results_explain_txt, results_explain_csv, lrp_estimate, attn_estimate_mean,\
attn_estimate_min, attn_estimate_max, lime_estimate, covid_indices,\
    noncovid_indices, healthy_indices, metric_name='sensitivity'):
    with open(results_explain_txt, 'a') as f:
        print(f'LRP {metric_name} mean: {lrp_estimate.mean()}', file=f)
        print(f'Attention mean {metric_name} mean: {attn_estimate_mean.mean()}', file=f)
        print(f'Attention min {metric_name} mean: {attn_estimate_min.mean()}', file=f)
        print(f'Attention max {metric_name} mean: {attn_estimate_max.mean()}', file=f)
        print(f'LIME {metric_name} mean: {lime_estimate.mean()}', file=f)
        print(f'LRP {metric_name} COVID mean: {lrp_estimate[covid_indices].mean()}', file=f)
        print(f'LRP {metric_name} Non-COVID mean: {lrp_estimate[noncovid_indices].mean()}', file=f)
        print(f'LRP {metric_name} Healthy mean: {lrp_estimate[healthy_indices].mean()}', file=f)
        print(f'Attention mean {metric_name} COVID mean: {attn_estimate_mean[covid_indices].mean()}', file=f)
        print(f'Attention mean {metric_name} Non-COVID mean: {attn_estimate_mean[noncovid_indices].mean()}', file=f)
        print(f'Attention mean {metric_name} Healthy mean: {attn_estimate_mean[healthy_indices].mean()}', file=f)
        print(f'Attention min {metric_name} COVID mean: {attn_estimate_min[covid_indices].mean()}', file=f)
        print(f'Attention min {metric_name} Non-COVID mean: {attn_estimate_min[noncovid_indices].mean()}', file=f)
        print(f'Attention min {metric_name} Healthy mean: {attn_estimate_min[healthy_indices].mean()}', file=f)
        print(f'Attention max {metric_name} COVID mean: {attn_estimate_max[covid_indices].mean()}', file=f)
        print(f'Attention max {metric_name} Non-COVID mean: {attn_estimate_max[noncovid_indices].mean()}', file=f)
        print(f'Attention max {metric_name} Healthy mean: {attn_estimate_max[healthy_indices].mean()}', file=f)
        print(f'LIME {metric_name} COVID mean: {lime_estimate[covid_indices].mean()}', file=f)
        print(f'LIME {metric_name} Non-COVID mean: {lime_estimate[noncovid_indices].mean()}', file=f)
        print(f'LIME {metric_name} Healthy mean: {lime_estimate[healthy_indices].mean()}', file=f)
        print('\n', file=f)
        np.savetxt(results_explain_csv, np.column_stack((lrp_estimate, attn_estimate_mean, attn_estimate_min,\
        attn_estimate_max,lime_estimate)), delimiter=',', header='LRP,Attention_mean,Attention_min,Attention_max,LIME')


#### Complexity ####
def calc_effective_complexity(a_batch_expl):
    complexity_estimator = quantus.EffectiveComplexity(
    eps=0.1,
    abs=False,
    normalise=False
    )
    complexity_estimate = complexity_estimator(model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch_expl,
        device=device
        )
    num_pixels = x_batch.shape[2] * x_batch.shape[3]
    return np.array(complexity_estimate)/num_pixels

print('Calculating effective complexity for LRP...')
effective_complexity_lrp = calc_effective_complexity(a_batch_lrp)
print('Calculating effective complexity for Attention...')
effective_complexity_attn_mean_0 = calc_effective_complexity(a_batch_attn_mean)
effective_complexity_attn_min_0 = calc_effective_complexity(a_batch_attn_min)
effective_complexity_attn_max_9 = calc_effective_complexity(a_batch_attn_max)
print('Calculating effective complexity for LIME...')
effective_complexity_lime = calc_effective_complexity(a_batch_lime)

save_results(results_explain_txt, results_explain_c_csv, effective_complexity_lrp,\
    effective_complexity_attn_mean_0, effective_complexity_attn_min_0,\
    effective_complexity_attn_max_9, effective_complexity_lime, covid_indices,\
    noncovid_indices, healthy_indices, metric_name='complexity')


#### Sensitivity ####
def calc_avg_sensitivity(a_batch_expl, **kwargs):
    name = kwargs['name']
    if name == 'lrp':
        explain_func = generate_lrp
        explain_func_kwargs = {'attribution_generator': attribution_generator}
    elif name == 'attn':
        head_fusion = kwargs['head_fusion']
        discard_ratio = kwargs['discard_ratio']
        explain_func = generate_attn
        explain_func_kwargs = {'head_fusion': head_fusion, 'discard_ratio':discard_ratio}
    elif name == 'lime':
        explain_func = generate_lime
        explain_func_kwargs = {'batch_predict': batch_predict}
    else:
        raise ValueError('name must be lrp, attn, or lime')

    step = 25
    avg_sensitivity_estimate_all = []
    avg_sensitivity_estimate = quantus.AvgSensitivity(
          nr_samples=3,
          lower_bound=.33
          )
    for i in range(0, len(x_batch), step): # CUDA out of memory
        print(i)
        estimator = avg_sensitivity_estimate(model=model, 
            x_batch=x_batch[i:i+step], 
            y_batch=y_batch[i:i+step],
            a_batch=a_batch_expl[i:i+step],
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            device=device,
            )
        avg_sensitivity_estimate_all += estimator

    return np.array(avg_sensitivity_estimate_all)



print('Generating sensitivity explanations for LRP...')
avg_sensitivity_estimate_lrp = calc_avg_sensitivity(a_batch_lrp, name='lrp')
print('Generating sensitivity explanations for Attention...')
avg_sensitivity_estimate_attn_mean_0 = calc_avg_sensitivity(a_batch_attn_mean, name='attn', head_fusion='mean', discard_ratio=0)
avg_sensitivity_estimate_attn_min_0 = calc_avg_sensitivity(a_batch_attn_min, name='attn', head_fusion='min', discard_ratio=0)
avg_sensitivity_estimate_attn_max_9 = calc_avg_sensitivity(a_batch_attn_max, name='attn', head_fusion='max', discard_ratio=0.99)
print('Generating sensitivity explanations for LIME...')
avg_sensitivity_estimate_lime = calc_avg_sensitivity(a_batch_lime, name='lime')
save_results(results_explain_txt, results_explain_s_csv, avg_sensitivity_estimate_lrp,\
    avg_sensitivity_estimate_attn_mean_0, avg_sensitivity_estimate_attn_min_0,\
    avg_sensitivity_estimate_attn_max_9, avg_sensitivity_estimate_lime, covid_indices,\
    noncovid_indices, healthy_indices, metric_name='sensitivity')


#### Faithfulness ####

def calc_faithfullness(a_batch_expl):
    faithfullness_estimator = quantus.FaithfulnessCorrelation(
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        subset_size=224,  
        perturb_baseline="black",
        return_aggregate=False,
        normalise=False,
    )
    faithfullness_estimate = faithfullness_estimator(
        model=model, 
        x_batch=x_batch, 
        y_batch=y_batch,
        a_batch=a_batch_expl,
        device=device,
        )
    return np.array(faithfullness_estimate)

print('Generating faithfulness explanations for LRP...')
faithfullness_estimate_lrp = calc_faithfullness(a_batch_lrp)
print('Generating faithfulness explanations for Attention...')
faithfullness_estimate_attn_mean_0 = calc_faithfullness(a_batch_attn_mean)
faithfullness_estimate_attn_min_0 = calc_faithfullness(a_batch_attn_min)
faithfullness_estimate_attn_max_9 = calc_faithfullness(a_batch_attn_max)
print('Generating faithfulness explanations for LIME...')
faithfullness_estimate_lime = calc_faithfullness(a_batch_lime)

save_results(results_explain_txt, results_explain_f_csv, faithfullness_estimate_lrp,\
    faithfullness_estimate_attn_mean_0, faithfullness_estimate_attn_min_0,\
    faithfullness_estimate_attn_max_9, faithfullness_estimate_lime, covid_indices,\
    noncovid_indices, healthy_indices, metric_name='faithfulness')