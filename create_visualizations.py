import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils_explain.visualize import show_mask_on_image

explanations_name = 'explanations'
explanations_file = f'{explanations_name}.npz'
explanations = np.load(explanations_file)
print(f'Loaded explanations from {explanations_file}')

x_batch = explanations['x_batch']
y_batch = explanations['y_batch']
a_batch_lrp = explanations['a_batch_lrp']
a_batch_attn_mean = explanations['a_batch_attn_mean']
a_batch_attn_min = explanations['a_batch_attn_min']
a_batch_attn_max = explanations['a_batch_attn_max']
a_batch_lime = explanations['a_batch_lime']
CLS2IDX = {0: 'COVID-19', 1: 'Non-COVID', 2: 'Normal'}

# Set the directory path
main_directory = f'images_{explanations_name}'

# Create the main directory if it doesn't exist
if not os.path.exists(main_directory):
    os.makedirs(main_directory)

explanations = [a_batch_lrp, a_batch_attn_mean, a_batch_attn_min,\
    a_batch_attn_max, a_batch_lime, x_batch]
names = ['LRP', 'Attention_mean', 'Attention_min',\
    'Attention_max', 'LIME', 'Original']

print('Generating explanation images...')
for explanation, name in zip(explanations, names):
    explanation_directory = os.path.join(main_directory, name)
    if not os.path.exists(explanation_directory):
        os.makedirs(explanation_directory)

    for i in range(explanation.shape[0]):
        # Get the true class of the image
        true_class = CLS2IDX[y_batch[i]]
        original_image = x_batch[i].transpose(1, 2, 0)

        # Create the subdirectory for the true class if it doesn't exist
        class_directory = os.path.join(explanation_directory, f'class_{true_class}')
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(5, 5))
        if name == 'Original':
            ax.imshow(show_mask_on_image(original_image))
        else:
            ax.imshow(show_mask_on_image(original_image, explanation[i]))
        plt.axis('off')
        # Save the figure in the subdirectory for the true class
        plt.savefig(os.path.join(class_directory, f'image_{i}.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)