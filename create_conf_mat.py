import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from modules.layers_ours import Linear
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP_base

# Set device to CPU or GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.58, 0.58, 0.58], std=[0.21, 0.21, 0.21])

test_dataset = datasets.ImageFolder(
    'data/lung/Test',
    transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    
# Load test data
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = vit_LRP_base().to(device)
model.head = Linear(model.head.in_features, 3).cuda()
model.load_state_dict(torch.load('results_model/model_best.pth.tar')['state_dict'])
model.eval()

# Initialize lists to store true labels and predicted labels
true_labels = []
pred_labels = []

# Loop over test data
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Make predictions
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Append true and predicted labels to lists
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

# Create confusion matrix
conf_mat = confusion_matrix(true_labels, pred_labels)
print(f'Classes: {test_dataset.class_to_idx}')
print(conf_mat)
acc = (conf_mat[0,0] + conf_mat[1,1] + conf_mat[2,2]) / np.sum(conf_mat)
print(f'Accuracy: {acc}')

acc_covid = conf_mat[0,0] / np.sum(conf_mat[0,:])
print(f'Accuracy COVID: {acc_covid}')
acc_noncovid = conf_mat[1,1] / np.sum(conf_mat[1,:])
print(f'Accuracy Non-COVID: {acc_noncovid}')
acc_normal = conf_mat[2,2] / np.sum(conf_mat[2,:])
print(f'Accuracy Normal: {acc_normal}')