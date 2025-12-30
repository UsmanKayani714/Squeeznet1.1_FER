import torch
import torch.nn as nn
import argparse
import os
import sys
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_score, f1_score, accuracy_score
import torchvision
import sys
import os

# Add parent directory to path to import FER2013
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from FER2013 import FER2013

# Import PEFT model from local models directory
sys.path.insert(0, os.path.dirname(__file__))
from models.peft_squeezemodel import PEFTSqueezeNetModel
import itertools

parser = argparse.ArgumentParser(description='PyTorch FER2013 PEFT-SqueezeNet 1.1 Evaluation')
parser.add_argument('--dataset', type=str, default='fer2013_peft_squeezenet', help='Dataset name')
parser.add_argument('--model', type=str, default='PEFTmodel', help='Model name')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--adapter_reduction', default=16, type=int, help='Adapter reduction factor')

opt = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

# Model
print('==> Loading PEFT-SqueezeNet model..')
num_classes = 7
net = PEFTSqueezeNetModel(num_classes=num_classes, adapter_reduction=opt.adapter_reduction)

# Count parameters
total_params = net.count_total_parameters()
trainable_params = net.count_trainable_parameters()
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')
print(f'Trainable percentage: {100.0 * trainable_params / total_params:.2f}%')

correct = 0
total = 0
all_targets = []
all_predicted = []

# Load the model checkpoint
path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))
checkpoint = torch.load(os.path.join(path, 'Test_model.t7'), map_location=device, weights_only=False)

if isinstance(checkpoint['net'], dict):
    net.load_state_dict(checkpoint['net'])
else:
    net.load_state_dict(checkpoint['net'].state_dict())

net.to(device)
net.eval()

testset = FER2013(split='Testing', fold=opt.fold, transform=transforms_valid)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

print('==> Evaluating model..')
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    inputs, targets = Variable(inputs), Variable(targets)
    
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    if batch_idx == 0:
        all_predicted = predicted.cpu()
        all_targets = targets.data.cpu()
    else:
        all_predicted = torch.cat((all_predicted, predicted.cpu()), 0)
        all_targets = torch.cat((all_targets, targets.data.cpu()), 0)

# Calculate metrics
all_predicted_np = all_predicted.numpy()
all_targets_np = all_targets.numpy()

accuracy = accuracy_score(all_targets_np, all_predicted_np) * 100
precision = precision_score(all_targets_np, all_predicted_np, average='weighted') * 100
f1 = f1_score(all_targets_np, all_predicted_np, average='weighted') * 100

print(f'\n==> Evaluation Results:')
print(f'Accuracy: {accuracy:.3f}%')
print(f'Precision (weighted): {precision:.3f}%')
print(f'F1-score (weighted): {f1:.3f}%')

# Compute confusion matrix
matrix = confusion_matrix(all_targets_np, all_predicted_np)
np.set_printoptions(precision=2)

# Print classification report
print('\nClassification Report:\n', classification_report(all_targets_np, all_predicted_np, target_names=class_names))

# Plot normalized confusion matrix
os.makedirs(opt.dataset + '_' + opt.model, exist_ok=True)
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                      title=f'PEFT-SqueezeNet Confusion Matrix (Accuracy: {accuracy:.3f}%)')
plt.savefig(os.path.join(opt.dataset + '_' + opt.model, 'confusion_matrix.png'))
plt.close()
print(f"\nConfusion matrix saved to {opt.dataset + '_' + opt.model}/confusion_matrix.png")

# Save metrics to file
metrics_file = os.path.join(opt.dataset + '_' + opt.model, 'metrics.txt')
with open(metrics_file, 'w') as f:
    f.write(f'PEFT-SqueezeNet 1.1 Evaluation Metrics\n')
    f.write(f'=====================================\n\n')
    f.write(f'Accuracy: {accuracy:.3f}%\n')
    f.write(f'Precision (weighted): {precision:.3f}%\n')
    f.write(f'F1-score (weighted): {f1:.3f}%\n\n')
    f.write(f'Total parameters: {total_params:,}\n')
    f.write(f'Trainable parameters: {trainable_params:,}\n')
    f.write(f'Trainable percentage: {100.0 * trainable_params / total_params:.2f}%\n')

print(f'\nMetrics saved to {metrics_file}')

