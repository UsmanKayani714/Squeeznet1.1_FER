import torch
import torch.nn as nn
import argparse
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torchvision
from models import squeezemodel
from FER2013 import FER2013

parser = argparse.ArgumentParser(description='PyTorch FER2013 SqueezeNet 1.1 Confusion Matrix')
parser.add_argument('--dataset', type=str, default='fer2013_squeezenet', help='Dataset name')
parser.add_argument('--model', type=str, default='Ourmodel', help='Model name')

opt = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#cut_size = 44

transforms_vaild = torchvision.transforms.Compose([
                                     torchvision.transforms.ToPILImage(),
                                     torchvision.transforms.Resize((224,)),
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225,))
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

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

# Model
if opt.model == 'Ourmodel':
   num_classes = 6
   net = squeezemodel.SqueezeNetModel(num_classes)

correct = 0
total = 0
all_target = []

# Load the model checkpoint
path = os.path.join(opt.dataset + '_' + opt.model)
checkpoint = torch.load(os.path.join(path, 'Test_model.t7'))

net.load_state_dict(checkpoint['net'])
net.to(device)
net.eval()
testset = FER2013(split='Testing', transform=transforms_vaild)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    if batch_idx == 0:
        all_predicted = predicted
        all_targets = targets
    else:
        all_predicted = torch.cat((all_predicted, predicted), 0)
        all_targets = torch.cat((all_targets, targets), 0)

acc = 100. * correct / total
print("FER2013 Test Accuracy: %0.3f%%" % acc)

# Compute confusion matrix
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)

# Print classification report
print('\nClassification Report:\n', classification_report(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy(), target_names=class_names))

# Plot normalized confusion matrix
os.makedirs(opt.dataset + '_' + opt.model, exist_ok=True)
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                      title='FER2013 Confusion Matrix (SqueezeNet 1.1 Accuracy: %0.3f%%)' % acc)
plt.savefig(os.path.join(opt.dataset + '_' + opt.model, 'confusion_matrix.png'))
plt.close()
print(f"\nConfusion matrix saved to {opt.dataset + '_' + opt.model}/confusion_matrix.png")
