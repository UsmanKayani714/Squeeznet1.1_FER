import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import os
import sys
import argparse
import time
from torch.autograd import Variable
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Add parent directory to path to import utils and FER2013
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)
import utils
from FER2013 import FER2013

# Import PEFT model from local models directory
sys.path.insert(0, os.path.dirname(__file__))
from models.peft_squeezemodel import PEFTSqueezeNetModel

parser = argparse.ArgumentParser(description='PyTorch FER2013 PEFT-SqueezeNet 1.1 Training')
parser.add_argument('--model', type=str, default='PEFTmodel', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='fer2013_peft_squeezenet', help='dataset')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--bs', default=32, type=int, help='batch_size')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--adapter_reduction', default=16, type=int, help='Adapter reduction factor (default: 16)')
opt = parser.parse_args()

use_mps = torch.backends.mps.is_available()
device = "mps" if use_mps else "cpu"

best_Test_acc = 0
best_Test_acc_epoch = 0
start_epoch = 0

train_accuracy_values = []
test_accuracy_values = []
train_loss_values = []
test_loss_values = []

total_epoch = 55

path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))

# Data
print('==> Preparing data..')
print(f'Using device: {device}')

def repeat_channels(x):
    return x.repeat(3, 1, 1)

transforms_valid = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(repeat_channels),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(repeat_channels),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = FER2013(split='Training', fold=opt.fold, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
testset = FER2013(split='Testing', fold=opt.fold, transform=transforms_valid)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

# Model
print('==> Building PEFT-SqueezeNet model..')
num_classes = 7
net = PEFTSqueezeNetModel(num_classes=num_classes, adapter_reduction=opt.adapter_reduction)

# Count parameters
total_params = net.count_total_parameters()
trainable_params = net.count_trainable_parameters()
frozen_params = total_params - trainable_params

print(f'\n==> Model Parameters:')
print(f'    Total parameters: {total_params:,}')
print(f'    Trainable parameters: {trainable_params:,}')
print(f'    Frozen parameters: {frozen_params:,}')
print(f'    Trainable percentage: {100.0 * trainable_params / total_params:.2f}%')

# Move model to device
net = net.to(device)

if opt.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path, 'Test_model.t7'), map_location=device)
    
    if isinstance(checkpoint['net'], dict):
        net.load_state_dict(checkpoint['net'])
    else:
        net.load_state_dict(checkpoint['net'].state_dict())
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = best_Test_acc_epoch + 1

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

print(f'\n==> Training Configuration:')
print(f'    Model: PEFT-SqueezeNet 1.1')
print(f'    Dataset: FER2013')
print(f'    Number of classes: {num_classes}')
print(f'    Batch size: {opt.bs}')
print(f'    Learning rate: {opt.lr}')
print(f'    Total epochs: {total_epoch}')
print(f'    Device: {device}')
print(f'    Adapter reduction factor: {opt.adapter_reduction}')
print(f'    Optimizer: Adam (weight_decay=1e-4)')
print(f'    Scheduler: StepLR (step_size=15, gamma=0.5)')
print('')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_time = elapsed_time - elapsed_hours * 3600
    elapsed_mins = int(elapsed_time // 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_hours, elapsed_mins, elapsed_secs

total_processing_time_train = 0
total_processing_time_test = 0
epoch_times = []

def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    global total_processing_time_train
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    epoch_start_time = time.monotonic()
  
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        batch_start_time = time.time()
        outputs = net(inputs)
        batch_end_time = time.time()
        processing_time = batch_end_time - batch_start_time
        total_processing_time_train += processing_time
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
    
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        
        utils.progress_bar(batch_idx, len(trainloader), 'TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    Train_acc = 100.*correct/total
    train_accuracy_values.append(Train_acc)
    train_loss_values.append(train_loss / (batch_idx + 1))

def test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    global total_processing_time_test
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            batch_start_time = time.time()
            outputs = net(inputs)
            batch_end_time = time.time()
            processing_time = batch_end_time - batch_start_time
            total_processing_time_test += processing_time
    
            loss = criterion(outputs, targets)
            PrivateTest_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()
            
            utils.progress_bar(batch_idx, len(testloader), 'TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)'
                % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    Test_acc = 100.*correct/total
    test_accuracy_values.append(Test_acc)
    test_loss_values.append(PrivateTest_loss / (batch_idx + 1))
    
    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % Test_acc)
        state = {
            'net': net.state_dict(),
            'best_Test_acc': Test_acc,
            'best_Test_acc_epoch': epoch,
            'trainable_params': trainable_params,
            'total_params': total_params,
        }
        if not os.path.isdir(opt.dataset + '_' + opt.model):
            os.mkdir(opt.dataset + '_' + opt.model)
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'Test_model.t7'))
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch

total_start_time = time.monotonic()
for epoch in range(start_epoch, total_epoch):
    epoch_start = time.monotonic()
    train(epoch)
    test(epoch)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    epoch_end = time.monotonic()
    epoch_hours, epoch_mins, epoch_secs = epoch_time(epoch_start, epoch_end)
    epoch_time_seconds = (epoch_end - epoch_start)
    epoch_times.append(epoch_time_seconds)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_hours}h {epoch_mins}m {epoch_secs}s | LR: {current_lr:.6f}')

total_end_time = time.monotonic()
total_hours, total_mins, total_secs = epoch_time(total_start_time, total_end_time)
total_time_estimate_hours = total_hours + (total_mins / 60) + (total_secs / 3600)

# Calculate average epoch time
avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
avg_epoch_hours, avg_epoch_mins, avg_epoch_secs = epoch_time(0, avg_epoch_time)

print(f'\n==> Training Summary:')
print(f'Total Time: {total_hours}h {total_mins}m {total_secs}s | Estimated Total Time: {total_time_estimate_hours:.2f} hours')
print(f'Average Epoch Time: {avg_epoch_hours}h {avg_epoch_mins}m {avg_epoch_secs:.2f}s')
print(f"best_Test_acc: %0.3f" % best_Test_acc)
print(f"best_Test_acc_epoch: %d" % best_Test_acc_epoch)
print(f'\n==> Model Efficiency:')
print(f'    Trainable parameters: {trainable_params:,}')
print(f'    Average training time per epoch: {avg_epoch_time:.2f} seconds')

