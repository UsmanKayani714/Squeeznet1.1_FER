import torch
import torch.nn as nn
import argparse
import os
import sys
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
import torchvision
import time

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Add parent directory to path to import FER2013
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
sys.path.insert(0, parent_dir)

from FER2013 import FER2013

# Import baseline model directly from file to avoid namespace conflicts
import importlib.util
baseline_model_path = os.path.join(parent_dir, 'models', 'squeezemodel.py')
spec = importlib.util.spec_from_file_location("squeezemodel", baseline_model_path)
squeezemodel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(squeezemodel)
SqueezeNetModel = squeezemodel.SqueezeNetModel

# Import PEFT model directly from file to avoid namespace conflicts
peft_model_path = os.path.join(script_dir, 'models', 'peft_squeezemodel.py')
spec = importlib.util.spec_from_file_location("peft_squeezemodel", peft_model_path)
peft_squeezemodel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(peft_squeezemodel)
PEFTSqueezeNetModel = peft_squeezemodel.PEFTSqueezeNetModel

parser = argparse.ArgumentParser(description='Compare Baseline vs PEFT-SqueezeNet Models')
parser.add_argument('--baseline_dataset', type=str, default='fer2013_squeezenet', help='Baseline dataset name')
parser.add_argument('--peft_dataset', type=str, default='fer2013_peft_squeezenet', help='PEFT dataset name')
parser.add_argument('--baseline_model', type=str, default='Ourmodel', help='Baseline model name')
parser.add_argument('--peft_model', type=str, default='PEFTmodel', help='PEFT model name')
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

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

def evaluate_model(model, testloader, device, model_name):
    """Evaluate a model and return metrics"""
    model.eval()
    all_targets = []
    all_predicted = []
    inference_times = []
    
    print(f'\n==> Evaluating {model_name}...')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            inference_times.append((end_time - start_time) / inputs.size(0))  # per sample
            
            _, predicted = torch.max(outputs.data, 1)
            
            if batch_idx == 0:
                all_predicted = predicted.cpu()
                all_targets = targets.data.cpu()
            else:
                all_predicted = torch.cat((all_predicted, predicted.cpu()), 0)
                all_targets = torch.cat((all_targets, targets.data.cpu()), 0)
    
    all_predicted_np = all_predicted.numpy()
    all_targets_np = all_targets.numpy()
    
    accuracy = accuracy_score(all_targets_np, all_predicted_np) * 100
    precision = precision_score(all_targets_np, all_predicted_np, average='weighted') * 100
    f1 = f1_score(all_targets_np, all_predicted_np, average='weighted') * 100
    avg_inference_time = np.mean(inference_times) * 1000  # in milliseconds
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'f1': f1,
        'avg_inference_time': avg_inference_time,
        'predictions': all_predicted_np,
        'targets': all_targets_np
    }

def count_parameters(model):
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Load baseline model
print('==> Loading Baseline SqueezeNet model..')
baseline_path = os.path.join(opt.baseline_dataset + '_' + opt.baseline_model, str(opt.fold))
baseline_net = SqueezeNetModel(num_classes=7)
baseline_checkpoint = torch.load(os.path.join(baseline_path, 'Test_model.t7'), map_location=device, weights_only=False)

if isinstance(baseline_checkpoint['net'], dict):
    baseline_net.load_state_dict(baseline_checkpoint['net'])
else:
    baseline_net.load_state_dict(baseline_checkpoint['net'].state_dict())

baseline_net.to(device)
baseline_total_params, baseline_trainable_params = count_parameters(baseline_net)

# Load PEFT model
print('==> Loading PEFT-SqueezeNet model..')
peft_path = os.path.join(opt.peft_dataset + '_' + opt.peft_model, str(opt.fold))
peft_net = PEFTSqueezeNetModel(num_classes=7, adapter_reduction=opt.adapter_reduction)
peft_checkpoint = torch.load(os.path.join(peft_path, 'Test_model.t7'), map_location=device, weights_only=False)

if isinstance(peft_checkpoint['net'], dict):
    peft_net.load_state_dict(peft_checkpoint['net'])
else:
    peft_net.load_state_dict(peft_checkpoint['net'].state_dict())

peft_net.to(device)
peft_total_params, peft_trainable_params = count_parameters(peft_net)

# Load test data
testset = FER2013(split='Testing', fold=opt.fold, transform=transforms_valid)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Evaluate both models
baseline_results = evaluate_model(baseline_net, testloader, device, 'Baseline SqueezeNet')
peft_results = evaluate_model(peft_net, testloader, device, 'PEFT-SqueezeNet')

# Print comparison
print('\n' + '='*80)
print('MODEL COMPARISON: Baseline vs PEFT-SqueezeNet')
print('='*80)

print('\n--- Recognition Performance ---')
print(f'{"Metric":<20} {"Baseline":<20} {"PEFT":<20} {"Difference":<20}')
print('-'*80)
print(f'{"Accuracy (%)":<20} {baseline_results["accuracy"]:<20.3f} {peft_results["accuracy"]:<20.3f} {peft_results["accuracy"] - baseline_results["accuracy"]:<20.3f}')
print(f'{"Precision (%)":<20} {baseline_results["precision"]:<20.3f} {peft_results["precision"]:<20.3f} {peft_results["precision"] - baseline_results["precision"]:<20.3f}')
print(f'{"F1-score (%)":<20} {baseline_results["f1"]:<20.3f} {peft_results["f1"]:<20.3f} {peft_results["f1"] - baseline_results["f1"]:<20.3f}')

print('\n--- Efficiency Metrics ---')
print(f'{"Metric":<30} {"Baseline":<25} {"PEFT":<25} {"Reduction":<25}')
print('-'*105)
print(f'{"Total Parameters":<30} {baseline_total_params:<25,} {peft_total_params:<25,} {"N/A":<25}')
print(f'{"Trainable Parameters":<30} {baseline_trainable_params:<25,} {peft_trainable_params:<25,} {100*(1-peft_trainable_params/baseline_trainable_params):<25.2f}%')
print(f'{"Trainable %":<30} {100*baseline_trainable_params/baseline_total_params:<25.2f}% {100*peft_trainable_params/peft_total_params:<25.2f}% {100*(1-peft_trainable_params/baseline_trainable_params):<25.2f}%')
print(f'{"Avg Inference Time (ms)":<30} {baseline_results["avg_inference_time"]:<25.3f} {peft_results["avg_inference_time"]:<25.3f} {100*(1-peft_results["avg_inference_time"]/baseline_results["avg_inference_time"]):<25.2f}%')

# Calculate parameter reduction
param_reduction = 100 * (1 - peft_trainable_params / baseline_trainable_params)
print(f'\nParameter Reduction: {param_reduction:.2f}%')
print(f'PEFT uses {peft_trainable_params / baseline_trainable_params:.2%} of baseline trainable parameters')

# Save comparison results
comparison_dir = 'model_comparison'
os.makedirs(comparison_dir, exist_ok=True)
comparison_file = os.path.join(comparison_dir, 'baseline_vs_peft_comparison.txt')

with open(comparison_file, 'w') as f:
    f.write('MODEL COMPARISON: Baseline vs PEFT-SqueezeNet\n')
    f.write('='*80 + '\n\n')
    
    f.write('--- Recognition Performance ---\n')
    f.write(f'{"Metric":<20} {"Baseline":<20} {"PEFT":<20} {"Difference":<20}\n')
    f.write('-'*80 + '\n')
    f.write(f'{"Accuracy (%)":<20} {baseline_results["accuracy"]:<20.3f} {peft_results["accuracy"]:<20.3f} {peft_results["accuracy"] - baseline_results["accuracy"]:<20.3f}\n')
    f.write(f'{"Precision (%)":<20} {baseline_results["precision"]:<20.3f} {peft_results["precision"]:<20.3f} {peft_results["precision"] - baseline_results["precision"]:<20.3f}\n')
    f.write(f'{"F1-score (%)":<20} {baseline_results["f1"]:<20.3f} {peft_results["f1"]:<20.3f} {peft_results["f1"] - baseline_results["f1"]:<20.3f}\n\n')
    
    f.write('--- Efficiency Metrics ---\n')
    f.write(f'{"Metric":<30} {"Baseline":<25} {"PEFT":<25} {"Reduction":<25}\n')
    f.write('-'*105 + '\n')
    f.write(f'{"Total Parameters":<30} {baseline_total_params:<25,} {peft_total_params:<25,} {"N/A":<25}\n')
    f.write(f'{"Trainable Parameters":<30} {baseline_trainable_params:<25,} {peft_trainable_params:<25,} {100*(1-peft_trainable_params/baseline_trainable_params):<25.2f}%\n')
    f.write(f'{"Trainable %":<30} {100*baseline_trainable_params/baseline_total_params:<25.2f}% {100*peft_trainable_params/peft_total_params:<25.2f}% {100*(1-peft_trainable_params/baseline_trainable_params):<25.2f}%\n')
    f.write(f'{"Avg Inference Time (ms)":<30} {baseline_results["avg_inference_time"]:<25.3f} {peft_results["avg_inference_time"]:<25.3f} {100*(1-peft_results["avg_inference_time"]/baseline_results["avg_inference_time"]):<25.2f}%\n\n')
    
    f.write(f'Parameter Reduction: {param_reduction:.2f}%\n')
    f.write(f'PEFT uses {peft_trainable_params / baseline_trainable_params:.2%} of baseline trainable parameters\n')

print(f'\nComparison results saved to {comparison_file}')

