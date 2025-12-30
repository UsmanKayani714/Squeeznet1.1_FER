# PEFT-SqueezeNet Extension

This folder contains the Parameter-Efficient Fine-Tuning (PEFT) extension of the baseline SqueezeNet 1.1 model.

## Overview

The PEFT-SqueezeNet model extends the baseline by:
- **Freezing all original convolutional layers** from the pre-trained SqueezeNet 1.1
- **Inserting small adapter blocks** (1×1 conv + ReLU + 1×1 conv) within selected Fire modules
- **Training only adapter parameters** and the final classifier layer

This approach significantly reduces the number of trainable parameters while maintaining competitive performance.

## Architecture

### Adapter Block
Each adapter block consists of:
1. **Down-projection**: 1×1 convolution that reduces channels by a reduction factor (default: 16)
2. **ReLU activation**
3. **Up-projection**: 1×1 convolution that restores original channel count
4. **Residual connection**: `output = input + adapter(input)`

### Adapter Placement
By default, adapters are inserted after Fire modules: `fire4`, `fire5`, `fire6`, `fire7`, `fire8` (middle layers).

## Files

- `models/peft_squeezemodel.py`: PEFT-SqueezeNet model implementation
- `train_peft_squeezenet.py`: Training script for PEFT model
- `evaluate_peft.py`: Evaluation script with Accuracy, Precision, and F1-score
- `compare_models.py`: Comparison script for baseline vs PEFT models

## Usage

### 1. Train PEFT-SqueezeNet Model

```bash
cd peft_extension
python train_peft_squeezenet.py --bs 32 --lr 0.001 --adapter_reduction 16
```

**Arguments:**
- `--bs`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--adapter_reduction`: Adapter reduction factor (default: 16)
- `--fold`: K-fold number (default: 1)
- `--resume`: Resume from checkpoint

### 2. Evaluate PEFT Model

```bash
python evaluate_peft.py --adapter_reduction 16
```

This will compute:
- Accuracy
- Precision (weighted)
- F1-score (weighted)
- Confusion matrix
- Classification report

### 3. Compare Baseline vs PEFT Models

```bash
python compare_models.py --adapter_reduction 16
```

This will compare:
- **Recognition Performance**: Accuracy, Precision, F1-score
- **Efficiency Metrics**: 
  - Number of trainable parameters
  - Training time per epoch
  - Inference time

## Expected Results

The PEFT model should achieve:
- **Significant parameter reduction**: ~90-95% fewer trainable parameters
- **Competitive performance**: Similar or slightly lower accuracy compared to baseline
- **Faster training**: Reduced training time per epoch due to fewer parameters

## Model Parameters

The PEFT model tracks:
- Total parameters (frozen + trainable)
- Trainable parameters (adapters + classifier)
- Trainable percentage

These metrics are printed during training and saved in the checkpoint.

## Notes

- The baseline model must be trained first before comparison
- Both models use the same FER2013 dataset
- Adapter reduction factor controls the size of adapter blocks (higher = smaller adapters)

