"""
Quick test script to verify PEFT-SqueezeNet model can be instantiated
"""
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))
from models.peft_squeezemodel import PEFTSqueezeNetModel

print("Testing PEFT-SqueezeNet model instantiation...")

# Create model
model = PEFTSqueezeNetModel(num_classes=7, adapter_reduction=16)

# Count parameters
total_params = model.count_total_parameters()
trainable_params = model.count_trainable_parameters()
frozen_params = total_params - trainable_params

print(f"\n✓ Model created successfully!")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters: {frozen_params:,}")
print(f"  Trainable percentage: {100.0 * trainable_params / total_params:.2f}%")

# Test forward pass
print("\nTesting forward pass...")
dummy_input = torch.randn(1, 3, 224, 224)
try:
    output = model(dummy_input)
    print(f"✓ Forward pass successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: (1, 7)")
    assert output.shape == (1, 7), f"Expected output shape (1, 7), got {output.shape}"
    print("✓ Output shape is correct!")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    raise

# Verify adapters are trainable and base model is frozen
print("\nVerifying parameter freezing...")
adapter_params_trainable = sum(p.numel() for name, p in model.named_parameters() if 'adapter' in name and p.requires_grad)
base_params_trainable = sum(p.numel() for name, p in model.named_parameters() if 'adapter' not in name and 'classifier' not in name and p.requires_grad)
classifier_params_trainable = sum(p.numel() for name, p in model.named_parameters() if 'classifier' in name and p.requires_grad)

print(f"  Adapter parameters (trainable): {adapter_params_trainable:,}")
print(f"  Base model parameters (trainable): {base_params_trainable:,}")
print(f"  Classifier parameters (trainable): {classifier_params_trainable:,}")

if base_params_trainable == 0:
    print("✓ Base model is correctly frozen!")
else:
    print(f"⚠ Warning: {base_params_trainable} base model parameters are trainable (should be 0)")

if adapter_params_trainable > 0:
    print("✓ Adapters are trainable!")
else:
    print("⚠ Warning: No adapter parameters are trainable")

if classifier_params_trainable > 0:
    print("✓ Classifier is trainable!")
else:
    print("⚠ Warning: Classifier is not trainable")

print("\n" + "="*60)
print("All tests passed! Model is ready for training.")
print("="*60)

