import torch
import torch.nn as nn
import torchvision.models as models


class AdapterBlock(nn.Module):
    """
    Adapter block: 1×1 conv + ReLU + 1×1 conv
    This is inserted within Fire modules for parameter-efficient fine-tuning.
    """
    def __init__(self, in_channels, reduction_factor=16):
        super(AdapterBlock, self).__init__()
        # First 1×1 convolution (down-projection)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_factor, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # Second 1×1 convolution (up-projection)
        self.conv2 = nn.Conv2d(in_channels // reduction_factor, in_channels, kernel_size=1, bias=False)
        
        # Initialize adapters with small weights
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        
    def forward(self, x):
        # Residual connection: x + adapter(x)
        return x + self.conv2(self.relu(self.conv1(x)))


class PEFTSqueezeNetModel(nn.Module):
    """
    Parameter-Efficient Fine-Tuning SqueezeNet 1.1
    - Freezes all original convolutional layers
    - Inserts adapter blocks within selected Fire modules
    - Only adapter parameters are trainable
    """
    def __init__(self, num_classes=7, adapter_reduction=16, adapter_positions=None):
        super(PEFTSqueezeNetModel, self).__init__()
        
        # Load pre-trained SqueezeNet
        self.base_model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        
        # Freeze all original parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Modify the classifier (this will be trainable)
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, 0.0)
        self.base_model.classifier[1] = final_conv
        
        # Insert adapter blocks in Fire modules
        # SqueezeNet 1.1 has Fire modules in features: fire2, fire3, fire4, fire5, fire6, fire7, fire8, fire9
        # Default: insert adapters in fire4, fire5, fire6, fire7, fire8 (middle layers)
        if adapter_positions is None:
            adapter_positions = ['fire4', 'fire5', 'fire6', 'fire7', 'fire8']
        
        self.adapters = nn.ModuleDict()
        self.adapter_positions = adapter_positions
        
        # Insert adapters after the specified Fire modules
        # We need to determine output channels of each Fire module
        # Fire modules output: expand1x1_channels + expand3x3_channels
        for name, module in self.base_model.features.named_children():
            if name in adapter_positions:
                # Fire module outputs concatenated expand1x1 and expand3x3
                # Calculate total output channels
                out_channels = 0
                if hasattr(module, 'expand1x1'):
                    out_channels += module.expand1x1.out_channels
                if hasattr(module, 'expand3x3'):
                    out_channels += module.expand3x3.out_channels
                
                if out_channels > 0:
                    adapter = AdapterBlock(out_channels, reduction_factor=adapter_reduction)
                    self.adapters[f'{name}_adapter'] = adapter
        
        # Make adapters and classifier trainable
        for param in self.adapters.parameters():
            param.requires_grad = True
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        # Forward through features with adapters
        for name, module in self.base_model.features.named_children():
            x = module(x)
            # Apply adapter if this Fire module has one
            if name in self.adapter_positions:
                adapter_key = f'{name}_adapter'
                if adapter_key in self.adapters:
                    x = self.adapters[adapter_key](x)
        
        # Forward through classifier
        x = self.base_model.classifier(x)
        # SqueezeNet outputs (batch, num_classes, 1, 1), reshape to (batch, num_classes)
        return x.view(x.size(0), -1)
    
    def count_trainable_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_parameters(self):
        """Count the total number of parameters"""
        return sum(p.numel() for p in self.parameters())

