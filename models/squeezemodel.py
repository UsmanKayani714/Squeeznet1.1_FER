import torch
import torch.nn as nn
import torchvision.models as models

class SqueezeNetModel(nn.Module):
    def __init__(self, num_classes=7): # Default to 7 for FER2013
        super(SqueezeNetModel, self).__init__()
        
        # Load pre-trained SqueezeNet
        self.base_model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        
        # Modify the classifier
        # SqueezeNet classifier[1] is the Conv2d layer we need to replace
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # Initialize weights
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, 0.0)
            
        self.base_model.classifier[1] = final_conv
        
    def forward(self, x):
        x = self.base_model(x)
        # SqueezeNet outputs (batch, num_classes, 1, 1), reshape to (batch, num_classes)
        return x.view(x.size(0), -1)