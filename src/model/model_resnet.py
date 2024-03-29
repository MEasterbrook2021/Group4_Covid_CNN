import torch
import torchvision

from src.data.source import *

class ResnetModel(torch.nn.Module):
    def __init__(self, freeze_layers=True):
        super(ResnetModel, self).__init__()

        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1) # Does not take in pretrained parameter anymore

        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        num_features = self.resnet.fc.in_features # Extracting number of input features from the fully connected layer in the resnet50.
        self.resnet.fc = torch.nn.Linear(num_features, 1)

    def __str__(self):
        return "Adapted Resnet50 (Binary Classification)"

    def forward(self, x):
        x = self.resnet(x)
        x = torch.sigmoid(x)

        return x
    
    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True