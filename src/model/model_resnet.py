import torch
import torchvision

from src.data.source import *

class ResnetModel(torch.nn.Module):
    def __init__(self):
        super(ResnetModel, self).__init__()

        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1) # Does not take in pretrained parameter anymore

        num_features = self.resnet.fc.in_features # Extracting number of input features from the fully connected layer in the resnet50.
        self.resnet.fc = torch.nn.Linear(num_features, len(Covidx_CXR2.CLASSES))

    def forward(self, x):
        x = self.resnet(x)
        x = torch.sigmoid(x)

        return x