import torch
import math

class CustomModel(torch.nn.Module):
    def __init__(self, img_size):
        super(CustomModel, self).__init__()

        # Define our own layers? Perfomance will not be good!
        self.conv1 = CustomModel.ConvLayer(3, 20)
        self.conv2 = CustomModel.ConvLayer(20, 40)
        i_x = img_size[0] / (2 ** len([self.conv1, self.conv2]))
        i_y = img_size[1] / (2 ** len([self.conv1, self.conv2]))
        fc_in = 40 * i_x * i_y
        self.fc1 = CustomModel.FCLayer(int(fc_in), 1200)
        self.fc2 = CustomModel.FCLayer(1200, 100)
        self.fc3 = CustomModel.FCLayer(100, 1, torch.nn.Sigmoid())

    def __str__(self):
        return "Custom CNN Binary Classification Model"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    class ConvLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride=2):
            super().__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.seq(x)
        
    class FCLayer(torch.nn.Module):
        def __init__(self, in_channels, out_channels, activation=torch.nn.ReLU()):
            super().__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels),
                activation
            )

        def forward(self, x):
            return self.seq(x)