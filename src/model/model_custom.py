import torch

class CustomModel(torch.nn.Module):
    def __init__(self, input_size):
        super(CustomModel, self).__init__()
        # Define our own layers? Perfomance will not be good!
        self.conv_layer1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, padding=1, stride=1)
        self.conv_layer2 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1, stride=1)
        self.pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2) 
        self.fc_layer1 = torch.nn.Linear(in_features=40 * 56 * 56, out_features=1200)
        self.fc_layer2 = torch.nn.Linear(in_features=1200, out_features=100)
        self.fc_layer3 = torch.nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = torch.relu(x)
        x = self.pooling_layer(x)

        x = self.conv_layer2(x)
        x = torch.relu(x)
        x = self.pooling_layer(x)

        x = torch.flatten(x, 1)
        x = self.fc_layer1(x)
        x = torch.relu(x)
        x = self.fc_layer2(x)
        x = torch.relu(x)
        x = self.fc_layer3(x)
        x = torch.sigmoid(x)

        return x