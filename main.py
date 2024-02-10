import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision.datasets.folder import default_loader
import os
import cv2
import torch
import matplotlib.pyplot as plt


# def load_images(folder_path, limit, resize_shape):
#     images = []
#     for filename in os.listdir(folder_path):
#         if(len(images) >= limit):
#             break
#         img_path = os.path.join(folder_path, filename)
#         if os.path.isfile(img_path):
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is not None:
#                 img = cv2.resize(img, resize_shape)
#                 images.append(img)
#     return images

# def read_labels(folder_path, limit):
#     labels = []
#     negative = 0
#     positive = 0
#     count = 0
#     with open(folder_path, 'r') as file:
#         for line in file:
#             words = line.split()
            
#             if(words[2] == 'negative'):
#                 negative += 1
#                 labels.append(0)
#             else:
#                 positive += 1
#                 labels.append(1)
#             # labels.append(words[2])
#             count += 1
#             if count >= limit:
#                 break
#     print(positive, negative)
#     return labels

# def stack_tensors(img_array):
#     stack_array = np.stack(img_array)
#     images_tensor = transforms.ToTensor()(stack_array).permute(1, 0, 2) # PyTorch takes in tensors of with this shape.
#     print(images_tensor.shape)

#     return images_tensor

transform = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_total_lables(filepaths):
    labels = []
    
    count = 0
    for path in filepaths:
        negative = 0
        positive = 0
        with open(path, 'r') as file:
            for line in file:
                words = line.split()

                if words[2] == "negative":
                    negative += 1
                elif words[2] == "positive":
                    positive += 1
                else:
                    count += 1
        print("In {} there are {} negative datapoints and {} positive datapoints".format(path, negative, positive))
    return negative, positive

class NNModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(NNModel, self).__init__()
        # Define our own layers? Perfomance will not be good!
        self.conv_layer1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, padding=1, stride=1)
        self.conv_layer2 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1, stride=1)
        self.pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2) 
        self.fc_layer1 = torch.nn.Linear(in_features=40 * 56 * 56, out_features=1200)
        self.fc_layer2 = torch.nn.Linear(in_features=1200, out_features=100)
        self.fc_layer3 = torch.nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = self.pooling_layer(torch.relu(self.conv_layer1(x)))
        x = self.pooling_layer(torch.relu(self.conv_layer2(x)))

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc_layer1(x))
        x = torch.relu(self.fc_layer2(x))
        x = self.fc_layer3(x)
        x = torch.sigmoid(x)

        return x
    
class PretrainedModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(PretrainedModel, self).__init__()

        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1) # Does not take in pretrained parameter anymore

        num_features = self.resnet.fc.in_features # Extracting number of input features from the fully connected layer in the resnet50.
        self.resnet.fc = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.sigmoid(x)

        return x

class Trainer():
    def __init__(self, model, batch_size, learning_rate, nb_epochs, image_filepaths, label_filepaths, data_limit, resize_shape):
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = nb_epochs
        self.train_img_filepaths, self.val_img_filepaths, self.test_img_filepaths = image_filepaths
        self.train_label_filepaths, self.val_label_filepaths, self.test_label_filepaths = label_filepaths
        self.data_limit = data_limit
        self.resize_shape = resize_shape

    def load_images(self, folder_path, limit, resize_shape):
        images = []
        for filename in os.listdir(folder_path):
            if(len(images) >= limit):
                break
            img_path = os.path.join(folder_path, filename)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (resize_shape, resize_shape))
                    images.append(img)
        return images
    
    def read_labels(self, folder_path, limit):
        labels = []
        count = 0
        negative_count = 0
        positive_count = 0
        with open(folder_path, 'r') as file:
            for line in file:
                words = line.split()

                if(words[2] == 'negative'):
                    labels.append(0)
                    negative_count += 1

                else:
                    labels.append(1)
                    positive_count += 1
                count += 1
                
                if count >= limit:
                    break
                
        return labels
        
        
    def preprocessor(self, image_filepath, label_filepath, limit, resize_shape):
        images = self.load_images(image_filepath, limit, resize_shape)
        labels = self.read_labels(label_filepath, limit)
        stack_array = np.stack(images)
        images_tensor = transforms.ToTensor()(stack_array).permute(1, 0, 2)
        images_tensor = images_tensor.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        dataset = TensorDataset(images_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        return loader


    def train(self):
        loss_func = torch.nn.BCELoss()
        self.train_loader = self.preprocessor(self.train_img_filepaths, self.train_label_filepaths, self.data_limit, self.resize_shape)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # Maybe add in weight decay? Parameter to look into...

        for epoch in range(self.epochs):
            print("Epoch {}".format(epoch + 1))
            self.model.train()
            total_training_loss = 0

            for batch_inputs, batch_labels in self.train_loader:

                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device) # Moving tensors to GPU if available

                optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                outputs = outputs.squeeze(1)
                loss = loss_func(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_training_loss += loss.item()

        return self.model

    def eval(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        loss_func = torch.nn.BCELoss()
        self.test_loader = self.preprocessor(self.test_img_filepaths, self.test_label_filepaths, self.data_limit, self.resize_shape)

        with torch.no_grad():
            for batch_inputs, batch_labels in self.test_loader:

                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

                outputs = self.model(batch_inputs) # Since the output is from the sigmoid activation function, it can take on any value between 0 and 1.
                outputs = outputs.squeeze(1)
                predictions = torch.sigmoid(outputs) > 0.5 # We will set the threshold to be exactly at the center (THIS CAN BE TUNED!)
                predictions = predictions.float()
                
                loss = loss_func(predictions, batch_labels)
                total_loss += loss.item()
                correct_predictions += (predictions == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)

        avg_loss = total_loss / len(self.test_loader.dataset)
        accuracy = correct_predictions / total_predictions
        # Can also look at false positives/false negatives and metrics like that

        return accuracy, avg_loss


def main():
    # print("current directory:", os.getcwd())
    # root_dir = "./data/train"

    label_filepaths = ["./data/train.txt", "./data/val.txt", "./data/test.txt"]
    img_filepaths = ["./data/train", "./data/val", "./data/test"]

    model = PretrainedModel(1)
    model = model.to(device)
    trainer = Trainer(model, batch_size=32, learning_rate=0.001, nb_epochs=1, image_filepaths=img_filepaths, label_filepaths=label_filepaths, data_limit=3000, resize_shape=224)
    trainer.train()
    accuracy, avg_loss = trainer.eval()
    print(accuracy)    

    # negative, positive = check_total_lables(filepaths=filepaths)
    # print(negative, positive)

    # categories = ["negative", "positive"]
    # plt.bar(categories, [negative, positive])

    # plt.xlabel("categories")
    # plt.ylabel("count")
    # plt.title("number of negative datapoints vs positive datapoints")

    # plt.show()

    # labels = read_labels("./data/train.txt", 1500)
    # labels_tensor = torch.tensor(labels)

    # print(labels_tensor.shape)


if __name__ == "__main__":
    main()