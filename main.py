import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision.datasets.folder import default_loader
import os
import cv2

def load_images(folder_path, limit, resize_shape):
    images = []
    for filename in os.listdir(folder_path):
        if(len(images) >= limit):
            break
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, resize_shape)
                images.append(img)
    return images

def read_labels(folder_path, limit):
    labels = []
    negative = 0
    positive = 0
    count = 0
    with open(folder_path, 'r') as file:
        for line in file:
            words = line.split()
            
            if(words[2] == 'negative'):
                negative += 1
            else:
                positive += 1
            labels.append(words[2])
            count += 1
            if count >= limit:
                break
    print(positive, negative)
    return labels
labels = read_labels("./data/train.txt", 1500)
print(len(labels))

print("current directory:", os.getcwd())

root_dir = "./data/train"

train_dataset = load_images(root_dir, 1500, (224, 224))
# print((train_dataset[0].shape))
print(len(train_dataset))