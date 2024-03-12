from src.data import read_annotations_file
from .source import *

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np

from pathlib import Path
from random import Random


class CovidxDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: Path, image_size, noise=None, normalize=True):
        self.df: pd.DataFrame = dataframe
        self.img_dir = img_dir
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
        ])
        self.noise = noise
        self.normalize = normalize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        img_filename = self.df["filename"].iloc[idx]
        img_path = self.img_dir / img_filename
        lbl = self.df["label"].iloc[idx]
        lbl = 1 if lbl == Covidx_CXR2.CLASS_POSITIVE else 0
        return self.__load_image(img_path), lbl

    def get_random(self) -> tuple[torch.Tensor, int]:
        return self[torch.randint(low=0, high=len(self), size=()).item()]
    
    def get_random(self, label: str) -> tuple[torch.Tensor, int]:
        df = self.df[self.df.label == label]
        i = torch.randint(low=0, high=df.shape[0], size=()).item()
        item = df.iloc[i]
        img_filename = item["filename"]
        img_path = self.img_dir / img_filename
        lbl = item["label"]
        lbl = 1 if lbl == Covidx_CXR2.CLASS_POSITIVE else 0
        return self.__load_image(img_path), lbl
    
    def __load_image(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = self.img_transform(img)
        if self.noise is not None:
            (mean, std) = self.noise
            img += torch.randn(img.shape) * std + mean
        if self.normalize:
            img = self.__normalize(img)
        img = img.repeat((3, 1, 1))
        return img

    def __normalize(self, img):
        mean = img.mean()
        std = img.std()
        norm = transforms.Normalize(mean=mean, std=std)
        return norm(img)