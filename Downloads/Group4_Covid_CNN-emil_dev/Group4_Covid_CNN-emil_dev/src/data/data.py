from src.data import read_annotations_file
from .source import *

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import pandas as pd

from pathlib import Path
from random import Random


class CovidxDataset(Dataset):
    def __init__(self, annotations_file: Path, img_dir: Path, image_size, rng=Random()):
        df = read_annotations_file(annotations_file)
        self.__init__(df, img_dir, image_size, rng)

    def __init__(self, dataframe: pd.DataFrame, img_dir: Path, image_size, rng=Random()):
        self.df: pd.DataFrame = dataframe
        self.img_dir = img_dir
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
        ])
        self.rng = rng

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        img_filename = self.df["filename"].iloc[idx]
        img_path = self.img_dir / img_filename
        lbl = self.df["label"].iloc[idx]
        lbl = 1 if lbl == Covidx_CXR2.CLASS_POSITIVE else 0
        return self.__load_image(img_path), lbl

    def get_random(self) -> tuple[torch.Tensor, int]:
        return self[self.rng.randint(0, len(self) - 1)]
    
    def get_random(self, label: str) -> tuple[torch.Tensor, int]:
        item = self.df[self.df.label == label].sample(1).iloc[0]
        img_filename = item["filename"]
        img_path = self.img_dir / img_filename
        lbl = item["label"]
        lbl = 1 if lbl == Covidx_CXR2.CLASS_POSITIVE else 0
        return self.__load_image(img_path), lbl
    
    def __load_image(self, img_path):
        img = cv2.imread(str(img_path))
        img = self.img_transform(img)
        img = self.__standardize(img)
        return img

    def __standardize(self, img):
        mean = img.mean()
        std = img.std()
        norm = transforms.Normalize(mean=mean, std=std)
        return norm(img)