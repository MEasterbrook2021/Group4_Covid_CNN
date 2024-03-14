from .source import *

import pandas as pd
import os
from pathlib import Path

class DataDir:
    PATH       = Path("data")
    PATH_TRAIN = PATH / Covidx_CXR2.TRAIN
    PATH_TEST  = PATH / Covidx_CXR2.TEST
    PATH_VAL   = PATH / Covidx_CXR2.VAL
    DEMO_PREF  = "demo"
    SANIT_PREF = "s"

    def prefix(path, prefix):
        return path.parent / (f"{prefix}_{path.name}")

    def demo_file(path):
        return DataDir.prefix(path, DataDir.DEMO_PREF)
    
    def sanit_file(path):
        return DataDir.prefix(path, DataDir.SANIT_PREF)

def read_annotations_file(file: Path) -> pd.DataFrame:
    print(f"Reading {file}...")
    # Decouple later
    return pd.read_csv(file, sep=" ", names=Covidx_CXR2.COLUMNS, header=None, dtype=str)

def save_annotations_file(df: pd.DataFrame, path: Path):
    if not path.parent.is_dir():
        os.makedirs(path.parent)
    df.to_csv(path, sep=" ", index=False, header=None)

from .dataset import *