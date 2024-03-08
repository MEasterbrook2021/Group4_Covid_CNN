from .source import *

import pandas as pd
import os
from pathlib import Path

class DataDir:
    PATH = Path("data")
    PATH_RAW = PATH / "raw"
    PATH_DEMO = PATH / "demo"

def read_annotations_file(file: Path) -> pd.DataFrame:
    print(f"Reading {file}...")
    # Decouple later
    return pd.read_csv(file, sep=" ", names=Covidx_CXR2.COLUMNS, header=None, dtype=str)

def save_annotations_file(df: pd.DataFrame, path: Path):
    if not path.parent.is_dir():
        os.makedirs(path.parent)
    df.to_csv(path, sep=" ", index=False, header=None)

from .data import *