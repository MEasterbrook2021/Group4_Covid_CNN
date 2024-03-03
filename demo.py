from src.data.source import *
from src.data.download import CovidxDownloader
from src.data.process import *

from pathlib import Path
import os
from shutil import copy


def create_demo_annots(file: Path, output: Path, num, split=0.5, copy_imgs=True):
    # Create the annotations file under data/demo/ based on data/raw/
    df = read_annotations_file(file)
    df = pd.concat([
        df[df.label == Covidx_CXR2.CLASS_POSITIVE].sample(n = int(num * split)), 
        df[df.label == Covidx_CXR2.CLASS_NEGATIVE].sample(n = int(num * (1 - split)))
    ])
    save_annotations_file(df, output)
    return df


def demo(limits):
    num_train, num_test, _ = limits

    dfs = list()
    for af, num in zip(Covidx_CXR2.ANNOTATIONS_FILES, limits):
        # Create the demo annotations files and download the dataset if necessary
        demo_af = Data.DEMO / af
        if not demo_af.is_file():
            if not Data.RAW.is_dir() or len(os.listdir(Data.RAW)):
                CovidxDownloader().download(Data.RAW)
            df = create_demo_annots(file=Data.RAW / af, output=demo_af, num=num)
        else:
            df = read_annotations_file(demo_af)
        # Copy files to be used in the demo
        file_stem = Path(af).stem
        for _, row in df.iterrows():
            img_filename = row["filename"]
            dest = (Data.DEMO / file_stem) / img_filename
            if not dest.parent.is_dir():
                os.makedirs(dest.parent)
            if not dest.is_file():
                copy((Data.RAW / file_stem) / img_filename, dest.parent)
        dfs.append(df)
    train_df, test_df, _ = tuple(dfs)
    del dfs




if __name__ == "__main__":
    demo((200, 200, 0))