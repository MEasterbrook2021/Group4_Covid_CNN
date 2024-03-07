from src.data.source import *
from src.data.download import CovidxDownloader
from src.data import *
from src.data import viz

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

    files = list()
    for af, num in zip(Covidx_CXR2.ANNOTATIONS_FILES, limits):
        # Create the demo annotations files and download the dataset if necessary
        demo_af = DataDir.PATH_DEMO / af
        if not demo_af.is_file():
            if not DataDir.PATH_RAW.is_dir() or len(os.listdir(DataDir.PATH_RAW)):
                CovidxDownloader().download(DataDir.PATH_RAW)
            df = create_demo_annots(file=DataDir.PATH_RAW / af, output=demo_af, num=num)
        else:
            df = read_annotations_file(demo_af)
        # Copy files to be used in the demo
        file_stem = Path(af).stem
        for _, row in df.iterrows():
            img_filename = row["filename"]
            dest = (DataDir.PATH_DEMO / file_stem) / img_filename
            if not dest.parent.is_dir():
                os.makedirs(dest.parent)
            if not dest.is_file():
                copy((DataDir.PATH_RAW / file_stem) / img_filename, dest.parent)
        files.append(demo_af)
    train_file, test_file, _ = tuple(files)
    del files

    # Load the training dataset and visualise some examples from each class
    train_df = read_annotations_file(train_file)
    train_dataset = CovidxDataset(train_df, DataDir.PATH_DEMO / Covidx_CXR2.TRAIN, image_size=(128, 128))
    viz.show_examples(train_dataset, title="Training Examples (Standardized)", num_examples=5)


if __name__ == "__main__":
    demo((200, 200, 0))