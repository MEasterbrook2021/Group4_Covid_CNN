from src.data.download import CovidxDownloader
from src.data import *
from src.data import viz
from src.model import *

from pathlib import Path
import os
from shutil import copy


NUM_TRAIN, NUM_TEST, NUM_VAL = 200, 200, 0

IMGS_DIR = DataDir.PATH_RAW / Covidx_CXR2.TRAIN
IMAGE_SIZE = (128, 128)
BATCH_SIZE_TRAIN = 10
BATCH_SIZE_TEST = 10

MODEL_TYPE = ModelTypes.RESNET
LEARNING_RATE = 0.001
NUM_EPOCHS = 10


def create_demo_annots(file: Path, output: Path, num, split=0.5):
    # Create the annotations file under data/demo/ based on data/raw/
    df = read_annotations_file(file)
    df = pd.concat([
        df[df.label == Covidx_CXR2.CLASS_POSITIVE].sample(n = int(num * split)), 
        df[df.label == Covidx_CXR2.CLASS_NEGATIVE].sample(n = int(num * (1 - split)))
    ])
    save_annotations_file(df, output)
    return df


def demo(limits):
    num_train, num_test, num_val = limits

    print("_______________________________________________________________________________________________DATA LOADING")
    files = list()
    for af, num in zip(Covidx_CXR2.ANNOTATIONS_FILES, limits):
        # Create the demo annotations files and download the dataset if necessary
        demo_af = DataDir.PATH_DEMO / af
        if not demo_af.is_file():
            if not DataDir.PATH_RAW.is_dir() or len(os.listdir(DataDir.PATH_RAW)) == 0:
                CovidxDownloader().download(DataDir.PATH_RAW)
            df = create_demo_annots(file=DataDir.PATH_RAW / af, output=demo_af, num=num)
        else:
            df = read_annotations_file(demo_af)
        
        # Copy files to be used in the demo
        # file_stem = Path(af).stem
        # for _, row in df.iterrows():
        #     img_filename = row["filename"]
        #     dest = (DataDir.PATH_DEMO / file_stem) / img_filename
        #     if not dest.parent.is_dir():
        #         os.makedirs(dest.parent)
        #     if not dest.is_file():
        #         copy((DataDir.PATH_RAW / file_stem) / img_filename, dest.parent)
        files.append(demo_af)
    train_file, test_file, _ = tuple(files)
    del files
    print(f"Training examples: {num_train}")
    print(f"Testing examples: {num_test}")
    print(f"Validation examples: {num_val}")

    print("______________________________________________________________________________________________VISUALIZATION")
    # Load the training dataset and visualise some examples from each class
    train_df = read_annotations_file(train_file)
    train_dataset = CovidxDataset(train_df, IMGS_DIR, image_size=IMAGE_SIZE)
    viz.show_examples(train_dataset, title="Training Examples (Standardized)", num_examples=5)

    print("___________________________________________________________________________________________________TRAINING")
    # Perform training
    if MODEL_TYPE == ModelTypes.RESNET:
        model = ResnetModel()
    elif MODEL_TYPE == ModelTypes.CUSTOM:
        model = CustomModel(IMAGE_SIZE)

if __name__ == "__main__":
    demo((NUM_TRAIN, NUM_TEST, NUM_VAL))