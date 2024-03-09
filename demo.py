from src.data.download import CovidxDownloader
from src.data import *
from src.data import viz
from src.model import *
from src.model.train import Trainer
from src.model.eval import Evaluator

from torchsummary import summary
from torch.utils.data import DataLoader

from pathlib import Path
import os
from shutil import copy


STEPS = [
    # "download",
    # "viz",
    "model",
    "train",
    "valid",
    "eval",
    "save",
    "stats",
]

TRAIN_IMGS_DIR = DataDir.PATH_RAW / Covidx_CXR2.TRAIN
TEST_IMGS_DIR = DataDir.PATH_RAW / Covidx_CXR2.TEST
VAL_IMGS_DIR = DataDir.PATH_RAW / Covidx_CXR2.VAL
COPY_DEMO_FILES = False

NUM_TRAIN, NUM_TEST, NUM_VAL = 50, 50, 50
IMAGE_SIZE = (224, 224)
BATCH_SIZE_TRAIN = 10
BATCH_SIZE_TEST = 10

MODEL_TYPE = ModelTypes.RESNET
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
EVAL_AFTER_EPOCHS = int(NUM_EPOCHS / 5)
SAVE_AFTER_EPOCHS = NUM_EPOCHS


def print_section_header(str: str):
    print(f"\n{f" {str.upper()} ":═^120}")

def print_info(str: str, val):
    print(f"> {str}: {val}")

def create_demo_annots(file: Path, output: Path, num, split=0.5):
    # Create the annotations file under data/demo/ based on data/raw/
    df = read_annotations_file(file)
    df = pd.concat([
        df[df.label == Covidx_CXR2.CLASS_POSITIVE].sample(n = int(num * split)), 
        df[df.label == Covidx_CXR2.CLASS_NEGATIVE].sample(n = int(num * (1 - split)))
    ])
    save_annotations_file(df, output)
    return df

def copy_demo_files(df: pd.DataFrame, file_stem):
    for _, row in df.iterrows():
        img_filename = row["filename"]
        dest = (DataDir.PATH_DEMO / file_stem) / img_filename
        if not dest.parent.is_dir():
            os.makedirs(dest.parent)
        if not dest.is_file():
            copy((DataDir.PATH_RAW / file_stem) / img_filename, dest.parent)


def demo(limits):
    num_train, num_test, num_val = limits

    print_section_header("DATA LOADING")
    dfs = list()
    for af, num in zip(Covidx_CXR2.ANNOTATIONS_FILES, limits):
        # Create the demo annotations files and download the dataset if necessary
        demo_af = DataDir.PATH_DEMO / af
        if not demo_af.is_file():
            if "download" in STEPS and (not DataDir.PATH_RAW.is_dir() or len(os.listdir(DataDir.PATH_RAW)) == 0):
                CovidxDownloader().download(DataDir.PATH_RAW)
            df = create_demo_annots(file=DataDir.PATH_RAW / af, output=demo_af, num=num)
        else:
            df = read_annotations_file(demo_af)
        # Copy files to be used in the demo
        if COPY_DEMO_FILES:
            copy_demo_files(df, Path(af).stem)
        dfs.append(df)
    train_df, test_df, val_df = tuple(dfs)
    del dfs
    print_info("Training examples",   num_train)
    print_info("Testing examples",    num_test)
    print_info("Validation examples", num_val)

    # Load the training dataset and visualise some examples from each class
    train_dataset = CovidxDataset(train_df, TRAIN_IMGS_DIR, image_size=IMAGE_SIZE)
    if "viz" in STEPS:
        print_section_header("visualization")
        viz.show_examples(train_dataset, title="Training Examples (Standardized)", num_examples=5)

    # Get the device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if MODEL_TYPE == ModelTypes.RESNET:
        model = ResnetModel()
    elif MODEL_TYPE == ModelTypes.CUSTOM:
        model = CustomModel(IMAGE_SIZE)

    # Print stats and info about the model
    if "model" in STEPS:
        print_section_header("model")
        print_info("Model", model)
        print_info("Device", device)
        summary(model, input_size=(3, IMAGE_SIZE[0], IMAGE_SIZE[1]), device=device_name)
    
    # Perform training
    if "train" in STEPS:
        print_section_header("training")
        train_dl = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE_TRAIN, 
            shuffle=True, 
            pin_memory=(device_name=="cuda")
        )
        print_info("Total epochs", NUM_EPOCHS)
        print_info("Training batch size", BATCH_SIZE_TRAIN)
        trainer = Trainer(model, device, train_dl, LEARNING_RATE, NUM_EPOCHS)
        trainer.train(NUM_EPOCHS)
        print(f"Losses: {trainer.epoch_losses}")
    
    # Perform final evaluation
    if "eval" in STEPS:
        print_section_header("evaluation")
        test_dataset = CovidxDataset(test_df, TEST_IMGS_DIR, image_size=IMAGE_SIZE)
        test_dl = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_TEST
        )
        print_info("Testing batch size", BATCH_SIZE_TEST)
        evaluator = Evaluator(model, device, test_dl)
        evaluator.eval()
        print(f"Accuracy: {evaluator.accuracy}")
    
    # Save the model
    if "save" in STEPS:
        print_section_header("saving")
        if not ModelDir.PATH_OUTPUT.is_dir():
            os.makedirs(ModelDir.PATH_OUTPUT)
        torch.save(
            model.state_dict(),
            ModelDir.PATH_OUTPUT / 
                f"{Covidx_CXR2.NAME}-{MODEL_TYPE.value}-{NUM_EPOCHS}epochs.pt"
        )


if __name__ == "__main__":
    demo((NUM_TRAIN, NUM_TEST, NUM_VAL))