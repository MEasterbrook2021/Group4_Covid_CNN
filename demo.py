from src.data.download import CovidxDownloader
from src.data import *
from src.data import viz
from src.model import *
from src.model.train import Trainer
from src.model.eval import Evaluator

from torchsummary import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pathlib import Path
import os


STEPS = [
    "download",
    "viz",
    "model",
    "train",
    # "valid",
    "eval",
    "stats",
    # "save",
]

NUM_TRAIN, NUM_TEST, NUM_VAL = 20, 50, 50
IMAGE_SIZE        = (224, 224)
BATCH_SIZE_TRAIN  = 10
BATCH_SIZE_TEST   = 10
AUGMENT_NOISE     = (0.0, 0.05)

MODEL_TYPE        = ModelTypes.RESNET
LEARNING_RATE     = 0.001
NUM_EPOCHS        = 5
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


def demo(limits):
    num_train, num_test, num_val = limits

    print_section_header("DATA LOADING")
    dfs = list()
    for af, num in zip(Covidx_CXR2.ANNOTATIONS_FILES, limits):
        # Create the demo annotations files and download the dataset if necessary
        demo_af = DataDir.demo_file(DataDir.PATH / af)
        if not demo_af.is_file():
            if "download" in STEPS:
                CovidxDownloader().download(DataDir.PATH)
            df = create_demo_annots(file=DataDir.PATH / af, output=demo_af, num=num)
        else:
            df = read_annotations_file(demo_af)
            if df.shape[0] != num:
                os.remove(demo_af)
                df = create_demo_annots(file=DataDir.PATH / af, output=demo_af, num=num)

        assert(df.shape[0] == num)
        dfs.append(df)
    train_df, test_df, val_df = tuple(dfs)
    del dfs
    print_info("Training examples",   num_train)
    print_info("Testing examples",    num_test)
    print_info("Validation examples", num_val)

    # Load the training dataset and visualise some examples from each class
    train_dataset = CovidxDataset(train_df, DataDir.PATH_TRAIN, image_size=IMAGE_SIZE, noise=AUGMENT_NOISE)
    if "viz" in STEPS:
        viz.show_examples(train_dataset, title="Training Examples (Normalized and Augmented)", num_examples=5)

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
    train_losses = []
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
        train_losses = trainer.epoch_losses
    
    # Perform final evaluation
    if "eval" in STEPS:
        print_section_header("evaluation")
        test_dataset = CovidxDataset(test_df, DataDir.PATH_TEST, image_size=IMAGE_SIZE)
        test_dl = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_TEST
        )
        print_info("Testing batch size", BATCH_SIZE_TEST)
        evaluator = Evaluator(model, device, test_dl)
        evaluator.eval()
        print(f"Accuracy: {evaluator.accuracy}")
        
    # Show some stats and graphs about the model
    if "stats" in STEPS:
        fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
        viz.plot_loss(axarr[0, 0], train_losses, [])
        plt.show()
    
    # Save the model
    if "save" in STEPS:
        print_section_header("saving")
        out_pref = f"{Covidx_CXR2.NAME}-{MODEL_TYPE.value}-{NUM_EPOCHS}epochs"
        max_ver = 0
        for file in Path(ModelDir.PATH_OUTPUT).glob(f"{out_pref}-*.pt"):
            ver = abs(int(file.stem.split("-")[-1]))
            if ver > max_ver:
                max_ver = ver
        out = ModelDir.PATH_OUTPUT / f"{out_pref}-{max_ver + 1:04d}.pt"
        print(f"Saving model to {out}...")
        if not ModelDir.PATH_OUTPUT.is_dir():
            os.makedirs(ModelDir.PATH_OUTPUT)
        torch.save(model.state_dict(), out)
        print("Saved!")

    print_section_header("finished")


if __name__ == "__main__":
    demo((NUM_TRAIN, NUM_TEST, NUM_VAL))