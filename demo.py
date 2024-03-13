from src.data.download import CovidxDownloader
from src.data import *
from src.data import viz
from src.model import *
from src.model.train import Trainer
from src.model.eval import Evaluator
from src.model.hyperparameter import TuneHyperparameters

from torchsummary import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pathlib import Path
import os


STEPS = [
    "download",
    # "viz",
    "model",
    "train",
    "valid",
    # "save",
    "test",
    "stats",
    "tune",
    "save",
]
NUM_WORKERS = 0
LOAD_MODEL_NAME = "covidx-cxr2-resnet-10epochs-0003"

NUM_TRAIN, NUM_TEST, NUM_VAL = 50, 50, 50
IMAGE_SIZE        = (224, 224)
BATCH_SIZE_TRAIN  = 10
BATCH_SIZE_TEST   = 10
AUGMENT_NOISE     = (0.0, 0.01)
THRESHOLD         = 0.5

MODEL_TYPE        = ModelTypes.RESNET
FREEZE            = True
LEARNING_RATE     = 0.01
NUM_EPOCHS        = 3
VAL_AFTER_EPOCHS  = 1
SAVE_AFTER_EPOCHS = NUM_EPOCHS


def print_section_header(str: str):
    # print(f"\n{f" {str.upper()} ":═^120}")
    print(f"\n{' ' + str.upper() + '':═^120}")


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
        df = df.sample(frac=1).reset_index(drop=True)
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
        model = ResnetModel(freeze_layers=FREEZE)
    elif MODEL_TYPE == ModelTypes.CUSTOM:
        model = CustomModel(IMAGE_SIZE)
    if "train" not in STEPS:
        model_path = ModelDir.PATH_OUTPUT / f"{LOAD_MODEL_NAME}.pt"
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    model.to(device)

    # Print stats and info about the model
    if "model" in STEPS:
        print_section_header("model")
        print_info("Model", model)
        print_info("Device", device)
        summary(model, input_size=(3, IMAGE_SIZE[0], IMAGE_SIZE[1]), device=device_name)
    
    # Perform training
    train_losses, val_losses = None, None
    if "train" in STEPS:
        print_section_header("training")
        train_dl = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE_TRAIN, 
            shuffle=True, 
            pin_memory=(device_name=="cuda"),
            num_workers=NUM_WORKERS
        )
        val_dataset = CovidxDataset(val_df, DataDir.PATH_VAL, image_size=IMAGE_SIZE)
        val_dl = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE_TRAIN,
            shuffle=True,
            pin_memory=(device_name=="cuda"),
            num_workers=NUM_WORKERS
        )
        print_info("Total epochs", NUM_EPOCHS)
        print_info("Training batch size", BATCH_SIZE_TRAIN)
        trainer = Trainer(
            model, device, 
            train_dl, val_dl,
            LEARNING_RATE, NUM_EPOCHS, VAL_AFTER_EPOCHS if "valid" in STEPS else None
        )
        trainer.train()
            
        train_losses = trainer.training_losses
        val_losses = trainer.validation_losses
    
    # Perform final evaluation
    if "test" in STEPS:
        print_section_header("final evaluation")
        test_dataset = CovidxDataset(test_df, DataDir.PATH_TEST, image_size=IMAGE_SIZE)
        test_dl = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_TEST,
            num_workers=NUM_WORKERS
        )
        print_info("Testing batch size", BATCH_SIZE_TEST)
        evaluator = Evaluator(model, device, test_dl)
        evaluator.eval()
        print(f"Accuracy: {evaluator.accuracy}")
        
    # Show some stats and graphs about the model
    if "stats" in STEPS:
        fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
        viz.plot_loss(axarr[0, 0], train_losses, val_losses)
        plt.show()

    if "tune" in STEPS:
        params_to_tune = {"learning_rate" : [0.001, 0.002, 0.005, 0.01],
                          "epochs" : [5, 10, 20, 40],
                          "thresholds" : [0.48, 0.49, 0.50, 0.51, 0.52]}
        
        tuner = TuneHyperparameters(model, device, train_dl, val_dl, params_to_tune)
        best_params = tuner.tune()
        print(f"Best parameters: {best_params}")
    
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
    
    # Perform final evaluation
    if "test" in STEPS:
        print_section_header("final evaluation")
        test_dataset = CovidxDataset(test_df, DataDir.PATH_TEST, image_size=IMAGE_SIZE)
        test_dl = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE_TEST,
            num_workers=NUM_WORKERS
        )
        print_info("Testing batch size", BATCH_SIZE_TEST)
        evaluator = Evaluator(model, device, test_dl, threshold=THRESHOLD)
        evaluator.eval()
        evaluator.print_stats()
        
        # Show some stats and graphs about the model
        if "stats" in STEPS:
            fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
            viz.plot_loss(axarr[0, 0], train_losses, val_losses)
            plt.show()

    print_section_header("finished")


if __name__ == "__main__":
    demo((NUM_TRAIN, NUM_TEST, NUM_VAL))