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
import math


STEPS = [
    "download",
    # "viz",
    "model",
    "train",
    "valid",
    "save",
    "test",
    "stats",
    # "tune",
    "save",
]
NUM_WORKERS = 4
LOAD_MODEL_NAME = "covidx-cxr2-resnet-20epochs-0003"

NUM_TRAIN, NUM_TEST, NUM_VAL = 20_000, 8_000, 8_000
USE_SANITISED     = True
IMAGE_SIZE        = (224, 224)
BATCH_SIZE_TRAIN  = 32
BATCH_SIZE_TEST   = 32
AUGMENT_NOISE     = None # (0.0, 0.01)
THRESHOLD         = 0.5

MODEL_TYPE        = ModelTypes.RESNET
FREEZE            = True
LEARNING_RATE     = 0.001
NUM_EPOCHS        = 25
VAL_AFTER_EPOCHS  = 1
SAVE_AFTER_EPOCHS = NUM_EPOCHS


def print_section_header(str: str):
    # print(f"\n{f" {str.upper()} ":═^120}")
    print(f"\n{' ' + str.upper() + '':═^120}")


def print_info(str: str, val):
    print(f"> {str}: {val}")

def create_demo_annots(file: Path, output: Path, num_instances):
    df = read_annotations_file(file)
    df_len = df.shape[0]
    num_positive = int(math.floor(0.5 * num_instances))
    num_negative = int(math.ceil(0.5 * num_instances))
    out = []
    # Add positive instances
    df_pos = df[df["label"] == Covidx_CXR2.CLASS_POSITIVE]
    sample_pos = df_pos.sample(min(df_pos.shape[0], num_positive))
    out.append(sample_pos)
    num_positive -= sample_pos.shape[0]
    if num_positive > 0:
        # Upsample
        sample_pos = df_pos.sample(num_positive, replace=True)
        out.append(sample_pos)
        num_positive -= sample_pos.shape[0]
    # Add negative instances
    df_neg = df[df["label"] == Covidx_CXR2.CLASS_NEGATIVE]
    sample_neg = df_neg.sample(min(df_neg.shape[0], num_negative))
    out.append(sample_neg)
    num_negative -= sample_neg.shape[0]
    if num_negative > 0:
        # Upsample
        sample_neg = df_neg.sample(num_negative, replace=True)
        out.append(sample_neg)
        num_negative -= sample_neg.shape[0]
    # Concat all and save
    df = pd.concat(out)
    assert(num_positive == 0)
    assert(num_negative == 0)
    save_annotations_file(df, output)
    return df


def demo(limits):
    num_train, num_test, num_val = limits

    print_section_header("DATA LOADING")
    dfs = list()
    for af, num in zip(Covidx_CXR2.ANNOTATIONS_FILES, limits):
        # Create the demo annotations files and download the dataset if necessary
        af = DataDir.PATH / af
        if USE_SANITISED:
            af = DataDir.sanit_file(af)
        demo_af = DataDir.demo_file(af)
        if not demo_af.is_file():
            if "download" in STEPS:
                CovidxDownloader().download(DataDir.PATH)
            df = create_demo_annots(file=af, output=demo_af, num_instances=num)
        else:
            df = read_annotations_file(demo_af)
            if df.shape[0] != num:
                os.remove(demo_af)
                df = create_demo_annots(file=af, output=demo_af, num_instances=num)
        assert(df.shape[0] == num)
        df = df.sample(frac=1).reset_index(drop=True)
        dfs.append(df)

    train_df, test_df, val_df = tuple(dfs)
    del dfs
    print_info("Total training examples",    train_df.shape[0])
    print_info("Training positive examples", train_df[train_df["label"] == Covidx_CXR2.CLASS_POSITIVE].shape[0])
    print_info("Training negative examples", train_df[train_df["label"] == Covidx_CXR2.CLASS_NEGATIVE].shape[0])
    print_info("Total testing examples",    test_df.shape[0])
    print_info("Testing positive examples", test_df[test_df["label"] == Covidx_CXR2.CLASS_POSITIVE].shape[0])
    print_info("Testing negative examples", test_df[test_df["label"] == Covidx_CXR2.CLASS_NEGATIVE].shape[0])
    print_info("Total validation examples",    val_df.shape[0])
    print_info("Validation positive examples", val_df[val_df["label"] == Covidx_CXR2.CLASS_POSITIVE].shape[0])
    print_info("Validation negative examples", val_df[val_df["label"] == Covidx_CXR2.CLASS_NEGATIVE].shape[0])

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
            fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
            viz.plot_loss(axarr[0, 0], train_losses, val_losses)
            (fpr, tpr, _) = evaluator.roc
            viz.plot_roc(axarr[1, 0], fpr, tpr)
            viz.plot_confusion(axarr[0, 1], evaluator.confusion)
            viz.plot_stats(axarr[1, 1], 
                evaluator.loss_avg, evaluator.accuracy, evaluator.precision, evaluator.recall, evaluator.f1
            )
            plt.show()

    print_section_header("finished")


if __name__ == "__main__":
    demo((NUM_TRAIN, NUM_TEST, NUM_VAL))