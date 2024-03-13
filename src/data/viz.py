from .dataset import CovidxDataset
from .source import *
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

def show_examples(dataset: CovidxDataset, title, num_examples):
    print("Visualizing:")
    n_rows = 2
    n_cols = num_examples + 1
    fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 5))
    fig.suptitle(title, fontsize=16)
    
    pos_row = axarr[0]
    neg_row = axarr[1]

    print(f"{num_examples} {Covidx_CXR2.CLASS_POSITIVE} examples")
    ax = pos_row[0]
    ax.text(0.5, 0.5, Covidx_CXR2.CLASS_POSITIVE, fontsize=12, horizontalalignment="center", verticalalignment="center")
    ax.axis("off")
    for i in range(0, num_examples):
        img, _ = dataset.get_random(Covidx_CXR2.CLASS_POSITIVE)
        img = img.squeeze().permute(1, 2, 0)
        img = (img + 1.0) / 2.0
        img = img.clamp(min=0.0, max=1.0)
        ax = pos_row[i + 1]
        ax.axis("off")
        ax.imshow(img)

    print(f"{num_examples} {Covidx_CXR2.CLASS_NEGATIVE} examples")
    ax = neg_row[0]
    ax.text(0.5, 0.5, Covidx_CXR2.CLASS_NEGATIVE, fontsize=12, horizontalalignment="center", verticalalignment="center")
    ax.axis("off")
    for i in range(0, num_examples):
        img, _ = dataset.get_random(Covidx_CXR2.CLASS_NEGATIVE)
        img = img.squeeze().permute(1, 2, 0)
        img = (img + 1.0) / 2.0
        img = img.clamp(min=0.0, max=1.0)
        ax = neg_row[i + 1]
        ax.axis("off")
        ax.imshow(img)

    plt.show()


def plot_loss(axis: Axes, train_losses=None, val_losses=None):
    axis.set_title("Training and Validation Losses over Epochs")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    if train_losses is not None and len(train_losses) > 0:
        (train_epochs, train_losses) = tuple(zip(*train_losses))
    else:
        train_epochs, train_losses = [], []
    axis.plot(train_epochs, train_losses, label="Training")
    if val_losses is not None and len(val_losses) > 0:
        (val_epochs, val_losses) = tuple(zip(*val_losses))
    else:
        val_epochs, val_losses = [], []
    axis.plot(val_epochs, val_losses, label="Validation")
    axis.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2, fancybox=True, shadow=True)

def plot_roc(axis: Axes, fpr, tpr):
    axis.set_title("ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.plot([0, 1], [0, 1], label="Baseline", linestyle="dotted", color="red")
    axis.plot(fpr, tpr, label="Model", color="black")
    axis.legend(loc="lower right", bbox_to_anchor=(0.9, 0.1), ncol=1, fancybox=True, shadow=True)

def plot_confusion(axis: Axes, confusion):
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Labels")
    axis.set_ylabel("True Labels")
    width, height = confusion.shape
    for i in range(width):
        for j in range(height):
            axis.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black")
    offset = 0.5    
    axis.set_xlim(-offset, width - offset)
    axis.set_ylim(-offset, height - offset)
    axis.hlines(y=np.arange(height + 1) - offset, xmin=-offset, xmax=width-offset)
    axis.vlines(x=np.arange(width + 1) - offset, ymin=-offset, ymax=height-offset)
    axis.set_xticks(np.arange(confusion.shape[0]))
    axis.set_yticks(np.arange(confusion.shape[1]))
    axis.set_xticklabels(Covidx_CXR2.CLASSES)
    axis.set_yticklabels(Covidx_CXR2.CLASSES)

def plot_stats(axis: Axes, loss_avg, accuracy, precision, recall, f1):
    axis.axis("off")
    text1 = reduce(lambda s1, s2: s1 + "\n" + s2, [
        "Average Loss",
        "Accuracy", 
        "Precision",
        "Recall", 
        "F1 Score",
    ])
    text2 = reduce(lambda s1, s2: str(s1) + "\n" + str(s2), [
        f"{loss_avg:.4f}",
        f"{accuracy*100:.1f}%",
        f"{precision:.4f}",
        f"{recall:.4f}",
        f"{f1:.4f}"
    ])
    axis.text(0.2, 0.2, text1, fontsize=18)
    axis.text(0.6, 0.2, text2, fontsize=18)