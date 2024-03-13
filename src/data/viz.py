from .dataset import CovidxDataset
from .source import *
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

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
    axis.xaxis.set_label("Epoch")
    axis.yaxis.set_label("Loss")
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
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True, shadow=True)