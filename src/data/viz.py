from .dataset import CovidxDataset
from .source import *
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
        ax = pos_row[i + 1]
        ax.axis("off")
        ax.imshow(img.squeeze().permute(1, 2, 0), cmap="gray")

    print(f"{num_examples} {Covidx_CXR2.CLASS_NEGATIVE} examples")
    ax = neg_row[0]
    ax.text(0.5, 0.5, Covidx_CXR2.CLASS_NEGATIVE, fontsize=12, horizontalalignment="center", verticalalignment="center")
    ax.axis("off")
    for i in range(0, num_examples):
        img, _ = dataset.get_random(Covidx_CXR2.CLASS_NEGATIVE)
        ax = neg_row[i + 1]
        ax.axis("off")
        ax.imshow(img.squeeze().permute(1, 2, 0), cmap="gray")

    plt.show()
