from enum import Enum
import os


class Kaggle(Enum):
    URL = "www.kaggle.com"
    DATASETS = "datasets"

    def dataset_id(author, name):
        return f"{author}/{name}"

    def dataset_link(author, name):
        return Kaggle.dataset_link(Kaggle.dataset_id(author, name))
    
    def dataset_link(id):
        return f"https://{Kaggle.URL.value}/{Kaggle.DATASETS.value}/{id}"


class Covidx_CXR2(Enum):
    NAME = "covidx-cxr2"
    AUTHOR = "andyczhao"
    ID = Kaggle.dataset_id(AUTHOR, NAME)
    LINK = Kaggle.dataset_link(ID)

    TRAIN = "train"
    TEST = "test"
    VAL = "val"

    TRAIN_TXT = os.path.join(NAME, f"{TRAIN}.txt")
    TEST_TXT = os.path.join(NAME, f"{TEST}.txt")
    VAL_TXT = os.path.join(NAME, f"{VAL}.txt")
    
    COLUMNS = ["patient_id", "filename", "label", "data_source"]
    CLASS_POSITIVE = "positive"
    CLASS_NEGATIVE = "negative"