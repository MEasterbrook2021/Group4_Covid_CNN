from pathlib import Path


class Kaggle:
    URL = "www.kaggle.com"
    DATASETS = "datasets"

    def dataset_id(author, name):
        return f"{author}/{name}"

    def dataset_link(author, name):
        return Kaggle.dataset_link(Kaggle.dataset_id(author, name))
    
    def dataset_link(id):
        return f"https://{Kaggle.URL}/{Kaggle.DATASETS}/{id}"


class Covidx_CXR2:
    NAME = "covidx-cxr2"
    AUTHOR = "andyczhao"
    ID = Kaggle.dataset_id(AUTHOR, NAME)
    LINK = Kaggle.dataset_link(ID)

    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    
    ANNOTATIONS_FILES = [f"{s}.txt" for s in [TRAIN, TEST, VAL]]
    COLUMNS = ["patient_id", "filename", "label", "data_source"]
    CLASS_POSITIVE = "positive"
    CLASS_NEGATIVE = "negative"
    CLASSES = [CLASS_POSITIVE, CLASS_NEGATIVE]