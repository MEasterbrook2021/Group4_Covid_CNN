from src.data.datasource import *
import urllib.parse
import kaggle


class KaggleDownloader:
    def __init__(self, kaggle_link, download_path, limits=(None, None, None)):
        pr = urllib.parse.urlparse(kaggle_link)
        if pr.netloc != Kaggle.URL.value:
            raise ValueError("kaggle_link must be a www.kaggle.com link")
        
        path = [s for s in pr.path.split("/") if len(s) > 0]
        if path[0] != Kaggle.DATASETS.value:
            raise ValueError("kaggle_link must be a link to a dataset")
        
        self.kaggle_link = kaggle_link
        self.download_path = download_path
        self.dataset_author = path[1]
        self.dataset_name = path[2]
        self.dataset_id = f"{self.dataset_author}/{self.dataset_name}"
        self.limits = limits

    def download(self):
        kaggle.api.authenticate()
        if self.limits == (None, None, None):
            kaggle.api.dataset_download_files(
                self.dataset_id, 
                path=self.download_path,
                force=False,
                quiet=False,
                unzip=True
            )
        else:
            self.__download_limits(*self.limits)

    def __download_limits(self, train_num: int, test_num: int, val_num: int):
        pass