from src.data.datasource import Covidx_CXR2
from src.data.download import KaggleDownloader

if __name__ == "__main__":
    dl = KaggleDownloader(
        kaggle_link=Covidx_CXR2.LINK.value,
        download_path="data/"
    )
    dl.download()