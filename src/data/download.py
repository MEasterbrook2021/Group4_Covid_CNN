from pathlib import Path
from urllib.parse import urlparse
import kaggle
import pandas as pd
from zipfile import ZipFile
import os

from .source import *
from .process import read_annotations_file


class CovidxDownloader:
    def __init__(self):
        pr = urlparse(Covidx_CXR2.LINK)
        if pr.netloc != Kaggle.URL:
            raise ValueError("Covidx_CXR2.LINK must be a www.kaggle.com link")
        
        path = [s for s in pr.path.split("/") if len(s) > 0]
        if path[0] != Kaggle.DATASETS:
            raise ValueError("Covidx_CXR2.LINK must be a link to a dataset")
        
        self.kaggle_link = Covidx_CXR2.LINK
        self.dataset_author = path[1]
        self.dataset_name = path[2]
        self.dataset_id = Kaggle.dataset_id(self.dataset_author, self.dataset_name)
        kaggle.api.authenticate()

    def download(self, download_dir=Data.RAW):
        if not download_dir.is_dir() or len(os.listdir(download_dir)) == 0:
            print("--- DOWNLOADING DATASET")
            kaggle.api.dataset_download_files(
                self.dataset_id,
                path=download_dir,
                force=False,
                quiet=False,
                unzip=True
            )
        else:
            print(f"{Covidx_CXR2.NAME} already downloaded!")

    def __download_limits(self, train_num: int, test_num: int, val_num: int):
        # This shit doesn't work because Kaggle returns 404 whenever you try to download a specific file 
        # that isn't displayed in the first 100 or so files in a folder
        #
        # There's probably a workaround but i'm going to suck it up and waste 30GB of disk space on storing 80,000
        # lung scans

        out = list()
        # Assume all these files exist and are in root of dataset
        for file_txt, num in zip(Covidx_CXR2.ANNOTATIONS_FILES, [train_num, test_num, val_num]):
            file_stem = Path(file_txt).stem
            # Assume file is .txt
            file_path: Path = (self.download_path / self.dataset_name) / file_txt
            if not file_path.is_file():
                kaggle.api.dataset_download_file(
                    self.dataset_id,
                    file_txt,
                    path=file_path.parent,
                    force=False,
                    quiet=False
                )
                zip_file = f"{file_path}.zip"
                print(f"Extracting {zip_file}...")
                with ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(file_path.parent)
                os.remove(zip_file)

            # Create annotations file that only has {num} entries
            l_file_path = file_path.parent / f"{file_stem}_{num}.txt"
            if not l_file_path.is_file():
                df = self.__create_limited_annotations_file(file_path, l_file_path, num)
            else:
                df = read_annotations_file(l_file_path)

            # Download all the images in the limited annotations file
            img_dir_path = l_file_path.parent / file_stem
            for _, row in df.iterrows():
                img_filename = row["filename"]
                img_filepath = img_dir_path / img_filename
                dl_file = f"{file_stem}/{img_filename}"
                if not img_filepath.is_file():
                    kaggle.api.dataset_download_file(
                        self.dataset_id,
                        file_name=dl_file,
                        path=img_dir_path,
                        force=False,
                        quiet=False
                    )
            out.append(file_path)
        return tuple(out)
    
    def __create_limited_annotations_file(self, file, out_path, num, split=0.5):
        print(f"Creating {out_path}...")
        df = read_annotations_file(file)
        df = pd.concat([
            df[df.label == Covidx_CXR2.CLASS_POSITIVE].sample(n = int(num * split)), 
            df[df.label == Covidx_CXR2.CLASS_NEGATIVE].sample(n = int(num * (1 - split)))
        ])
        df.to_csv(out_path, sep=" ", index=False, header=None)
        return df