"""
The `t_res.utils.get_data` module contains utility functions for downloading 
benchmark datasets from the [British Library 
repository](https://bl.iro.bl.uk/concern/datasets/f3686eb9-4227-45cb-9acb-0453d35e6a03) 
and from the [HIPE repository](https://impresso.github.io/CLEF-HIPE-2020/datasets.html).
"""

import os
import zipfile
from pathlib import Path

import wget


def download_lwm_data(news_path: str):
    """
    Download the LwM dataset from the BL repository and unzip it.

    Arguments:
        news_path (str): The directory path where the dataset will be stored.
    """
    url = (
        "https://bl.iro.bl.uk/downloads/0192d762-7277-46d0-8363-1636079e7afd?locale=en"
    )
    if not Path(
        os.path.join(news_path, "topRes19th_v2", "train", "metadata.tsv")
    ).exists() or (
        not Path(
            os.path.join(news_path, "topRes19th_v2", "test", "metadata.tsv")
        ).exists()
    ):
        Path(os.path.join(news_path)).mkdir(parents=True, exist_ok=True)
        lwm_dataset = wget.download(url, out=news_path)
        with zipfile.ZipFile(lwm_dataset) as zip_ref:
            zip_ref.extractall(news_path)


def download_hipe_data(hipe_path: str):
    """
    Download the HIPE dataset from the HIPE repository and unzip it.

    Arguments:
        hipe_path (str): The directory path where the dataset will be stored.
    """
    dev_url = "https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/hipe2020/en/HIPE-2022-v2.1-hipe2020-dev-en.tsv"
    test_url = "https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/hipe2020/en/HIPE-2022-v2.1-hipe2020-test-en.tsv"

    Path(hipe_path).mkdir(parents=True, exist_ok=True)
    if not Path(
        os.path.join(f"{hipe_path}", "HIPE-2022-v2.1-hipe2020-dev-en.tsv")
    ).exists():
        wget.download(dev_url, out=hipe_path, bar=None)
        wget.download(test_url, out=hipe_path, bar=None)
