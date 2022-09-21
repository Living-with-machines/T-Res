import os
import wget
import zipfile
from pathlib import Path


# ------------------------------
# LwM data
# ------------------------------

# ------------------------------
# Download data from BL repository and unzip it.
def download_lwm_data(news_path):
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
        Path(os.path.join(news_path)).mkdir(parents=True, exist_ok=True)
        lwm_dataset = wget.download(url, out=news_path)
        with zipfile.ZipFile(lwm_dataset) as zip_ref:
            zip_ref.extractall(news_path)


# ------------------------------
# HIPE data
# ------------------------------

# ------------------------------
# Download data from HIPE repository and unzip it.
def download_hipe_data(hipe_path):
    dev_url = "https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/hipe2020/en/HIPE-2022-v2.1-hipe2020-dev-en.tsv"
    test_url = "https://raw.githubusercontent.com/hipe-eval/HIPE-2022-data/main/data/v2.1/hipe2020/en/HIPE-2022-v2.1-hipe2020-test-en.tsv"

    Path(hipe_path).mkdir(parents=True, exist_ok=True)
    if not Path(
        os.path.join(f"{hipe_path}", "HIPE-2022-v2.1-hipe2020-dev-en.tsv")
    ).exists():
        wget.download(dev_url, out=hipe_path, bar=None)
        wget.download(test_url, out=hipe_path, bar=None)
