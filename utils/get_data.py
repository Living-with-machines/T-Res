import wget
import pathlib
import zipfile

def download_lwm_data():
    url = "https://bl.iro.bl.uk/downloads/ff44881f-97ca-4c68-97f8-097324bdba94?locale=en"
    save_to = "resources"
    pathlib.Path(save_to).mkdir(parents=True, exist_ok=True)
    if not pathlib.Path("resources/topRes19th.zip").is_file():
        lwm_dataset = wget.download(url, out=save_to)
        with zipfile.ZipFile(lwm_dataset) as zip_ref:
            zip_ref.extractall(save_to)