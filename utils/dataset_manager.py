import os
import urllib.request
import ssl
import zipfile

from pathlib import Path
from ogb.nodeproppred import PygNodePropPredDataset
from utils.data_loader import load_deezer_europe, DATASET_CONFIGS

DATA_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = DATA_ROOT / 'data'
ssl._create_default_https_context = ssl._create_unverified_context

def download_url(url, folder):
    folder.mkdir(parents=True, exist_ok=True)
    filename = url.split('/')[-1]
    file_path = folder / filename

    if not file_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, str(file_path))

    return file_path

def unzip(zip_path, folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        first_file = next(m for m in zip_file.infolist() if not m.is_dir())
        first_name = first_file.filename.split('/', 1)[-1]

        if (folder / first_name).exists():
            print(f" Data already unzipped {zip_path.name}")
            return

        for member in zip_file.infolist():
            if member.is_dir():
                continue
            parts = member.filename.split('/', 1)
            if len(parts) > 1:
                member.filename = parts[1]
            zip_file.extract(member, str(folder))


def download_datasets():
    twitch_url = "https://snap.stanford.edu/data/twitch.zip"
    twitch_zip = download_url(twitch_url, DATA_DIR/ "twitch_gamers")
    unzip(twitch_zip, DATA_DIR/ "twitch_gamers")

    deezer_url = "https://snap.stanford.edu/data/deezer_europe.zip"
    download_url(deezer_url, DATA_DIR / "deezer_europe")
    deezer_cfg = DATASET_CONFIGS['deezer_europe']
    _ = load_deezer_europe(deezer_cfg)

    arxiv_root = DATA_DIR / "ogbn_arxiv"
    _ = PygNodePropPredDataset(name='ogbn-arxiv', root=str(arxiv_root))

if __name__ == '__main__':
    download_datasets()