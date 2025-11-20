import zipfile
import pandas as pd
from pathlib import Path

def load_dataset(zip_path: Path, extract_dir: Path, feature_filename, edges_filename):
    """
    Loads dataset from a zip file containing feature and edge CSV files.
    """
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zip_f:
            zip_f.extractall(extract_dir)

        feature_file = extract_dir / feature_filename
        edges_file = extract_dir / edges_filename

        features_df = pd.read_csv(feature_file)
        edges_df = pd.read_csv(edges_file)

        return features_df, edges_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
