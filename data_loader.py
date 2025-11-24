import zipfile
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = Path(__file__).resolve().parent / 'data'

DATASET_CONFIGS = {
    'twitch_gamers': {
        'zip_path': DATA_ROOT / 'twitch_gamers' / 'twitch_gamers_de.zip',
        'extract_dir': DATA_ROOT / 'twitch_gamers',
        'feature_filename': 'twitch_gamers_features_de.csv',
        'edges_filename': 'twitch_gamers_edges_de.csv'
    },
    'presidential_election': {
        'zip_path': DATA_ROOT / 'presidential_election' / 'presidential_election.zip',
        'extract_dir': DATA_ROOT / 'presidential_election',
        'feature_filename': 'presidential_election_nodes.csv',
        'edges_filename': 'presidential_election_edges.csv'
    }
}

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

def create_dataobj (features_df, edges_df):
    """
    Creates a PyG Data object from features and edges DataFrames.
    """
    x = torch.tensor(features_df, dtype=torch.float)
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
    y = torch.tensor(features_df.values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

def get_graph_data(dataset_name: str):
    """
    Main function to get graph data for a specified dataset.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not found.")

    cfg = DATASET_CONFIGS[dataset_name]
    feat_df,edges_df = load_dataset(
        zip_path=cfg['zip_path'],
        extract_dir=cfg['extract_dir'],
        feature_filename=cfg['feature_filename'],
        edges_filename=cfg['edges_filename']
    )

    return create_dataobj(feat_df, edges_df)