import zipfile
import pandas as pd
import torch
import json
from pathlib import Path
from torch_geometric.data import Data

DATA_ROOT = Path(__file__).parent.parent.resolve()
print(DATA_ROOT)

DATASET_CONFIGS = {
    'twitch_gamers': {
        'zip_path': DATA_ROOT / "data"/ 'twitch_gamers' / 'twitch_gamers_de.zip',
        'extract_dir': DATA_ROOT/ "data" / 'twitch_gamers',
        'feature_filename': 'twitch_gamers_features_de.csv',
        'edges_filename': 'twitch_gamers_edges_de.csv'
    },
    'presidential_election': {
        'zip_path': DATA_ROOT /'data'/ 'presidential_election' / 'presidential_election.zip',
        'extract_dir': DATA_ROOT/'data' / 'presidential_election',
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

def save_data_obj (data_obj, dataset_name):
    """
    Saves a PyG Data object as a pytorch object.
    """
    save_path = DATA_ROOT/"split_data"/ dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{dataset_name}_data.pt"
    torch.save(data_obj, file_path)
    print(f"Saved {dataset_name}_data.pt to {file_path}")


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

def load_data_object(dataset_name: str, base_dir="split_data"):
    """
    Loads a saved PyG Data object for a dataset.
    """
    data_path = DATA_ROOT/ f"{base_dir}/{dataset_name}/{dataset_name}_data.pt"
    data = torch.load(data_path, weights_only=False)
    return data


def load_model(dataset_name, model_type, split_name, model_config, device='cpu'):
    """
    Loads a saved model for a given datset, architecture should be consistent with model_config
    """
    if split_name is not None:
        model_fname= f"{model_type}_{dataset_name}_{split_name}.pth"
    else:
        print(f"Model {model_type} not found.")

    model_path = DATA_ROOT / "saved_models" / model_fname

    input_dim = model_config['input_dim']
    hidden_dim = model_config['hidden_dim']
    output_dim = model_config['output_dim']
    dropout = model_config.get('dropout', 0.5)

    if model_type == 'GCN':
        from models.gcn import GCN
        model = GCN(input_dim, hidden_dim, output_dim, dropout)
    elif model_type == 'MLP':
        from models.mlp import MLP
        model = MLP(input_dim, hidden_dim, output_dim, dropout)
    else:
        raise ValueError(f"Model {model_type} not recognized.")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model= model.to(device)
    model.eval()

    return model
