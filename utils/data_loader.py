import zipfile
import pandas as pd
import torch
import json

from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent.resolve()
print(DATA_ROOT)

DATASET_CONFIGS = {
    'twitch_gamers': {
        'extract_dir': DATA_ROOT/ "data" / 'twitch_gamers'
    },
    'presidential_election': {
        'extract_dir': DATA_ROOT/'data' / 'presidential_election',
        'feature_filename': 'presidential_election_nodes.csv',
        'edges_filename': 'presidential_election_edges.csv'
    },
    'deezer_europe': {
        'zip_path': DATA_ROOT / 'data' / 'deezer_europe' / 'deezer_europe.zip',
        'extract_dir': DATA_ROOT / 'data',
        'feature_filename': 'deezer_europe/deezer_europe_features.json',
        'edges_filename': 'deezer_europe/deezer_europe_edges.csv',
        'target_filename': 'deezer_europe/deezer_europe_target.csv'
    }

}

def load_dataset(extract_dir: Path, feature_filename, edges_filename):
    """
    Loads dataset from a zip file containing feature and edge CSV files.
    """

    feature_file = extract_dir / feature_filename
    edges_file = extract_dir / edges_filename

    features_df = pd.read_csv(feature_file)
    edges_df = pd.read_csv(edges_file)

    return features_df, edges_df

def save_data_obj (data_obj, dataset_name):
    """
    Saves a PyG Data object as a pytorch object.
    """
    save_path = DATA_ROOT/"split_data"/ dataset_name
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{dataset_name}_data.pt"
    torch.save(data_obj, file_path)
    print(f"Saved {dataset_name}_data.pt to {file_path}")

def load_data_object(dataset_name: str, base_dir="split_data", split_name: str | None = None):
    """
    Loads a saved PyG Data object for a dataset.
    """
    if dataset_name in ['twitch_gamers', 'presidential_election']:
        data_path = DATA_ROOT / base_dir / dataset_name / split_name / f"{split_name}_data.pt"
    else:
        data_path = DATA_ROOT/ f"{base_dir}/{dataset_name}/{dataset_name}_data.pt"
    print(f"Loading {dataset_name}_data.pt from {data_path}")
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

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model file at: {model_path}")
    config = torch.load(model_path, map_location=device)
    model_config = config.get("model_config")
    if model_config is None:
        # Debug print to see what IS in there
        print(f"DEBUG: Keys found in checkpoint: {list(config.keys())}")
        raise KeyError(f"The file {model_fname} does not contain 'model_config'. ")

    input_dim = model_config['input_dim']
    hidden_dim = model_config['hidden_dim']
    output_dim = model_config['output_dim']
    dropout = model_config.get('dropout', 0.5)
    model_type = model_config.get('name', model_type).upper()

    if model_type == 'GCN':
        from models.gcn import GCN
        model = GCN(input_dim, hidden_dim, output_dim, dropout)
    elif model_type == 'MLP':
        from models.mlp import MLP
        model = MLP(input_dim, hidden_dim, output_dim, dropout)
    elif model_type == 'SAGE':
        from models.graphSage import SAGE
        model = SAGE(input_dim, hidden_dim, output_dim, dropout)
    else:
        raise ValueError(f"Model {model_type} not recognized.")

    state_dict = config["model_state_dict"]
    model.load_state_dict(state_dict)
    model= model.to(device)
    model.eval()

    return model, model_config

def load_deezer_europe(cfg):
    zip_path = cfg['zip_path']
    extract_dir = cfg['extract_dir']

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zip_f:
        zip_f.extractall(extract_dir)

    edges_path = extract_dir / cfg['edges_filename']
    features_path = extract_dir / cfg['feature_filename']
    target_path = extract_dir / cfg['target_filename']

    edges_df = pd.read_csv(edges_path)

    target_df = pd.read_csv(target_path)

    with open(features_path, 'r') as f:
        features_data = json.load(f)

    return features_data, edges_df, target_df

def load_twitch_gamers(cfg):
    extract_dir = cfg['extract_dir']
    SOURCE_LANG = 'DE'
    TARGET_LANG = ['ENGB', 'ES', 'FR', 'PTBR', 'RU']
    ALL_LANG = [SOURCE_LANG] + TARGET_LANG
    all_datasets = {}

    for lang in ALL_LANG:
        source_path = extract_dir / lang

        with open(source_path / f"musae_{lang}_features.json", 'r') as f:
            features_df = json.load(f)

        target_df = pd.read_csv(source_path / f"musae_{lang}_target.csv")
        edges_df = pd.read_csv(source_path / f"musae_{lang}_edges.csv")

        all_datasets[lang] = {
            'features_df': features_df,
            'edges_df': edges_df,
            'target_df': target_df
        }

    return all_datasets

if __name__ == '__main__':
    print(DATA_ROOT)