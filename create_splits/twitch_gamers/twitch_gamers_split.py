import pandas as pd
import numpy as np
import torch
import warnings
import os

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from torch_geometric.data import Data
from pathlib import Path
from torch_geometric.data.remote_backend_utils import num_nodes

from utils.data_loader import load_twitch_gamers, DATASET_CONFIGS, save_data_obj
from utils.mask_creation import _create_mask
from create_splits.split_manager import save_split

DATASET_NAME = 'twitch_gamers'
CONFIG = DATASET_CONFIGS[DATASET_NAME]
CURRENT_DIR = Path(__file__).parent.parent.resolve()
PROJECT_ROOT = CURRENT_DIR
DATA_ROOT = PROJECT_ROOT / 'split_data'/ DATASET_NAME

SPLIT_REGISTRY = {
    'split_0': 'ENGB',
    'split_1': 'ES',
    'split_2': 'FR',
    'split_3': 'PTBR',
    'split_4': 'RU'
}
TRAIN_COUNTRY = 'DE'
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def preprocess_twitch_gamers (df):
    """
    takes the dataframes of each country and returns the normalized columns (view, days)
    """
    final_features_df = {}

    target_df_de = df[TRAIN_COUNTRY]['target_df']
    features_df_de = df[TRAIN_COUNTRY]['features_df']
    games_list_de = [features_df_de.get(str(user_id), []) for user_id in target_df_de['new_id']]

    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb.fit(games_list_de)

    norm_target_df = target_df_de[['days', 'views']].copy()
    norm_target_df['views'] = np.log1p(norm_target_df['views'])

    scaler = StandardScaler()
    scaler.fit(norm_target_df)

    for lang, data in df.items():
        target_df = data['target_df']
        features_df = data['features_df']

        game_lists = [features_df.get(str(user_id), []) for user_id in target_df['new_id']]
        games = mlb.transform(game_lists)
        game_cols = [f"game_{g}" for g in mlb.classes_]
        df_games = pd.DataFrame(games, columns=game_cols, index=target_df.index)

        curr_norm_df = target_df[['days', 'views']].copy()
        curr_norm_df['views'] = np.log1p(curr_norm_df['views'])

        scaled_vals = scaler.transform(curr_norm_df)
        df_numeric = pd.DataFrame(scaled_vals, columns=['days', 'views'], index=target_df.index)

        df_numeric['partner']= target_df['partner'].values.astype(float)
        df_numeric ['mature'] = target_df['mature'].values.astype(float)

        features_df = pd.concat([df_numeric, df_games], axis=1)
        final_features_df[lang] = features_df

    return final_features_df

def create_country_splits(num_train_nodes, num_test_nodes):
    """
    Creates 5 splits (DE -> ENGB, DE -> ES,...) and saves masks
    """
    num_nodes = num_train_nodes + num_test_nodes

    train_indices = list (range(0, num_train_nodes))
    val_indices = list(range(num_train_nodes, num_nodes))

    split_pt = int(len(val_indices) * 0.2)
    val_nodes = val_indices[:split_pt]
    test_nodes = val_indices[split_pt:]

    return _create_mask(num_nodes, train_indices, val_nodes, test_nodes)


def save_all_splits_json():
    data = load_twitch_gamers(CONFIG)
    features_df = preprocess_twitch_gamers(data)
    train_df = features_df[TRAIN_COUNTRY]
    num_train_nodes = len(train_df)

    for split_name, test_country in SPLIT_REGISTRY.items():
        test_df = features_df[test_country]
        num_test_nodes = len(test_df)
        total_nodes = num_train_nodes + num_test_nodes

        train_indices = list(range(0, num_train_nodes))
        test_indices = list(range(num_train_nodes, total_nodes))

        split_ratio = int(len(test_indices)* 0.2)
        val_indices = test_indices[:split_ratio]
        test_indices = test_indices[split_ratio:]

        train_mask, val_mask, test_mask = _create_mask(total_nodes, train_indices, val_indices, test_indices)
        save_split(DATASET_NAME, split_name, train_mask, val_mask, test_mask)


def check_feature_overlap(features_map, source_country, target_country):

    df_train = features_map[source_country]
    df_test = features_map[target_country]
    game_cols = [c for c in df_train.columns if c.startswith('game_')]

    print(f"Total unique games learned from {source_country}: {len(game_cols)}")
    x_test_games = df_test[game_cols].values
    total_game_entries = x_test_games.sum()
    avg_games_per_user = total_game_entries / len(df_test)
    print(f"Average 'Known Games' per user in {target_country}: {avg_games_per_user:.2f}")

    if avg_games_per_user < 1.0:
        print("CRITICAL ISSUE: The Target users play almost NONE of the games the Source users played.")

def save_splits_pt():
    data = load_twitch_gamers(DATASET_CONFIGS[DATASET_NAME])
    features_df = preprocess_twitch_gamers(data)

    ALL_COUNTRIES = [TRAIN_COUNTRY]+ list(SPLIT_REGISTRY.keys())
    unqiue_countries = set(ALL_COUNTRIES)

    for country in unqiue_countries:
        df_feats = features_df[country]
        Y_tensor = torch.from_numpy(df_feats['mature'].values).long()
        X_np = df_feats.drop(columns=['mature'], errors='ignore').astype(float).values
        X_tensor = torch.from_numpy(X_np).float()

        edges_df = data[country]['edges_df']
        edge_index_np = edges_df.values.T
        edge_index_tensor = torch.from_numpy(edge_index_np).long()

        country_data = Data(x=X_tensor, edge_index=edge_index_tensor, y=Y_tensor)
        country_specific_name = f"{DATASET_NAME}_{country}"
        save_data_obj(country_data, country_specific_name)


def get_dataset(split_name = None):

    train_country = TRAIN_COUNTRY
    test_country = SPLIT_REGISTRY[split_name]

    base_path = DATA_ROOT / 'split_data'

    train_name = f"{DATASET_NAME}_{train_country}"
    test_name = f"{DATASET_NAME}_{test_country}"

    train_file = base_path / train_name / f"{train_name}_data.pt"
    test_file = base_path / test_name / f"{test_name}_data.pt"

    train_data = torch.load(train_file)
    test_data = torch.load(test_file)

    X = torch.cat([train_data.x, test_data.x], dim=0)
    Y = torch.cat([train_data.y, test_data.y], dim=0)

    num_train_nodes = train_data.x.shape[0]
    num_test_nodes = test_data.x.shape[0]

    test_edges = train_data.edge_index + num_train_nodes

    edge_index = torch.cat([train_data.edge_index, test_edges], dim=1)

    train_mask, val_mask, test_mask = create_country_splits(num_train_nodes, num_test_nodes)

    train_mask = torch.from_numpy(train_mask).bool()
    val_mask = torch.from_numpy(val_mask).bool()
    test_mask = torch.from_numpy(test_mask).bool()

    data = Data(x=X, edge_index=edge_index, y=Y)

    return data, train_mask, val_mask, test_mask


if __name__ == '__main__':

    data = load_twitch_gamers(DATASET_CONFIGS[DATASET_NAME])
    features_df = preprocess_twitch_gamers(data)
    #save_all_splits()
    save_splits_pt()