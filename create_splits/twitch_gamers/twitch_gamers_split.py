import pandas as pd
import numpy as np
import torch
import warnings

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from torch_geometric.data import Data
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

from utils.data_loader import load_twitch_gamers, DATASET_CONFIGS, DATA_ROOT
from utils.mask_creation import _create_mask
from create_splits.split_manager import save_split

DATASET_NAME = 'twitch_gamers'
CONFIG = DATASET_CONFIGS[DATASET_NAME]
PROJECT_ROOT = DATA_ROOT / 'split_data'/ DATASET_NAME

SPLIT_REGISTRY = {
    'split_0': 'ENGB',
    'split_1': 'ES',
    'split_2': 'FR',
    'split_3': 'PTBR',
    'split_4': 'RU'
}
TRAIN_COUNTRY = 'DE'
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from sklearn.feature_extraction.text import TfidfTransformer


def preprocess_twitch_gamers(df):
    final_features_df = {}
    svd_components = 128  # Increased from 32 for better signal

    all_games_lists = []
    country_codes = list(df.keys())

    for lang in country_codes:
        target_df = df[lang]['target_df']
        features_df = df[lang]['features_df']
        game_lists = [features_df.get(str(u_id), []) for u_id in target_df['new_id']]
        all_games_lists.extend(game_lists)

    mlb = MultiLabelBinarizer(sparse_output=True)  # Use sparse for memory efficiency
    games_binary = mlb.fit_transform(all_games_lists)

    tfidf = TfidfTransformer()
    games_tfidf = tfidf.fit_transform(games_binary)

    svd = TruncatedSVD(n_components=svd_components, algorithm='arpack', random_state=42)
    svd.fit(games_tfidf)

    # 4. Fit Global Scaler (on ALL countries to prevent distribution shift)
    all_numeric_data = []
    for lang in country_codes:
        temp_df = df[lang]['target_df'][['days', 'views']].copy()
        temp_df['views'] = np.log1p(temp_df['views'])
        all_numeric_data.append(temp_df)

    global_scaler = StandardScaler()
    global_scaler.fit(pd.concat(all_numeric_data))

    # 5. Transform each country
    for lang in country_codes:
        target_df = df[lang]['target_df']
        features_df = df[lang]['features_df']

        # Process Games (MLB -> TFIDF -> SVD)
        game_lists = [features_df.get(str(u_id), []) for u_id in target_df['new_id']]
        curr_games_bin = mlb.transform(game_lists)
        curr_games_tfidf = tfidf.transform(curr_games_bin)
        curr_games_svd = svd.transform(curr_games_tfidf)

        df_games = pd.DataFrame(curr_games_svd,
                                columns=[f"svd_{i}" for i in range(svd_components)],
                                index=target_df.index)

        # Process Numeric
        curr_norm_df = target_df[['days', 'views']].copy()
        curr_norm_df['views'] = np.log1p(curr_norm_df['views'])
        scaled_vals = global_scaler.transform(curr_norm_df)
        df_numeric = pd.DataFrame(scaled_vals, columns=['days', 'views'], index=target_df.index)

        # Add Binary Features
        df_numeric['partner'] = target_df['partner'].values.astype(float)
        df_numeric['mature'] = target_df['mature'].values.astype(float)

        # Add Country One-Hot (Helping the model realize it's a different graph)
        for i, code in enumerate(country_codes):
            df_numeric[f'is_{code}'] = 1.0 if lang == code else 0.0

        final_features_df[lang] = pd.concat([df_numeric, df_games], axis=1)

    return final_features_df

def create_country_splits(num_train_nodes, num_test_nodes):
    """
    Creates 5 splits (DE -> ENGB, DE -> ES,...) and saves masks
    """
    num_nodes = num_train_nodes + num_test_nodes

    train_indices = list (range(0, num_train_nodes))
    test_indices = list(range(num_train_nodes, num_nodes))

    split_pt = int(len(train_indices) * 0.8)
    train_nodes = train_indices[:split_pt]
    val_nodes = train_indices[split_pt:]
    test_nodes = test_indices

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)


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

    df_train = features_df[TRAIN_COUNTRY]
    x_train = torch.from_numpy(df_train.drop(columns =['mature'], errors='ignore').values).float()
    y_train = torch.from_numpy(df_train['mature'].values).long()

    edges_train_df = data[TRAIN_COUNTRY]['edges_df']
    edge_index_train= torch.from_numpy(edges_train_df.values.T).long()

    num_train_nodes = x_train.size(0)

    for split_name, test_country in SPLIT_REGISTRY.items():
        print(f"Processing {split_name} [Source: {TRAIN_COUNTRY} -> Target: {test_country}]")
        df_test = features_df[test_country]
        x_test = torch.from_numpy(df_test.drop(columns =['mature'], errors='ignore').values).float()
        y_test = torch.from_numpy(df_test['mature'].values).long()
        edges_test_df = data[test_country]['edges_df']
        edge_index_test = torch.from_numpy(edges_test_df.values.T).long()

        num_test_nodes = x_test.size(0)

        x_combined = torch.cat((x_train, x_test), dim=0)
        y_combined = torch.cat((y_train, y_test), dim=0)

        edge_index_shifted = edge_index_test + num_train_nodes
        edge_index_combined = torch.cat((edge_index_train, edge_index_shifted), dim=1)

        train_mask,val_mask,test_mask = create_country_splits(num_train_nodes, num_test_nodes)

        data_obj = Data(x=x_combined, edge_index=edge_index_combined, y=y_combined)
        data_obj.train_mask = torch.from_numpy(train_mask).bool()
        data_obj.val_mask = torch.from_numpy(val_mask).bool()
        data_obj.test_mask = torch.from_numpy(test_mask).bool()

        save_dir = DATA_ROOT / "split_data" / DATASET_NAME / split_name
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f"{split_name}_data.pt"
        torch.save(data_obj, file_path)
        print(f"Saved {split_name} data to {file_path}")

def get_dataset(split_name):

    file_path = DATA_ROOT / "split_data" / DATASET_NAME / split_name / f"{split_name}_data.pt"
    data = torch.load(file_path, weights_only=False)

    return data, data.train_mask, data.val_mask, data.test_mask


if __name__ == '__main__':

    data = load_twitch_gamers(DATASET_CONFIGS[DATASET_NAME])
    features_df = preprocess_twitch_gamers(data)
    #save_all_splits_json()
    #save_splits_pt()