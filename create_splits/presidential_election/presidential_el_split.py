import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Data
from utils.data_loader import load_dataset, DATASET_CONFIGS, save_data_obj, DATA_ROOT
from create_splits.split_manager import save_split
from create_splits.presidential_election.random_split import create_random_split
from create_splits.presidential_election.geographic_split import create_geographic_split
from create_splits.presidential_election.coast_split import create_coast_split
from create_splits.presidential_election.metropolitan_split import create_metro_split,METRO_DROP_COLS

DATASET_NAME = 'presidential_election'
CONFIG = DATASET_CONFIGS[DATASET_NAME]
SPLIT_REGISTRY= {
    'split_0': 'random',
    'split_1': 'geographic',
    'split_2': 'coast',
    'split_3': 'metropolitan'
}

TRAIN_RATIO = 0.5
VALIDATION_RATIO = 0.1
TEST_RATIO = 1.0 - TRAIN_RATIO - VALIDATION_RATIO

NORM_COLS = [
        'Mean income (dollars)',
        'Total Population',
        'Population with less than 9th grade education',
        'Population with 9th to 12th grade education, no diploma',
        'High School graduate and equivalent',
        'Some College,No Degree',
        'Associates Degree',
        'Bachelors Degree',
        'Graduate or professional degree',
        'Gini Index',
        'Density per square km',
        'Hispanic or Latino percentage',
        'NH-White percentage',
        'NH-Black percentage',
        'Percentage engaged in Management, business, science, and arts occupations',
        'Percentage engaged in Service Occupations',
        'Percentage engaged in Sales and Office',
        'Percentage engaged in Resources and Construction',
        'Percentage engaged in Transportation'
]

def load_data():
    features_df, edges_df = load_dataset(
        extract_dir= CONFIG['extract_dir'],
        feature_filename= CONFIG['feature_filename'],
        edges_filename=CONFIG['edges_filename'],
    )
    return features_df, edges_df

def encode_labels(features_df):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(features_df['Label (County)'].values)
    return encoded_labels

def preprocess_features(df, used_cols):
    df_clean = df.copy()
    df_clean = df_clean[used_cols].apply(pd.to_numeric, errors='coerce')
    df_clean = df_clean[used_cols].fillna(df_clean[used_cols].mean())
    return df_clean

def normalize_features(features_df, train_mask, used_cols):
    """
    Normalization of features using Z-score normalization
    return: normalized and cleaned features_df
    """
    scaler = StandardScaler()
    scaler.fit(features_df.loc[train_mask,used_cols])
    scaled_features_df = scaler.transform(features_df[used_cols])

    return scaled_features_df.copy()

def manage_splits(split_name, num_nodes, create_sp):
    train_mask, val_mask, test_mask = create_sp()
    save_split(DATASET_NAME, split_name, train_mask, val_mask, test_mask)
    return train_mask, val_mask, test_mask

def save_splits_pt():
    features_df, edges_df = load_data()
    num_nodes = len(features_df)
    Y_np = encode_labels(features_df)

    edge_index = torch.from_numpy(edges_df.values.T).long()
    Y_tensor = torch.from_numpy(Y_np).float()

    for split_name, split_type in SPLIT_REGISTRY.items():
        current_cols = NORM_COLS.copy()

        if split_type == 'geographic':
            train_mask, val_mask, test_mask = manage_splits(split_name, num_nodes, lambda : create_geographic_split(features_df, VALIDATION_RATIO))
        elif split_type == 'random':
            train_mask, val_mask, test_mask = manage_splits(split_name, num_nodes, lambda : create_random_split(features_df, TEST_RATIO, VALIDATION_RATIO))
        elif split_type == 'coast':
            train_mask, val_mask, test_mask = manage_splits(split_name, num_nodes, lambda : create_coast_split(features_df))
        elif split_type == 'metropolitan':
            train_mask, val_mask, test_mask = manage_splits(split_name, num_nodes, lambda : create_metro_split(features_df))
            for col in METRO_DROP_COLS:
                if col in current_cols:
                    current_cols.remove(col)

        features_clean = preprocess_features(features_df, current_cols)
        scaler = StandardScaler()
        scaler.fit(features_clean.loc[train_mask, current_cols])

        X_np = scaler.transform(features_clean[current_cols])
        X_tensor = torch.from_numpy(X_np).float()

        data = Data(x=X_tensor, edge_index=edge_index, y=Y_tensor)

        data.train_mask = torch.from_numpy(train_mask).bool()
        data.val_mask = torch.from_numpy(val_mask).bool()
        data.test_mask = torch.from_numpy(test_mask).bool()

        save_dir = DATA_ROOT/"split_data"/DATASET_NAME/split_name
        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir/f"{split_name}_data.pt"
        torch.save(data, file_path)


def get_dataset(split_name):

    file_path = DATA_ROOT/"split_data"/DATASET_NAME/split_name/f"{split_name}_data.pt"
    data = torch.load(file_path, weights_only=False)

    return data, data.train_mask, data.val_mask, data.test_mask


if __name__ == '__main__':
    save_splits_pt()
    data, train_mask, val_mask, test_mask = get_dataset(split_name='split_1')
