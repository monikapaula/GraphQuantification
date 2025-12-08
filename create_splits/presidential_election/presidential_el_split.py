import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Data
from utils.data_loader import load_dataset, DATASET_CONFIGS, save_data_obj
from create_splits.split_manager import save_split, load_split
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

TRAIN_RATIO = 0.6
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
    #1: Republican, 0: Democrat
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(features_df['Label (County)'].values)
    return encoded_labels

def preprocess_features(df):
    df_clean = df.copy()
    df_clean = df_clean[NORM_COLS].apply(pd.to_numeric, errors='coerce')
    df_clean = df_clean[NORM_COLS].fillna(df_clean[NORM_COLS].mean())
    return df_clean

def normalize_features(features_df, train_mask):
    """
    Normalization of features using Z-score normalization
    return: normalized and cleaned features_df
    """
    scaler = StandardScaler()
    scaler.fit(features_df.loc[train_mask,NORM_COLS])
    scaled_features_df = scaler.transform(features_df[NORM_COLS])

    return scaled_features_df.copy()

def compute_class_weights(y):
    """
    Compute class weights to handle class imbalance
    total_samples / (num_classes * num_samples_per_class)
    """
    classes, counts = y.unique(return_counts=True)
    total_samples = y.size(0)
    num_classes = len(classes)

    weights = total_samples/ (num_classes * counts.float())

    return weights

def manage_splits(split_name, num_nodes, create_sp):
    train_mask, val_mask, test_mask = create_sp()
    save_split(DATASET_NAME, split_name, train_mask, val_mask, test_mask)
    return train_mask, val_mask, test_mask

def get_dataset(split_name= None, split_type = None):

    features_df, edges_df = load_data()
    num_nodes = len(features_df)
    Y_np = encode_labels(features_df)
    cols = NORM_COLS.copy()

    if split_type is None:
        if split_name in SPLIT_REGISTRY:
            split_type = SPLIT_REGISTRY[split_name]
        else:
            print(f"Split name {split_name} not in registry")

    if split_type == "geographic":
        create_sp = lambda: create_geographic_split(features_df,VALIDATION_RATIO)
    elif split_type == "random":
        create_sp = lambda: create_random_split(num_nodes, TRAIN_RATIO,VALIDATION_RATIO)
    elif split_type == "coast":
        create_sp = lambda: create_coast_split(features_df,Y_np)
    elif split_type == "metropolitan":
        create_sp = lambda: create_metro_split(features_df)
        for col in METRO_DROP_COLS:
            if col in cols:
                cols.remove(col)
    else:
        raise ValueError("Unknown split type")

    train_mask, val_mask, test_mask = manage_splits(split_name, num_nodes, create_sp)

    Y_np = encode_labels(features_df)
    features_df = preprocess_features(features_df)
    X_np = normalize_features(features_df, train_mask)
    edge_index_np = edges_df.values.T


    X_tensor = torch.from_numpy(X_np).float()
    edge_index = torch.from_numpy(edge_index_np).long()
    Y_tensor = torch.from_numpy(Y_np).long()

    data = Data(x=X_tensor, edge_index=edge_index, y=Y_tensor)

    train_mask =torch.from_numpy(train_mask).to(torch.bool)
    val_mask = torch.from_numpy(val_mask).to(torch.bool)
    test_mask = torch.from_numpy(test_mask).to(torch.bool)

    return data, train_mask, val_mask, test_mask


if __name__ == '__main__':
    data, train_mask, val_mask, test_mask = get_dataset(split_name='split_3')
    print("size training:", train_mask.sum())
    print("size val:", val_mask.sum())
    print("size test:", test_mask.sum())
    save_data_obj(data, 'presidential_election')

