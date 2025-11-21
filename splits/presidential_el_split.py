import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
from torch_geometric.data import Data
from data_loader import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ZIP_PATH = BASE_DIR/'data/presidential_election/presidential_election.zip'
EXTRACT_DIR = BASE_DIR/'data/presidential_election/'

TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.1
TEST_RATIO = 1.0 - TRAIN_RATIO - VALIDATION_RATIO

def load_data():
    feature_name = 'presidential_election_nodes.csv'
    edges_name = 'presidential_election_edges.csv'

    return load_dataset(
        DATA_ZIP_PATH,
        EXTRACT_DIR,
        feature_name,
        edges_name
    )

def normalize_features(features_df, train_mask):
    """
    Normalization of features using Z-score normalization
    return: normalized and cleaned features_df
    """
    norm_cols = [
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
    features_df[norm_cols] = features_df[norm_cols].apply(pd.to_numeric, errors='coerce').fillna(features_df[norm_cols].mean())
    scaler= StandardScaler()
    scaler.fit(features_df.loc[train_mask, norm_cols])
    features_df[norm_cols] = scaler.transform(features_df[norm_cols])
    return features_df [norm_cols].copy()

def create_split(features_df):
    """
    trying a random split where different counties are randomly assigned to train, val and test (should mimic real-world)
    :return: train, validation and test mask
    """
    num_nodes = len(features_df.index)

    train_size = int(num_nodes * TRAIN_RATIO)
    val_size = int(num_nodes * VALIDATION_RATIO)

    shuffle_ind = np.random.permutation(num_nodes)

    train_nodes = shuffle_ind[:train_size]
    val_nodes = shuffle_ind[train_size:train_size + val_size]
    test_nodes = shuffle_ind[train_size + val_size:]

    #Create masks
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    return train_mask, val_mask, test_mask

def compute_class_weights(y):
    """
    Compute class weights to handle class imbalance
    total_samples / (num_classes * num_samples_per_class)
    """
    #classes, counts = y.unqiue(return_counts=True)
    #total_samples = y.size(0)
    #num_classes = len(classes)

    #weights = total_samples/ (num_classes * counts.float())
    weights = torch.tensor([8.0,1.0])
    return weights

def get_mask():

    features_df, edges_df = load_data()
    # Label encoding
    # 1: Republican, 0: Democrat
    Y = features_df['Label (County)'].values
    print('Label values:', Y)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    train_mask, val_mask, test_mask = create_split(features_df)
    X = normalize_features(features_df, train_mask)

    edge_index = edges_df.values.T
    print('edge index:', edge_index)
    X_np = X.values
    print(X_np.shape)
    X_tensor = torch.from_numpy(X_np).float()
    print('X_tensor:', X_tensor)
    Y_tensor = torch.from_numpy(Y).long()
    edge_index_tensor = torch.from_numpy(edge_index).long()

    data = Data(x=X_tensor, edge_index=edge_index_tensor, y=Y_tensor)

    train_mask =torch.from_numpy(train_mask).to(torch.bool)
    val_mask = torch.from_numpy(val_mask).to(torch.bool)
    test_mask = torch.from_numpy(test_mask).to(torch.bool)

    return data, train_mask, val_mask, test_mask


if __name__ == '__main__':
    data, train_mask, val_mask, test_mask = get_mask()

