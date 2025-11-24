import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from pathlib import Path
from torch_geometric.data import Data
from data_loader import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ZIP_PATH = BASE_DIR/'data/twitch_gamers/twitch_gamers_de.zip'
EXTRACT_DIR = BASE_DIR/'data/twitch_gamers/'

TRAIN_RATIO = 0.010 #Top 10% (by views)
VALIDATION_RATIO = 0.010
TEST_RATIO = 1.0 - TRAIN_RATIO - VALIDATION_RATIO

def load_data():
    feature_name = 'twitch_gamers_features_de.csv'
    edges_name = 'twitch_gamers_edges_de.csv'

    return load_dataset(
        DATA_ZIP_PATH,
        EXTRACT_DIR,
        feature_name,
        edges_name
    )

def normalize_features(features_df):
    """
    Normalization of features_df
    :param features_df:
    :return: normalized features_df
    """
    features_df = features_df[['views','mature','life_time','affiliate']].copy()
    scaler= StandardScaler()
    norm_cols = ['views','life_time']
    features_df [norm_cols] = scaler.fit_transform(features_df[norm_cols])

    #print('Normalization:', features_df.head())
    return features_df

def create_split(features_df):
    """
    Creates training, validation and test split based on the top 10% of streamers (in views)
    :return: train, validation and test mask
    """
    num_nodes = len(features_df)
    sorted_nodes = features_df.sort_values('views', ascending=False).index.values

    train_size = int(num_nodes * TRAIN_RATIO)
    val_size = int(num_nodes * VALIDATION_RATIO)

    train_nodes = sorted_nodes[:train_size] #10%
    val_nodes = sorted_nodes[train_size:train_size + val_size] #10 %
    test_nodes = sorted_nodes[train_size + val_size:]   #remaining 80%

    #Create masks
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True
    print('Train Nodes:', len(train_nodes))

    return train_mask, val_mask, test_mask

def get_masks():

    features_df, edges_df = load_data()

    Y = features_df['dead_account'].values
    X = normalize_features(features_df)
    #print ('Normalized features:', X)
    edge_index = edges_df.values.T
    #print ('Edge index shape:', edge_index.shape)
    #print( 'Features shape:', X.shape)
    #print( 'Labels shape:', Y.shape)

    X_np = X.values
    X_tensor = torch.from_numpy(X_np).float()
    edge_index = torch.from_numpy(edge_index).long()
    Y_tensor = torch.from_numpy(Y).long()
    data = Data(x=X_tensor, edge_index=edge_index, y=Y_tensor)

    print(X_tensor.shape)
    train_np, val_np, test_np = create_split(X)

    train_mask =torch.from_numpy(train_np).to(torch.bool)
    val_mask = torch.from_numpy(val_np).to(torch.bool)
    test_mask = torch.from_numpy(test_np).to(torch.bool)

    print('Training mask:', len(train_mask))
    return data, train_mask, val_mask, test_mask

if __name__ == "__main__":
    data, train_m, val_m, test_m = get_masks()
