import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = 'data/twitch_gamers/twitch_gamers_features_de.csv'
EDGES_PATH = 'data/twitch_gamers/twitch_gamers_edges_de.csv'

TRAIN_RATIO = 0.10 #Top 10% (by views)
VALIDATION_RATIO = 0.10
TEST_RATIO = 1.0 - TRAIN_RATIO - VALIDATION_RATIO

def load_data(features_path, edges_path):

    try:
        features_df = pd.read_csv(features_path)
        edges_df = pd.read_csv(edges_path)
        print("Loaded features and edges")
        return features_df, edges_df
    except FileNotFoundError:
        print('File not found')
        return None, None

def normalize_features(features_df):
    """
    Normalization of features_df
    :param features_df:
    :return: X,Y
    """
    features_df = features_df[['views','mature','life_time','affiliate']].copy()
    scaler= StandardScaler()
    norm_cols = ['views','life_time']
    features_df [norm_cols] = scaler.fit_transform(features_df[norm_cols])

    print('Normalization:', features_df.head())
    X = features_df.values
    Y = features_df['dead_account'].values

    return X, Y

def create_split(features_df, edges_df):
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

    return train_mask, val_mask, test_mask


