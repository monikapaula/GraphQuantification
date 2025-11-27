import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Data
from utils.data_loader import load_dataset, DATASET_CONFIGS, save_data_obj
from utils.mask_creation import _create_mask
from create_splits.split_manager import save_split, load_split

DATASET_NAME = 'presidential_election'
CONFIG = DATASET_CONFIGS[DATASET_NAME]
SPLIT_REGISTRY= {
    'split_0': 'random',
    'split_1': 'geographic',
    'split_2': 'coast'
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
        zip_path= CONFIG['zip_path'],
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


def create_random_split(num_nodes):
    """
    trying a random split where different counties are randomly assigned to train, val and test (should mimic real-world)
    split_name: split_0
    """

    train_size = int(num_nodes * TRAIN_RATIO)
    val_size = int(num_nodes * VALIDATION_RATIO)

    shuffle_ind = np.random.permutation(num_nodes)

    train_nodes = shuffle_ind[:train_size]
    val_nodes = shuffle_ind[train_size:train_size + val_size]
    test_nodes = shuffle_ind[train_size + val_size:]

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)

def create_geographic_split(features_df):
    """
    Train and val set on non-Eastern states, test on Eastern states, to create natural shift
    between democrats and Republicans
    split_name: split_1
    """
    state_lables = features_df['state'].values
    num_nodes = len(state_lables)

    eastern_states = {
        'connecticut', 'maine', 'massachusetts', 'newhampshire', 'rhodeisland',
        'vermont', 'newjersey', 'newyork', 'pennsylvania',
        'delaware', 'maryland', 'virginia', 'westvirginia', 'northcarolina',
        'southcarolina', 'georgia', 'florida'
    }

    eastern_idx = np.array([i for i in range(num_nodes)
                            if state_lables[i] in eastern_states])
    non_eastern_idx = np.array([i for i in range(num_nodes)
                                if state_lables[i] not in eastern_states])

    val_size = int(len(non_eastern_idx) * VALIDATION_RATIO)
    val_nodes = non_eastern_idx[:val_size]
    train_nodes = non_eastern_idx[val_size:]
    test_nodes = eastern_idx

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)

def create_coast_shift(features_df, y_np ):
    """
    creates a shift between interior states (republican) and coastal states (democratic)
    split_name: split_2
    """
    train_rep = 0.9
    test_dem = 0.8

    val_states = {
        'arizona', 'georgia', 'nevada', 'northcarolina'
    }

    test_states = {
        'california', 'newyork', 'illinois', 'newjersey', 'virginia', 'washington', 'massachusetts',
        'maryland', 'colorado', 'minnesota', 'oregon', 'connecticut', 'hawaii', 'delaware',
        'rhodeisland', 'vermont', 'maine', 'newmexico', 'newhampshire', 'districtofcolumbia'
    }

    all_states = set(features_df['state'].unique())
    train_states = all_states.difference(test_states.union(val_states))

    num_nodes = len(features_df)
    rng = np.random.default_rng(seed=42)
    states = features_df['state'].values

    train_nodes_pool = np.array([i for i, s in enumerate(states) if s in train_states])
    test_nodes_pool = np.array([i for i, s in enumerate(states) if s in test_states])
    val_nodes_pool = np.array([i for i, s in enumerate(states) if s in val_states])

    rng.shuffle(train_nodes_pool); rng.shuffle(test_nodes_pool); rng.shuffle(val_nodes_pool)
    train_list = []; test_list = []; val_list = [val_nodes_pool]

    # MAXIMIZE REPUBLICANS in TRAIN
    y_tr = y_np[train_nodes_pool]
    tr_dem = train_nodes_pool[y_tr == 0] #Minorty = Democrtas
    tr_rep = train_nodes_pool[y_tr == 1] #Majority = Republicans
    rng.shuffle(tr_dem); rng.shuffle(tr_rep)

    tr_rep_keep = tr_rep
    ratio_tr = (1.0 - train_rep) / train_rep
    num_dem_nodes = int(len(tr_rep_keep)* ratio_tr)

    tr_dem_keep = tr_dem[:num_dem_nodes]
    tr_dem_dis = tr_dem[num_dem_nodes:]

    val_list.append(tr_dem_dis)
    train_list.append(np.concatenate([tr_rep_keep, tr_dem_keep]))

    # MAXIMIZE DEMOCRATS in TEST
    y_te = y_np[test_nodes_pool]
    te_dem = test_nodes_pool[y_te == 0] #Majority
    te_rep = test_nodes_pool[y_te == 1] # Minority
    rng.shuffle(te_dem); rng.shuffle(te_rep)

    te_dem_keep = te_dem
    ratio_te = (1.0 - test_dem) / test_dem
    num_rep_nodes = int(len(te_dem_keep) * ratio_te)
    num_rep_nodes = min(num_rep_nodes, len(te_rep))

    te_rep_keep = te_rep[:num_rep_nodes]
    te_rep_dis = te_rep[num_rep_nodes:]

    test_list.append(np.concatenate([te_dem_keep, te_rep_keep]))
    train_list.append(np.concatenate([te_dem_keep, te_rep_keep]))
    train_list.append(te_rep_dis)

    train_nodes = np.concatenate(train_list)
    test_nodes = np.concatenate(test_list)
    val_nodes = np.concatenate(val_list)

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)


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

    if split_type is None:
        if split_name in SPLIT_REGISTRY:
            split_type = SPLIT_REGISTRY[split_name]
        else:
            print(f"Split name {split_name} not in registry")

    if split_type == "geographic":
        create_sp = lambda: create_geographic_split(features_df)
    elif split_type == "random":
        create_sp = lambda: create_random_split(num_nodes)
    elif split_type == "coast":
        create_sp = lambda: create_coast_shift(features_df,Y_np)
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
    data, train_mask, val_mask, test_mask = get_dataset(split_name='split_2')
    print("size training:", train_mask.sum())
    print("size val:", val_mask.sum())
    print("size test:", test_mask.sum())
    save_data_obj(data, 'presidential_election')

