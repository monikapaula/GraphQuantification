import torch
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np

from utils.data_loader import load_deezer_europe, DATASET_CONFIGS, save_data_obj
from create_splits.split_manager import save_split
from create_splits.deezer_europe.gender_split import create_gender_split
from utils.metrics import class_balance

DATASET_NAME = 'deezer_europe'
CONFIG = DATASET_CONFIGS[DATASET_NAME]
SPLIT_REGISTRY = {
    'split_0': 'male_dominated',
    'split_1': 'female_dominated'
}

def load_data():
    cfg= DATASET_CONFIGS[DATASET_NAME]
    features_df, edges_df, target_df = load_deezer_europe(cfg)
    return features_df, edges_df, target_df

def manage_splits(split_name, num_nodes, create_sp):
    train_mask, val_mask, test_mask = create_sp()
    save_split(DATASET_NAME, split_name, train_mask, val_mask, test_mask)
    return train_mask, val_mask, test_mask

def user_embeddings(features_df, embed_dim=64):

    user_ids = [int(u) for u in features_df.keys()]
    num_users = max(user_ids) + 1

    all_artists = set()
    for user in range(num_users):
        artists = features_df.get(str(user), [])
        all_artists.update(artists)
    artist_to_idx = {a: i for i, a in enumerate(sorted(all_artists))}
    num_artists = len(artist_to_idx)

    torch.manual_seed(0)
    artist_emb = nn.Embedding(num_artists, embed_dim)
    X_emb = np.zeros((num_users, embed_dim), dtype=np.float32)
    for u in range(num_users):
        artists = features_df.get(str(u), [])
        if not artists:
            continue
        idxs = [artist_to_idx[a] for a in artists if a in artist_to_idx]
        if not idxs:
            continue
        artist_t = torch.tensor(idxs, dtype=torch.long)
        vec = artist_emb(artist_t).mean(dim=0)
        X_emb[u] = vec.detach().numpy()

    return X_emb

def get_dataset(split_name= None, split_type = None):
    features_df, edges_df, target_df = load_data()
    num_nodes = target_df['id'].max()+1

    if split_name:
        if split_name in SPLIT_REGISTRY:
            split_type = SPLIT_REGISTRY[split_name]
        else:
            print(f"Split name {split_name} not in registry")

    if split_type == "male_dominated":
        create_sp = lambda : create_gender_split(features_df, target_df)
    elif split_type == "female_dominated":
        create_sp = lambda : create_gender_split(features_df, target_df)

    else:
        raise ValueError("Unknown split type")

    train_mask, val_mask, test_mask = manage_splits(split_name, num_nodes, create_sp)

    X_np = user_embeddings(features_df, embed_dim=64)

    Y_np = target_df.sort_values('id')['target'].values
    edge_index_np = edges_df.values.T

    X_tensor = torch.from_numpy(X_np).float()
    edge_index = torch.from_numpy(edge_index_np).long()
    Y_tensor = torch.from_numpy(Y_np).long()

    data = Data(x=X_tensor, edge_index=edge_index, y=Y_tensor)

    train_mask = torch.from_numpy(train_mask).to(torch.bool)
    val_mask = torch.from_numpy(val_mask).to(torch.bool)
    test_mask = torch.from_numpy(test_mask).to(torch.bool)

    return data, train_mask, val_mask, test_mask

if __name__ == "__main__":
    data, train_mask, val_mask, test_mask = get_dataset(split_name="split_0")
    save_data_obj(data, DATASET_NAME)
    y = data.y
    class_balance(y, train_mask, "TRAIN")
    class_balance(y, val_mask, "VAL")
    class_balance(y, test_mask, "TEST")