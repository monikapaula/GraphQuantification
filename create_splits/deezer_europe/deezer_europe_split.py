import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np

from utils.data_loader import load_deezer_europe, DATASET_CONFIGS, save_data_obj ,DATA_ROOT
from create_splits.split_manager import save_split, load_split
from create_splits.deezer_europe.male_gender_split import male_gender_split
from create_splits.deezer_europe.female_gender_split import female_gender_split
from create_splits.deezer_europe.popularity_split import create_popularity_split
from create_splits.deezer_europe.user_activity import create_user_activity_split
from utils.metrics import class_balance

DATASET_NAME = 'deezer_europe'
CONFIG = DATASET_CONFIGS[DATASET_NAME]
SPLIT_REGISTRY = {
    'split_0': 'male_dominated',
    'split_1': 'female_dominated',
    'split_2': 'popular_artists',
    'split_3': 'user_activity'
}

def load_data():
    cfg= DATASET_CONFIGS[DATASET_NAME]
    features_df, edges_df, target_df = load_deezer_europe(cfg)
    return features_df, edges_df, target_df

def svd_embeddings(features_df, embed_dim=32):
    """
    creates user embeddings using TruncatedSVD
    """

    user_ids = [int(u) for u in features_df.keys()]
    num_users = max(user_ids) + 1

    valid_indices = []
    valid_artists = []

    for u in range(num_users):
        artists = features_df.get(str(u), [])
        if artists:
            valid_indices.append(u)
            valid_artists.append(artists)

    mlb = MultiLabelBinarizer(sparse_output=True)
    user_artist_matrix = mlb.fit_transform(valid_artists)
    user_artist_matrix = user_artist_matrix.astype(np.float64)

    svd = TruncatedSVD(n_components=embed_dim, algorithm='arpack', random_state=42)
    X_emb = svd.fit_transform(user_artist_matrix)

    scaler = Normalizer(norm='l2')
    X_emb = scaler.fit_transform(X_emb)

    X_embeddings = np.zeros((num_users, embed_dim), dtype=np.float32)
    X_embeddings[valid_indices] = X_emb.astype(np.float32)

    return X_embeddings

def save_splits_pt():
    features_df, edges_df, target_df = load_data()

    X_np = svd_embeddings(features_df, embed_dim=32)
    X_tensor = torch.from_numpy(X_np).float()

    Y_np = target_df.sort_values('id')['target'].values
    Y_tensor = torch.from_numpy(Y_np).long()

    edge_index_np = edges_df.values.T
    edge_index = torch.from_numpy(edge_index_np).long()

    data = Data(x=X_tensor, edge_index=edge_index, y=Y_tensor)
    save_data_obj(data, DATASET_NAME)

    for split_name, split_type in SPLIT_REGISTRY.items():
        if split_type == "male_dominated":
             masks = male_gender_split(features_df, target_df)
        elif split_type == "female_dominated":
            masks = female_gender_split(features_df, target_df)
        elif split_type == "popular_artists":
            masks = create_popularity_split(features_df, target_df)
        elif split_type == "user_activity":
            masks = create_user_activity_split(features_df, target_df)

        train_mask,val_mask,test_mask = masks

        save_split(DATASET_NAME, split_name, train_mask, val_mask, test_mask)

def get_dataset(split_name):
    file_path = DATA_ROOT/"split_data"/ DATASET_NAME / f"{DATASET_NAME}_data.pt"
    data = torch.load(file_path, weights_only=False)
    num_nodes = data.num_nodes
    train_mask_np, val_mask_np, test_mask_np = load_split(DATASET_NAME, split_name, num_nodes)

    train_mask = torch.from_numpy(train_mask_np).bool()
    val_mask = torch.from_numpy(val_mask_np).bool()
    test_mask = torch.from_numpy(test_mask_np).bool()

    return data, train_mask, val_mask, test_mask

if __name__ == "__main__":
    #save_splits_pt()
    features, _, _ = load_data()
    embeddings = svd_embeddings(features, embed_dim=32)
    print(f"Output shape: {embeddings.shape}")
    print(f"Vector user 0: {embeddings[0][:10]}...")
    data, train_mask, val_mask, test_mask = get_dataset("split_0")
    y = data.y
    class_balance(y, train_mask, "TRAIN")
    class_balance(y, val_mask, "VAL")
    class_balance(y, test_mask, "TEST")