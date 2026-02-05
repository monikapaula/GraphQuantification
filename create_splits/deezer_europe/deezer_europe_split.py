import torch
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer, StandardScaler
from torch_geometric.data import Data
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
    'split_0': {'type': 'male_dominated', 'featureless': False},
    'split_1': {'type': 'female_dominated', 'featureless': False},
    'split_2': {'type': 'popular_artists', 'featureless': False},
    'split_3': {'type': 'male_dominated', 'featureless': True},
    'split_4': {'type': 'female_dominated', 'featureless': True},
    'split_5': {'type': 'popular_artists', 'featureless': True}
}

def load_data():
    cfg= DATASET_CONFIGS[DATASET_NAME]
    features_df, edges_df, target_df = load_deezer_europe(cfg)
    return features_df, edges_df, target_df

def svd_embeddings(features_df, edges_df, embed_dim=1024):
    """
    creates user embeddings using TruncatedSVD
    """
    rw = 3
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

    tfidf = TfidfTransformer()
    weighted = tfidf.fit_transform(user_artist_matrix)
    svd = TruncatedSVD(n_components=embed_dim, algorithm='arpack', random_state=42)
    X_emb = svd.fit_transform(weighted)

    X_embeddings = np.zeros((num_users, embed_dim + 1), dtype=np.float32)
    X_embeddings[valid_indices, :embed_dim] = X_emb
    X_embeddings[valid_indices, -1] = 1.0

    edge_index = edges_df.values
    adj = sp.coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
                        shape=(num_users, num_users))
    adj = adj + adj.T
    degrees = np.array(adj.sum(axis=1)).flatten()
    log_degrees = np.log1p(degrees).reshape(-1,1)

    X_embeddings = np.zeros((num_users, embed_dim + 1), dtype=np.float32)
    X_embeddings[valid_indices, :embed_dim] = X_emb
    X_embeddings[valid_indices, -2] = 1.0
    X_embeddings[:, -1]= log_degrees.flatten()

    d_inv = sp.diags(1.0 / np.maximum(degrees, 1.0))
    a_tilde = d_inv @ adj

    X_propagated = X_embeddings.copy()
    for _ in range(rw):
        X_propagated = a_tilde @ X_propagated

    scaler = StandardScaler()
    X_emb = scaler.fit_transform(X_propagated)
    var = svd.explained_variance_ratio_.sum()
    print(f"SVD Explained Variance (dim={embed_dim}): {var:.2%}")

    return X_emb

def save_splits_pt():
    features_df, edges_df, target_df = load_data()

    X_np = svd_embeddings(features_df, edges_df, embed_dim=1024)
    X_tensor = torch.from_numpy(X_np).float()

    Y_np = target_df.sort_values('id')['target'].values
    Y_tensor = torch.from_numpy(Y_np).long()

    edge_index_np = edges_df.values.T
    edge_index = torch.from_numpy(edge_index_np).long()

    data = Data(x=X_tensor, edge_index=edge_index, y=Y_tensor)
    save_data_obj(data, DATASET_NAME)

    func_map = {
        'male_dominated': male_gender_split,
        'female_dominated': female_gender_split,
        'popular_artists': create_popularity_split,
    }

    for split_name, config in SPLIT_REGISTRY.items():
        split_func = func_map[config['type']]

        if config['type'] in ['male_dominated', 'female_dominated', 'popular_artists']:
            masks= split_func(features_df, target_df, include_featureless_nodes=config['featureless'])
        else:
            masks = split_func(features_df, target_df)

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
    save_splits_pt()
    features, edges, _ = load_data()
    embeddings = svd_embeddings(features,edges, embed_dim=1024)
    data, train_mask, val_mask, test_mask = get_dataset("split_1")
    y = data.y
    class_balance(y, train_mask, "TRAIN")
    class_balance(y, val_mask, "VAL")
    class_balance(y, test_mask, "TEST")

