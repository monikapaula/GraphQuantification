import numpy as np
from networkx.algorithms.bipartite.basic import density

from utils.mask_creation import _create_mask

def create_random_split(features_df, TRAIN_RATIO, VALIDATION_RATIO):
    """
    trying a random split where different counties are randomly assigned to train, val and test
    split_name: split_0
    """
    randomness_factor = 0.2
    rng = np.random.default_rng(42)
    num_nodes = len(features_df)
    rank_idx = features_df['Density per square km'].rank(pct=True)
    noise = rng.uniform(-randomness_factor, randomness_factor, size=num_nodes)
    mix = rank_idx + noise

    sorted_idx = np.argsort(mix)

    train_size = int(num_nodes * TRAIN_RATIO)
    val_size = int(num_nodes * VALIDATION_RATIO)

    train_nodes = sorted_idx[:train_size]
    val_nodes = sorted_idx[train_size : train_size + val_size]
    test_nodes = sorted_idx[train_size + val_size:]

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)