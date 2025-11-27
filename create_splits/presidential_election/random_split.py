import numpy as np
from utils.mask_creation import _create_mask

def create_random_split(num_nodes, TRAIN_RATIO, VALIDATION_RATIO):
    """
    trying a random split where different counties are randomly assigned to train, val and test
    (should mimic real-world)
    split_name: split_0
    """

    train_size = int(num_nodes * TRAIN_RATIO)
    val_size = int(num_nodes * VALIDATION_RATIO)

    shuffle_ind = np.random.permutation(num_nodes)

    train_nodes = shuffle_ind[:train_size]
    val_nodes = shuffle_ind[train_size:train_size + val_size]
    test_nodes = shuffle_ind[train_size + val_size:]

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)