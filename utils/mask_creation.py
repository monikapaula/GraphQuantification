import numpy as np

def _create_mask(num_nodes, train_nodes, val_nodes, test_nodes):
    # Create masks
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    return train_mask, val_mask, test_mask