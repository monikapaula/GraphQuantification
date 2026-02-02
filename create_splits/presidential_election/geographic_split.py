import numpy as np
from utils.mask_creation import _create_mask

def create_geographic_split(features_df, VALIDATION_RATIO):
    """
    Train and val set on southern states, test on northern states
    split_name: split_1
    """
    state_lables = features_df['state'].values
    num_nodes = len(state_lables)
    rng = np.random.default_rng(seed=42)

    north_states = {
        'washington', 'oregon', 'idaho', 'montana',
        'minnesota','wyoming', 'northdakota', 'southdakota', 'nebraska','wisconsin', 'michigan',
        'illinois', 'colorado', 'iowa', 'indiana', 'ohio', 'maryland',
        'newyork', 'newjersey', 'pennsylvania', 'vermont', 'newhampshire',
        'massachusetts', 'connecticut', 'rhodeisland', 'maine', 'delaware', 'districtofcolumbia'
    }

    test_idx = np.array([i for i in range(num_nodes)
                            if state_lables[i] in north_states])
    train_idx = np.array([i for i in range(num_nodes)
                                if state_lables[i] not in north_states])
    rng.shuffle(train_idx)
    val_size = int(len(train_idx) * VALIDATION_RATIO)
    val_nodes = train_idx[:val_size]
    train_nodes = train_idx[val_size:]
    test_nodes = test_idx

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)