import numpy as np
from utils.mask_creation import _create_mask

def create_geographic_split(features_df, VALIDATION_RATIO):
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