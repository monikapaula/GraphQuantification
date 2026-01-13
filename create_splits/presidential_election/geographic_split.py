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

    north_states = {
        'washington', 'oregon', 'kansas', 'california', 'idaho'
        'minnesota','wyoming', 'northdakota', 'southdakota', 'nebreaska','wisconsin', 'michigan', 'illinois', 'colorado', 'iowa', 'indiana', 'ohio'
        'newyork', 'newjersey', 'pennsylvania', 'vermont', 'newhampshire',
        'massachusetts', 'connecticut', 'rhodeisland', 'maine',
        'maryland', 'delaware', 'districtofcolumbia', 'virginia'
    }

    test_idx = np.array([i for i in range(num_nodes)
                            if state_lables[i] in north_states])
    train_idx = np.array([i for i in range(num_nodes)
                                if state_lables[i] not in north_states])
    np.random.shuffle(train_idx)
    val_size = int(len(train_idx) * VALIDATION_RATIO)
    val_nodes = train_idx[:val_size]
    train_nodes = train_idx[val_size:]
    test_nodes = test_idx

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)