import numpy as np
from utils.mask_creation import _create_mask

def create_coast_split(features_df):
    """
    creates a shift between interior states (republican) and coastal states (democratic)
    split_name: split_2
    """

    val_states = {
        'arizona', 'georgia', 'nevada', 'northcarolina'
    }

    test_states = {
        'alaska', 'florida', 'california', 'louisiana','newyork', 'illinois', 'newjersey', 'virginia', 'washington', 'massachusetts',
        'maryland', 'colorado', 'minnesota', 'oregon', 'connecticut', 'hawaii', 'delaware', 'southcarolina'
        'rhodeisland', 'vermont', 'maine', 'newmexico', 'newhampshire', 'districtofcolumbia'
    }
    num_nodes = len(features_df)
    all_states = set(features_df['state'].unique())
    train_states = all_states.difference(test_states.union(val_states))

    states = features_df['state'].values

    train_mask = np.isin(states, list(train_states))
    val_mask = np.isin(states, list(val_states))
    test_mask = np.isin(states, list(test_states))

    train_nodes = np.where(train_mask)[0]
    val_nodes = np.where(val_mask)[0]
    test_nodes = np.where(test_mask)[0]

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)