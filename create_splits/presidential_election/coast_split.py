import numpy as np
from utils.mask_creation import _create_mask

def create_coast_split(features_df, y_np ):
    """
    creates a shift between interior states (republican) and coastal states (democratic)
    split_name: split_2
    """
    train_rep = 0.9
    test_dem = 0.8

    val_states = {
        'arizona', 'georgia', 'nevada', 'northcarolina'
    }

    test_states = {
        'california', 'newyork', 'illinois', 'newjersey', 'virginia', 'washington', 'massachusetts',
        'maryland', 'colorado', 'minnesota', 'oregon', 'connecticut', 'hawaii', 'delaware',
        'rhodeisland', 'vermont', 'maine', 'newmexico', 'newhampshire', 'districtofcolumbia'
    }

    all_states = set(features_df['state'].unique())
    train_states = all_states.difference(test_states.union(val_states))

    num_nodes = len(features_df)
    rng = np.random.default_rng(seed=42)
    states = features_df['state'].values

    train_nodes_pool = np.array([i for i, s in enumerate(states) if s in train_states])
    test_nodes_pool = np.array([i for i, s in enumerate(states) if s in test_states])
    val_nodes_pool = np.array([i for i, s in enumerate(states) if s in val_states])

    rng.shuffle(train_nodes_pool); rng.shuffle(test_nodes_pool); rng.shuffle(val_nodes_pool)
    train_list = []; test_list = []; val_list = [val_nodes_pool]

    # MAXIMIZE REPUBLICANS in TRAIN
    y_tr = y_np[train_nodes_pool]
    tr_dem = train_nodes_pool[y_tr == 0] #Minorty = Democrtas
    tr_rep = train_nodes_pool[y_tr == 1] #Majority = Republicans
    rng.shuffle(tr_dem); rng.shuffle(tr_rep)

    tr_rep_keep = tr_rep
    ratio_tr = (1.0 - train_rep) / train_rep
    num_dem_nodes = int(len(tr_rep_keep)* ratio_tr)

    tr_dem_keep = tr_dem[:num_dem_nodes]
    tr_dem_dis = tr_dem[num_dem_nodes:]

    val_list.append(tr_dem_dis)
    train_list.append(np.concatenate([tr_rep_keep, tr_dem_keep]))

    # MAXIMIZE DEMOCRATS in TEST
    y_te = y_np[test_nodes_pool]
    te_dem = test_nodes_pool[y_te == 0] #Majority
    te_rep = test_nodes_pool[y_te == 1] # Minority
    rng.shuffle(te_dem); rng.shuffle(te_rep)

    te_dem_keep = te_dem
    ratio_te = (1.0 - test_dem) / test_dem
    num_rep_nodes = int(len(te_dem_keep) * ratio_te)
    num_rep_nodes = min(num_rep_nodes, len(te_rep))

    te_rep_keep = te_rep[:num_rep_nodes]
    te_rep_dis = te_rep[num_rep_nodes:]

    test_list.append(np.concatenate([te_dem_keep, te_rep_keep]))
    train_list.append(np.concatenate([te_dem_keep, te_rep_keep]))
    train_list.append(te_rep_dis)

    train_nodes = np.concatenate(train_list)
    test_nodes = np.concatenate(test_list)
    val_nodes = np.concatenate(val_list)

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)