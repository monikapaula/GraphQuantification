import numpy as np

from utils.mask_creation import _create_mask

def flatten_arts (arts):
    flat = []
    for a in arts:
        if isinstance(a, (list, tuple,set)):
            flat.extend(a)
        else:
            flat.append(a)
    return flat

def create_popularity_split(features_df, target_df, include_featureless_nodes=False):
    """
    creates a shift between users based on the popularity of the artists they listen to
    split_name: split_2
    """
    val_frac = 0.2
    rng = np.random.default_rng(42)
    num_nodes = target_df['id'].max()+1

    artist_counts = {}
    for arts in features_df.values():
        flat_arts = flatten_arts(arts)
        for a in set(flat_arts):
            artist_counts[a] = artist_counts.get(a, 0) + 1
    sorted_arts = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)
    n_unique_arts = len(sorted_arts)
    top_art_cutoff = int(0.2 * n_unique_arts)
    mainstream_art_idx = set([a[0] for a in sorted_arts[:top_art_cutoff]])

    train_idx = []
    test_idx = []
    dropped_count = 0

    for user_id, arts in features_df.items():
        user_id = int(user_id)
        if user_id >= num_nodes:
            continue
        if not arts:
            if include_featureless_nodes:
                train_idx.append(user_id)
            else:
                dropped_count += 1
            continue

        m_hits = len([a for a in arts if a in mainstream_art_idx])
        m_share = m_hits / len(arts)

        if m_share >= 0.75:
            train_idx.append(user_id)
        else:
            test_idx.append(user_id)

    mainstream_pool = np.array(train_idx, dtype=int)
    niche_pool = np.array(test_idx, dtype=int)
    rng.shuffle(mainstream_pool)

    n_val = int(len(mainstream_pool) * val_frac)
    train_nodes = mainstream_pool[n_val:]
    val_nodes = mainstream_pool[:n_val]
    test_nodes = niche_pool

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)