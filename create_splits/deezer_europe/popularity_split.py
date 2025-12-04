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

def create_popularity_split(features_df, target_df ):
    """
    creates a shift between users based on the popularity of the artists they listen to
    mainstream artist: top 40% artists by total number of listeners
    niche artist: bottom 60% artists by total number of listeners
    split_name: split_2
    """
    top_frac = 0.4
    val_frac = 0.1

    rng = np.random.default_rng(42)
    num_nodes = target_df['id'].max()+1

    artist_counts = {}
    for arts in features_df.values():
        flat_arts = flatten_arts(arts)
        for a in set(flat_arts):
            artist_counts[a] = artist_counts.get(a, 0) + 1

    mainstream_scores = np.zeros(num_nodes, dtype=np.float32)

    for user_id, arts in features_df.items():
        user_id = int(user_id)
        if user_id >= num_nodes:
            continue
        popular = [artist_counts[a] for a in arts if a in artist_counts]
        if not popular:
            continue
        mainstream_scores[user_id] = np.mean(popular)

    all_users = np.arange(num_nodes)
    sorted_users = all_users[np.argsort(mainstream_scores[all_users])]

    n_users = len(sorted_users)
    cutoff = int ((1.0 - top_frac) * n_users)

    niche_users = sorted_users[:cutoff]
    mainstream_users = sorted_users[cutoff:]

    rng.shuffle(niche_users)
    rng.shuffle(mainstream_users)

    n_val = int(len(mainstream_users) * val_frac)
    train_nodes = mainstream_users[n_val:]
    val_nodes = mainstream_users[:n_val]
    test_nodes = niche_users

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)