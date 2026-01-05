import numpy as np
import pandas as pd

from utils.mask_creation import _create_mask

def user_activity(features_df):
    artist_counts = {int(u_id): len(set(artists)) for u_id, artists in features_df.items()}
    count_df = pd.Series(artist_counts).sort_values(ascending=False)
    split_idx = int(len(count_df) * 0.80)

    high_activity_users = count_df.index[:split_idx].tolist()
    low_activity_users = count_df.index[split_idx:].tolist()

    return high_activity_users, low_activity_users

def create_user_activity_split(features_df, target_df = None):
    """
    Create a split based on the activity of users
    high_activity_users are the top 20% (messured by the number of aritsts they listen to)
    low_activity_users: rest
    """
    val_frac = 0.25
    num_nodes = max([int(u) for u in features_df.keys()]) + 1

    high_activity_users, low_activity_users = user_activity(features_df)
    print(f"Number of low activity users: {len(low_activity_users)}")
    print(f"Number of high activity users: {len(high_activity_users)}")

    high_activity_users = np.array(high_activity_users, dtype = int)
    low_activity_users = np.array(low_activity_users, dtype = int)
    rng = np.random.default_rng(42)
    rng.shuffle(high_activity_users)
    rng.shuffle(low_activity_users)

    v_val = int(len(high_activity_users) * val_frac)
    val_nodes = high_activity_users[:v_val]
    train_nodes = high_activity_users[v_val:]
    test_nodes = low_activity_users

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)