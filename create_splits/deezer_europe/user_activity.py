import numpy as np
import pandas as pd

from utils.mask_creation import _create_mask

def user_activity(features_df):
    active_threshold = 0.8
    min_artist = 15
    # excluding users with 0-14 artists
    # Number of users with zero artists: 6159

    artist_counts = {}
    for u_id, artist in features_df.items():
        u = int(u_id)
        artist_counts[u] = len(set(artist))

    count_df = pd.DataFrame(artist_counts.items(), columns=['user', 'artist_count'])
    count_df = count_df.set_index('user')
    count_df = count_df['artist_count'].copy()

    active = count_df.quantile(active_threshold)

    high_activity_users = count_df[count_df >= active].index.tolist()
    low_activity_users = count_df[count_df < active]

    filtered_low_activity_users = low_activity_users[low_activity_users >= min_artist]
    low_activity_users = filtered_low_activity_users.index.tolist()


    avg_high = count_df[count_df >= active].mean()
    avg_low = count_df[count_df < active].mean()
    print(f"Number of artists for high activity users: {avg_high:.2f}")
    print(f"Number of artists for low activity users: {avg_low:.2f}")

    return high_activity_users, low_activity_users

def create_user_activity_split(features_df, target_df = None):
    """
    Create a split based on the activity of users
    high_activity_users are the top 20% (messured by the number of aritsts they listen to)
    low_activity_users: rest
    """
    val_frac = 0.2
    num_nodes = len(features_df)
    print("Totoal number of users", num_nodes)

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

    #overlap = len(np.intersect1d(train_nodes, val_nodes))
    #overlap_2 = len(np.intersect1d(test_nodes, val_nodes))
    #print("Overlap between train and val:", overlap)
    #print("Overlap between val and test:", overlap_2)

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)