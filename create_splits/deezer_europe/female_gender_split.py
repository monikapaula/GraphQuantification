import numpy as np
import pandas as pd
from utils.mask_creation import _create_mask


def gender_artist_df (features_df, target_df):
    rows = []
    for user_id, arts in features_df.items():
        u = int(user_id)
        for art in arts:
            rows.append((u, art))
    user_artist_df = pd.DataFrame(rows, columns=['id', 'artist_id'])
    user_artist_df= user_artist_df.merge(target_df, on='id', how='inner')
    return user_artist_df

def artist_gender_majority(user_artist_df):
    counts= (
        user_artist_df.groupby(['artist_id','target'])['id']
        .nunique()
        .unstack(fill_value=0)
    )
    counts = counts.rename(columns={0: 'female_cnt', 1: 'male_cnt'})
    counts['total']= counts['female_cnt'] + counts['male_cnt']
    counts['female_share']= counts['female_cnt']/counts['total'].replace(0,np.nan)
    counts['male_share']= counts['male_cnt']/counts['total'].replace(0,np.nan)

    return counts.reset_index()

def categorize_by_maj(stats):
    min_listeners = 5
    majority = 0.55

    s = stats[stats['total'] >= min_listeners].copy()

    mostly_male = s[s['male_share'] >= majority]['artist_id'].tolist()
    mostly_female = s[s['female_share'] >= majority]['artist_id'].tolist()
    return mostly_male, mostly_female

def female_gender_split(features_df, target_df, include_featureless_nodes:False):
    """
    creates a shift between male and female users based on the
    artistes they listen to

    """
    val_frac = 0.2
    rng = np.random.default_rng(42)
    num_nodes = len(features_df)

    user_artist_df = gender_artist_df(features_df, target_df)
    stats = artist_gender_majority(user_artist_df)
    mostly_male, mostly_female = categorize_by_maj(stats)

    male_art_set = set(mostly_male)
    female_art_set = set(mostly_female)

    train_idx = []
    test_idx = []
    dropped_count = 0

    for uid_str, arts in features_df.items():
        u = int(uid_str)
        if not arts:
            if include_featureless_nodes:
                train_idx.append(u)
            else:
                dropped_count += 1
                continue
            continue

        aset = set(arts)
        m_cnt = len(aset & male_art_set)
        f_cnt = len(aset & female_art_set)
        total = m_cnt + f_cnt

        if total == 0:
            if include_featureless_nodes:
                train_idx.append(u)
            else:
                dropped_count += 1
            continue
        f_share = f_cnt / total

        if f_share >= 0.40:
            test_idx.append(u)
        else:
            train_idx.append(u)

    train_idx = np.array(train_idx, dtype=int)
    rng.shuffle(train_idx)

    n_val = int(len(train_idx) * val_frac)
    val_nodes = train_idx[:n_val]
    train_nodes = train_idx[n_val:]
    test_nodes = np.array(test_idx, dtype=int)


    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)