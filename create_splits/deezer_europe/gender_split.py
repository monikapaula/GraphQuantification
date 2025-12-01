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

    if 0 not in counts.columns:
        counts[0] = 0
    if 1 not in counts.columns:
        counts[1] = 0
    counts = counts.rename(columns={0: 'gender1_cnt', 1: 'gender2_cnt'})
    counts['total']= counts['gender1_cnt'] + counts['gender2_cnt']
    counts['gender1_share']= counts['gender1_cnt']/counts['total'].replace(0,np.nan)
    counts['gender2_share']= counts['gender2_cnt']/counts['total'].replace(0,np.nan)

    return counts.reset_index()

def categorize_by_maj(stats):
    min_listeners = 10
    majority = 0.8

    s = stats[stats['total'] >= min_listeners].copy()

    male_only = s[(s['gender2_cnt'] == 0) & (s['gender1_cnt'] > 0)]['artist_id'].tolist()
    female_only = s[(s['gender1_cnt'] == 0) & (s['gender1_cnt'] > 0)]['artist_id'].tolist()

    mostly_male = s[(s['gender2_share'] >= majority) & (s['gender1_cnt'] > 0)]['artist_id'].tolist()
    mostly_female = s[(s['gender1_share'] >= majority) & (s['gender2_cnt'] > 0)]['artist_id'].tolist()
    return mostly_male, mostly_female

def create_gender_split(features_df, target_df ):
    """
    creates a shift between male and female users based on the
    artistes they listen to

    """
    user_majority = 0.6
    val_frac = 0.1
    rng = np.random.default_rng(42)
    num_nodes = len(features_df)

    user_artist_df = gender_artist_df(features_df, target_df)
    stats = artist_gender_majority(user_artist_df)
    mostly_male, mostly_female = categorize_by_maj(stats)

    male_set = set(mostly_male)
    female_set = set(mostly_female)

    male_users = []
    female_users = []

    for uid_str, arts in features_df.items():
        u = int(uid_str)
        if u >= num_nodes:
            continue
        if not arts:
            continue
        aset = set(arts)
        m_cnt = len(aset & male_set)
        f_cnt = len(aset & female_set)
        if m_cnt + f_cnt == 0:
            continue
        m_share = m_cnt / (m_cnt + f_cnt)
        f_share = f_cnt / (m_cnt + f_cnt)
        if m_share >= user_majority:
            male_users.append(u)
        if f_share >= user_majority:
            female_users.append(u)

    male_users = np.array(male_users, dtype=int)
    female_users = np.array(female_users, dtype=int)
    rng.shuffle(male_users)
    rng.shuffle(female_users)

    # split_0: training_data dominated by 1, test_data dominated by 0 (male)
    # split_1: training_data_dominated by 0, test_data dominated by 1 (female)
    n_val = int(len(female_users) * val_frac)
    val_nodes = female_users[:n_val]
    train_nodes = female_users[n_val:]
    test_nodes = male_users

    overlap = np.intersect1d(train_nodes, test_nodes)
    if len(overlap) > 0:
        mask = ~np.isin(test_nodes, overlap)
        test_nodes = test_nodes[mask]

    print("Num male-dominated artists:", len(male_set))
    print("Num female-dominated artists:", len(female_set))
    print("Num typical male users:", len(male_users))
    print("Num typical female users:", len(female_users))

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)



