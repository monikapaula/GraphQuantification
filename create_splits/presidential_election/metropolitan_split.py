import numpy as np
from utils.mask_creation import _create_mask

METRO_DROP_COLS = ['Total Population', 'Density per square km']

def create_metro_split(features_df):
    """
    Create a split based on population/ density to simulate metropolitan vs non-metropolitan areas
    based on this definition: https://en.wikipedia.org/wiki/Metropolitan_statistical_area
    TRAIN = MSAs
    VAL= µSAs
    TEST = Rural areas
    leave the remaining counties out, so not all nodes are used
    split_name: split_3
    """

    population = features_df['Total Population'].values
    num_nodes = len(features_df)

    train_nodes = np.where(population >= 50000)[0]  # MSAs
    val_indices = np.where((population < 50000) & (population >= 10000))[0]  # µSAs
    test_nodes = np.where(population < 10000)[0]  # Rural areas

    print(f"Number of total population: {len(population)}")
    print(f"Number of MSA nodes: {len(train_nodes)}")


    rng = np.random.default_rng(seed=42)
    rng.shuffle(train_nodes)
    rng.shuffle(val_indices)
    rng.shuffle(test_nodes)

    val_target = int(num_nodes * 0.1)
    val_nodes = val_indices[:val_target]

    unused_val_nodes = val_indices[val_target:]
    num_left_out = len(unused_val_nodes)
    percent_left_out = (num_left_out / num_nodes) * 100

    # 2. Verify if any nodes were lost in the initial population split (e.g. NaNs)
    # The sum of all groups should equal num_nodes
    total_captured = len(train_nodes) + len(val_indices) + len(test_nodes)
    unaccounted_nodes = num_nodes - total_captured

    print("-" * 30)
    print(f"Total Nodes: {num_nodes}")
    print(f"Validation Target (10%): {val_target}")
    print(f"Available µSAs: {len(val_indices)}")
    print(f"Used µSAs: {len(val_nodes)}")
    print(f"Counties Left Out (Unused µSAs): {num_left_out} ({percent_left_out:.2f}%)")

    if unaccounted_nodes > 0:
        print(f"WARNING: {unaccounted_nodes} nodes were not categorized (possibly NaN population data).")
    print("-" * 30)

    return _create_mask(num_nodes, train_nodes, val_nodes, test_nodes)






