import numpy as np
from utils.mask_creation import _create_mask

METRO_DROP_COLS = ['Total Population', 'Density per square km']

def create_metro_split(features_df):
    """
    Create a split based on population/ density to simulate metropolitan vs non-metropolitan areas
    based on this definition: https://en.wikipedia.org/wiki/Metropolitan_statistical_area
    TRAIN = MSAs and µSAs
    VAL= µSAs
    TEST = Rural areas
    split_name: split_3
    """

    population = features_df['Total Population'].values
    num_nodes = len(features_df)
    rng = np.random.default_rng(seed=42)

    train_indices = np.where(population >= 50000)[0]
    rng.shuffle(train_indices)# MSAs
    val_indices = np.where(population < 50000) [0] # µSAs
    sorted_indices = val_indices[np.argsort(population[val_indices])][::-1]
    val_target = int(num_nodes * 0.1)
    val_nodes = sorted_indices[:val_target]
    rng.shuffle(val_nodes)
    test_nodes = val_indices[val_target:]
    rng.shuffle(test_nodes)

    return _create_mask(num_nodes, train_indices, val_nodes, test_nodes)






