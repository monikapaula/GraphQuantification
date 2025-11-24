import json
import torch
import os
import numpy as np

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DATA_DIR = PROJECT_ROOT/'split_data'

def save_split(dataset_name, split_name, train_mask, val_mask, test_mask):

    target_dir = SPLIT_DATA_DIR/dataset_name/split_name
    os.makedirs(target_dir, exist_ok=True)

    def to_indices(mask):
        if isinstance(mask,torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.dtype == bool or mask.dtype == np.bool_:
            return np.where(mask)[0].tolist()
        return list(mask)

    data_map = {
        'train.json': to_indices(train_mask),
        'val.json': to_indices(val_mask),
        'test.json': to_indices(test_mask)
    }

    for filename, indices in data_map.items():
        with open(target_dir/filename, 'w') as f:
            json.dump(indices, f)

def load_split(dataset_name,split_name,num_nodes ):
    """
    Loads JSON index files and converts them back to boolean mask
    """
    target_dir = SPLIT_DATA_DIR/dataset_name/split_name
    masks = {}
    for part in ['train','val','test']:
        filepath = target_dir/f"{part}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Split file not found: {filepath}")
        with open(filepath,'r') as f:
            indices = json.load(f)

        mask_array = np.zeros(num_nodes, dtype=bool)
        if len(indices)>0:
            mask_array[indices] = True
        masks[part] = mask_array
    return masks['train'], masks['val'], masks['test']