import numpy as np
import torch
import torch_geometric.transforms as T
from pathlib import Path
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset

from utils.mask_creation import _create_mask
from create_splits.split_manager import save_split

DATASET_NAME = 'ogbn_arxiv'
DATA_ROOT = Path(__file__).parent.parent.parent.resolve()

def load_arxiv_dataset():
    root = DATA_ROOT / 'data'
    transform = T.Compose([
        T.ToUndirected(),
        T.AddSelfLoops()
    ])
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=str(root), transform=transform)
    data = dataset[0]

    return data

def split_0(years):
    train_idx = np.where(years <= 2017)[0]
    val_idx = np.where(years == 2018)[0]
    test_idx = np.where(years >= 2019)[0]

    return train_idx, val_idx, test_idx

def split_1(years):
    train_idx = np.where(years >= 2018)[0]
    val_idx = np.where((years >= 2016) & (years <= 2017))[0]
    test_idx = np.where(years <= 2015)[0]

    return train_idx, val_idx, test_idx

def split_2(years):
    train_idx = np.where((years >= 2014) & (years <= 2018))[0]
    val_idx = np.where((years >= 2012) & (years <= 2013))[0]
    test_idx = np.where((years <= 2011) | (years >= 2019))[0]

    return train_idx, val_idx, test_idx

def create_splits():
    data = load_arxiv_dataset()
    num_nodes = data.num_nodes
    years = data.node_year.flatten().numpy()

    SPLIT_REGISTRY = {
        'split_0': split_0,
        'split_1': split_1,
        'split_2': split_2,
    }

    for split_name,func in SPLIT_REGISTRY.items():
        train_idx, val_idx, test_idx = func(years)
        train_mask, val_mask, test_mask = _create_mask(num_nodes, train_idx, val_idx, test_idx)

        save_split(DATASET_NAME, split_name, train_mask, val_mask, test_mask)

        data_obj = Data(x=data.x,edge_index=data.edge_index, y=data.y.squeeze().long())

        data_obj.train_mask = torch.from_numpy(train_mask).bool()
        data_obj.val_mask = torch.from_numpy(val_mask).bool()
        data_obj.test_mask = torch.from_numpy(test_mask).bool()

        save_path = DATA_ROOT / "split_data" / DATASET_NAME / split_name
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(data_obj, save_path / f"{split_name}_data.pt")

    return data_obj, train_mask, val_mask, test_mask

def get_dataset(split_name):
    file_path = DATA_ROOT / "split_data" / DATASET_NAME / split_name / f"{split_name}_data.pt"
    data = torch.load(file_path, weights_only=False)

    return data, data.train_mask, data.val_mask, data.test_mask

def class_prevalences(data):
    y = data.y.squeeze()
    classes = int(y.max().item()) + 1
    train_mask = data.train_mask
    test_mask = data.test_mask

    y_train = y[train_mask]
    y_test = y[test_mask]

    train_counts = torch.bincount(y_train, minlength=classes).float()
    test_counts = torch.bincount(y_test, minlength=classes).float()

    train_prev = train_counts / train_counts.sum().clamp(min=1)
    test_prev = test_counts / test_counts.sum().clamp(min=1)

    top5 = torch.topk(train_prev, k =5).indices

    for c in top5.tolist():
        print(f"{c:5d} | {train_prev[c]:10.4f} | {test_prev[c]:10.4f}")

    return top5, train_prev, test_prev


if __name__ == '__main__':

    data = load_arxiv_dataset()
    num_nodes = data.num_nodes
    years = data.node_year.flatten().numpy()

    SPLIT_REGISTRY = {
        'split_0': split_0,
        'split_1': split_1,
        'split_2': split_2,
    }

    for name, func in SPLIT_REGISTRY.items():
        train_idx, val_idx, test_idx = func(years)

        n_train = len(train_idx)
        n_val = len(val_idx)
        n_test = len(test_idx)
        n_total = n_train + n_val + n_test

        print(f"\n[ {name.upper()} ]")
        print(f"  Train: {n_train:>6} ({n_train / num_nodes:>7.2%})")
        print(f"  Val:   {n_val:>6} ({n_val / num_nodes:>7.2%})")
        print(f"  Test:  {n_test:>6} ({n_test / num_nodes:>7.2%})")
        print(f"  Coverage:     {n_total / num_nodes:>7.2%}")
    create_splits()

    data, train_mask, val_mask, test_mask = get_dataset("split_2")
    top5_classes, train_prev, test_prev = class_prevalences(data)
