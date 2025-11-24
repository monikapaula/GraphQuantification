import sys
import torch
import argparse
import os
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import CC,ACC,PCC,PACC,EMQ

from models.gcn import GCN
from models.mlp import MLP
from quantification.wrapper import WrapperClassifier
from train_classifier import train
from quantification.run_quantification import run_quantification
from data_loader import get_graph_data
from create_splits.split_manager import load_split


CONFIG = {
    'model': 'GCN',  # 'GCN' or 'MLP'
    'dataset': 'presidential_election',  # 'presidential' or 'twitch'
    'split': 'split_0',
    'quantifier': 'all', # 'CC', 'ACC', 'PCC', 'PACC', 'EMQ'
    'device': 'cpu',
    'train': False,
    'epochs': 300,
    'lr': 0.001
}

def get_args():
    parser = argparse.ArgumentParser(description="Graph Quantification")
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN','MLP'])
    parser.add_argument('--dataset', type=str, default='presidential_election',choices=['presidential_election','twitch_gamers'])
    parser.add_argument('--split', type=str, default='split_0')
    parser.add_argument('--quantifier', type=str, default='PACC', choices=['CC','ACC','PCC','PACC','EMQ'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    return parser.parse_args()

def get_model(args, data, train_mask, val_mask, test_mask):
    if hasattr(data, 'y') and data.y is not None:
        output_dim = int(data.y.max()) + 1
    else:
        output_dim = 2  # Default to binary classification if y is not available
    config = {
        'name': args.model,
        'input_dim': data.x.size(1),
        'hidden_dim': 64,
        'output_dim': output_dim,
        'dropout': 0.5,
        'lr': 1e-3,
        'epochs': 300,
        'save_model': True
    }

    run_name = f"{args.dataset}_{args.split}"
    save_filename = f"{args.model}_{run_name}.pth"
    save_path = os.path.join("saved_models",save_filename)

    if args.train:
        class_weights = None
        model = train(
            config =config,
            x = data.x,
            edge_index = data.edge_index,
            y = data.y,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
            class_weights = class_weights,
            dataset_name = run_name
        )

        return model
    else:
        if args.model == 'GCN':
            model = GCN(config['input_dim'], config['hidden_dim'], config['output_dim'], config['dropout'])
        else:
            model = MLP(config['input_dim'], config['hidden_dim'], config['output_dim'])

        model.load_state_dict(torch.load(save_path, map_location=args.device))
        model.to(args.device)
        model.eval()
        return model

def main():
    args = get_args()
    data = get_graph_data(args.dataset)
    try:
        train_mask, val_mask, test_mask = data.load_split(args.dataset, args.split, data.num_nodes)
    except FileNotFoundError:
        print(f"Split '{args.split}' not found for dataset '{args.dataset}'. Please create the split first.")
        sys.exit(1)

    data = data.to(args.device)
    train_mask = train_mask.to(args.device)
    val_mask = val_mask.to(args.device)
    test_mask = test_mask.to(args.device)

    model = get_model(args, data, train_mask, val_mask, test_mask)
    wrapper = WrapperClassifier(model, data, device=args.device)

    run_quantification(
        wrapper= wrapper,
        data= data,
        val_mask = val_mask,
        test_mask= test_mask,
        quantifier_model= args.quantifier
    )

if __name__ == '__main__':
    main()