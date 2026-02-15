import copy
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import pandas as pd
import datetime

from create_splits.presidential_election.presidential_el_split import get_dataset as get_election_dataset
from create_splits.deezer_europe.deezer_europe_split import get_dataset as get_deezer_dataset
from create_splits.twitch_gamers.twitch_gamers_split import get_dataset as get_twitch_gamers_dataset
from create_splits.ogbn_arxiv.ogbn_arxiv_split import get_dataset as get_ogbn_arxiv_dataset

from utils.metrics import classifier_mae, extensive_evaluate, class_balance, macro_f1, compute_class_weights, print_confusion_matrix
from models.gcn import GCN
from models.mlp import MLP
from models.graphSage import SAGE
from models.gcnh import GCNH
#from utils.focal_loss import FocalLoss
from utils.save_model import save_model
from utils.early_stopping import EarlyStopper

MODEL_CONFIG = {
    'name': 'GCN',
    'input_dim': None,
    'hidden_dim': 256,
    'output_dim': None,
    'dropout': 0.2,
    'lr': 0.01,
    'save_model': True,
    'nlayers': 3
}

def load_model(config:dict):
    name = config.get('name').upper()
    in_dim = config['input_dim']
    hidden_dim = config['hidden_dim']
    output_dim = config['output_dim']
    dropout = config['dropout']
    nlayers = config.get('nlayers', 2)

    if name == 'GCN':
        return GCN(in_dim, hidden_dim, output_dim,dropout)
    elif name == 'MLP':
        return MLP(in_dim, hidden_dim, output_dim,dropout)
    elif name == 'SAGE':
        return SAGE(in_dim, hidden_dim, output_dim,dropout)
    elif name == 'GCNH':
        return GCNH(nfeat=in_dim,
            nhid=hidden_dim,
            nclass=output_dim,
            dropout=dropout,
            nlayers=nlayers,
            maxpool=False)
    else:
        raise ValueError(f"Model {name} not recognized.")

def evaluate(model, x, edge_index, mask, y):
    model.eval()
    with torch.no_grad():
        log_probabilities = model(x, edge_index)
        pred = log_probabilities[mask].max(1)[1]
        corr = pred.eq(y[mask]).sum().item()
        acc = corr / mask.sum().item()
    return acc


def train (config: dict, x, edge_index, y, train_mask, val_mask, test_mask, class_weights=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['input_dim'] = x.size(1)
    model = load_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    y = y.long()
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = nn.NLLLoss(weight=class_weights)
    early_stopper = EarlyStopper(patience=150, min_delta=0.001 )
    best_model_state = None
    best_val_metric = -float('inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        log_probabilities = model(x, edge_index)
        loss = criterion(log_probabilities[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(x, edge_index)
            val_loss = criterion(out_val[val_mask], y[val_mask]).item()
            val_acc = evaluate(model, x, edge_index, val_mask, y)
            train_acc = evaluate(model, x, edge_index, train_mask, y)
            val_macro_f1 = macro_f1(model, x, edge_index, val_mask, y)

        #if epoch % 10 == 0 or epoch == 0:
            #print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_macro_f1 > best_val_metric:
            best_val_metric = val_macro_f1
            best_model_state = copy.deepcopy(model.state_dict())
        if early_stopper.early_stop(val_loss):
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # evaluation on test set using F1 and MAE
    #test_acc = extensive_evaluate(model, x, edge_index, test_mask,y)

    print(f"\n--- FINAL EVALUATION (Confusion Matrices) ---")
    print_confusion_matrix(model, x, edge_index, train_mask, y, "TRAIN")
    print_confusion_matrix(model, x, edge_index, val_mask, y, "VALIDATION")
    print_confusion_matrix(model, x, edge_index, test_mask, y, "TEST")

    with torch.no_grad():
        test_f1 = macro_f1(model, x, edge_index, test_mask, y)
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        y_true = y[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()

        mae, true_prev, pred_prev = classifier_mae(y_pred, y_true)
        print(f"Macro-F1 on test set: {test_f1:.4f}")
        print(f"Classifier MAE on test set: {mae:.4f}")

    if config.get('save_model', False):
        os.makedirs('saved_models', exist_ok=True)
        save_model(model, config, dataset_name=run_name, split_name=SPLIT_NAME)

    return test_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classifier model')
    parser.add_argument('--datasets', type=str, nargs= '+', choices=['deezer_europe', 'presidential_election', 'twitch_gamers', 'ogbn_arxiv'],
                        help='Name of datasets')
    parser.add_argument('--splits', type=str, nargs='+')
    parser.add_argument('--models', type=str, nargs='+', choices=['GCN', 'MLP','SAGE', 'GCNH'])
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    results_table = []
    timestamp= datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = 'quantification/results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"classification_results_{timestamp}.csv")

    for DATASET in args.datasets:
        for SPLIT_NAME in args.splits:
            try:
                if DATASET == 'presidential_election':
                    data, train_mask, val_mask, test_mask = get_election_dataset(split_name=SPLIT_NAME)
                elif DATASET == 'deezer_europe':
                    data, train_mask, val_mask, test_mask = get_deezer_dataset(split_name=SPLIT_NAME)
                elif DATASET == 'twitch_gamers':
                    data,train_mask,val_mask,test_mask = get_twitch_gamers_dataset(split_name=SPLIT_NAME)
                elif DATASET == 'ogbn_arxiv':
                    data,train_mask,val_mask,test_mask = get_ogbn_arxiv_dataset(split_name=SPLIT_NAME)
                else:
                    raise ValueError(f"Unknown dataset '{DATASET}'")
            except Exception as e:
                print(f"Fehler beim Laden von {DATASET}: {e}")
                continue

            for MODEL_NAME in args.models:
                print(f"\n{'=' * 50}")
                print(f"RUNNING: Dataset={DATASET}, Split={SPLIT_NAME}, Model={MODEL_NAME}")

                current_config = MODEL_CONFIG.copy()
                current_config['name'] = MODEL_NAME
                current_config['output_dim'] = int(data.y.max().item()) + 1
                current_config['epochs'] = args.epochs

                y_train = data.y[train_mask]
                weights = compute_class_weights(y_train)
                run_name = f"{DATASET}_{SPLIT_NAME}"

                macro_f1_score= train(
                    config=current_config,
                    x=data.x,
                    edge_index=data.edge_index,
                    y=data.y,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    class_weights=weights,
                )
                y = data.y

                results_table.append([DATASET, SPLIT_NAME, MODEL_NAME, f"{macro_f1_score:.4f}"])

                headers = ["Dataset", "Split", "Model", "Macro F1"]
                df = pd.DataFrame(results_table, columns=headers)
                file_exists = os.path.isfile(output_file)
                df.to_csv(output_file, mode='a', index=False, header=not file_exists)


