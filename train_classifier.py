import copy
import torch
import torch.optim as optim
import argparse

from create_splits.presidential_election.presidential_el_split import compute_class_weights, get_dataset as get_election_dataset
from create_splits.deezer_europe.deezer_europe_split import get_dataset as get_deezer_dataset
from create_splits.twitch_gamers.twitch_gamers_split import get_dataset as get_twitch_gamers_dataset
from utils.metrics import classifier_mae, extensive_evaluate, class_balance
from models.gcn import GCN
from models.mlp import MLP
from utils.focal_loss import FocalLoss
from utils.save_model import save_model
from utils.early_stopping import EarlyStopper

MODEL_CONFIG = {
    'name': 'GCN',
    'input_dim': None,
    'hidden_dim': 64,
    'output_dim': 2,
    'dropout': 0.3,
    'lr': 1e-3,
    'save_model': True
}

def load_model(config:dict):
    name = config.get('name').upper()
    in_dim = config['input_dim']
    hidden_dim = config.get('hidden_dim', 32)
    output_dim = config.get('output_dim',2)
    dropout = config.get('dropout', 0.3)

    if name == 'GCN':
        return GCN(in_dim, hidden_dim, output_dim,dropout)
    elif name == 'MLP':
        return MLP(in_dim, hidden_dim, output_dim,dropout)
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


def train (config: dict, x, edge_index, y, train_mask, val_mask, test_mask, class_weights=None, dataset_name=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['input_dim'] = x.size(1)
    model = load_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    #gamma parameter for focusing on hard examples
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    early_stopper = EarlyStopper(patience=200, min_delta=0.001 )
    best_model_state = None

    for epoch in range(args.epochs):
        #Training step
        model.train()
        optimizer.zero_grad()

        #output the log-probabilities
        log_probabilities = model(x, edge_index)
        loss = criterion(log_probabilities[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out_val = model(x, edge_index)
            val_loss = criterion(out_val[val_mask], y[val_mask]).item()
            val_acc = evaluate(model, x, edge_index, val_mask, y)
            train_acc = evaluate(model, x, edge_index, train_mask, y)

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        #Early Stopping
        if val_loss < early_stopper.min_validation_loss:
            best_model_state = copy.deepcopy(model.state_dict())
        if early_stopper.early_stop(val_loss):
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # evaluation on test set using F1 and MAE
    test_acc = extensive_evaluate(model, x, edge_index, test_mask,y)
    print(f"Macro F1-Score: {test_acc:.4f}")

    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        y_true = y[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()

        mae, true_prev, pred_prev = classifier_mae(y_pred, y_true)
        print(f"Classifier MAE on test set: {mae:.4f}")

    if config.get('save_model', False):
        save_model(model, config, dataset_name=run_name)


    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classifier model')
    parser.add_argument('--dataset', type=str, choices=['deezer_europe', 'presidential_election', 'twitch_gamers'],
                        help='Name of datasets')
    parser.add_argument('--split', type=str)
    parser.add_argument('--model', type=str, choices=['GCN', 'MLP'], default='GCN')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()

    DATASET = args.dataset
    SPLIT_NAME = args.split
    MODEL_CONFIG['name'] = args.model
    MODEL_CONFIG['epochs'] = args.epochs
    print(f"Loading dataset '{DATASET}' with split '{SPLIT_NAME}'")
    if DATASET == 'presidential_election':
        data, train_mask, val_mask, test_mask = get_election_dataset(split_name=SPLIT_NAME)
    elif DATASET == 'deezer_europe':
        data, train_mask, val_mask, test_mask = get_deezer_dataset(split_name=SPLIT_NAME)
    elif DATASET == 'twitch_gamers':
        data,train_mask,val_mask,test_mask = get_twitch_gamers_dataset(SPLIT_NAME)
    else:
        raise ValueError(f"Unknown dataset '{DATASET}'")

    y_train = data.y[train_mask]
    weights = compute_class_weights(y_train)
    run_name = f"{DATASET}_{SPLIT_NAME}"
    y = data.y

    class_balance(y, train_mask, "TRAIN")
    class_balance(y, val_mask, "VAL")
    class_balance(y, test_mask, "TEST")

    train(
        config=MODEL_CONFIG,
        x = data.x,
        edge_index = data.edge_index,
        y = data.y,
        train_mask = train_mask,
        val_mask = val_mask,
        test_mask = test_mask,
        class_weights = weights,
        dataset_name = run_name
    )