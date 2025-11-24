import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from create_splits.twitch_gamers_split import get_masks
from create_splits.presidential_el_split import get_mask, compute_class_weights
from models.gcn import GCN
from models.mlp import MLP
from loss.focal_loss import FocalLoss

MODEL_CONFIG = {
    'name': 'GCN',
    'input_dim': None,
    'hidden_dim': 64,
    'output_dim': 2,
    'dropout': 0.5,
    'lr': 1e-3,
    'epochs': 300,
    'save_model': True
}

def load_model(config:dict):
    name = config.get('name').upper()
    in_dim = config['input_dim']
    hidden_dim = config.get('hidden_dim', 32)
    output_dim = config.get('output_dim',2)
    dropout = config.get('dropout', 0.5)

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

def extensive_evaluate(model, x, edge_index, mask, y):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)

        y_true = y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

        cm = confusion_matrix(y_true, y_pred)
        print("\n--- Confusion Matrix ---")
        print(f"[[TN (Dem richtig): {cm[0][0]}, FP (Dem falsch als Rep): {cm[0][1]}]")
        print(f" [FN (Rep falsch als Dem): {cm[1][0]}, TP (Rep richtig): {cm[1][1]}]]")

        print("\n--- Classification Report ---")
        print(classification_report(y_true, y_pred, target_names=['Democrat', 'Republican']))

    marco_f1= f1_score(y_true, y_pred, average='macro')
    return marco_f1


def train (config: dict, x, edge_index, y, train_mask, val_mask, test_mask, class_weights=None, dataset_name='presidential_election'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['input_dim'] = x.size(1)
    model = load_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    #gamma parameter for focusing on hard examples
    criterion = FocalLoss(alpha=class_weights, gamma=4.0)

    print(f"---Starting training for {config['epochs']} epochs---")

    for epoch in range(1, config['epochs']+1):
        #Training step
        model.train()
        optimizer.zero_grad()

        #output the log-probabilities
        log_probabilities = model(x, edge_index)
        #loss = F.nll_loss(log_probabilities[train_mask], y[train_mask], weight=class_weights)
        loss = criterion(log_probabilities[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        #Evaluation step
        train_acc = evaluate(model, x, edge_index, train_mask, y)
        val_acc = evaluate(model, x, edge_index, val_mask, y)

        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


    test_acc = extensive_evaluate(model, x, edge_index, test_mask,y)
    print(f"Macro F1-Score: {test_acc:.4f}")

    if config.get('save_model', False):
        model_name = config.get('name','model')

        filename = f"{model_name}_{dataset_name}.pth"
        save_dir = "saved_models"

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        torch.save(model.state_dict(), save_path)
    return model

if __name__ == '__main__':
    DATASET = 'presidential_election'
    SPLIT_NAME = 'split_0'
    data, train_mask, val_mask, test_mask = get_mask(split_name=SPLIT_NAME)
    y_train = data.y[train_mask]
    weights = compute_class_weights(y_train)
    run_name = f"{DATASET}_{SPLIT_NAME}"
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





