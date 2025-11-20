import torch
import torch.optim as optim
import torch.nn.functional as F

from splits.twitch_gamers_split import get_masks
from splits.presidential_el_split import get_mask
from models.gcn import GCN
from models.mlp import MLP

MODEL_CONFIG = {
    'name': 'GCN',
    'input_dim': None,
    'hidden_dim': 16,
    'output_dim': 2,
    'dropout': 0.5,
    'lr': 1e-3,
    'epochs': 100
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

def train (config: dict, x, edge_index, y, train_mask, val_mask, test_mask):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['input_dim'] = x.size(1)
    model = load_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    print(f"---Starting training for {config['epochs']} epochs---")

    for epoch in range(1, config['epochs']+1):
        #Training step
        model.train()
        optimizer.zero_grad()

        #output the log-probabilities
        log_probabilities = model(x, edge_index)
        loss = F.nll_loss(log_probabilities[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        #Evaluation step
        train_acc = evaluate(model, x, edge_index, train_mask, y)
        val_acc = evaluate(model, x, edge_index, val_mask, y)

        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


    test_acc = evaluate(model, x, edge_index, test_mask,y)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    data, train_mask, val_mask, test_mask = get_mask()
    train(
        config=MODEL_CONFIG,
        x = data.x,
        edge_index = data.edge_index,
        y = data.y,
        train_mask = train_mask,
        val_mask = val_mask,
        test_mask = test_mask
    )




