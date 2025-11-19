import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import edge_index

from models.gcn import GCN
from models.mlp import MLP

MODEL_CONFIG = {
    'name': 'GCN'
    'input_dim':


}
def load_model(config:dict):


def train (config: dict):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(config['epochs']+1):
        model.train()
        optimizer.zero_grad()
        log_probabilites = model(x, edge_index)
        loss = F.nll_loss(log_probabilites[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():




