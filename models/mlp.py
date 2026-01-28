import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) for only using node features and ignores
    graph-structure
    """

    def __init__(self, in_dim:int, hidden_dim: int, out_dim:int, dropout:float):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)
        #self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim)
        #self.lin3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index=None):

        x = self.lin1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.lin2(x)
        #x = self.bn2(x)
        #x = F.relu(x)
        #x = self.dropout(x)
        #x = self.lin3(x)
        return F.log_softmax(x, dim=-1)
