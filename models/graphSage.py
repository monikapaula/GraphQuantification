import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    """
    The GraphSAGE operator from the "Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216> paper.
    """
    def __init__(self, in_dim, hidden_dim, output_dim, dropout):
        super(SAGE, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p = dropout)

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x= self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.lin_out(x)
        return F.log_softmax(x, dim=-1)
