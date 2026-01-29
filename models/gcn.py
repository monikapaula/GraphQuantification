import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network (GCN) after the paper by Kipf et al.
    (https://github.com/senadkurtisi/pytorch-GCN/tree/main/data)
    """

    def __init__(self, in_dim: int, hidden_dim:int, out_dim: int, dropout, use_bias=True):
        super(GCN,self).__init__()
        self.in_dim = in_dim
        self.out_channels = out_dim
        self.use_bias = use_bias

        self.conv1 = GCNConv(in_dim, hidden_dim, use_bias, bias=use_bias)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, bias=use_bias)
        self.conv3 = GCNConv(hidden_dim, out_dim, bias=use_bias)
        #self.linear1 = nn.Linear(hidden_dim,out_dim, bias=use_bias) #ogbn
        #self.linear1 = nn.Linear(hidden_dim,out_dim, bias=use_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index ):
        """
        x : node features
        edge_index : adjacency matrix of edges
        return: tensor of posterior probabilities
        """
        x= self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x =self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        #x = self.linear1(x)
        return F.log_softmax(x, dim=1)
