import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class GCNHLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNHLayer, self).__init__()
        self.W_ego = nn.Linear(in_channels, out_channels)
        self.W_neigh = nn.Linear(in_channels, out_channels)
        self.beta = nn.Parameter(torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_ego.weight)
        nn.init.xavier_uniform_(self.W_neigh.weight)
        self.beta.data.fill_(0.5)

    def forward(self, x, edge_index):
        row, col = edge_index
        neigh_msg = x[col]
        agg_neigh = scatter(neigh_msg, row, dim=0, dim_size=x.size(0), reduce='mean')

        h_ego = self.W_ego(x)
        h_neigh = self.W_neigh(agg_neigh)

        sig_beta = torch.sigmoid(self.beta)
        out = (1 - sig_beta) * h_neigh + sig_beta * h_ego

        return F.relu(out), sig_beta


class GCNH(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers=3):
        super(GCNH, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        self.layers.append(GCNHLayer(nfeat, nhid))
        for _ in range(nlayers - 1):
            self.layers.append(GCNHLayer(nhid, nhid))
        self.cls = nn.Linear(nhid, nclass)

        nn.init.xavier_uniform_(self.cls.weight)
        self.cls.bias.data.fill_(0.01)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x, beta = layer(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.cls(x)
        return F.log_softmax(x, dim=1)