import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class GCNHLayer(nn.Module):
    """
    from the paper "GCNH: A Simple Method For Representation Learning On Heterophilous Graphs, 2023", [https://github.com/SmartData-Polito/GCNH]
    """
    def __init__(self, nfeat, nhid, maxpool=False):
        super(GCNHLayer, self).__init__()
        self.nhid = nhid
        self.maxpool = maxpool

        self.MLPfeat = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.LeakyReLU()
        )

        self.MLPmsg = nn.Sequential(
            nn.Linear(nfeat, nhid),
            nn.LeakyReLU()
        )
        self.beta = nn.Parameter(torch.zeros(1, nhid))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x, edge_index):
        row, col = edge_index
        h = self.MLPfeat(x)
        z = self.MLPmsg(x)

        if not self.maxpool:
            agg_h = scatter(h[col], row, dim=0, dim_size=x.size(0), reduce='mean')
        else:
            agg_h = scatter(h[col], row, dim=0, dim_size=x.size(0), reduce='max')

        sig_beta = torch.sigmoid(self.beta)
        hp = sig_beta * z + (1 - sig_beta) * agg_h

        return hp, sig_beta


class GCNH(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers=3, maxpool=False):
        super(GCNH, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nhid = nhid

        layer_sizes = [nfeat] + [nhid] * (self.nlayers - 1)
        self.layers = nn.ModuleList([
            GCNHLayer(layer_sizes[i], nhid, maxpool)
            for i in range(self.nlayers)
        ])

        self.MLPcls = nn.Sequential(
            nn.Linear(self.nhid, nclass),
            nn.LogSoftmax(dim=1)
        )

        self.init_weights(self.MLPcls)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x, _ = layer(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.MLPcls(x)