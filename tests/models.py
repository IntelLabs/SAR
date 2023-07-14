import dgl
import torch.nn.functional as F
from torch import nn

class GNNModel(nn.Module):
    def __init__(self,  in_dim: int, out_dim: int):
        super().__init__()

        self.convs = nn.ModuleList([
            dgl.nn.GraphConv(in_dim, out_dim, weight=True, bias=False),
        ])

    def forward(self,  graph, features):
        for idx, conv in enumerate(self.convs):
            features = conv(graph, features)
        return features