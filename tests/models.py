########################################################################################
# File's content was partially taken from
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py
########################################################################################

import dgl
import dgl.nn as dglnn
from dgl.nn import HeteroEmbedding
from dgl import DGLGraph
import torch.nn.functional as F
from torch import nn

class GNNModel(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int):
        super().__init__()

        self.convs = nn.ModuleList([
            dgl.nn.GraphConv(in_dim, h_dim, activation=F.relu, bias=False),
            dgl.nn.GraphConv(h_dim, h_dim, activation=F.relu, bias=False),
            dgl.nn.GraphConv(h_dim, out_dim, activation=None, bias=False),
        ])

    def forward(self, graph, features):
        if isinstance(graph, list):
            # Message Flow Graph
            for conv, block in zip(self.convs, graph):
                features = conv(block, features)
        else:
            # Whole graph
            for conv in self.convs:
                features = conv(graph, features)
        return features


class HeteroGNNModel(nn.Module):
    def __init__(self, g: DGLGraph, in_dim: int, h_dim: int, out_dim: int):
        super().__init__()
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()

        self.layers = nn.ModuleList([
            RelGraphConvLayer(in_dim, h_dim, g.ntypes, self.rel_names, activation=F.relu),
            RelGraphConvLayer(h_dim, h_dim, g.ntypes, self.rel_names, activation=F.relu),
            RelGraphConvLayer(h_dim, out_dim, g.ntypes, self.rel_names, activation=None)
        ])

    def forward(self, graph, h):
        if isinstance(graph, list):
            # Message Flow Graph
            for layer, block in zip(self.layers, graph):
                h = layer(block, h)
        else:
            # Whole graph
            for layer in self.layers:
                h = layer(graph, h)
        return h
    
class RelGraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, ntypes, rel_names, activation=None):
        super().__init__()
        self.rel_names = rel_names
        self.activation = activation

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(in_feat, out_feat, norm="right", weight=False, bias=False)
                for rel in rel_names
            }
        )
        self.weight = nn.ModuleDict({
            rel_name: nn.Linear(in_feat, out_feat, bias=False) for rel_name in rel_names
            }
        )
        self.loop_weights = nn.ModuleDict({
            ntype: nn.Linear(in_feat, out_feat, bias=True) for ntype in ntypes
            }
        )

    def forward(self, g, inputs):
        with g.local_scope():
            wdict = {
                rel_name: {"weight": self.weight[rel_name].weight.T}
                for rel_name in self.rel_names
            }
            inputs_dst = {
                k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()
            }
            hs = self.conv(g, inputs, mod_kwargs=wdict)

            def _apply(ntype, h):
                h = h + self.loop_weights[ntype](inputs_dst[ntype])
                if self.activation:
                    h = self.activation(h)
                return h

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}
    
    
def extract_embed(node_embed, input_nodes, skip_type=None):
    emb = node_embed(
        {ntype: input_nodes[ntype] for ntype in input_nodes if ntype != skip_type}
    )
    return emb


def rel_graph_embed(graph, embed_size, num_nodes_dict=None, skip_type=None):
    node_num = {}
    for ntype in graph.ntypes:
        if ntype == skip_type:
            continue
        if num_nodes_dict != None:
            node_num[ntype] = num_nodes_dict[ntype]
        else:
            node_num[ntype] = graph.num_nodes(ntype)
    embeds = HeteroEmbedding(node_num, embed_size)
    return embeds
