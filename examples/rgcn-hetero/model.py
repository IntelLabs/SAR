################################################################################################################
# File's content taken from https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py
################################################################################################################

import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroEmbedding

def extract_embed(node_embed, input_nodes):
    emb = node_embed(
        {ntype: input_nodes[ntype] for ntype in input_nodes if ntype != "paper"}
    )
    return emb

def rel_graph_embed(graph, embed_size, num_nodes_dict):
    node_num = {}
    for ntype in graph.ntypes:
        if ntype == "paper":
            continue
        node_num[ntype] = num_nodes_dict[ntype]
    embeds = HeteroEmbedding(node_num, embed_size)
    return embeds


class RelGraphConvLayer(nn.Module):
    def __init__(
        self, in_feat, out_feat, ntypes, rel_names, activation=None, dropout=0.0, self_loop=True
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.ntypes = ntypes
        self.rel_names = rel_names
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_feat, out_feat, norm="right", weight=False, bias=False
                )
                for rel in rel_names
            }
        )

        self.weight = nn.ModuleDict(
            {
                rel_name: nn.Linear(in_feat, out_feat, bias=False)
                for rel_name in self.rel_names
            }
        )

        # weight for self loop
        if self.self_loop:
            self.loop_weights = nn.ModuleDict(
                {
                    ntype: nn.Linear(in_feat, out_feat, bias=True)
                    for ntype in self.ntypes
                }
            )

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.weight.values():
            layer.reset_parameters()
        if self.self_loop:
            for layer in self.loop_weights.values():
                layer.reset_parameters()

    def forward(self, g, inputs):
        """
        Parameters
        ----------
        g : DGLGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        with g.local_scope():
            wdict = {
                rel_name: {"weight": self.weight[rel_name].weight.T}
                for rel_name in self.rel_names
            }

            if self.self_loop:
                inputs_dst = {
                    k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                }

            hs = self.conv(g, inputs, mod_kwargs=wdict)

            def _apply(ntype, h):
                if self.self_loop:
                    h = h + self.loop_weights[ntype](inputs_dst[ntype])
                if self.activation:
                    h = self.activation(h)
                return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class EntityClassify(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(EntityClassify, self).__init__()
        self.in_dim = in_dim
        self.h_dim = 64
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.dropout = 0.5

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(
            RelGraphConvLayer(
                self.in_dim,
                self.h_dim,
                g.ntypes,
                self.rel_names,
                activation=F.relu,
                dropout=self.dropout,
                self_loop=g.tgt_in_src
            )
        )

        # h2o
        self.layers.append(
            RelGraphConvLayer(
                self.h_dim,
                self.out_dim,
                g.ntypes,
                self.rel_names,
                activation=None,
                self_loop=g.tgt_in_src
            )
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph, h):
        if isinstance(graph, list):
            # Message Flow Graph
            for layer, block in zip(self.layers, graph):
                h = layer(block, h)
        else:
            for layer in self.layers:
                h = layer(graph, h)
        return h
