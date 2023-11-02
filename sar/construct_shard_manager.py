# Copyright (c) 2022 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import List, Tuple,  Dict
import torch
from torch import Tensor
import dgl  # type: ignore
import numpy as np  # type: ignore
from .comm import exchange_tensors, rank
from .common_tuples import ShardEdgesAndFeatures, PartitionData
from .core import GraphShardManager, GraphShard
from .data_loading import _mask_features_dict, _get_type_ordered_edges, load_dgl_partition_data_from_graph


def map_to_contiguous_range(active_indices: Tensor, sampled_indices: Tensor) -> Tensor:
    '''
    Maps active indices to the range (0,len(active_indices)). Uses the same mapping to map
    sampled_indices to this range and returns the result. Assumes active_indices
    has unique elements and that every element in sampled_indices is also an element of
    active_indices
    '''

    assert torch.all(torch.isin(sampled_indices, active_indices)),\
        'sampled indices not in active indices'

    if active_indices.size(0) == 0:
        return active_indices.new(0)
    active_indices_sorted, active_indices_argsort = torch.sort(active_indices)

    u_si, i_si = torch.unique(
        sampled_indices, sorted=True, return_inverse=True)
    temp = torch.arange(len(active_indices))[
        torch.isin(active_indices_sorted, u_si)][i_si]
    return active_indices_argsort[temp]


def make_induced_graph_shard_manager(full_graph_shards: List[ShardEdgesAndFeatures],
                                     seed_nodes: Tensor,
                                     node_ranges: List[Tuple[int, int]],
                                     edge_type_names: List[str],
                                     partition_book : dgl.distributed.GraphPartitionBook,
                                     keep_seed_nodes: bool = True) -> GraphShardManager:
    '''
    Creates new graph shards that only contain edges to the seed nodes. Adjusts the target
    nodes indices in the resulting shards so that they lie in the
    contiguous range (0,len(unique(seed_nodes))). seed_nodes must be unique and sorted.

    Compacts the source indices in each shard[i] so
    that they lie in a contiguous range starting at node_ranges[i][0]. The compaction
    is consistent across all workers

    Returns the new graph shards as well as the new seed nodes
    '''

    assert torch.unique(seed_nodes).size(0) == seed_nodes.size(0)

    active_edges_src = []
    compacted_active_edges_dst = []
    active_edge_features: List[Dict[str, Tensor]] = []
    for shard in full_graph_shards:
        active_edges_loc = torch.isin(shard.edges[1], seed_nodes)
        active_edges_src.append(shard.edges[0][active_edges_loc])
        compacted_active_edges_dst.append(map_to_contiguous_range(
            seed_nodes, shard.edges[1][active_edges_loc]) + node_ranges[rank()][0])

        if not shard.edge_features:
            active_edge_features.append({})
        else:
            new_shard_edge_features = {k: shard.edge_features[k][active_edges_loc]
                                       for k in shard.edge_features}
            active_edge_features.append(new_shard_edge_features)

    src_compact_data = \
        compact_src_ranges(active_edges_src, seed_nodes,
                           node_ranges, keep_seed_nodes)

    graph_shard_list: List[GraphShard] = []
    for edges_src, edges_dst, edges_features, node_range, remote_n_nodes in \
        zip(src_compact_data['compacted_active_edges_src'], compacted_active_edges_dst,
            active_edge_features, node_ranges, src_compact_data['remote_n_nodes']):

        shard_edges_features = ShardEdgesAndFeatures(
            (edges_src, edges_dst), edges_features)
        src_range = (node_range[0], node_range[0] + remote_n_nodes)
        tgt_range = (node_ranges[rank()][0],
                     node_ranges[rank()][0] + len(seed_nodes))
        graph_shard_list.append(GraphShard(shard_edges_features,
                                           src_range, tgt_range, edge_type_names))

    return GraphShardManager(graph_shard_list, src_compact_data['local_src_seed_nodes'], seed_nodes, partition_book)


def compact_src_ranges(active_edges_src, seed_nodes, node_ranges, keep_seed_nodes):
    '''
    active_edges_src[i] are the source nodes for the current graph partitions located at worker i.
    Consistently compacts all source node indices in all workers so that they lie in a contiguous
    range starting at node_ranges[i][0] where i is the worker idx containing the src nodes
    '''

    all_unique_src_nodes = [torch.unique(x) for x in active_edges_src]

    local_src_seed_nodes_list = exchange_tensors(all_unique_src_nodes)
    if keep_seed_nodes:
        local_src_seed_nodes_list += [seed_nodes]

    local_src_seed_nodes = torch.unique(
        torch.cat(local_src_seed_nodes_list), sorted=True)

    assert torch.all(torch.isin(
        active_edges_src[rank()], local_src_seed_nodes))
    # Make sure the seed nodes come first
    if keep_seed_nodes:
        mask = torch.isin(local_src_seed_nodes, seed_nodes)
        local_src_seed_nodes = torch.cat(
            (seed_nodes, local_src_seed_nodes[~mask]))

    remote_all_active_src_nodes = exchange_tensors(
        [local_src_seed_nodes] * len(active_edges_src))

    compacted_active_edges_src = [map_to_contiguous_range(x, y)+node_ranges[i][0] for i, (x, y) in
                                  enumerate(zip(remote_all_active_src_nodes, active_edges_src))]

    remote_n_nodes = [len(x) for x in remote_all_active_src_nodes]
    return {'compacted_active_edges_src': compacted_active_edges_src,
            'local_src_seed_nodes': local_src_seed_nodes - node_ranges[rank()][0],
            'remote_n_nodes': remote_n_nodes}


def construct_mfgs(partition_data: PartitionData,
                   seed_nodes: Tensor,
                   n_layers: int,
                   keep_seed_nodes: bool = True) -> List[GraphShardManager]:
    """
    Constructs a list of GraphShardManager objects (one for each GNN layer) to compute only the node
    features needed for producing the output features for the  ``seed_nodes`` at the top layer.
    This is analoguous to the Message Flow Graphs (MFG) created by DGL's sampling DataLoaders.
    MFGs are particularly useful in node classification tasks where they can avoid a large
    amount of redundant computations compared to using the full graph.


    :param partition_data: The local partition data
    :type partition_data: PartitionData
    :param seed_nodes: The global indices of the graph nodes whose features need to be\
    computed at the top layer. Typically, these are the labeled nodes in a node classification\
    task.
    :type seed_nodes: Tensor
    :param n_layers: The number of layers in the GNN
    :type n_layers: int
    :param keep_seed_nodes: Keep the seed nodes as part of the source nodes. Default: True 
    :type keep_seed_nodes: bool


    :returns: A list of GraphShardManager objects, one for each layer

    """

    seed_nodes = seed_nodes.cpu()
    graph_shard_manager_list: List[GraphShardManager] = []
    for _ in range(n_layers):
        gsm = make_induced_graph_shard_manager(partition_data.all_shard_edges,
                                               seed_nodes,
                                               partition_data.node_ranges,
                                               partition_data.edge_type_names,
                                               partition_data.partition_book,
                                               keep_seed_nodes)
        graph_shard_manager_list.append(gsm)
        seed_nodes = gsm.input_nodes + partition_data.node_ranges[rank()][0]

    return graph_shard_manager_list[::-1]


def construct_full_graph(partition_data: PartitionData) -> GraphShardManager:
    """
    Constructs a GraphShardManager object from the partition data. The GraphShardManager
    object can serve as a drop-in replacemet to DGL's native graph in most GNN layers

    :param partition_data: The local partition data
    :type partition_data: PartitionData
    :returns: The constructed GraphShardManager object

    """
    num_splits = len(partition_data.all_shard_edges)
    graph_shard_list = [GraphShard(partition_data.all_shard_edges[part_idx],
                                   partition_data.node_ranges[part_idx],
                                   partition_data.node_ranges[rank()],
                                   partition_data.edge_type_names)
                        for part_idx in range(num_splits)]
    seed_nodes = torch.arange(partition_data.node_ranges[rank()][1] -
                              partition_data.node_ranges[rank()][0])
    return GraphShardManager(graph_shard_list,
                             seed_nodes, seed_nodes, partition_data.partition_book)
    
def convert_dist_graph(dist_graph: dgl.distributed.DistGraph) -> GraphShardManager:
    partition_data = load_dgl_partition_data_from_graph(dist_graph, dist_graph.device)
    return construct_full_graph(partition_data)