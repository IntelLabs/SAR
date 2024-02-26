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
import itertools
from torch import Tensor
import dgl  # type: ignore
import numpy as np  # type: ignore
from .comm import exchange_tensors, rank
from .common_tuples import ShardEdgesAndFeatures, PartitionData
from .core import HeteroGraphShardManager, GraphShardManager, GraphShard
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
                                     local_input_nodes: Tensor,
                                     node_ranges: List[Tuple[int, int]],
                                     edge_type_names: List[str],
                                     partition_book : dgl.distributed.GraphPartitionBook,
                                     etype_id: int) -> GraphShardManager:
    '''
    Creates new graph shards that only contain edges to the seed nodes of the relation
    given by etype_id. Adjusts the target nodes indices in the resulting shards so that
    they lie in the contiguous range (0,len(unique(seed_nodes))).

    Compacts the source indices in each shard[i] so
    that they lie in a contiguous range starting at node_ranges[i][0]. The compaction
    is consistent across all workers

    Returns the new graph shards as well as the new seed nodes
    '''

    assert torch.unique(seed_nodes).size(0) == seed_nodes.size(0)
    assert torch.unique(local_input_nodes).size(0) == local_input_nodes.size(0)

    active_edges_src = []
    compacted_active_edges_dst = []
    active_edge_features: List[Dict[str, Tensor]] = []
    for shard in full_graph_shards:
        active_edges_loc = torch.logical_and(torch.isin(shard.edges[1], seed_nodes),
                                             shard.edge_features[dgl.ETYPE] == etype_id)
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
        compact_src_ranges(active_edges_src, local_input_nodes, node_ranges)

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
        
    return GraphShardManager(graph_shard_list, src_compact_data['local_src_seed_nodes'],
                             seed_nodes, partition_book._canonical_etypes[etype_id])


def compact_src_ranges(active_edges_src, local_src_seed_nodes, node_ranges):
    """
    active_edges_src[i] are the source nodes for the current graph partitions located at worker i.
    Consistently compacts all source node indices in all workers so that they lie in a contiguous
    range starting at node_ranges[i][0] where i is the worker idx containing the src nodes
    """
    assert torch.all(torch.isin(active_edges_src[rank()], local_src_seed_nodes))
    
    remote_all_active_src_nodes = exchange_tensors(
        [local_src_seed_nodes] * len(active_edges_src))

    compacted_active_edges_src = [map_to_contiguous_range(x, y) + node_ranges[i][0] for i, (x, y) in
                                  enumerate(zip(remote_all_active_src_nodes, active_edges_src))]

    remote_n_nodes = [len(x) for x in remote_all_active_src_nodes]
    return {'compacted_active_edges_src': compacted_active_edges_src,
            'local_src_seed_nodes': local_src_seed_nodes - node_ranges[rank()][0],
            'remote_n_nodes': remote_n_nodes}


def determine_src_dst_nodes_for_hgsm(partition_data, seed_nodes):
    """
    Determine indices of the input nodes and destination nodes for the created
    HeteroGraphShardManager in the form of Message Flow Graph. Determining these
    indices is required, because number of destination nodes of given type 
    should be the same for each relation (even if they are not connected by the edge
    with that relation). The same goes for source nodes.
    """
    type_seed_nodes_indices = {}
    type_input_nodes_indices = {stype: torch.tensor([], dtype=torch.int64) for (stype, _, _) in partition_data.edge_type_names}
    
    seed_nodes_types = partition_data.node_features[dgl.NTYPE][seed_nodes - partition_data.node_ranges[rank()][0]]
    for rel_idx, (stype, _, dtype) in enumerate(partition_data.edge_type_names):
        if dtype not in type_seed_nodes_indices:
            relation_seeds_mask = (seed_nodes_types == partition_data.partition_book.ntypes.index(dtype))
            relation_seeds = seed_nodes[relation_seeds_mask]
            type_seed_nodes_indices[dtype] = relation_seeds
        
        active_edges_src = []
        for shard in partition_data.all_shard_edges:
            active_edges_loc = torch.logical_and(torch.isin(shard.edges[1], seed_nodes),
                                                shard.edge_features[dgl.ETYPE] == rel_idx)
            active_edges_src.append(shard.edges[0][active_edges_loc])
            
        local_active_edges_src_list = exchange_tensors(active_edges_src)
        local_active_edges_src = torch.unique(torch.cat(local_active_edges_src_list))
        type_input_nodes_indices[stype] = torch.cat((type_input_nodes_indices[stype], local_active_edges_src))
        
    type_input_nodes_indices = {k: torch.unique(v) for (k, v) in type_input_nodes_indices.items()}
    
    return type_input_nodes_indices, type_seed_nodes_indices


def construct_mfgs(partition_data: PartitionData,
                   seed_nodes: Tensor,
                   n_layers: int,
                   keep_seed_nodes: bool = True) -> List[GraphShardManager]:
    """
    Constructs a list of HeteroGraphShardManager objects (one for each GNN layer) to compute only the node
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

    :returns: A list of HeteroGraphShardManager objects, one for each layer
    """
    seed_nodes = torch.sort(seed_nodes)[0].cpu()
    hetero_graph_shard_manager_list: List[HeteroGraphShardManager] = []
    for _ in range(n_layers):
        graph_shard_managers = {}
        type_input_nodes_indices, type_seed_nodes_indices = determine_src_dst_nodes_for_hgsm(partition_data, seed_nodes)
        
        if keep_seed_nodes:
            for stype in type_input_nodes_indices.keys():
                if stype not in type_seed_nodes_indices:
                    continue
                updated_inputs = torch.cat((type_input_nodes_indices[stype], type_seed_nodes_indices[stype]))
                updated_inputs = torch.unique(updated_inputs)
                mask = torch.isin(updated_inputs, type_seed_nodes_indices[stype])
                updated_inputs = torch.cat((type_seed_nodes_indices[stype], updated_inputs[~mask]))
                type_input_nodes_indices[stype] = updated_inputs
            
        for rel_idx, (stype, _, dtype) in enumerate(partition_data.edge_type_names):
            relation_input_nodes = type_input_nodes_indices[stype]
            relation_seeds = type_seed_nodes_indices[dtype]
            if relation_seeds.nelement() == 0:
                    continue
                
            graph_shard_managers[rel_idx] = make_induced_graph_shard_manager(partition_data.all_shard_edges,
                                                                             relation_seeds,
                                                                             relation_input_nodes,
                                                                             partition_data.node_ranges,
                                                                             partition_data.edge_type_names,
                                                                             partition_data.partition_book,
                                                                             etype_id=rel_idx)

        input_nodes = torch.tensor([], dtype=torch.int64)
        for val in type_input_nodes_indices.values():
            input_nodes = torch.cat((input_nodes, val))
        input_nodes -= partition_data.node_ranges[rank()][0]
        hgsm = HeteroGraphShardManager(graph_shard_managers, input_nodes, seed_nodes,
                                       partition_data.partition_book, partition_data.node_features[dgl.NTYPE],
                                       partition_data.node_ranges, keep_seed_nodes)
        hetero_graph_shard_manager_list.append(hgsm)
        seed_nodes = input_nodes + partition_data.node_ranges[rank()][0]

    return hetero_graph_shard_manager_list[::-1]


def construct_full_graph(partition_data: PartitionData) -> HeteroGraphShardManager:
    """
    Constructs a HeteroGraphShardManager object from the partition data. The HeteroGraphShardManager
    object can serve as a drop-in replacemet to DGL's native graph in most GNN layers

    :param partition_data: The local partition data
    :type partition_data: PartitionData
    :returns: The constructed HeteroGraphShardManager object
    """
    graph_shard_managers = {}
    for rel_idx, (stype, _, dtype) in enumerate(partition_data.edge_type_names):
        stype_id = partition_data.partition_book.ntypes.index(stype)
        dtype_id = partition_data.partition_book.ntypes.index(dtype)
        seeds = (partition_data.node_features[dgl.NTYPE] == dtype_id).nonzero().view(-1) + partition_data.node_ranges[rank()][0]
        input_nodes = (partition_data.node_features[dgl.NTYPE] == stype_id).nonzero().view(-1) + partition_data.node_ranges[rank()][0]
        graph_shard_managers[rel_idx] = make_induced_graph_shard_manager(partition_data.all_shard_edges,
                                                                         seeds,
                                                                         input_nodes,
                                                                         partition_data.node_ranges,
                                                                         partition_data.edge_type_names,
                                                                         partition_data.partition_book,
                                                                         etype_id=rel_idx)
    seed_nodes = torch.arange(partition_data.node_ranges[rank()][1] -
                              partition_data.node_ranges[rank()][0])
    return HeteroGraphShardManager(graph_shard_managers, seed_nodes, seed_nodes,
                                   partition_data.partition_book, partition_data.node_features[dgl.NTYPE],
                                   partition_data.node_ranges)

    
def convert_dist_graph(dist_graph: dgl.distributed.DistGraph) -> GraphShardManager:
    partition_data = load_dgl_partition_data_from_graph(dist_graph, dist_graph.device)
    return construct_full_graph(partition_data)
