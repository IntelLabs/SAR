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

from typing import List, Tuple, Dict, Optional
import torch
from torch import Tensor
import dgl  # type: ignore
from dgl.distributed.partition import load_partition  # type: ignore
from .common_tuples import PartitionData, ShardEdgesAndFeatures


def suffix_key_lookup(feature_dict: Dict[str, Tensor], key: str,
                      expand_to_all: bool = False,
                      type_list: Optional[List[str]] = None) -> Tensor:
    """
    Looks up the provided key in the provided dictionary. Uses suffix matching, where a dictionary
    key matches if ends with the provided key. This allows feature name lookup in the edge/node
    feature dictionaries in DGL's partition data whose keys have the form
    ``{node or edge type name}/{feature_name}``. In heterogeneous graphs, some features might only
    be present for certain node/edge types. Set the ``expand_to_all`` flag to expand the
    feature tensor to all nodes/edges in the graph. The expanded tensor will be zero for all 
    nodes/edges where the requested feature is not present


    :param feature_dict: Node or edge feature dictionary
    :type feature_dict: Dict[str, Tensor]
    :param key: Key to look up.
    :type key: str
    :param expand_to_all: Expand feature tensor to all nodes/edges.
    :type expand_to_all: bool
    :param type_list: List of edge or node type names. Required if ``expand_to_all`` is ``True``
    :type type_list: Optional[List[str]]
    :returns:   The matched (possibly expanded) feature tensor 

    """
    matched_keys = [k for k in feature_dict if k.endswith(key)]
    if len(matched_keys) == 0:
        return torch.LongTensor([])
    assert len(matched_keys) == 1
    matched_features = feature_dict[matched_keys[0]]
    if expand_to_all:
        assert type_list is not None
        if len(type_list) > 1 and dgl.NTYPE in feature_dict:
            type_id = feature_dict[dgl.NTYPE]
            key_node_type = matched_keys[0].split('/')[0]
            node_type_idx = type_list.index(key_node_type)

            expanded_features = matched_features.new(
                type_id.size(0), *matched_features.size()[1:]).zero_()
            expanded_features[type_id == node_type_idx] = matched_features
            return expanded_features

    return matched_features


def _mask_features_dict(edge_features: Dict[str, Tensor],
                        mask: Tensor, device: torch.device) -> Dict[str, Tensor]:
    return {k: edge_features[k][mask].to(device) for k in edge_features}


def _get_type_ordered_edges(edge_mask: Tensor, edge_types: Tensor,
                            n_edge_types: int) -> Tensor:
    reordered_edge_mask: List[Tensor] = []
    for edge_type_idx in range(n_edge_types):
        edge_mask_typed = torch.logical_and(
            edge_mask, edge_types == edge_type_idx)
        reordered_edge_mask.append(
            edge_mask_typed.nonzero(as_tuple=False).view(-1))

    return torch.cat(reordered_edge_mask)


def create_partition_data(graph: dgl.DGLGraph,
                          own_partition_idx: int,
                          node_features: Dict[str, torch.Tensor],
                          edge_features: Dict[str, Tensor],
                          partition_book: dgl.distributed.GraphPartitionBook,
                          node_type_list: List[str],
                          edge_type_list: List[str],
                          device: torch.device) -> PartitionData:
    """
    Creates SAR's PartitionData object basing on graph partition and features.

    :param graph: The graph partition structure for specific ``own_partition_idx``
    :type graph: dgl.DGLGraph
    :param own_partition_idx: The index of the partition to create. This is typically the\
    worker/machine rank
    :type own_partition_idx: int
    :param node_features: Dictionary containing node features for graph partition
    :type node_features: Dict[str, Tensor]
    :param edge_features: Dictionary containing edge features for graph partition
    :type edge_features: Dict[(str, str, str), Tensor]
    :param partition_book: The graph partition information
    :type partition_book: dgl.distributed.GraphPartitionBook
    :param node_type_list: List of node types
    :type node_type_list: List[str]
    :param edge_type_list: List of edge types
    :type edge_type_list: List[str]
    :param device: Device on which to place the loaded partition data
    :type device: torch.device
    :returns: The loaded partition data
    """
    is_heterogeneous = (len(edge_type_list) > 1)
    # Delete redundant edge features with keys {relation name}/reltype. graph.edata[dgl.ETYPE ] already contains
    # the edge type in a heterogeneous graph
    if is_heterogeneous:
        for edge_feat_key in list(edge_features.keys()):
            if 'reltype' in edge_feat_key:
                del edge_features[edge_feat_key]

    # Obtain the node ranges in each partition in the homogenized graph
    start_node_idx = 0
    node_ranges: List[Tuple[int, int]] = []
    for part_metadata in partition_book.metadata():
        node_ranges.append(
            (start_node_idx, start_node_idx + part_metadata['num_nodes']))
        start_node_idx += part_metadata['num_nodes']

    # Include the node types in the node feature dictionary
    if dgl.NTYPE in graph.ndata:
        node_features[dgl.NTYPE] = graph.ndata[dgl.NTYPE][graph.ndata['inner_node'].bool()]
    else:
        node_features[dgl.NTYPE] = torch.zeros(graph.num_nodes(), dtype=torch.int32)[graph.ndata['inner_node'].bool()]

    # Include the edge types in the edge feature dictionary
    inner_edge_mask = graph.edata['inner_edge'].bool()
    if dgl.ETYPE in graph.edata:
        edge_features[dgl.ETYPE] = graph.edata[dgl.ETYPE][inner_edge_mask]
    else:
        edge_features[dgl.ETYPE] = torch.zeros(graph.num_edges(), dtype=torch.int32)[inner_edge_mask]

    # Obtain the inner edges. These are the partition edges
    local_partition_edges = torch.stack(graph.all_edges())[:, inner_edge_mask]
    # Use global node ids in partition_edges
    partition_edges = graph.ndata[dgl.NID][local_partition_edges]

    # Check that all target nodes lie in the current partition
    assert partition_edges[1].min() >= node_ranges[own_partition_idx][0] \
        and partition_edges[1].max() < node_ranges[own_partition_idx][1]

    all_shard_edges: List[ShardEdgesAndFeatures] = []

    for part_idx in range(partition_book.num_partitions()):
        # obtain the mask for edges originating from partition part_idx
        edge_mask = torch.logical_and(partition_edges[0] >= node_ranges[part_idx][0],
                                      partition_edges[0] < node_ranges[part_idx][1])

        # Reorder the edges in each shard so that edges with the same type
        # follow each other
        if is_heterogeneous:
            edge_mask = _get_type_ordered_edges(
                edge_mask, edge_features[dgl.ETYPE], len(edge_type_list))

        all_shard_edges.append(ShardEdgesAndFeatures(
            (partition_edges[0, edge_mask], partition_edges[1, edge_mask]),
            _mask_features_dict(edge_features, edge_mask, device)
        ))

    return PartitionData(all_shard_edges,
                         node_ranges,
                         node_features,
                         node_type_list,
                         edge_type_list,
                         partition_book
                         )


def load_dgl_partition_data(partition_json_file: str,
                            own_partition_idx: int, device: torch.device) -> PartitionData:
    """
    Loads partition data created by DGL's ``partition_graph`` function

    :param partition_json_file: Path to the .json file containing partitioning data
    :type partition_json_file: str
    :param own_partition_idx: The index of the partition to load. This is typically the\
    worker/machine rank
    :type own_partition_idx: int
    :param device: Device on which to place the loaded partition data
    :type device: torch.device
    :returns: The loaded partition data

    """
    (graph, node_features,
     edge_features, partition_book, _,
     node_type_list, edge_type_list) = load_partition(partition_json_file, own_partition_idx)

    return create_partition_data(graph, own_partition_idx,
                                 node_features, edge_features,
                                 partition_book, node_type_list,
                                 edge_type_list, device)

def load_dgl_partition_data_from_graph(graph: dgl.distributed.DistGraph,
                                       device: torch.device) -> PartitionData:
    """
    Loads partition data from DistGraph object

    :param graph: The distributed graph
    :type graph: dgl.distributed.DistGraph
    :param device: Device on which to place the loaded partition data
    :type device: torch.device
    :returns: The loaded partition data

    """
    own_partition_idx = graph.rank()
    local_g = graph.local_partition

    assert dgl.NID in local_g.ndata
    assert dgl.EID in local_g.edata

    # get originalmapping for node and edge ids
    orig_n_ids = local_g.ndata[dgl.NID][local_g.ndata['inner_node'].bool().nonzero().view(-1)]
    orig_e_ids = local_g.edata[dgl.EID][local_g.edata['inner_edge'].bool().nonzero().view(-1)]

    # fetch local features from DistTensor
    node_features = {key : torch.Tensor(graph.ndata[key][orig_n_ids]) for key in list(graph.ndata.keys())}
    edge_features = {key : torch.Tensor(graph.edata[key][orig_e_ids]) for key in list(graph.edata.keys())}

    partition_book = graph.get_partition_book()
    node_type_list = local_g.ntypes
    edge_type_list = [local_g.to_canonical_etype(etype) for etype in graph.etypes]

    return create_partition_data(local_g, own_partition_idx,
                                 node_features, edge_features,
                                 partition_book, node_type_list,
                                 edge_type_list, device)