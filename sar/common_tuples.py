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

'''
Tuples for grouping related data
'''
from typing import NamedTuple, Dict, Tuple, List, Optional, Any, TYPE_CHECKING
from enum import Enum
from torch import Tensor
import dgl  # type: ignore

if TYPE_CHECKING:
    from .core.graphshard import GraphShardManager
    from .core.sar_aggregation import BackwardManager


class TensorPlace(Enum):
    SRC = 0
    DST = 1
    EDGE = 2
    PARAM = 3


class ShardEdgesAndFeatures(NamedTuple):
    '''
    Stores the edge information for all edges connecting nodes in one partition to
    nodes in another partition. For an N-way partition, each worker will have N ShardEdgesAndFeatures object,
    where each  object contains data for incoming edges from each  partition (including the worker's own
    partition)


    .. py:attribute:: edges : Tuple[Tensor,Tensor]

        The source and destination global node ids for each edge in the shard


    .. py:attribute:: edge_features : Dict[str,Tensor]

        A dictionary of the edge features

    '''
    edges: Tuple[Tensor, Tensor]
    edge_features: Dict[str, Tensor]


class GraphShardManagerData(NamedTuple):
    all_shard_edges: List[ShardEdgesAndFeatures]
    src_node_ranges: List[Tuple[int, int]]
    tgt_node_range: Tuple[int, int]
    tgt_seed_nodes: Tensor
    local_src_seed_nodes: Tensor


class PartitionData(NamedTuple):
    '''
    Stores all the data for the local partition


    .. py:attribute:: all_shard_edges : List[ShardEdgesAndFeatures]

        A list of ShardEdgesAndFeatures objects. One for edges incoming from each partition


    .. py:attribute:: node_ranges : List[Tuple[int,int]]

        node_ranges[i] is a tuple of the start and end global node indices for nodes in partition i.

    .. py:attribute:: node_features : Dict[str,Tensor]

        Dictionary of node features for nodes in local partition

    .. py:attribute:: node_type_names : List[str]

        List of node type names. Use in conjunction with dgl.NTYPE node features to get\
    the node type of each node

    .. py:attribute:: edge_type_names : List[str]

        List of edge type names. Use in conjunction with dgl.ETYPE edge features to get\
    the edge type of each edge
    
    .. py:attribute:: partition_book : dgl.distributed.GraphPartitionBook
    
        The graph partition information


    '''

    all_shard_edges: List[ShardEdgesAndFeatures]
    node_ranges: List[Tuple[int, int]]
    node_features: Dict[str, Tensor]
    node_type_names: List[str]
    edge_type_names: List[str]
    partition_book: dgl.distributed.GraphPartitionBook


class AggregationData(NamedTuple):
    graph_shard_manager: "GraphShardManager"
    message_func: Any
    reduce_func:  Any
    etype: Any
    all_input_names: List[Tuple[TensorPlace, str]]
    n_params: int
    grad_enabled: bool
    remote_data: bool


class ShardInfo(NamedTuple):
    shard_idx: int
    src_node_range: Tuple[int, int]
    tgt_node_range: Tuple[int, int]
    edge_range: Tuple[int, int]


class SocketInfo(NamedTuple):
    name: str
    ip_addr: str
