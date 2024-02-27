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
GraphShard manages the part of the graph containing only edges
between the nodes present in two workers (or between the nodes present
in a single worker)
'''

from typing import Any, Tuple, Dict, List, Optional, Union, Callable
import inspect
import os
import itertools
import logging
import types
from collections.abc import MutableMapping
from contextlib import contextmanager
import torch
import dgl  # type:ignore
from dgl import DGLGraph
from dgl.function.base import TargetCode  # type:ignore
import dgl.function as fn  # type: ignore
from dgl.distributed.constants import DEFAULT_ETYPE, DEFAULT_NTYPE
from torch import Tensor
import torch.distributed as dist


from ..common_tuples import ShardEdgesAndFeatures, AggregationData, TensorPlace, ShardInfo
from ..comm import exchange_tensors, rank, all_reduce
from .sar_aggregation import sar_op
from .full_partition_block import DistributedBlock

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)


class GraphShard:
    """
    Encapsulates information for all edges incoming from one partition to the local partition.

    :param shard_edges_features:  Edges with global node ids and edge features for \
    all edges incoming from one remote partition
    :type shard_edges_features: ShardEdgesAndFeatures
    :param src_range: The global start and end indices for nodes in the source partition from which the edges originate
    :type src_range: Tuple[int, int]
    :param tgt_range: The global start and end indices for the nodes in the local partition
    :type tgt_range: Tuple[int, int]
    :param edge_type_names: List of edge type names
    :type edge_type_names:
    """

    def __init__(self,
                 shard_edges_features: ShardEdgesAndFeatures,
                 src_range: Tuple[int, int],
                 tgt_range: Tuple[int, int],
                 edge_type_names
                 ) -> None:
        self.tgt_range = tgt_range
        self.src_range = src_range

        self.unique_src_nodes, unique_src_nodes_inverse = \
            torch.unique(shard_edges_features.edges[0], return_inverse=True)

        self.unique_tgt_nodes, unique_tgt_nodes_inverse = \
            torch.unique(shard_edges_features.edges[1], return_inverse=True)

        self.graph = dgl.create_block((unique_src_nodes_inverse, unique_tgt_nodes_inverse),
                                      num_src_nodes=self.unique_src_nodes.size(
                                          0),
                                      num_dst_nodes=self.unique_tgt_nodes.size(
                                          0)
                                      )
        self._graph_reverse: Optional[DGLGraph] = None
        self._shard_info: Optional[ShardInfo] = None
        self.graph.edata.update(shard_edges_features.edge_features)

        self.edge_type_names = edge_type_names

    def _set_shard_info(self, shard_info: ShardInfo):
        self._shard_info = shard_info

    @property
    def shard_info(self) -> Optional[ShardInfo]:
        return self._shard_info

    @property
    def graph_reverse(self) -> DGLGraph:
        if self._graph_reverse is None:
            edges_src, edges_tgt = self.graph.all_edges()
            self._graph_reverse = dgl.create_block((edges_tgt, edges_src),
                                                   num_src_nodes=self.unique_tgt_nodes.size(
                                                       0),
                                                   num_dst_nodes=self.unique_src_nodes.size(0))
            self._graph_reverse.edata.update(self.graph.edata)
        return self._graph_reverse

    def to(self, device: torch.device):
        self.graph = self.graph.to(device)
        if self._graph_reverse is not None:
            self._graph_reverse = self._graph_reverse.to(device)


class ChainedDataView(MutableMapping):
    """
    A dictionary that chains to children dictionary on missed __getitem__ calls
    """

    def __init__(self, base_dict: Optional["ChainedDataView"] = None):
        self._base_dict = base_dict
        self._store: Dict[str, Dict[str, Tensor]] = {}
        self._valid_types = []
        self._valid_counts = []
    
    def __getitem__(self, key: str) ->  Dict[str, Tensor]:
        if key in self._store:
            return self._store[key]
        if self._base_dict is not None:
            return self._base_dict[key]
        raise KeyError(f'key {key} not found')
                       
    def rewind(self) -> Optional['ChainedDataView']:
        self._store.clear()
        return self._base_dict

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        if self._base_dict is None:
            return iter(self._store)
        return itertools.chain(self._base_dict, self._store)

    def __len__(self):
        return len(self._store)
    
    @property
    def acceptable_entries(self):
        return zip(self._valid_types, self._valid_counts)

    @property
    def is_homogenous(self):
        return len(self._valid_types) == 1 and len(self._valid_counts) == 1


class ChainedNodeDataView(ChainedDataView):
    """
    A dictionary for graph's node types that chains to children dictionary on missed __getitem__ calls
    """

    def __init__(self,
                 valid_entries: List[Tuple[str, int]],
                 base_dict: Optional["ChainedDataView"] = None):
        super().__init__(base_dict)
        self._valid_types, self._valid_counts = zip(*valid_entries)

    def __getitem__(self, key: str) ->  Dict[str, Tensor]:
        return_tensor = super().__getitem__(key)
        if self.is_homogenous:
            return return_tensor[DEFAULT_NTYPE]
        return return_tensor

    def __setitem__(self, key: str, value: Union[Tensor, Dict[str, Tensor]]):
        if len(self._valid_types) == 1 and len(self._valid_counts) == 1 and not isinstance(value, Dict):
            value = {DEFAULT_NTYPE: value}
        assert isinstance(value, Dict), \
            "Graph has more than one node type, please specify node type and the data through a dict"
        for ntype, val in value.items():
            required_size = self._valid_counts[self._valid_types.index(ntype)]
            assert val.size(0) == required_size, \
                f'Tenosr size {val.size()} does not match graph data size {required_size} for node type {ntype}'
            if key not in self._store:
                self._store[key] = {}     
            self._store[key][ntype] = val
         
                
class ChainedEdgeDataView(ChainedDataView):
    """
    A dictionary for graph's edge types that chains to children dictionary on missed __getitem__ calls
    """

    def __init__(self, 
                 valid_entries: List[Tuple[Tuple[str, str, str], int]],
                 _etype_to_canonical: Callable[[Union[str, Tuple[str, str, str]]], Tuple[str, str, str]],
                 base_dict: Optional["ChainedDataView"] = None):
        super().__init__(base_dict)
        self._valid_types, self._valid_counts = zip(*valid_entries)
        self.etype_to_canonical = _etype_to_canonical

    def __getitem__(self, key: str) ->  Dict[str, Tensor]:
        return_tensor = super().__getitem__(key)
        if self.is_homogenous:
            return return_tensor[DEFAULT_ETYPE]
        return return_tensor
    
    def __setitem__(self, key: str, value: Union[Tensor, Dict[str, Tensor]]):
        if len(self._valid_types) == 1 and len(self._valid_counts) == 1 and not isinstance(value, Dict):
            value = {DEFAULT_ETYPE: value}
        assert isinstance(value, Dict), \
            "Graph has more than one edge type, please specify edge type and the data through a dict"
        for etype, val in value.items():
            if isinstance(etype, str):
                canonical_etype = self.etype_to_canonical(etype)
            else:
                canonical_etype = etype
            required_size = self._valid_counts[self._valid_types.index(canonical_etype)]
            assert val.size(0) == required_size, \
                f'Tenosr size {val.size()} does not match graph data size {required_size} for edge type {canonical_etype}'
            if key not in self._store:
                self._store[key] = {}
            self._store[key][canonical_etype] = val


class GraphShardManagerDataViewProxy():
    """
    Helper class for accessing specific type from ChainedDataView in GraphShardManager
    """
    
    def __init__(self, type: str, chained_data_view: ChainedDataView):
        self._type = type
        self._chained_data_view = chained_data_view
        
    def __getitem__(self, key: str):
        bag_of_values = self._chained_data_view[key]
        if self._chained_data_view.is_homogenous:
            return bag_of_values
        
        if self._type in bag_of_values:
            return self._chained_data_view[key][self._type]
        raise KeyError(key)
        
    def __setitem__(self, key: str, value: Tensor):
        self._chained_data_view[key] = {self._type: value}
        

class GraphShardManager:
    """
    Manages the local graph partition and exposes a subset of the interface
    of dgl.heterograph.DGLGraph. Most importantly, it implements a
    distributed version of the ``update_all`` and ``apply_edges`` functions
    which are extensively used by GNN layers to exchange messages. By default, 
    both  ``update_all`` and  ``apply_edges`` use sequential aggregation and 
    rematerialization (SAR) to minimize data duplication across the workers. 
    In some cases, this might introduce extra communication and computation overhead. 
    SAR can be disabled by setting :attr:`Config.disable_sr` to False to avoid this overhead. 
    Memory consumption may jump up siginifcantly, however. You should not construct
    GraphShardManager directly, but should use :func:`sar.construct_mfgs` and :func:`sar.construct_full_graph`
    instead.

    :param graph_shards: List of N graph shards where N is the number of partitions/workers.\
    graph_shards[i] contains information for edges originating from partition i
    :type graph_shards: List[GraphShard]
    :param local_src_seeds: The indices of the input nodes relative to the starting node index of the local partition\
    The input nodes are the nodes needed to produce the output node features assuming one-hop aggregation
    :type local_src_seeds: torch.Tensor
    :param local_tgt_seeds: The node indices of the output nodes relative to the starting node index of the local partition
    :type local_tgt_seeds: torch.Tensor
    :param canonical_relation: canonical edge type representing the relation that is handled by this GraphShardManager
    :type canonical_relation: Tuple[str, str, str]
    """

    def __init__(self,
                 graph_shards: List[GraphShard],
                 local_src_seeds: Tensor,
                 local_tgt_seeds: Tensor,
                 canonical_relation: Tuple[str, str, str]) -> None:
        super().__init__()
        self.graph_shards = graph_shards
        self.canonical_relation = canonical_relation
        self.parent_hgsm = None
        self.srcdata = None
        self.dstdata = None
        self.edata = None

        # source nodes and target nodes are all the same
        # srcdata, dstdata and ndata should be also the same
        self.src_is_tgt = local_src_seeds is local_tgt_seeds

        assert all(self.tgt_node_range ==
                   x.tgt_range for x in self.graph_shards[1:])

        current_edge_index = 0
        for idx, g_shard in enumerate(graph_shards):
            g_shard._set_shard_info(
                ShardInfo(
                    idx,
                    g_shard.src_range,
                    g_shard.tgt_range,
                    (current_edge_index, current_edge_index +
                     g_shard.graph.number_of_edges())
                )
            )
            current_edge_index += g_shard.graph.number_of_edges()

        self.input_nodes = local_src_seeds
        self.seeds = local_tgt_seeds

        assert self.input_nodes.size(0) == \
            self.local_src_node_range[1] - self.local_src_node_range[0]

        assert self.seeds.size(
            0) == self.tgt_node_range[1] - self.tgt_node_range[0]

        self.indices_required_from_me = self._update_boundary_nodes_indices()
        self.sizes_expected_from_others = [
            shard.unique_src_nodes.size(0) for shard in self.graph_shards]
        
        self._sampling_graph = None   
        self._in_degrees_cache = None
        self._out_degrees_cache = None
        self._num_edges_cache = None
        if self.src_is_tgt:
            self._num_all_nodes = self.input_nodes.shape[0]
        else:
            self._num_all_nodes = torch.unique(torch.cat((self.input_nodes, self.seeds - self.tgt_node_range[0]))).shape[0]
                               
    def __getattr__(self, attr):
        if hasattr(self.parent_hgsm, attr):
            attr_type = getattr(type(self.parent_hgsm), attr)
            if isinstance(attr_type, property):
                fget_attr = attr_type.fget
                if hasattr(fget_attr, "_gsm_prop"):
                    prop = getattr(self.parent_hgsm, attr)
                    return prop
            elif isinstance(attr_type, types.FunctionType):
                func = getattr(self.parent_hgsm, attr)
                if hasattr(func, "_gsm_func"):
                    return func
        raise AttributeError(f"'{__class__.__name__ }' object has no attribute '{attr}'")

    @property
    def srctypes(self):
        """
        Returns source node type represented by GraphShardManager in a list
        """
        return [self.canonical_relation[0]]
    
    @property
    def dsttypes(self):
        """
        Returns destination node type represented by GraphShardMnager in a list
        """
        return [self.canonical_relation[2]]

    @property
    def ntypes(self) -> List[str]:
        """
        Returns list of node types in the relation represented by GraphShardManager
        """
        if self.canonical_relation[0] == self.canonical_relation[2]:
            return [self.canonical_relation[0]]
        return [self.canonical_relation[0], self.canonical_relation[2]]

    @property
    def etypes(self) -> List[str]:
        """
        Returns edge type represented by the GraphShardManager in a list
        """
        return [self.canonical_relation[1]]

    @property
    def canonical_etypes(self) -> List[Tuple[str, str, str]]:
        """
        Returns canonical edge type represented by GraphShardManager in alist
        """
        return [self.canonical_relation]
    
    @property
    def tgt_node_range(self) -> Tuple[int, int]:
        return self.graph_shards[0].tgt_range
    
    @property
    def local_src_node_range(self) -> Tuple[int, int]:
        return self.graph_shards[rank()].src_range
   
    @property
    def ndata(self):
        assert self.src_is_tgt, "ndata shouldn't be used with MFGs"
        return self.srcdata

    @property
    def sampling_graph(self):
        """
        Returns a non-compacted graph for sampling. The node ids in the returned
        graph are the same as their global ids. Message passing on the sampling_graph
        will be very memory-inefficient as the node feature tensor will have to include
        a feature vector for each node in the whole graph

        """
        raise NotImplementedError("sampling_graph property is not adjusted to work with HeteroGraphShardManager")
        if self._sampling_graph is not None:
            return self._sampling_graph

        global_src_nodes = []
        global_tgt_nodes = []
        for shard in self.graph_shards:
            global_src_nodes.append(
                shard.unique_src_nodes[shard.graph.all_edges()[0]])
            global_tgt_nodes.append(
                shard.unique_tgt_nodes[shard.graph.all_edges()[1]])

        # We only need the csc format for sampling
        sampling_graph = dgl.graph((torch.cat(global_src_nodes),
                                    torch.cat(global_tgt_nodes)),
                                   num_nodes=self.graph_shards[-1].src_range[1]).shared_memory(
                                       'sample_graph'+repr(os.getpid()), formats=['csc'])
        del global_src_nodes, global_tgt_nodes

        edge_feat_dict: Dict[str, Tensor] = {}
        for edge_feat_key in self.graph_shards[0].graph.edata:
            edge_feat_dict[edge_feat_key] = \
                torch.cat([g_shard.graph.edata[edge_feat_key]
                           for g_shard in self.graph_shards]).share_memory_()

        sampling_graph.edata.update(edge_feat_dict)

        self._sampling_graph = sampling_graph
        return sampling_graph

    def nodes(self, ntype: str = None) -> Tensor:
        if ntype is None:
            if len(self.canonical_relation[0] != self.canonical_relation[2]):
                raise RuntimeError("Node type must be specified if the graph consists of more than one node type")
        if ntype == self.canonical_relation[0]:
            return self.input_nodes
        return self.seeds if self.src_is_tgt else self.seeds - self.tgt_node_range[0]
        
    def srcnodes(self, ntype: str = None) -> Tensor:
        if ntype is None:
            if len(self.srctypes) == 1:
                return self.input_nodes
            raise RuntimeError("Node type must be specified if the graph consists of more than one node source type")
        if ntype == self.canonical_relation[0]:
            return self.input_nodes
        raise KeyError(f"Cannot find node with specified node type: '{ntype}'")
        
    def dstnodes(self, ntype: str = None) -> Tensor:
        if ntype is None:
            if len(self.dsttypes) == 1:
                return self.seeds if self.src_is_tgt else self.seeds - self.node_ranges[rank()][0]
            raise RuntimeError("Node type must be specified if the graph consists of more than one node destination type")
        if ntype == self.canonical_relation[2]:
            return self.seeds if self.src_is_tgt else self.seeds - self.node_ranges[rank()][0]
        raise KeyError(f"Cannot find node with specified node type: '{ntype}'")

    def num_nodes(self, ntype: str = None) -> int:
        """
        Returns the number of nodes of a given node type in a GraphShard in the local partition.
        If node type is not specified, returns the number of all nodes in the local partition.
        Local nodes are the nodes in a given partition without HALO nodes.
        
        :param ntype: node type
        :type ntype: str
        """
        self._validate_ntype(ntype)
        if ntype == self.canonical_relation[0]:
            return self.num_src_nodes(ntype)
        if ntype == self.canonical_relation[2]:
            return self.num_dst_nodes(ntype)
        return self._num_all_nodes

    def number_of_nodes(self, ntype: str = None) -> int:
        """
        Alias of :func:`num_nodes`
        """
        return self.num_nodes(ntype)
    
    def num_src_nodes(self, ntype: str = None) -> int:
        """
        Returns the number of source nodes in the graph (local partition)
        
        :param ntype: node type
        :type ntype: str
        """
        self._validate_ntype(ntype, allow_dst=False)
        return self.input_nodes.shape[0]

    def number_of_src_nodes(self, ntype: str = None) -> int:
        """
        Alias of :func:`num_src_nodes`
        """
        return self.num_src_nodes(ntype)

    def num_dst_nodes(self, ntype: str = None) -> int:
        """
        Returns the number of destination nodes in the graph (local partition)
        
        :param ntype: node type
        :type ntype: str
        """
        self._validate_ntype(ntype, allow_src=False)
        return self.seeds.shape[0]

    def number_of_dst_nodes(self, ntype: str = None) -> int:
        """
        Alias of :func:`num_dst_nodes`
        """
        return self.num_dst_nodes(ntype)

    def num_edges(self, etype: Union[str, Tuple[str, str, str]] = None) -> int:
        """
        Returns the number of edges in the GraphShardManager in the local partition.
        Since GraphShardManager is responsible for only one relation, specified etype should be equal
        to that relation or to ('_N', '_E', '_N'), or should be set to "None".
        
        Local edges are the edges between nodes in the local partition and edges which are 
        incoming from other partition (destination nodes are in the local partition)
        
        :param etype: edge type specified by a string or a string triplet (canonical form)
        :type etype: Union[str, Tuple[str, str, str]]
        """
        self._validate_etype(etype)
        if self._num_edges_cache is not None:
            return self._num_edges_cache
        
        self._num_edges_cache = 0
        for shard in self.graph_shards:
            self._num_edges_cache += shard.graph.num_edges()
        
        return self._num_edges_cache
        
    def number_of_edges(self, etype: Union[str, Tuple[str, str, str]] = None) -> int:
        """
        Alias of :func:`num_edges`
        """
        return self.num_edges(etype)
    
    @contextmanager
    def local_scope(self):
        """
        local_scope has no effect when called on a relation slice of a graph.
        It should be called via HeteroGraphShardManager. Not raising error to be
        consistent with DGL's behaviour.
        """
        yield

    def get_full_partition_graph(self,
                                 delete_shard_data: bool = True) -> DistributedBlock:
        """
        Returns a graph representing all the edges incoming to this partition.
        The ``update_all`` and ``apply_edges`` functions in this graph will
        execute one-shot communication and aggregation in the forward and backward
        passes.

        :param delete_shard_data: Delete shard information. Remove the graph data in\
        the GraphShardManager object. You almost always want this as you will not be using\
        the GraphShardManager object after obtaining the full partition graph
        :type delete_shard_data: bool
        :returns: A graph-like object representing all the incoming edges to nodes in the local\
        partition

        """
        raise NotImplementedError("get_full_partition_graph function is not adjusted to work with HeteroGraphShardManager")
        start_src_index = 0
        new_src_indices_l = []
        new_tgt_indices_l = []
        for shard in self.graph_shards:
            new_src_indices_l.append(shard.graph.all_edges()[
                                     0] + start_src_index)
            new_tgt_indices_l.append(
                shard.unique_tgt_nodes[shard.graph.all_edges()[1]] - self.tgt_node_range[0])
            start_src_index += shard.graph.number_of_src_nodes()

        n_total_src_nodes = start_src_index
        new_src_indices = torch.cat(new_src_indices_l)
        new_tgt_indices = torch.cat(new_tgt_indices_l)

        edge_feat_dict: Dict[str, Tensor] = {}
        for edge_feat_key in self.graph_shards[0].graph.edata:
            edge_feat_dict[edge_feat_key] = torch.cat([g_shard.graph.edata[edge_feat_key]
                                                       for g_shard in self.graph_shards])
        if dgl.ETYPE in edge_feat_dict:
            etype_sorting_indices = torch.argsort(edge_feat_dict[dgl.ETYPE])
            for edge_feat_key in list(edge_feat_dict.keys()):
                edge_feat_dict[edge_feat_key] = edge_feat_dict[edge_feat_key][etype_sorting_indices]
            new_src_indices = new_src_indices[etype_sorting_indices]
            new_tgt_indices = new_tgt_indices[etype_sorting_indices]

        full_partition_block = dgl.create_block(
            (new_src_indices, new_tgt_indices),
            num_src_nodes=n_total_src_nodes,
            num_dst_nodes=self.tgt_node_range[1] -
            self.tgt_node_range[0],
            device=self.graph_shards[0].graph.device
        )
        full_partition_block.edata.update(edge_feat_dict)

        src_ranges = [shard.src_range for shard in self.graph_shards]
        unique_src_nodes = [
            shard.unique_src_nodes for shard in self.graph_shards]

        distributed_block = DistributedBlock(full_partition_block, self.indices_required_from_me,
                                             self.sizes_expected_from_others,
                                             src_ranges,
                                             unique_src_nodes,
                                             self.input_nodes,
                                             self.seeds,
                                             self.graph_shards[0].edge_type_names)

        if delete_shard_data:
            del self.graph_shards

        return distributed_block

    def to(self, device=torch.device):
        for shard in self.graph_shards:
            shard.to(device)
        return self
    
    def in_degrees(self, vertices=dgl.ALL, etype=None) -> Tensor:
        self._validate_etype(etype)
        if self._in_degrees_cache is None:
            in_degrees = torch.zeros(
                self.tgt_node_range[1] - self.tgt_node_range[0], dtype=self.graph_shards[0].graph.idtype).to(self.graph_shards[0].graph.device)
            for shard in self.graph_shards:
                in_degrees[shard.unique_tgt_nodes - self.tgt_node_range[0]] += shard.graph.in_degrees()
            self._in_degrees_cache = in_degrees

        if vertices == dgl.ALL:
            return self._in_degrees_cache
        return self._in_degrees_cache[vertices]

    def out_degrees(self, vertices=dgl.ALL, etype=None) -> Tensor:
        self._validate_etype(etype)
        if self._out_degrees_cache is None:
            for comm_round, shard in enumerate(self.graph_shards):
                out_degrees = torch.zeros(
                    shard.src_range[1] - shard.src_range[0], dtype=shard.graph.idtype).to(shard.graph.device)

                out_degrees[shard.unique_src_nodes - shard.src_range[0]] = shard.graph.out_degrees()
                all_reduce(out_degrees, op=dist.ReduceOp.SUM, move_to_comm_device=True)
                if comm_round == rank():
                    self._out_degrees_cache = out_degrees.to(shard.graph.device)

        if vertices == dgl.ALL:
            return self._out_degrees_cache
        return self._out_degrees_cache[vertices]
    
    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None,
                   ):

        assert isinstance(reduce_func, dgl.function.reducer.SimpleReduceFunction), \
            'only simple reduce functions: sum, min, max, and mean are supported'

        if reduce_func.name == 'mean':
            reduce_func = fn.sum(reduce_func.msg_field,  # pylint: disable=no-member
                                 reduce_func.out_field)
            mean_postprocess = True
        else:
            mean_postprocess = False

        all_input_tensors, all_input_names, remote_data, n_params = self._get_active_tensors(
            message_func)

        aggregation_data = AggregationData(self,
                                           message_func,
                                           reduce_func,
                                           etype,
                                           all_input_names,
                                           n_params,
                                           torch.is_grad_enabled(),
                                           remote_data)

        result_val = sar_op(aggregation_data, *all_input_tensors)
        if mean_postprocess:
            in_degrees = self.in_degrees()
            in_degrees = torch.where(in_degrees == 0, torch.ones_like(in_degrees), in_degrees)
            in_degrees = in_degrees.view([-1] + [1] * (result_val.ndim - 1))
            result_val = result_val / in_degrees

        if apply_node_func is not None:
            result_val = apply_node_func(result_val)

        self.dstdata[reduce_func.out_field] = result_val

    def apply_edges(self,
                    message_func,
                    edges=dgl.ALL,
                    etype=None,
                    inplace=False
                    ):

        assert edges == dgl.ALL, 'GraphShardManager only supports updating all the edges'

        all_input_tensors, all_input_names, remote_data, n_params = self._get_active_tensors(
            message_func)

        aggregation_data = AggregationData(self,
                                           message_func,
                                           None,
                                           etype,
                                           all_input_names,
                                           n_params,
                                           torch.is_grad_enabled(),
                                           remote_data)

        result_val = sar_op(aggregation_data, *all_input_tensors)

        self.edata[message_func.out_field] = result_val

    def _get_active_tensors(self, message_func):
        message_params = ()
        if callable(message_func):
            arg_spec = inspect.getfullargspec(message_func)
            if '__get_params__' in list(arg_spec.kwonlyargs):
                message_params = message_func(__get_params__=True)

        active_src_dict = {}
        active_dst_dict = {}
        active_edge_dict = {}

        def _update_relevant_dict(code: int, field: str) -> None:
            data_name = TargetCode.CODE2STR[code]
            if data_name == 'u':
                active_src_dict.update({field: self.srcdata[field]})
            elif data_name == 'v':
                active_dst_dict.update({field: self.dstdata[field]})
            elif data_name == 'e':
                active_edge_dict.update({field: self.edata[field]})

        if isinstance(message_func, dgl.function.message.CopyMessageFunction):
            _update_relevant_dict(message_func.target, message_func.in_field)
        elif isinstance(message_func, dgl.function.message.BinaryMessageFunction):
            _update_relevant_dict(message_func.lhs, message_func.lhs_field)
            _update_relevant_dict(message_func.rhs, message_func.rhs_field)
        else:  # Unrecognized message function. Make use of all edata, srcdata and dstdata tensors
            active_src_dict = self.srcdata
            active_dst_dict = self.dstdata
            active_edge_dict = self.edata

        all_input_tensors = list(active_src_dict.values()) + list(active_edge_dict.values()) +\
            list(active_dst_dict.values()) + list(message_params)

        tensor_places = ([TensorPlace.SRC] * len(active_src_dict)) + \
            ([TensorPlace.EDGE] * len(active_edge_dict)) + \
            ([TensorPlace.DST] * len(active_dst_dict))

        tensor_names = list(active_src_dict.keys()) + list(active_edge_dict.keys()) + \
            list(active_dst_dict.keys())

        remote_data = (len(active_src_dict) > 0)

        return all_input_tensors, list(zip(tensor_places, tensor_names)), remote_data, len(message_params)
    
    def _update_boundary_nodes_indices(self) -> List[Tensor]:
        all_my_sources_indices = [
            x.unique_src_nodes for x in self.graph_shards]

        indices_required_from_me = exchange_tensors(all_my_sources_indices)
        for ind in indices_required_from_me:
            ind.sub_(self.tgt_node_range[0])
        return indices_required_from_me

    def _validate_etype(self, etype: Union[str, Tuple[str, str, str]]) -> None:
        """
        GraphShardManager is responsible for managing only one relation. Because of that
        functions that allow to specify etype are allowd only to take this exact etype as the argument
        or  DEFAULT_ETYPE ('_N', '_E', '_N') type or simply "None"
        """
        if isinstance(etype, tuple):
            etype = etype[1]
            
        if (etype == self.etypes[0]) or \
            (etype == DEFAULT_ETYPE) or \
            (etype == None):
            return
        raise RuntimeError(f'Edge type "{etype}" does not exist.')
        
    def _validate_ntype(self, ntype: str, allow_src: bool = True, allow_dst: bool = True) -> None:
        """
        GraphShardManager is responsible for managing only one relation. Because of that
        functions that allow to specify ntype are allowd only to take srctype, dsttype, 
        DEFAULT_NTYPE ("_N") type or simply "None"
        """
        if (ntype == DEFAULT_NTYPE) or \
            (ntype == self.srctypes[0] and allow_src) or \
            (ntype == self.dsttypes[0] and allow_dst) or \
            (ntype == None):
            return
        raise RuntimeError(f'Node type "{ntype}" does not exist.')
        
    def _initialize_data(self,
                         hgsm_srcdata: ChainedNodeDataView,
                         hgsm_dstdata: ChainedNodeDataView,
                         hgsm_edata: ChainedEdgeDataView):
        if self.src_is_tgt:
            assert hgsm_srcdata == hgsm_dstdata
        self.edata = GraphShardManagerDataViewProxy(self.canonical_relation, hgsm_edata)
        self.srcdata = GraphShardManagerDataViewProxy(self.canonical_relation[0], hgsm_srcdata)
        self.dstdata = GraphShardManagerDataViewProxy(self.canonical_relation[2], hgsm_dstdata)
        

class HeteroGraphShardManager:
    """
    Manages R GraphShardManager's, where R is the number of relations in the graph
    
    :param graph_shard_managers: Dict of GraphShardManagers, where key is the relation index and value is GraphShardManager responsible for that relation
    :type graph_shard_managers: Dict[int, GraphShardManager]
    :param local_src_seeds: The indices of the input nodes relative to the starting node index of the local partition\
    The input nodes are the nodes needed to produce the output node features assuming one-hop aggregation
    :type local_src_seeds: torch.Tensor
    :param local_tgt_seeds: The node indices of the output nodes relative to the starting node index of the local partition
    :type local_tgt_seeds: torch.Tensor
    :param partition_book: The graph partition information
    :type partition_book: dgl.distributed.GraphPartitionBook
    :param node_types: tensor with node types in local partition
    :type node_types: torch.Tensor
    :param node_ranges: list of indices of the first and the last node for each partitioned part of the graph
    :type node_ranges: List[Tuple[int, int]]
    :param tgt_in_src: specifies if the seed nodes are a part of the source nodes 
    :type tgt_in_src: bool
    """
    
    def __init__(self,
                 graph_shard_managers: Dict[int, GraphShardManager],
                 local_src_seeds: Tensor,
                 local_tgt_seeds: Tensor,
                 partition_book: dgl.distributed.GraphPartitionBook,
                 node_types: Tensor,
                 node_ranges: List[Tuple[int, int]],
                 tgt_in_src: bool = True):
        self.graph_shard_managers = graph_shard_managers
        self._partition_book = partition_book
        self._node_types = node_types
        self._node_ranges = node_ranges
        self.input_nodes = local_src_seeds
        self.seeds = local_tgt_seeds
        self.src_is_tgt = local_src_seeds is local_tgt_seeds
        self.tgt_in_src = tgt_in_src
            
        for gsm in self.graph_shard_managers.values():
            gsm.parent_hgsm = self
                   
        self._etype2canonical = {}
        for idx, etype in enumerate(self.etypes):
            if etype in self._etype2canonical:
                self._etype2canonical[etype] = tuple()
            else:
                self._etype2canonical[etype] = self.canonical_etypes[idx]
                
        # retrieve ids of each node type in the HGSM
        self._src_node_types_indices_dict = {}
        self._dst_node_types_indices_dict = {}
        self._all_node_types_indices_dict = {}
        self._update_node_type_indices_dicts()
        
        # Construction of srcdata, dstdata, edata views
        self.srcdata = None
        self.dstdata = None
        self.edata = None
        self._construct_data_views()
            
        # Update necessary information in each child GraphShardMnager
        for gsm in self.graph_shard_managers.values():
            gsm._initialize_data(self.srcdata, self.dstdata, self.edata)
          
    def __getitem__(self, key: Union[str, Tuple[str, str, str]]) ->GraphShardManager:
        """
        Return GraphShardManager responsible for given relation specified by key parameter (relation slice of this graph).
        Key is the edge type which can be either a string representing a relation or a string triplet representing
        canonical form of the relation.
        key might be None only if there is one relation in the graph.
        
        :param key: edge type represented by a string or a string triplet (canonical form)
        :type key: Union[str, Tuple[str, str, str]]
        :returns: GraphShardManager
        """
        if key is None: 
            if len(self.etypes) != 1:
                 raise RuntimeError("Edge type name must be specified if there are more than one edge types.")
            return self.graph_shard_managers[0]
        
        if isinstance(key, str):
            canonical_etype = self._etype_to_canonical(key)
        else:
            assert isinstance(key, tuple) and len(key) == 3, f'Canonical form of the edge type "{key}" is incorrect'
            canonical_etype = key
            
        etype_id = self.canonical_etypes_global.index(canonical_etype)
        try:
            slice_gsm = self.graph_shard_managers[etype_id]
        except:
            raise ValueError(f"Graph does not contain relation specified by key: '{key}'")
        return slice_gsm
    
    def _gsm_function(func):
        """
        To avoid code duplication in GraphShardManager, decorate a function with 
        this decorator, and then this function will be available in all of the
        GraphShardManagers managed by the HeteroGraphShardManager
        """
        func._gsm_func = True
        return func
    
    def _gsm_property(func):
        """
        To avoid code duplication in GraphShardManager, decorate a property with this decorator
        instead of @property, and then this function will act like a property and will be available
        in all of the GraphShardManagers managed by the HeteroGraphShardManager
        """
        def wrapper(self):
            return func(self)
        wrapped_prop = property(wrapper)
        wrapped_prop.fget._gsm_prop = True
        return wrapped_prop
            
    @_gsm_property
    def partition_book(self) -> dgl.distributed.GraphPartitionBook:
        """
        Returns the partition book object, which stores all 
        graph partition information
        """
        return self._partition_book
        
    @property
    def srctypes(self) -> List[str]:
        """
        Returns list of source node types in the local graph
        """
        return list(self._src_node_types_indices_dict.keys())
        
    @property
    def dsttypes(self) -> List[str]:
        """
        Returns list of destination node types in the local graph
        """
        return list(self._dst_node_types_indices_dict.keys())
    
    @property
    def ntypes(self) -> List[str]:
        """
        Returns list of node types in the local graph
        """
        return list(self._all_node_types_indices_dict.keys())
    
    @_gsm_property
    def ntypes_global(self) -> List[str]:
        """
        Returns list of node types in the distributed graph
        """
        return self.partition_book.ntypes
    
    @property
    def etypes(self) -> List[str]:
        """
        Returns list of edge types in the local graph
        """
        collected_etypes = [gsm.etypes[0] for gsm in self.graph_shard_managers.values()]
        return collected_etypes
    
    @_gsm_property
    def etypes_global(self) -> List[str]:
        """
        Returns list of edge types in the distributed graph
        """
        return self.partition_book.etypes
    
    @property
    def canonical_etypes(self) -> List[Tuple[str, str, str]]:
        """
        Returns list of canonical edge types in the local graph
        """
        collected_etypes = [gsm.canonical_etypes[0] for gsm in self.graph_shard_managers.values()]
        return collected_etypes
    
    @_gsm_property
    def canonical_etypes_global(self) -> List[Tuple[str, str, str]]:
        """
        Returns list of canonical edge types in the distributed graph
        """
        return self.partition_book.canonical_etypes
    
    @_gsm_property
    def is_block(self) -> bool:
        """
        HeteroGraphShardManager is always a block, because it is used to manage graphs consistisng
        of two sets of nodes: source (input) and destination (seed) nodes
        """
        return True
    
    @_gsm_property
    def node_ranges(self) -> List[Tuple[int, int]]:
        return self._node_ranges
    
    @property
    def ndata(self):
        assert self.src_is_tgt, "ndata shouldn't be used with MFGs"
        return self.srcdata
    
    def nodes(self, ntype: str = None) -> Tensor:
        if ntype is None:
            if len(self.ntypes) == 1:
                return self.input_nodes
            raise RuntimeError("Node type must be specified if the graph consists of more than one node type")
        return self._all_node_types_indices_dict[ntype]
        
    def srcnodes(self, ntype: str = None) -> Tensor:
        if ntype is None:
            if len(self.srctypes) == 1:
                return self.input_nodes
            raise RuntimeError("Node type must be specified if the graph consists of more than one node source type")
        return self._src_node_types_indices_dict[ntype]
        
    def dstnodes(self, ntype: str = None) -> Tensor:
        if ntype is None:
            if len(self.dsttypes) == 1:
                return self.seeds if self.src_is_tgt else self.seeds - self.node_ranges[rank()][0]
            raise RuntimeError("Node type must be specified if the graph consists of more than one node destination type")
        return self._dst_node_types_indices_dict[ntype]
    
    def num_nodes(self, ntype: str = None) -> int:
        """
        Returns the number of nodes of a given node type in the local partition.
        If node type is not specified, returns the number of all nodes in the local partition.
        Local nodes are the nodes in a given partition without HALO nodes.
        
        :param ntype: node type
        :type ntype: str
        """
        if self.src_is_tgt:
            if ntype is not None:
                return self._src_node_types_indices_dict[ntype].shape[0] 
            return self.input_nodes.shape[0]
        elif self.tgt_in_src:
            return self.num_src_nodes(ntype) 
        elif ntype is None:
            return sum([v.shape[0] for _, v in self._all_node_types_indices_dict.items()])
        return self._all_node_types_indices_dict[ntype].shape[0]
    
    @_gsm_function
    def num_nodes_global(self, ntype: str = None) -> int:
        """
        Returns the number of nodes of a given node type in the distributed graph.
        If node type is not specified, returns the number of all nodes in the distributed graph.
        
        :param ntype: node type
        :type ntype: str
        """
        if ntype is not None:
            return self.partition_book._num_nodes(ntype)
        return self.partition_book._num_nodes()
    
    def number_of_nodes(self, ntype: str = None) -> int:
        """
        Alias of :func:`num_nodes`
        """
        return self.num_nodes(ntype)
    
    @_gsm_function
    def number_of_nodes_global(self, ntype: str = None) -> int:
        """
        Alias of :func:`num_nodes_global`
        """
        return self.num_nodes_global(ntype)
    
    def num_src_nodes(self, ntype: str = None) -> int:
        """
        Returns the number of source nodes in the graph (local partition)
        
        :param ntype: node type
        :type ntype: str
        """
        if ntype is not None:
            return self._src_node_types_indices_dict[ntype].shape[0]
        return sum([src_indices.shape[0] for src_indices in self._src_node_types_indices_dict.values()])

    def number_of_src_nodes(self, ntype: str = None) -> int:
        """
        Alias of :func:`num_src_nodes`
        """
        return self.num_src_nodes(ntype)

    def num_dst_nodes(self, ntype: str = None) -> int:
        """
        Returns the number of destination nodes in the graph (local partition)
        
        :param ntype: node type
        :type ntype: str
        """
        if ntype is not None:
            return self._dst_node_types_indices_dict[ntype].shape[0]
        return sum([dst_indices.shape[0] for dst_indices in self._dst_node_types_indices_dict.values()])

    def number_of_dst_nodes(self, ntype: str = None) -> int:
        """
        Alias of :func:`num_dst_nodes`
        """
        return self.num_dst_nodes(ntype)
    
    def num_edges(self, etype: Union[str, Tuple[str, str, str]] = None) -> int:
        """
        Returns the number of edges of a given edge type in the local partition.
        If edge type is not specified, returns the number of all edges in the local partition.
        Local edges are the edges between nodes in the local partition and edges which are 
        incoming from other partition (destination nodes are in the local partition).
        
        :param etype: edge type specified by a string or a string triplet (canonical form)
        :type etype: Union[str, Tuple[str, str, str]]
        """
        if etype is not None:
            return self[etype].num_edges()
        return sum([gsm.num_edges() for gsm in self.graph_shard_managers.values()])
    
    @_gsm_function
    def num_edges_global(self, etype: Union[str, Tuple[str, str, str]] = None) -> int:
        """
        Returns the number of edges of a given edge type in the distributed graph.
        If edge type is not specified, returns the number of all edges in the distributed graph.
        
        :param etype: edge type
        :type etype: str
        """
        if etype is not None:
            return self.partition_book._num_edges(etype)
        return self.partition_book._num_edges()
    
    def number_of_edges(self, etype: Union[str, Tuple[str, str, str]] = None) -> int:
        """
        Alias of :func:`num_edges`
        """
        return self.num_edges(etype)
    
    @_gsm_function
    def number_of_edges_global(self, etype: Union[str, Tuple[str, str, str]] = None) -> int:
        """
        Alias of :func:`num_edges_global`
        """
        return self.num_edges_global(etype) 

    @contextmanager
    def local_scope(self):
        self.srcdata = ChainedNodeDataView(self.srcdata.acceptable_entries, self.srcdata)
        self.edata = ChainedEdgeDataView(self.edata.acceptable_entries, self.edata.etype_to_canonical, self.edata)
        
        if self.src_is_tgt:
            self.dstdata = self.srcdata
        else:
            self.dstdata = ChainedNodeDataView(self.dstdata.acceptable_entries, self.dstdata)
        for gsm in self.graph_shard_managers.values():
            gsm._initialize_data(self.srcdata, self.dstdata, self.edata)
        yield
        self.srcdata = self.srcdata.rewind()
        self.edata = self.edata.rewind()
        if self.src_is_tgt:
            self.dstdata = self.srcdata
        else:
            self.dstdata = self.dstdata.rewind()
        for gsm in self.graph_shard_managers.values():
            gsm._initialize_data(self.srcdata, self.dstdata, self.edata)
         
    def to(self, device=torch.device):
        """
        Specify device for a HeteroGraphShardManager
        """
        for graph_shard_manager in self.graph_shard_managers.values():
            graph_shard_manager.to(device)
        return self
            
    def in_degrees(self, vertices=dgl.ALL, etype=None) -> Tensor:
        return self[etype].in_degrees(vertices, etype)
        
    def out_degrees(self, vertices=dgl.ALL, etype=None) -> Tensor:
        return self[etype].out_degrees(vertices, etype)
            
    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None,
                   ):
        if len(self.graph_shard_managers) == 1:
            self.graph_shard_managers[0].update_all(message_func, reduce_func, apply_node_func, etype)
        else:
            raise RuntimeError("When HeteroGraphShardManager handles graph with more than one relation. \
                               Update_all must be called on a specific relation.")

    def apply_edges(self,
                    message_func,
                    edges=dgl.ALL,
                    etype=None,
                    inplace=False
                    ):
        if len(self.graph_shard_managers) == 1:
            self.graph_shard_managers[0].apply_edges(message_func, edges, etype, inplace)
        else:
            raise RuntimeError("When HeteroGraphShardManager handles graph with more than one relation. \
                               Aply_edges must be called on a specific relation.")
            
    def _etype_to_canonical(self, etype: Union[str, Tuple[str, str, str]]) -> Tuple[str, str, str]:
        """
        Convert etype passed as an input parameter to its canonical form, which is represented
        by a string triplet ``(str, str, str)``, where the first element is a source node type,
        second one is an edge type, and the third one is a destination node type.
        If etype is already represented by its canonical form, then the functions returns 
        input etype as the output.
        
        :param etype: edge type, for which canonical form will be returned
        :type etype: Union[str, Tuple[str, str, str]]
        """
        if etype is None:
            if len(self.canonical_etypes) != 1:
                raise RuntimeError("Edge type name must be specified if there are more than one edge types.")
            etype = self.canonical_etypes[0]
            
        if isinstance(etype, tuple):
            return etype
        else:
            ret = self._etype2canonical.get(etype, None)
            if ret is None:
                raise RuntimeError(f'Edge type "{etype}" does not exist.')
            if len(ret) == 0:
                raise RuntimeError(f'Edge type "{etype}" is ambiguous. Please use canonical edge type in the form of (srctype, etype, dsttype)')
            return ret

    def _update_node_type_indices_dicts(self):
        local_seeds = self.seeds if self.src_is_tgt else self.seeds - self.node_ranges[rank()][0]
        if self.tgt_in_src is False:
            all_nodes = torch.unique(torch.cat((local_seeds, self.input_nodes)))
        else:
            all_nodes = self.input_nodes
            
        for ntype in self._partition_book.ntypes:
            type_index = self._partition_book.ntypes.index(ntype)
            
            for indices_dict, indices in zip(
                [
                    self._src_node_types_indices_dict,
                    self._dst_node_types_indices_dict,
                    self._all_node_types_indices_dict,
                ],
                [self.input_nodes, local_seeds, all_nodes]
            ):
                type_indices = (self._node_types == type_index).nonzero().view(-1)
                indices_mask = torch.isin(indices, type_indices)
                global_type_indices = indices[indices_mask] + self.node_ranges[rank()][0]
                type_wise_indices = self._partition_book.map_to_per_ntype(global_type_indices)[1]
                local_type_wise_indices = self._partition_book.nid2localnid(type_wise_indices, rank(), ntype)
                indices_dict[ntype] = local_type_wise_indices

    def _construct_data_views(self):
        valid_src_entries = []
        for ntype in self.ntypes:
            valid_src_entries.append(((ntype), self.num_src_nodes(ntype)))
        valid_edge_entries = []
        for canonical_etype in self.canonical_etypes:
            valid_edge_entries.append(((canonical_etype), self.num_edges(canonical_etype)))
        self.srcdata = ChainedNodeDataView(valid_src_entries)
        self.edata = ChainedEdgeDataView(valid_edge_entries, self._etype_to_canonical)
        if self.src_is_tgt:
            assert self.num_src_nodes() == self.num_dst_nodes()
            self.dstdata = self.srcdata
        else:
            valid_dst_entries = []
            for ntype in self.ntypes:
                valid_dst_entries.append(((ntype), self.num_dst_nodes(ntype)))
            self.dstdata = ChainedNodeDataView(valid_dst_entries)
