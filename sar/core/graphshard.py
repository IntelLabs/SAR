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

from typing import Tuple, Dict, List,  Optional
import inspect
import itertools
import logging
from collections.abc import MutableMapping
from contextlib import contextmanager
import torch
import dgl  # type:ignore
from dgl import DGLHeteroGraph
from dgl.function.base import TargetCode  # type:ignore
import dgl.function as fn  # type: ignore
from torch import Tensor
import torch.distributed as dist


from ..common_tuples import ShardEdgesAndFeatures, AggregationData, TensorPlace, ShardInfo
from ..comm import exchange_tensors,  rank, all_reduce
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

        edges_src_nodes = torch.arange(self.unique_src_nodes.size(0))[
            unique_src_nodes_inverse]

        edges_tgt_nodes = torch.arange(self.unique_tgt_nodes.size(0))[
            unique_tgt_nodes_inverse]

        self.graph = dgl.create_block((edges_src_nodes, edges_tgt_nodes),
                                      num_src_nodes=self.unique_src_nodes.size(
                                          0),
                                      num_dst_nodes=self.unique_tgt_nodes.size(
                                          0)
                                      )
        self._graph_reverse: Optional[DGLHeteroGraph] = None
        self._shard_info: Optional[ShardInfo] = None
        self.graph.edata.update(shard_edges_features.edge_features)

        self.edge_type_names = edge_type_names

    def _set_shard_info(self, shard_info: ShardInfo):
        self._shard_info = shard_info

    @property
    def shard_info(self) -> Optional[ShardInfo]:
        return self._shard_info

    @property
    def graph_reverse(self) -> DGLHeteroGraph:
        if self._graph_reverse is None:
            edges_src, edges_tgt = self.graph.all_edges()
            self._graph_reverse = dgl.create_block((edges_tgt, edges_src),
                                                   num_src_nodes=self.unique_tgt_nodes.size(0),
                                                   num_dst_nodes=self.unique_src_nodes.size(0))
            self._graph_reverse.edata.update(self.graph.edata)
        return self._graph_reverse

    def to(self, device: torch.device):
        self.graph = self.graph.to(device)
        if self._graph_reverse is not None:
            self._graph_reverse = self._graph_reverse.to(device)


class ChainedDataView(MutableMapping):
    """A dictionary that chains to children dictionary on missed __getitem__ calls"""

    def __init__(self, sz: int, base_dict: Optional["ChainedDataView"] = None):
        self._store: Dict[str, Tensor] = {}
        self._base_dict = base_dict
        self._sz = sz

    def __getitem__(self, key: str):
        if key in self._store:
            return self._store[key]

        if self._base_dict is not None:
            return self._base_dict[key]

        raise KeyError(f'key {key} not found')

    def __setitem__(self, key: str, value: Tensor):
        assert value.size(0) == self._sz, \
            f'Tenosr size {value.size()} does not match graph data size {self._sz}'
        self._store[key] = value

    @property
    def acceptable_size(self):
        return self._sz

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


class GraphShardManager:
    """
    Manages the local graph partition and exposes a subset of the interface
    of dgl.heterograph.DGLHeteroGraph. Most importantly, it implements a
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

    """

    def __init__(self, graph_shards: List[GraphShard], local_src_seeds: Tensor, local_tgt_seeds: Tensor) -> None:
        super().__init__()
        self.graph_shards = graph_shards

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

        self.indices_required_from_me = self.update_boundary_nodes_indices()
        self.sizes_expected_from_others = [
                    shard.unique_src_nodes.size(0) for shard in self.graph_shards]

        self.in_degrees_cache: Dict[Optional[str], Tensor] = {}
        self.out_degrees_cache: Dict[Optional[str], Tensor] = {}

        self.dstdata = ChainedDataView(self.num_dst_nodes())
        self.srcdata = ChainedDataView(self.num_src_nodes())
        self.edata = ChainedDataView(self.num_edges())

    @ property
    def tgt_node_range(self) -> Tuple[int, int]:
        return self.graph_shards[0].tgt_range

    @ property
    def local_src_node_range(self) -> Tuple[int, int]:
        return self.graph_shards[rank()].src_range

    def update_boundary_nodes_indices(self) -> List[Tensor]:
        all_my_sources_indices = [
            x.unique_src_nodes for x in self.graph_shards]

        indices_required_from_me = exchange_tensors(all_my_sources_indices)
        for ind in indices_required_from_me:
            ind.sub_(self.tgt_node_range[0])
        return indices_required_from_me

    @ contextmanager
    def local_scope(self):
        self.dstdata = ChainedDataView(
            self.dstdata.acceptable_size, self.dstdata)
        self.srcdata = ChainedDataView(
            self.srcdata.acceptable_size, self.srcdata)
        self.edata = ChainedDataView(self.edata.acceptable_size, self.edata)
        yield
        self.dstdata = self.dstdata.rewind()
        self.srcdata = self.srcdata.rewind()
        self.edata = self.edata.rewind()

    @ property
    def is_block(self):
        return True

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

    def num_nodes(self, ntype=None) -> int:
        return sum(x.graph.num_nodes(ntype) for x in self.graph_shards)

    def number_of_nodes(self, ntype=None) -> int:
        return self.num_nodes(ntype)

    def num_src_nodes(self, ntype=None) -> int:
        assert ntype is None, 'Node types not supported in GraphShardManager'
        return self.local_src_node_range[1] - self.local_src_node_range[0]

    def number_of_src_nodes(self, ntype=None) -> int:
        return self.num_src_nodes(ntype)

    def num_dst_nodes(self, ntype=None) -> int:
        assert ntype is None, 'Node types not supported in GraphShardManager'
        return self.tgt_node_range[1] - self.tgt_node_range[0]

    def number_of_dst_nodes(self, ntype=None) -> int:
        return self.num_dst_nodes(ntype)

    def number_of_edges(self, etype=None) -> int:
        return self.num_edges(etype)

    def num_edges(self, etype=None) -> int:
        return sum(x.graph.num_edges(etype) for x in self.graph_shards)

    def in_degrees(self, vertices=dgl.ALL, etype=None) -> Tensor:
        if etype not in self.in_degrees_cache:
            in_degrees = torch.zeros(
                self.tgt_node_range[1] - self.tgt_node_range[0], dtype=self.graph_shards[0].graph.idtype).to(self.graph_shards[0].graph.device)
            for shard in self.graph_shards:
                in_degrees[shard.unique_tgt_nodes - self.tgt_node_range[0]] += \
                    shard.graph.in_degrees(etype=etype)
            in_degrees[in_degrees == 0] = 1
            self.in_degrees_cache[etype] = in_degrees

        if vertices == dgl.ALL:
            return self.in_degrees_cache[etype]

        return self.in_degrees_cache[etype][vertices]

    def out_degrees(self, vertices=dgl.ALL, etype=None) -> Tensor:
        if etype not in self.out_degrees_cache:
            for comm_round, shard in enumerate(self.graph_shards):
                out_degrees = torch.zeros(
                    shard.src_range[1] - shard.src_range[0], dtype=shard.graph.idtype).to(shard.graph.device)

                out_degrees[shard.unique_src_nodes - shard.src_range[0]
                            ] = shard.graph.out_degrees(etype=etype)
                all_reduce(out_degrees, op=dist.ReduceOp.SUM,
                           move_to_comm_device=True)
                if comm_round == rank():
                    out_degrees[out_degrees == 0] = 1
                    self.out_degrees_cache[etype] = out_degrees.to(
                        shard.graph.device)

        if vertices == dgl.ALL:
            return self.out_degrees_cache[etype]

        return self.out_degrees_cache[etype][vertices]

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
