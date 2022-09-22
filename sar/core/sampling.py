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

from typing import List,  Tuple, Dict, Optional,   cast, Union
import functools
import os
import logging
import torch
import dgl  # type: ignore
from dgl.heterograph import DGLBlock  # type: ignore
from dgl.heterograph import DGLHeteroGraph as DGLGraph  # type: ignore

from dgl.sampling import sample_neighbors  # type: ignore
import dgl.partition  # type:ignore

from torch import Tensor
import torch.distributed as dist

from ..comm import rank, exchange_tensors, world_size,\
    master_ip, master_port, backend, comm_device, initialize_comms, all_reduce
from .graphshard import GraphShardManager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)


def minibatch_partition(sampling_graph: DGLGraph,
                        train_nodes: Tensor,
                        batch_size: int,
                        optimized_batches_cache: Optional[str] = None) -> List[Tensor]:
    """Creates balanced node minibatches where the nodes within each minibatch
    are strongly connected and nodes from different minibatches are weakly connected.
    This minimizes the size of the resulting minibatch graphs as it increases the number
    of nodes in each minibatch that have common neighbors. We run metis to create the
    balanced mini-batches which can be slow, so there is an option to cache the created
    minibatches to disk to speed up subsequent runs.

    :param sampling_graph: The graph with global node indices defining the node connectivity
    :type sampling_graph: DGLGraph
    :param train_nodes: The global indices of the nodes to create minibatches from
    :type train_nodes: Tensor
    :param batch_size: The size of the minibatches. The size the generated minibatches will be close\
    but not exactly the same as ``batch_size``
    :type batch_size: int
    :param optimized_batches_cache: The file name prefix for the cache files that will be used to\
    store the created minibatches. If provided, the files will be created if they do not exist.\
    If they exist, the minibatch data will be loaded from them.
    :type optimized_batches_cache: Optional[str]
    :returns: A list of node minibatches
    :rtype: List[Tensor]

    """
    train_nodes = train_nodes.cpu()
    n_batches = torch.tensor((len(train_nodes) + batch_size - 1) // batch_size)
    all_reduce(n_batches, dist.ReduceOp.MAX)
    logger.info(f'training using {n_batches} precomputed mini-batches')
    if n_batches == 1:
        return [train_nodes]

    if optimized_batches_cache is not None:
        own_batches_cache = optimized_batches_cache + '_' + \
            repr(rank()) + '_' + repr(n_batches.item())
        if os.path.isfile(own_batches_cache):
            batched_train_nodes = torch.load(own_batches_cache)
            logger.info(
                f'loading precomputed batches from {own_batches_cache}')
            return batched_train_nodes
    else:
        own_batches_cache = None

    labeled_mask = torch.zeros(
        sampling_graph.number_of_nodes())
    labeled_mask[train_nodes] = 1

    partitions = dgl.partition.metis_partition_assignment(
        sampling_graph, n_batches.item(),
        balance_ntypes=labeled_mask, balance_edges=True)

    node_assignment = [(partitions == idx).nonzero().view(-1)
                       for idx in range(n_batches)]

    batched_train_nodes = [train_nodes[torch.isin(
        train_nodes, x)] for x in node_assignment]

    if own_batches_cache is not None:
        logger.info(
            f'Saving precomputed batches to {own_batches_cache}')

        torch.save(batched_train_nodes, own_batches_cache)

    return batched_train_nodes


def sample_process_init_fn(worker_idx, _rank, _world_size,
                           _master_ip, _master_port, _backend, _comm_device):
    initialize_comms(_rank, _world_size, _master_ip, _backend, _comm_device,
                     master_port_number=_master_port + worker_idx+1)


class DistNeighborSampler:
    """
    A neighbor sampler that does multi-layer sampling on a distributed
    graph

    :param fanouts: A list of node fanouts where the ith entry is the sampling fanout\
    for nodes at layer i. The length of the list should match the number of layers in the GNN model
    :type fanouts: List[int]
    :param prob: Feature name used as the (unnormalized) probabilities associated with each\
    neighboring edge of a node.  The feature must have only one element for each edge
    :type prob: Optional[str]
    :param replace: If True, sample with replacement.
    :type replace: bool
    :param copy_edata: If True, the edge features of the new graph are copied from\
    the original graph.  If False, the new graph will not have any edge features.
    :type copy_edata: bool
    :param input_node_features: An optional dictionary of node features that should be added\
    to the ``srcdata`` of the sampled block closest to the input. Each feature tensor's first\
    dimension must be the number of nodes in the local partition. If not specified, the sampled\
    blocks will not have any input features
    :type input_node_features: Optional[Dict[str, Tensor]]
    :param output_node_features: An optional dictionary of node features that should be added\
    to the ``dstdata`` of the top sampled block. Each feature tensor's first dimension must be\
    the number of nodes in the local partition. In a node classification setting, this is typically\
    the node labels. If not specified, the sampled blocks will not have any output features
    :type output_node_features: Optional[Dict[str, Tensor]]
    :param output_device: The output device
    :type output_device:

    """

    def __init__(self, fanouts: List[int],  prob: Optional[str] = None,
                 replace: Optional[bool] = False,
                 copy_edata: Optional[bool] = True,
                 input_node_features: Optional[Dict[str, Tensor]] = None,
                 output_node_features: Optional[Dict[str, Tensor]] = None,
                 output_device: Optional[torch.device] = None):
        self.fanouts = fanouts
        self.prob = prob
        self.replace = replace
        self.copy_edata = copy_edata

        for node_features in [input_node_features, output_node_features]:
            if node_features is not None:
                for feature_name, feature_tensor in node_features.items():
                    if not feature_tensor.is_shared():
                        logger.info(f'node feature {feature_name} \
                            not in shared memory. Moving to shared memory')
                        feature_tensor.share_memory_()

        self.input_node_features = input_node_features
        self.output_node_features = output_node_features
        self.output_device = output_device

    def _sample_local(self, sampling_graph: DGLGraph,
                      fanout: int,
                      seed_nodes: Tensor):
        temp = sample_neighbors(sampling_graph, seed_nodes, fanout,
                                prob=self.prob, replace=self.replace, copy_edata=self.copy_edata,
                                output_device=self.output_device)
        return temp

    def _add_input_features(self, sampled_block: DGLBlock,
                            input_nodes,
                            node_ranges):
        if self.input_node_features is not None:
            per_partition_indices = [
                torch.logical_and(input_nodes >= start_idx,
                                  input_nodes < end_idx).nonzero(as_tuple=True)[0]
                for start_idx, end_idx in node_ranges]
            per_partition_input_nodes = [
                input_nodes[indices] - start_idx
                for indices, (start_idx, _) in zip(per_partition_indices, node_ranges)]
            local_nodes = per_partition_input_nodes[rank()]
            per_partition_input_nodes[rank()] = local_nodes.new(0)
            requested_nodes = exchange_tensors(per_partition_input_nodes)

            for feature_name in self.input_node_features:
                feature_tensor = self.input_node_features[feature_name]
                fetched_node_features = exchange_tensors([
                    feature_tensor[indices] for indices
                    in requested_nodes])
                graph_input_tensor = feature_tensor.new(
                    len(input_nodes), *feature_tensor.size()[1:])
                for part_idx in range(len(fetched_node_features)):
                    if part_idx == rank():
                        graph_input_tensor[per_partition_indices[part_idx]
                                           ] = feature_tensor[local_nodes]
                    else:
                        graph_input_tensor[per_partition_indices[part_idx]
                                           ] = fetched_node_features[part_idx]
                sampled_block.srcdata[feature_name] = graph_input_tensor.to(
                    sampled_block.device)

    def _add_output_features(self, sampled_block: DGLBlock,
                             node_ranges: List[Tuple[int, int]]):
        if self.output_node_features is not None:
            dst_id = sampled_block.dstdata[dgl.NID]
            for feature_key, feature_tensor in self.output_node_features.items():
                sampled_block.dstdata[feature_key] = \
                    feature_tensor[dst_id - node_ranges[rank()]
                                   [0]].to(sampled_block.device)

    def _make_edata(self, sampling_graph: DGLGraph,
                    sampled_graphs: List[DGLGraph]):
        local_sampled_graph_edata: Optional[Dict[str, Tensor]] = {}
        local_sampled_graph = sampled_graphs[rank()]
        if self.copy_edata:
            for edge_key in sorted(sampling_graph.edata.keys()):
                key_edge_features = [g.edata[edge_key]
                                     for g in sampled_graphs]

                key_edge_features[rank()] = key_edge_features[0].new(
                    0, *key_edge_features[0].size()[1:])
                assert local_sampled_graph_edata is not None  # for mypy
                local_sampled_graph_edata[edge_key] = torch.cat(
                    exchange_tensors(key_edge_features), dim=0)

        # dgl's add_edges has a problem with empty feature dictionaries
        # Set dictionary to None if empty
        if not local_sampled_graph_edata:
            local_sampled_graph_edata = None

        # dgl.sampling.sample_neighbors adds the
        # edge features dgl.EID that give the ID of the sampled edge from
        # the original graph. We do not care about that, so we delete it,
        # unless the original graph had a dgl.EID edge feature tensor
        if dgl.EID not in sampling_graph.edata:
            del local_sampled_graph.edata[dgl.EID]

        return local_sampled_graph, local_sampled_graph_edata

    def sample(self, full_graph_manager: GraphShardManager,
               seeds: Tensor) -> List[DGLBlock]:
        """
        Distributed sampling

        :param full_graph_manager: The distributed graph from which to sample
        :type full_graph_manager: GraphShardManager
        :param seeds: The seed nodes for sampling
        :type seeds: Tensor
        :returns: A list of ``DGLBlock`` objects with the same length as ``fanouts``
        :rtype: List[DGLBlock]
        """
        sampling_graph = full_graph_manager.sampling_graph
        node_ranges = full_graph_manager.node_ranges

        final_sampled_graphs = [dgl.to_block(self._sample_local(sampling_graph,
                                                                self.fanouts[-1], seeds), seeds)]
        self._add_output_features(final_sampled_graphs[0], node_ranges)
        input_nodes = final_sampled_graphs[0].srcdata[dgl.NID].to(
            sampling_graph.device)
        for fanout in self.fanouts[-2:: -1]:
            per_partition_input_nodes = [
                input_nodes[torch.logical_and(
                    input_nodes >= start_idx, input_nodes < end_idx)]
                for start_idx, end_idx in node_ranges]

            local_nodes_to_sample = exchange_tensors(per_partition_input_nodes)
            sampled_graphs = [self._sample_local(sampling_graph,
                                                 fanout, local_seed)
                              for local_seed in local_nodes_to_sample]

            sampled_src_nodes = [g.all_edges()[0] for g in sampled_graphs]
            sampled_src_nodes[rank()] = sampled_src_nodes[0].new(0).long()
            own_src_nodes = torch.cat(exchange_tensors(sampled_src_nodes))

            sampled_tgt_nodes = [g.all_edges()[1] for g in sampled_graphs]
            sampled_tgt_nodes[rank()] = sampled_tgt_nodes[0].new(0).long()
            own_tgt_nodes = torch.cat(exchange_tensors(sampled_tgt_nodes))

            local_sampled_graph, edata = self._make_edata(
                sampling_graph, sampled_graphs)

            del sampled_graphs

            local_sampled_graph.add_edges(own_src_nodes, own_tgt_nodes,
                                          edata)

            final_sampled_graphs.append(
                dgl.to_block(local_sampled_graph, input_nodes))
            del own_tgt_nodes
            input_nodes = final_sampled_graphs[-1].srcdata[dgl.NID].to(
                sampling_graph.device)

        self._add_input_features(
            final_sampled_graphs[-1], input_nodes, node_ranges)

        return final_sampled_graphs[::-1]


def DataLoader(full_graph_manager: GraphShardManager,
               seed_nodes: Tensor,
               graph_sampler: DistNeighborSampler,
               batch_size: int = 1,
               drop_last: bool = False,
               shuffle: bool = False,
               precompute_optimized_batches: bool = True,
               optimized_batches_cache: Optional[str] = None,
               num_workers: int = 0):
    """A dataloader for distributed node sampling 

    :param full_graph_manager: The distributed graph from which to sample
    :type full_graph_manager: GraphShardManager
    :param seed_nodes: The seed nodes for sampling
    :type seed_nodes: Tensor
    :param graph_sampler: The distributed sampling object. The object must expose the ``sample``\
    routine that will be used to sample the distributed graph
    :type graph_sampler: DistNeighborSampler
    :param batch_size: Batch size
    :type batch_size: int
    :param drop_last: Drop the last batch
    :type drop_last: bool
    :param shuffle: Shuffle the seed nodes each iteration
    :type shuffle: bool
    :param precompute_optimized_batches: Create balanced node minibatches that minimizes the number\
    of edges between nodes in different minibatches. 
    :type precompute_optimized_batches: bool
    :param optimized_batches_cache: The file name prefix for the cache files that will be used to\
    store the created minibatches. If provided, the files will be created if they do not exist.\
    If they exist, the minibatch data will be loaded from them.
    :type optimized_batches_cache: Optional[str]
    :param num_workers: The number of worker processes that will be spawned to do the distributed\
    sampling
    :type num_workers: int

    """
    node_collator = NodeCollator(full_graph_manager,
                                 graph_sampler)

    process_init_foo = functools.partial(sample_process_init_fn,
                                         _rank=rank(),
                                         _world_size=world_size(),
                                         _master_ip=master_ip(),
                                         _master_port=master_port(),
                                         _backend=backend(),
                                         _comm_device=comm_device()
                                         )

    adjusted_seed_nodes: Union[Tensor, List[Tensor]]
    if precompute_optimized_batches:
        adjusted_seed_nodes = minibatch_partition(
            full_graph_manager.sampling_graph,
            seed_nodes,
            batch_size,
            optimized_batches_cache)
        adjusted_batch_size = 1
    else:
        n_local_seeds = len(seed_nodes)
        max_n_seeds_t = torch.tensor(n_local_seeds)
        all_reduce(max_n_seeds_t, dist.ReduceOp.MAX,
                   move_to_comm_device=True)
        max_n_seeds = int(max_n_seeds_t.item())

        n_batches = (n_local_seeds + batch_size - 1) // batch_size
        adjusted_n_batches = (max_n_seeds + batch_size - 1) // batch_size
        if adjusted_n_batches > n_batches:
            expanded_seed_nodes = seed_nodes.repeat(
                (max_n_seeds + n_local_seeds - 1) // n_local_seeds)

            print(
                f'seed nodes size expanded from {n_local_seeds} to  {max_n_seeds}')
            adjusted_seed_nodes = expanded_seed_nodes[:max_n_seeds]
        else:
            adjusted_seed_nodes = seed_nodes
        adjusted_batch_size = batch_size

    return torch.utils.data.DataLoader(cast(torch.utils.data.Dataset, adjusted_seed_nodes),
                                       batch_size=adjusted_batch_size,
                                       shuffle=shuffle,
                                       drop_last=drop_last,
                                       num_workers=num_workers,
                                       worker_init_fn=process_init_foo,
                                       persistent_workers=(num_workers > 0),
                                       multiprocessing_context='spawn' if num_workers > 0 else None,
                                       collate_fn=node_collator.collate)


class NodeCollator:
    """Node collator for distributed neighbor sampling

    :param full_graph_manager: The distributed graph to sample
    :type full_graph_manager: GraphShardManager
    :param graph_sampler: The sampling object. Must expose the ``sample`` method to sample from\
    the distributed graph
    :type graph_sampler: DistNeighborSampler

    """

    def __init__(self, full_graph_manager: GraphShardManager,
                 graph_sampler: DistNeighborSampler):

        self.full_graph_manager = full_graph_manager
        self.graph_sampler = graph_sampler

    def collate(self, indices_parts):
        if indices_parts[0].ndim == 0:  # sampling from nodes
            final_indices = torch.LongTensor(indices_parts)
        else:  # using precomputed mini-batches
            final_indices = torch.cat(indices_parts)

        final_indices = torch.unique(final_indices)
        blocks = self.graph_sampler.sample(self.full_graph_manager,
                                           final_indices)

        return blocks
