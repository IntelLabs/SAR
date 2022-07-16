import enum
from math import ceil, floor, log2
from typing import List, Tuple
import logging
import dgl
from collections.abc import MutableMapping
from numpy import append
import torch
import torch.nn as nn
from torch import Tensor
from sar.comm import exchange_tensors, rank
from sar.config import Config
from sar.core.full_partition_block import DistributedBlock

logger = logging.getLogger(__name__)

class CompressorDecompressorBase(nn.Module):
    '''
    Base class for all communication compression modules
    '''

    def __init__(
        self):
        super().__init__()

    def compress(self, tensors_l: List[Tensor]):
        '''
        Take a list of tensors and return a list of compressed tensors
        '''
        return tensors_l

    def decompress(self, channel_feat: List[Tensor]):
        '''
        Take a list of compressed tensors and return a list of decompressed tensors
        '''
        return channel_feat


class FeatureCompressorDecompressor(CompressorDecompressorBase):
    def __init__(self, feature_dim: List[int], comp_ratio: List[float]):
        super().__init__()
        self.feature_dim = feature_dim
        self.compressors = nn.ModuleDict()
        self.decompressors = nn.ModuleDict()
        for i, f in enumerate(feature_dim):
            k = floor(f/comp_ratio[Config.current_layer_index])
            self.compressors[f"layer_{i}"] = nn.Sequential(
                nn.Linear(f, f),
                nn.ReLU(),
                nn.Linear(f, k),
                nn.ReLU()
            )
            self.decompressors[f"layer_{i}"] = nn.Sequential(
                nn.Linear(k, f),
                nn.ReLU(),
                nn.Linear(f, f)
            )
    
    def compress(self, tensors_l: List[Tensor], iter: int = 0):
        '''
        Take a list of tensors and return a list of compressed tensors
        '''
            # Send data to each client using same compression module
        logger.debug(f"index: {Config.current_layer_index}, tensor_sz: {tensors_l[0].shape}")
        tensors_l = [self.compressors[f"layer_{Config.current_layer_index}"](val)
                            if Config.current_layer_index < Config.total_layers - 1 
                                else val for val in tensors_l]
        return tensors_l

    def decompress(self, channel_feat: List[Tensor]):
        '''
        Take a list of compressed tensors and return a list of decompressed tensors
        '''
        decompressed_tensors = [self.decompressors[f"layer_{Config.current_layer_index}"](c)
                                    if Config.current_layer_index < Config.total_layers - 1 
                                        else c for c in channel_feat]
        return decompressed_tensors


class NodeCompressorDecompressor(CompressorDecompressorBase):
    def __init__(
        self, 
        feature_dim: List[int], 
        comp_ratio_b: List[float],
        comp_ratio_a: List[float],
        step: int):
        """
        Compresses and decompresses along node dimension by selecting
        a fraction of nodes at the compressor and replacing them by 0
        at the decompressor.
        """

        super().__init__()
        self.feature_dim = feature_dim
        self.scorer = nn.ModuleDict()
        self.comp_ratio_b = comp_ratio_b
        self.comp_ratio_a = comp_ratio_a
        self.step = step
        for i, f in enumerate(feature_dim):
            self.scorer[f"layer_{i}"] = nn.Sequential(
                nn.Linear(f, 1),
                nn.Sigmoid()
            )
    
    def _compute_CR_exp(self, step, iter):
        # Decrease CR from 2**p to 2**1
        return 2**(ceil((Config.total_train_iter - iter)/step))
    
    def _compute_CR_linear(self, init_CR, slope, step, iter):
        # Decrease CR using b - a*x
        return init_CR - slope * (iter// step)
            
    def compress(
        self, 
        tensors_l: List[Tensor],
        iter: int = 0,
        step: int = 32,
        vcr_type: str = "exp",
        scorer_type: str = "learnable"):
        """
        Take a list of tensors and return a list of compressed tensors

        :param tensors_l: List of send tensors for each graph shard
        :type List[Tensor]
        :param iter: The training iteration number
        :type int
        :param step: Number of steps for which CR is constant
        :type int
        :param vcr_type: Method by which CR will be changed through out the training
        :type str
        :param scorer_type: Module type by which the nodes will be ranked before sending
        :type str
        """

        compressed_tensors_l = []
        sel_indices = []
        if vcr_type == "exp":
            comp_ratio = self._compute_CR_exp(step, iter)
        elif vcr_type == "linear":
            comp_ratio = self._compute_CR_linear(
                                self.comp_ratio_b[Config.current_layer_index],
                                self.comp_ratio_a[Config.current_layer_index],
                                step, iter)
        else:
            raise NotImplementedError(
                "vcr_type should be either exp or linear")
        
        comp_ratio = max(1, comp_ratio)
        for val in tensors_l:
            if scorer_type == "learnable":
                score = self.scorer[f"layer_{Config.current_layer_index}"](val)
            elif scorer_type == "random":
                score = torch.rand(val.shape[0], 1)
            else:
                raise NotImplementedError(
                    "Scorer type should be either learnable or random")
            k = val.shape[0] // comp_ratio
            k = max(1, k) # Send at least 1 node if CR is too high.
            _, idx = torch.topk(score, k=k, dim=0)
            idx = idx.squeeze(-1)
            compressed_tensors_l.append(val[idx, :])
            sel_indices.append(idx)
        return compressed_tensors_l, sel_indices
    
    def decompress(
        self, 
        args):
        '''
        Decompress received tensors by creating properly shaped recv tensors
        '''

        channel_feat = args[0]
        sel_indices = args[1]
        sizes_expected_from_others = args[2]

        decompressed_tensors_l = []

        for i in range(len(sizes_expected_from_others)):
            new_val = channel_feat[i].new_zeros(
                sizes_expected_from_others[i], 
                channel_feat[i].shape[1])
            new_val[sel_indices[i], :] = channel_feat[i]
            decompressed_tensors_l.append(new_val)

        return decompressed_tensors_l


class SubgraphCompressorDecompressor(CompressorDecompressorBase):
    def __init__(
        self,
        feature_dim: List[int],
        full_local_graph: DistributedBlock,
        indices_required_from_me: List[Tensor],
        tgt_node_range: Tuple[int, int]
        ):
        super().__init__()

        self.full_local_graph = full_local_graph
        self.indices_required_from_me = indices_required_from_me
        self.tgt_node_range = tgt_node_range

        # Create learnable modules
        self.pack = nn.ModuleDict()
        self.scorer = nn.ModuleDict()
        self.unpack = nn.ModuleDict()

        for i, f in enumerate(feature_dim):
            self.pack[f"layer_{i}"] = nn.Sequential(
                dgl.nn.SAGEConv(f, f, aggregator_type='mean'),
                nn.ReLU(),
                dgl.nn.SAGEConv(f, f, aggregator_type='mean')
            )
            self.scorer[f"layer_{i}"] = nn.Sequential(
                nn.Linear(f, 1),
                nn.Sigmoid()
            )
            self.unpack[f"layer_{i}"] = nn.Sequential(
                dgl.nn.SAGEConv(f, f, aggregator_type='mean'),
                nn.ReLU(),
                dgl.nn.SAGEConv(f, f, aggregator_type='mean')
            )
        
        # Create induced subgraphs for source nodes
        self.induced_boundary_graphs = []
        self.remote_boundary_graphs = []
        for i, local_src_nodes in enumerate(indices_required_from_me):
            local_bg = self._construct_boundary_subgraph_local(local_src_nodes)
            remote_bg = self._construct_boundary_subgraph_remote(full_local_graph.unique_src_nodes[i])
            self.induced_boundary_graphs.append(local_bg)
            self.remote_boundary_graphs.append(remote_bg)
            
    def _construct_boundary_subgraph_local(self, seed_nodes: List[Tensor]):
        local_src_range = self.full_local_graph.src_ranges[rank()]
        # Select edges where target nodes are the local boundary nodes (seed nodes)
        # and source nodes are from local partition
        induced_edge_locs_tgt = torch.isin(self.full_local_graph.all_edges()[1], seed_nodes)
        induced_edge_locs_src = torch.isin(self.full_local_graph.all_edges()[0], 
                                    torch.range(local_src_range[0], local_src_range[1]))
        induced_edge_locs = induced_edge_locs_tgt and induced_edge_locs_src
        # Convert target and source nodes to global indices
        global_tgt_nodes = self.full_local_graph.all_edges()[1][induced_edge_locs] + self.tgt_node_range[0]
        graph_to_global = torch.cat(self.full_local_graph.unique_src_nodes)
        global_src_nodes = graph_to_global[self.full_local_graph.all_edges()[0][induced_edge_locs]]
        induced_graph = dgl.graph((global_src_nodes, global_tgt_nodes))
        return induced_graph
    
    def _construct_boundary_subgraph_remote(self, seed_nodes: List[Tensor]):
        # Select edges where target nodes are the local nodes
        # and source nodes are the remote nodes (seed nodes) for a partition
        induced_edge_locs = torch.isin(self.full_local_graph.all_edges()[0], seed_nodes)
        # Convert target and source nodes to global indices
        global_tgt_nodes = self.full_local_graph.all_edges()[1][induced_edge_locs] + self.tgt_node_range[0]
        graph_to_global = torch.cat(self.full_local_graph.unique_src_nodes)
        global_src_nodes = graph_to_global[self.full_local_graph.all_edges()[0][induced_edge_locs]]
        induced_graph = dgl.graph((global_src_nodes, global_tgt_nodes))
        return induced_graph

    def compress(
        self, 
        tensors_l: List[Tensor],
        iter: int = 0,
        step: int = 32,
        vcr_type: str = "exp"):
        """
        Take a list of tensors and return a list of compressed tensors

        :param tensors_l: List of send tensors for each graph shard
        :type List[Tensor]
        :param iter: The training iteration number
        :type int
        :param step: Number of steps for which CR is constant
        :type int
        :param vcr_type: Method by which CR will be changed through out the training
        :type str
        :param scorer_type: Module type by which the nodes will be ranked before sending
        :type str
        """

        compressed_tensors_l = []
        sel_indices = []
        if vcr_type == "exp":
            comp_ratio = self._compute_CR_exp(step, iter)
        elif vcr_type == "linear":
            comp_ratio = self._compute_CR_linear(
                                self.comp_ratio_b[Config.current_layer_index],
                                self.comp_ratio_a[Config.current_layer_index],
                                step, iter)
        else:
            raise NotImplementedError(
                "vcr_type should be either exp or linear")
        
        comp_ratio = max(1, comp_ratio)
        for i, val in enumerate(tensors_l):
            g = self.induced_boundary_graphs[i]
            val = self.pack[f"layer_{Config.current_layer_index}"](g, val)
            score = self.scorer[f"layer_{Config.current_layer_index}"](val)
            k = val.shape[0] // comp_ratio
            k = max(1, k) # Send at least 1 node if CR is too high.
            _, idx = torch.topk(score, k=k, dim=0)
            idx = idx.squeeze(-1)
            compressed_tensors_l.append(val[idx, :])
            sel_indices.append(idx)
        return compressed_tensors_l, sel_indices
    
    def decompress(
        self, 
        args):
        '''
        Decompress received tensors by creating properly shaped recv tensors
        '''

        channel_feat = args[0]
        sel_indices = args[1]
        sizes_expected_from_others = args[2]
        induced_subgraphs = args[3]

        decompressed_tensors_l = []

        for i in range(len(sizes_expected_from_others)):
            new_val = channel_feat[i].new_zeros(
                sizes_expected_from_others[i], 
                channel_feat[i].shape[1])
            new_val[sel_indices[i], :] = channel_feat[i]
            new_val = self.unpack[f"layer_{Config.current_layer_index}"](
                induced_subgraphs[i], new_val)
            decompressed_tensors_l.append(new_val)

        return decompressed_tensors_l
        
