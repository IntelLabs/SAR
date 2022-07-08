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

from typing import List, Dict, Optional, Tuple, Union
import logging
from collections.abc import MutableMapping
from numpy import append
import torch
import torch.nn as nn
import dgl  # type: ignore
from torch import Tensor
import torch.distributed as dist
from torch.autograd import profiler
from sar.comm import exchange_tensors
from sar.config import Config

from ..comm import all_to_all, world_size, rank, all_reduce

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)


class CompressorDecompressorBase(nn.Module):
    '''
    Base class for all communication compression modules
    '''

    def __init__(
        self,
        feature_dim: List[int],
        compressor_type: str,
        n_kernel: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.compressor_type = compressor_type
        self.compressors = nn.ModuleDict()
        self.decompressors = nn.ModuleDict()
        self.channel_type = "fixed"
        for i, f in enumerate(feature_dim):
            if compressor_type == "node":
                #TODO: Implement Graph pooling based compression
                self.compressors[f"layer_{i}"] = nn.Identity()
                self.decompressors[f"layer_{i}"] = nn.Identity()
            else:
                self.compressors[f"layer_{i}"] = nn.Sequential(
                    nn.Linear(f, 256),
                    nn.ReLU(),
                    nn.Linear(256, n_kernel)
                )
                self.decompressors[f"layer_{i}"] = nn.Sequential(
                    nn.Linear(n_kernel, 256),
                    nn.ReLU(),
                    nn.Linear(256, f)
                )
                

    def compress(self, tensors_l: List[Tensor]):
        '''
        Take a list of tensors and return a list of compressed tensors
        '''
            # Send data to each client using same compression module
        logger.debug(f"index: {Config.current_layer_index}, tensor_sz: {tensors_l[0].shape}")
        tensors_l = [self.compressors[f"layer_{Config.current_layer_index}"](val)
                            if i != rank() else val for i, val in enumerate(tensors_l)]
        return tensors_l

    def decompress(self, channel_feat: List[Tensor]):
        '''
        Take a list of compressed tensors and return a list of decompressed tensors
        '''
        decompressed_tensors = [self.decompressors[f"layer_{Config.current_layer_index}"](c) 
                                    if i != rank() else c for i, c in enumerate(channel_feat)]

        return decompressed_tensors


class SVD_Compressor(nn.Module):
    def __init__(self, n_kernel):
        self.n_kernel = n_kernel
    
    def compress(self, tensors_l: List[Tensor]):
        output_U = []
        output_S = []
        output_Vh = []
        for i, tensor in enumerate(tensors_l):
            if i == rank():
                output_U.append(U)
                output_S.append(S)
                output_Vh.append(Vh)
                continue
            U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
            output_U.append(U[:, :self.n_kernel])
            output_S.append(S[:self.n_kernel, :self.n_kernel])
            output_Vh.append(Vh[:self.n_kernel, :])
        return output_U, output_S, output_Vh
    
    def decompress(channel_feat: List[Tensor]):
        output = []
        for i, tensor in enumerate(channel_feat):
            if i == rank():
                output.append(tensor)
                continue
            U, S, Vh = tensor
            output.append(U @ torch.diag(S) @ Vh)
    


class ProxyDataView(MutableMapping):
    """A distributed dictionary"""

    def __init__(self, dist_block: "DistributedBlock",
                 tensor_sz: int, base_dict: MutableMapping,
                 indices_required_from_me: List[Tensor],
                 sizes_expected_from_others:  List[int],
                 compression_type: str):
        self.base_dict = base_dict
        self.tensor_sz = tensor_sz
        self.indices_required_from_me = indices_required_from_me
        self.sizes_expected_from_others = sizes_expected_from_others
        self.dist_block = dist_block
        self.compression_type = compression_type

    def set_base_dict(self, new_base_dict: MutableMapping):
        self.base_dict = new_base_dict

    def __setitem__(self, key: str, value: Tensor):
        assert value.size(0) == self.tensor_sz, \
            f'Tenosr size {value.size()} does not match graph data size {self.tensor_sz}'
        logger.debug(f'Distributing item {key} among all DistributedBlocks')

        with profiler.record_function("COMM_FETCH"):
            logger.debug(f'compression decompression: {rank()}')
            compressed_send_tensors = self.dist_block.compression_decompression.compress(
                [value[ind] for ind in self.indices_required_from_me])
            # Local received features have different shapes
            # Replace with dummy tensor to make sure all tensors have same shape
            dummy_tensor = torch.zeros_like(compressed_send_tensors[0]) if rank() != 0 \
                else torch.zeros_like(compressed_send_tensors[1])
            compressed_send_tensors = [v if i != rank() else dummy_tensor 
                                        for i, v in enumerate(compressed_send_tensors)]
        
            compressed_recv_tensors = simple_exchange_op(*compressed_send_tensors)
            compressed_recv_tensors = [v if i != rank() else
                            value[self.indices_required_from_me[i]] 
                            for i, v in enumerate(compressed_recv_tensors)]
            recv_tensors = self.dist_block.compression_decompression.decompress(
                list(compressed_recv_tensors))
            exchange_result = torch.cat(recv_tensors, dim=-2)

        logger.debug(f'exchange_result {exchange_result.size()}')

        self.base_dict[key] = exchange_result

    def __getitem__(self, key: str):
        return self.base_dict[key]

    def __delitem__(self, key: str):
        del self.base_dict[key]

    def __iter__(self):
        return iter(self.base_dict)

    def __len__(self):
        return len(self.base_dict)


class DistributedBlock:
    """
    A wrapper around a dgl.DGLBlock object. The DGLBlock object represents all the edges incoming
    to the local partition. It communicates with remote partitions to implement one-shot communication and
    aggregation in the forward and backward passes . You should not construct DistributedBlock directly,
    but instead use :meth:`GraphShardManager.get_full_partition_graph`

    :param block: A DGLBlock object representing all edges incoming to the local partition
    :type block:
    :param indices_required_from_me: The local node indices required by every other partition to carry out\
    one-hop aggregation
    :type indices_required_from_me: List[Tensor]
    :param sizes_expected_from_others: The number of channels from remote partitions that \
    we need to use to update the features of the nodes in the local partition
    :type sizes_expected_from_others: List[int]
    :param neighbors_indices_in_other_clients: The number of remote neighboring indices in\
    remote partition
    :type neighbors_indices_in_other_clients: List[int]
    :param src_ranges: The global node ids of the start node and end node in each partition. Nodes in each\
    partition have consecutive indices
    :type src_ranges: List[Tuple[int, int]]
    :param unique_src_nodes: The absolute node indices of the source nodes in each remote partition
    :type unique_src_nodes: List[Tensor]
    :param input_nodes: The indices of the input nodes relative to the starting node index of the local partition\
    The input nodes are the nodes needed to produce the output node features assuming one-hop aggregation
    :type input_nodes: Tensor
    :param seeds: The node indices of the output nodes relative to the starting node index of the local partition
    :type seeds: Tensor
    :param edge_type_names: A list of edge type names 
    :type edge_type_names: List[str]
    :param compressors: A list of learnable compressor modules for each remote client that compresses the outgoing
    node features
    :type compressors: List[nn.Module]

    """

    def __init__(self, block, indices_required_from_me: List[Tensor],
                 sizes_expected_from_others: List[int],
                 src_ranges: List[Tuple[int, int]],
                 unique_src_nodes: List[Tensor],
                 input_nodes: Tensor,
                 seeds: Tensor,
                 edge_type_names: List[str]):

        self._block = block
        self.indices_required_from_me = indices_required_from_me
        self.sizes_expected_from_others = sizes_expected_from_others
        self.src_ranges = src_ranges
        self.unique_src_nodes = unique_src_nodes
        self.edge_type_names = edge_type_names
        self.input_nodes = input_nodes
        self.seeds = seeds

        self.srcdata = ProxyDataView(self, input_nodes.size(0),
                                     block.srcdata, indices_required_from_me, 
                                     sizes_expected_from_others)

        self.out_degrees_cache: Dict[Optional[str], Tensor] = {}
        self._compression_decompression: CompressorDecompressorBase = None
    
    @property
    def compression_decompression(self):
        return self._compression_decompression

    @compression_decompression.setter
    def compression_decompression(self, mod: CompressorDecompressorBase):
        self._compression_decompression = mod

    def out_degrees(self, vertices=dgl.ALL, etype=None) -> Tensor:
        if etype not in self.out_degrees_cache:
            src_out_degrees = self._block.out_degrees(etype=etype)
            src_out_degrees_split = torch.split(src_out_degrees, self.sizes_expected_from_others)

            for comm_round in range(world_size()):
                out_degrees = torch.zeros(
                    self.src_ranges[comm_round][1] - self.src_ranges[comm_round][0],
                    dtype=self._block.idtype).to(self._block.device)

                out_degrees[self.unique_src_nodes[comm_round] - self.src_ranges[comm_round][0]
                            ] = src_out_degrees_split[comm_round]
                all_reduce(out_degrees, op=dist.ReduceOp.SUM, move_to_comm_device=True)
                if comm_round == rank():
                    out_degrees[out_degrees == 0] = 1
                    self.out_degrees_cache[etype] = out_degrees.to(self._block.device)

        if vertices == dgl.ALL:
            return self.out_degrees_cache[etype]

        return self.out_degrees_cache[etype][vertices]

    def to(self, device: torch.device):
        self._block = self._block.to(device)
        self.srcdata.set_base_dict(self._block.srcdata)
        return self

    def __getattr__(self, name):
        return getattr(self._block, name)

class SimpleExchangeOp(torch.autograd.Function):  # pylint: disable = abstract-method
    @ staticmethod
    # pylint: disable = arguments-differ,unused-argument
    def forward(ctx, *send_tensors) -> Tuple[Tensor, ...]:  # type: ignore
        send_tensors = [x.detach() for x in send_tensors]
        recv_tensors = exchange_tensors(list(send_tensors))
        return tuple(recv_tensors)

    @ staticmethod
    # pylint: disable = arguments-differ
    # type: ignore
    def backward(ctx, *grad):
        send_grad = exchange_tensors(list(grad))
        return tuple(send_grad)


simple_exchange_op = SimpleExchangeOp.apply

# class TensorExchangeOp(torch.autograd.Function):  # pylint: disable = abstract-method
#     @ staticmethod
#     # pylint: disable = arguments-differ,unused-argument
#     def forward(ctx, val: Tensor, indices_required_from_me: Tensor,  # type: ignore
#                 sizes_expected_from_others: Tensor, 
#                 compressors: Union[List[nn.Module], nn.Module],
#                 decompressors: Union[List[nn.Module], nn.Module],
#                 channel_type: str) -> Tensor:  # type: ignore
#         ctx.sizes_expected_from_others = sizes_expected_from_others
#         ctx.indices_required_from_me = indices_required_from_me
#         ctx.input_size = val.size()

#         send_tensors = [val[indices] for indices in indices_required_from_me]
#         if channel_type == "client":
#             # Client specific compression
#             send_tensors = [compressors[i](val) if i != rank() else val 
#                                 for i, val in enumerate(send_tensors)]
#         elif channel_type == "fixed":
#             # Send data to each client using same compression module
#             #TODO: check if i means rank or not
#             send_tensors = [compressors(val) if i != rank() else val 
#                                 for i, val in enumerate(send_tensors)]
        
#         recv_tensors = [val.new(sz_from_others, *val.size()[1:])
#                         for sz_from_others in sizes_expected_from_others]
#         all_to_all(recv_tensors, send_tensors, move_to_comm_device=True)

#         if channel_type == "client":
#             # Client specific decompression
#             recv_tensors = [decompressors[i](val) if i != rank() else val 
#                                 for i, val in enumerate(recv_tensors)]
#         elif channel_type == "fixed":
#             # All received features are decompressed using the same module
#             recv_tensors = [decompressors(val) if i != rank() else val
#                                 for i, val in enumerate(recv_tensors)]
        
#         return torch.cat(recv_tensors)

#     @ staticmethod
#     # pylint: disable = arguments-differ
#     # type: ignore
#     def backward(ctx, grad):
#         send_tensors = list(torch.split(grad, ctx.sizes_expected_from_others))
#         recv_tensors = [grad.new(len(indices), *grad.size()[1:])
#                         for indices in ctx.indices_required_from_me]
#         all_to_all(recv_tensors, send_tensors, move_to_comm_device=True)

#         input_grad = grad.new(ctx.input_size).zero_()
#         for r_tensor, indices in zip(recv_tensors, ctx.indices_required_from_me):
#             input_grad[indices] += r_tensor

#         return input_grad, None, None


# tensor_exchange_op = TensorExchangeOp.apply
