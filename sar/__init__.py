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
Top-level SAR package
'''

from .comm import initialize_comms, rank, world_size, comm_device,\
    nfs_ip_init, sync_params, gather_grads
from .core import GraphShardManager, message_has_parameters, DistributedBlock,\
    DistNeighborSampler, DataLoader
from .construct_shard_manager import construct_mfgs, construct_full_graph
from .data_loading import load_dgl_partition_data, suffix_key_lookup
from .distributed_bn import DistributedBN1D
from .config import Config
from .edge_softmax import edge_softmax
from .patch_dgl import patch_dgl, patched_edge_softmax, RelGraphConv
from .logging_setup import logging_setup, logger


__all__ = ['initialize_comms', 'rank', 'world_size', 'nfs_ip_init',
           'comm_device', 'DistributedBN1D',
           'construct_mfgs', 'construct_full_graph', 'GraphShardManager',
           'load_dgl_partition_data', 'suffix_key_lookup', 'Config', 'edge_softmax',
           'message_has_parameters', 'DistributedBlock', 'DistNeighborSampler', 'DataLoader',
           'logging_setup', 'logger', 'RelGraphConv', 'sync_params', 'gather_grads', 'patch_dgl', 'patched_edge_softmax']
