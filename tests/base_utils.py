import os
import sar
import torch
import torch.distributed as dist
import dgl
from constants import *
# IMPORTANT - This module should be imported independently
# only by the child processes - i.e. separate workers
 

def initialize_worker(rank, world_size, tmp_dir, backend="ccl"):
    """
    Boilerplate code for setting up connection between workers
    
    :param rank: Rank of the current machine
    :type rank: int
    :param world_size: Number of workers. The same as the number of graph partitions
    :type world_size: int
    :param tmp_dir: Path to the directory where ip file will be created
    :type tmp_dir: str
    """
    torch.seed()
    ip_file = os.path.join(tmp_dir, 'ip_file')
    master_ip_address = sar.nfs_ip_init(rank, ip_file)
    sar.initialize_comms(rank, world_size, master_ip_address, backend)


def load_partition_data(rank, graph_name, tmp_dir):
    """
    Boilerplate code for loading partition data with standard `full_graph_manager` (FGM)

    :param rank: Rank of the current machine
    :type rank: int
    :param graph_name: Name of the partitioned graph
    :type graph_name: str
    :param tmp_dir: Path to the directory where partition data is located
    :type tmp_dir: str
    :returns: Tuple consisting of GraphShardManager object, partition features and labels
    """
    partition_file = os.path.join(tmp_dir, f'{graph_name}.json')
    partition_data = sar.load_dgl_partition_data(partition_file, rank, "cpu")
    full_graph_manager = sar.construct_full_graph(partition_data).to('cpu')
    features = sar.suffix_key_lookup(partition_data.node_features, 'features')
    labels = sar.suffix_key_lookup(partition_data.node_features, 'labels')
    return full_graph_manager, features, labels


def load_partition_data_mfg(rank, graph_name, tmp_dir):
    """
    Boilerplate code for loading partition data with message flow graph (MFG)

    :param rank: Rank of the current machine
    :type rank: int
    :param graph_name: Name of the partitioned graph
    :type graph_name: str
    :param tmp_dir: Path to the directory where partition data is located
    :type tmp_dir: str
    :returns: Tuple consisting of GraphShardManager object, partition features and labels
    """
    partition_file = os.path.join(tmp_dir, f'{graph_name}.json')
    partition_data = sar.load_dgl_partition_data(partition_file, rank, "cpu")
    blocks = sar.construct_mfgs(partition_data,
                                (partition_data.node_features[dgl.NTYPE] == 0).nonzero(as_tuple=True)[0] +
                                partition_data.node_ranges[sar.comm.rank()][0],
                                3, True)
    blocks = [block.to('cpu') for block in blocks]
    features = sar.suffix_key_lookup(partition_data.node_features, 'features')
    labels = sar.suffix_key_lookup(partition_data.node_features, 'labels')
    return blocks, features, labels


def synchronize_processes():
    """
    Function that simulates dist.barrier (using all_reduce because there is an issue with dist.barrier() in ccl)
    """
    dummy_tensor = torch.tensor(1)
    dist.all_reduce(dummy_tensor, dist.ReduceOp.MAX)
