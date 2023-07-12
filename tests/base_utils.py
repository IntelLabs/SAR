
import os
import sar
import torch
import torch.distributed as dist
import dgl


def initialize_worker(rank, world_size, tmp_dir):
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
    sar.initialize_comms(rank, world_size, master_ip_address, 'ccl')


def get_random_graph():
    """
    Generates small homogenous graph with features and labels
    
    :returns: dgl graph
    """
    graph = dgl.rand_graph(1000, 2500)
    graph = dgl.add_self_loop(graph)
    graph.ndata.clear()
    graph.ndata['features'] = torch.rand((graph.num_nodes(), 10))
    graph.ndata['labels'] = torch.randint(0, 10, (graph.num_nodes(),))
    return graph


def load_partition_data(rank, graph_name, tmp_dir):
    """
    Boilerplate code for loading partition data

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


def synchronize_processes():
    """
    Function that simulates dist.barrier (using all_reduce because there is an issue with dist.barrier() in ccl)
    """
    dummy_tensor = torch.tensor(1)
    dist.all_reduce(dummy_tensor, dist.ReduceOp.MAX)
