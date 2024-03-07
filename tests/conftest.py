import os
import dgl
from dgl.heterograph import DGLGraph
import pytest
import tempfile
from constants import *
import torch
from torch import Tensor
import multiprocessing as mp
from typing import NamedTuple, Union, Dict


class FixtureEnv(NamedTuple):
    """
    Stores information about variables needed by tests
    
    .. py:attribute:: temp_dir : str
    
        Path to a temporary directory which will be 
        deleted after tests are finished
    
    .. py:attribute:: homo_graph : DGLGraph
    
        DGLGraph object representing homogenous graph
        
    .. py:attribute:: hetero_graph : DGLGraph
    
        DGLGraph object representing heterogeneous graph
        
    .. py:attribute:: node_map : Dict[str, Union[Tensor, Dict[str, Tensor]]]
    
        Dictionary of tensors representing mapping between shuffled node IDs
        and the original node IDs for a homogeneous graph.
        or, dict of dicts of tensors whose key is the node type and value
        is a tensor mapping between shuffled node IDs and the
        original node IDs for each node type for a heterogeneous graph.
        Each dict element represents node_mapping for different graph.
        Homogeneous/heterogeneous and different world_sizes
    """
    temp_dir: str
    homo_graph: DGLGraph
    hetero_graph: DGLGraph
    node_map: Dict[str, Union[Tensor, Dict[str, Tensor]]]
    

@pytest.fixture(autouse=True, scope="session")
def fixture_env():
    """
    Create temp directory that will be used by every test.
    Create and save partitioned graphs in that directory.
    """
    manager = mp.Manager()
    mp_dict = manager.dict()
    with tempfile.TemporaryDirectory() as temp_dir:
        p = mp.Process(target=graph_partitioning, args=(mp_dict, temp_dir,))
        p.start()
        p.join()
        yield FixtureEnv(temp_dir, 
                         mp_dict["homo_graph"],
                         mp_dict["hetero_graph"],
                         mp_dict["node_map"])
      
      
def graph_partitioning(mp_dict, temp_dir):
    """
    Create and partition both homogeneous and heterogeneous
    graphs for different world_sizes
    
    :param temp_dir: Path to the directory where graphs will be partitioned
    :type temp_dir: str
    """
    homo_g = get_random_graph()
    hetero_g = get_random_hetero_graph()
    
    node_mappings = {}
    world_sizes = [1, 2, 4, 8]
    for world_size in world_sizes:
        partition_homo_dir = os.path.join(temp_dir, f"homogeneous_{world_size}")
        os.makedirs(partition_homo_dir)
        node_map, _ = dgl.distributed.partition_graph(homo_g, HOMOGENEOUS_GRAPH_NAME,
                                                     world_size, partition_homo_dir,
                                                     num_hops=1, balance_edges=True,
                                                     return_mapping=True)
        node_mappings[f"homogeneous_{world_size}"] = node_map
        
        partition_hetero_dir = os.path.join(temp_dir, f"heterogeneous_{world_size}")
        os.makedirs(partition_hetero_dir)
        node_map, _ =dgl.distributed.partition_graph(hetero_g, HETEROGENEOUS_GRAPH_NAME,
                                                     world_size, partition_hetero_dir,
                                                     num_hops=1, balance_edges=True,
                                                     return_mapping=True)
        node_mappings[f"heterogeneous_{world_size}"] = node_map
        
    mp_dict["homo_graph"] = homo_g
    mp_dict["hetero_graph"] = hetero_g
    mp_dict["node_map"] = node_mappings

    
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


def get_random_hetero_graph():
    """
    Generates small heterogenous graph with node features and labels only for the first node type 
    
    :returns: dgl graph
    """
    graph_data = {
        ("n_type_1", "rel_1", "n_type_1"): (torch.randint(0, 800, (1000,)), torch.randint(0, 800, (1000,))),
        ("n_type_1", "rel_2", "n_type_3"): (torch.randint(0, 800, (1000,)), torch.randint(0, 800, (1000,))),
        ("n_type_4", "rel_3", "n_type_2"): (torch.randint(0, 800, (1000,)), torch.randint(0, 800, (1000,))),
        ("n_type_4", "rel_4", "n_type_1"): (torch.randint(0, 800, (1000,)), torch.randint(0, 800, (1000,))),
        ("n_type_1", "rev-rel_1", "n_type_1"): (torch.randint(0, 800, (1000,)), torch.randint(0, 800, (1000,))),
        ("n_type_3", "rev-rel_2", "n_type_1"): (torch.randint(0, 800, (1000,)), torch.randint(0, 800, (1000,))),
        ("n_type_2", "rev-rel_3", "n_type_4"): (torch.randint(0, 800, (1000,)), torch.randint(0, 800, (1000,))),
        ("n_type_1", "rev-rel_4", "n_type_4"): (torch.randint(0, 800, (1000,)), torch.randint(0, 800, (1000,)))
    }
    hetero_graph = dgl.heterograph(graph_data)
    hetero_graph.nodes["n_type_1"].data["features"] = torch.rand((hetero_graph.num_nodes("n_type_1"), 10))        
    hetero_graph.nodes["n_type_1"].data["labels"] = torch.randint(0, 10, (hetero_graph.num_nodes("n_type_1"),))
    return hetero_graph
