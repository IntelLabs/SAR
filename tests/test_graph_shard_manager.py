from multiprocessing_utils import *
# Do not import DGL and SAR - these modules should be
# independently loaded inside each process

@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_fgm_graph_properties_heterogenous_graph(world_size, backend):
    """
    Checks whether FGM's API regarding number of nodes and edges works properly for a heterogenous graph
    """
    import numpy as np
    def graph_properties(mp_dict, rank, world_size, tmp_dir, **kwargs):
        import dgl
        from base_utils import initialize_worker, get_random_hetero_graph,\
            synchronize_processes, load_partition_data
        try:
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
            graph_name = 'dummy_graph'
            if rank == 0:
                g = get_random_hetero_graph()
                mp_dict["expected_num_nodes"] = g.num_nodes()
                mp_dict["expected_num_edges"] = g.num_edges()
                mp_dict["expected_num_node_types"] = [g.num_nodes(type) for type in g.ntypes]
                mp_dict["expected_num_edge_types"] = [g.num_edges(type) for type in g.etypes]
                dgl.distributed.partition_graph(g, graph_name, world_size,
                                        tmp_dir, num_hops=1,
                                        balance_edges=True)
            synchronize_processes()
            fgm, _, _ = load_partition_data(rank, graph_name, tmp_dir)
            mp_dict[f"result_{rank}_num_nodes"] = fgm.num_nodes()
            mp_dict[f"result_{rank}_num_edges"] = fgm.num_edges()
            mp_dict[f"result_{rank}_num_node_types"] = [fgm.num_nodes(type) for type in fgm.ntypes]
            mp_dict[f"result_{rank}_num_edge_types"] = [fgm.num_edges(type) for type in fgm.canonical_etypes]
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            
    mp_dict = run_workers(graph_properties, world_size=world_size, backend=backend)
    
    assert mp_dict["expected_num_nodes"] == sum([mp_dict[f"result_{rank}_num_nodes"] for rank in range(world_size)]), "Number of nodes does not match"
    assert mp_dict["expected_num_edges"] == sum([mp_dict[f"result_{rank}_num_edges"] for rank in range(world_size)]), "Number of edges does not match"
    assert np.any([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)]) == True and \
        all(np.array(mp_dict["expected_num_node_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)])), "Number of nodes for specific types does not match"
    assert np.any([np.array(mp_dict[f"result_{rank}_num_edge_types"]) for rank in range(world_size)]) == True and \
        all(np.array(mp_dict["expected_num_edge_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_edge_types"]) for rank in range(world_size)])), "Number of edges for specific types does not match"


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_fgm_graph_properties_homogenous_graph(world_size, backend):
    """
    Checks whether FGM's API regarding number of nodes and edges works properly for a homogenous graph
    """
    import numpy as np
    def graph_properties(mp_dict, rank, world_size, tmp_dir, **kwargs):
        import dgl
        from base_utils import initialize_worker, get_random_graph,\
            synchronize_processes, load_partition_data
        try:
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
            graph_name = 'dummy_graph'
            if rank == 0:
                g = get_random_graph()
                mp_dict["expected_num_nodes"] = g.num_nodes()
                mp_dict["expected_num_edges"] = g.num_edges()
                mp_dict["expected_num_node_types"] = [g.num_nodes(type) for type in g.ntypes]
                mp_dict["expected_num_edge_types"] = [g.num_edges(type) for type in g.etypes]
                dgl.distributed.partition_graph(g, graph_name, world_size,
                                        tmp_dir, num_hops=1,
                                        balance_edges=True)
            synchronize_processes()
            fgm, _, _ = load_partition_data(rank, graph_name, tmp_dir)
            mp_dict[f"result_{rank}_num_nodes"] = fgm.num_nodes()
            mp_dict[f"result_{rank}_num_edges"] = fgm.num_edges()
            mp_dict[f"result_{rank}_num_node_types"] = [fgm.num_nodes(type) for type in fgm.ntypes]
            mp_dict[f"result_{rank}_num_edge_types"] = [fgm.num_edges(type) for type in fgm.canonical_etypes]
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            
    mp_dict = run_workers(graph_properties, world_size=world_size, backend=backend)
    
    assert mp_dict["expected_num_nodes"] == sum([mp_dict[f"result_{rank}_num_nodes"] for rank in range(world_size)]), "Number of nodes does not match"
    assert mp_dict["expected_num_edges"] == sum([mp_dict[f"result_{rank}_num_edges"] for rank in range(world_size)]), "Number of edges does not match"
    assert np.any([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)]) == True and \
        all(np.array(mp_dict["expected_num_node_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)])), "Number of nodes for specific types does not match"
    assert np.any([np.array(mp_dict[f"result_{rank}_num_edge_types"]) for rank in range(world_size)]) == True and \
        all(np.array(mp_dict["expected_num_edge_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_edge_types"]) for rank in range(world_size)])), "Number of edges for specific types does not match"


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_mfg_graph_properties_heterogenous_graph(world_size, backend):
    """
    Checks whether FGM's API for MFG regarding number of nodes and edges works properly for a heterogenous graph
    """
    import numpy as np
    def graph_properties(mp_dict, rank, world_size, tmp_dir, **kwargs):
        import dgl
        from base_utils import initialize_worker, get_random_hetero_graph,\
            synchronize_processes, load_partition_data_mfg
        try:
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
            graph_name = 'dummy_graph'
            if rank == 0:
                g = get_random_hetero_graph()
                dgl.distributed.partition_graph(g, graph_name, world_size,
                                        tmp_dir, num_hops=1,
                                        balance_edges=True)
            synchronize_processes()
            blocks, _, _ = load_partition_data_mfg(rank, graph_name, tmp_dir)
            for block_id, block in enumerate(blocks): 
                mp_dict[f"result_{rank}_{block_id}_num_nodes"] = block.num_nodes()
                mp_dict[f"result_{rank}_{block_id}_num_edges"] = block.num_edges()
                mp_dict[f"result_{rank}_{block_id}_num_node_types"] = [block.num_nodes(type) for type in block.ntypes]
                mp_dict[f"result_{rank}_{block_id}_num_edge_types"] = [block.num_edges(type) for type in block.canonical_etypes]
            mp_dict["number_of_blocks"] = len(blocks)
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            
    mp_dict = run_workers(graph_properties, world_size=world_size, backend=backend)
    
    for rank in range(world_size):
        for block_id in range(mp_dict["number_of_blocks"]):
            assert mp_dict[f"result_{rank}_{block_id}_num_nodes"] == sum(mp_dict[f"result_{rank}_{block_id}_num_node_types"]), "Sum of nodes of all types is not equal to the number of all nodes in mfg"
            assert mp_dict[f"result_{rank}_{block_id}_num_edges"] == sum(mp_dict[f"result_{rank}_{block_id}_num_edge_types"]), "Sum of edge of all types is not equal to the number of all edges in mfg"

@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_mfg_graph_properties_homogenous_graph(world_size, backend):
    """
    Checks whether FGM's API for MFG regarding number of nodes and edges works properly for a homogenous graph
    """
    import numpy as np
    def graph_properties(mp_dict, rank, world_size, tmp_dir, **kwargs):
        import dgl
        from base_utils import initialize_worker, get_random_graph,\
            synchronize_processes, load_partition_data_mfg
        try:
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
            graph_name = 'dummy_graph'
            if rank == 0:
                g = get_random_graph()
                dgl.distributed.partition_graph(g, graph_name, world_size,
                                        tmp_dir, num_hops=1,
                                        balance_edges=True)
            synchronize_processes()
            blocks, _, _ = load_partition_data_mfg(rank, graph_name, tmp_dir)
            for block_id, block in enumerate(blocks): 
                mp_dict[f"result_{rank}_{block_id}_num_nodes"] = block.num_nodes()
                mp_dict[f"result_{rank}_{block_id}_num_edges"] = block.num_edges()
                mp_dict[f"result_{rank}_{block_id}_num_node_types"] = [block.num_nodes(type) for type in block.ntypes]
                mp_dict[f"result_{rank}_{block_id}_num_edge_types"] = [block.num_edges(type) for type in block.canonical_etypes]
            mp_dict["number_of_blocks"] = len(blocks)
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            
    mp_dict = run_workers(graph_properties, world_size=world_size, backend=backend)
    
    for rank in range(world_size):
        for block_id in range(mp_dict["number_of_blocks"]):
            assert mp_dict[f"result_{rank}_{block_id}_num_nodes"] == sum(mp_dict[f"result_{rank}_{block_id}_num_node_types"]), "Sum of nodes of all types is not equal to the number of all nodes in mfg"
            assert mp_dict[f"result_{rank}_{block_id}_num_edges"] == sum(mp_dict[f"result_{rank}_{block_id}_num_edge_types"]), "Sum of edge of all types is not equal to the number of all edges in mfg"
