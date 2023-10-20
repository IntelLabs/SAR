from multiprocessing_utils import *
# Do not import DGL and SAR - these modules should be
# independently loaded inside each process

@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_graph_properties_heterogenousgraph(world_size, backend):
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
                mp_dict["expected_ntypes"] = g.ntypes
                mp_dict["expected_etypes"] = g.etypes
                mp_dict["expected_canonical_etypes"] = g.canonical_etypes
                dgl.distributed.partition_graph(g, graph_name, world_size,
                                        tmp_dir, num_hops=1,
                                        balance_edges=True)
            synchronize_processes()
            fgm, _, _ = load_partition_data(rank, graph_name, tmp_dir)
            mp_dict[f"result_{rank}_num_nodes"] = fgm.num_nodes()
            mp_dict[f"result_{rank}_num_edges"] = fgm.num_edges()
            mp_dict[f"result_{rank}_num_node_types"] = [fgm.num_nodes(type) for type in fgm.ntypes]
            mp_dict[f"result_{rank}_ntypes"] = fgm.ntypes
            mp_dict[f"result_{rank}_etypes"] = fgm.etypes
            mp_dict[f"result_{rank}_canonical_etypes"] = fgm.canonical_etypes            
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            
    mp_dict = run_workers(graph_properties, world_size=world_size, backend=backend)
    
    assert mp_dict["expected_num_nodes"] == sum([mp_dict[f"result_{rank}_num_nodes"] for rank in range(world_size)]) / world_size, "Number of nodes does not match"
    assert mp_dict["expected_num_edges"] == sum([mp_dict[f"result_{rank}_num_edges"] for rank in range(world_size)]) / world_size, "Number of edges does not match"
    assert all(np.array(mp_dict["expected_num_node_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)]) / world_size), "Number of nodes for specific types does not match"
    for rank in range(world_size):
        assert all(x == y for x, y in zip(mp_dict["expected_ntypes"], mp_dict[f"result_{rank}_ntypes"]) ), "Node types does not match"
        assert all(x == y for x, y in zip(mp_dict["expected_etypes"], mp_dict[f"result_{rank}_etypes"])), "Edge types does not match"
        assert all(x == y for x, y in zip(mp_dict["expected_canonical_etypes"], mp_dict[f"result_{rank}_canonical_etypes"])), "Canonical edge does not match"


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_graph_properties_homogenousgraph(world_size, backend):
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
                mp_dict["expected_ntypes"] = g.ntypes
                mp_dict["expected_etypes"] = g.etypes
                mp_dict["expected_canonical_etypes"] = g.canonical_etypes
                dgl.distributed.partition_graph(g, graph_name, world_size,
                                        tmp_dir, num_hops=1,
                                        balance_edges=True)
            synchronize_processes()
            fgm, _, _ = load_partition_data(rank, graph_name, tmp_dir)
            mp_dict[f"result_{rank}_num_nodes"] = fgm.num_nodes()
            mp_dict[f"result_{rank}_num_edges"] = fgm.num_edges()
            mp_dict[f"result_{rank}_num_node_types"] = [fgm.num_nodes(type) for type in fgm.ntypes]
            mp_dict[f"result_{rank}_ntypes"] = fgm.ntypes
            mp_dict[f"result_{rank}_etypes"] = fgm.etypes
            mp_dict[f"result_{rank}_canonical_etypes"] = fgm.canonical_etypes            
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            
    mp_dict = run_workers(graph_properties, world_size=world_size, backend=backend)
    
    assert mp_dict["expected_num_nodes"] == sum([mp_dict[f"result_{rank}_num_nodes"] for rank in range(world_size)]) / world_size, "Number of nodes does not match"
    assert mp_dict["expected_num_edges"] == sum([mp_dict[f"result_{rank}_num_edges"] for rank in range(world_size)]) / world_size, "Number of edges does not match"
    assert all(np.array(mp_dict["expected_num_node_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)]) / world_size), "Number of nodes for specific types does not match"
    for rank in range(world_size):
        assert all(x == y for x, y in zip(mp_dict["expected_ntypes"], mp_dict[f"result_{rank}_ntypes"]) ), "Node types does not match"
        assert all(x == y for x, y in zip(mp_dict["expected_etypes"], mp_dict[f"result_{rank}_etypes"])), "Edge types does not match"
        assert all(x == y for x, y in zip(mp_dict["expected_canonical_etypes"], mp_dict[f"result_{rank}_canonical_etypes"])), "Canonical edge does not match"
