from multiprocessing_utils import *
from constants import *
import os
# Do not import DGL and SAR - these modules should be
# independently loaded inside each process

@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_fgm_graph_properties_heterogenous_graph(world_size, backend, fixture_env):
    """
    Checks whether full graph manager's (FGM) API regarding number of nodes and edges works properly for a heterogenous graph
    """
    def graph_properties(mp_dict, rank, world_size, fixture_env, **kwargs):
        import numpy as np
        from base_utils import initialize_worker, synchronize_processes, load_partition_data
            
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        if rank == 0:
            g = fixture_env.hetero_graph
            mp_dict["expected_num_nodes"] = g.num_nodes()
            mp_dict["expected_num_edges"] = g.num_edges()
            mp_dict["expected_num_node_types"] = [g.num_nodes(type) for type in g.ntypes]
            mp_dict["expected_num_edge_types"] = [g.num_edges(type) for type in g.etypes]
            
        fgm, _, _ = load_partition_data(rank, HETEROGENEOUS_GRAPH_NAME, os.path.join(temp_dir, f"heterogeneous_{world_size}"))
        mp_dict[f"result_{rank}_num_nodes"] = fgm.num_nodes()
        mp_dict[f"result_{rank}_num_edges"] = fgm.num_edges()
        mp_dict[f"result_{rank}_num_node_types"] = [fgm.num_nodes(type) for type in fgm.ntypes]
        mp_dict[f"result_{rank}_num_edge_types"] = [fgm.num_edges(type) for type in fgm.canonical_etypes]
        
        synchronize_processes()
        assert mp_dict["expected_num_nodes"] == sum([mp_dict[f"result_{rank}_num_nodes"] for rank in range(world_size)]), \
            "Number of nodes does not match"
        assert mp_dict["expected_num_edges"] == sum([mp_dict[f"result_{rank}_num_edges"] for rank in range(world_size)]), \
            "Number of edges does not match"
        assert np.any([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)]) == True and \
            all(np.array(mp_dict["expected_num_node_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)])), \
                "Number of nodes for specific types does not match"
        assert np.any([np.array(mp_dict[f"result_{rank}_num_edge_types"]) for rank in range(world_size)]) == True and \
            all(np.array(mp_dict["expected_num_edge_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_edge_types"]) for rank in range(world_size)])), \
                "Number of edges for specific types does not match"
            
    run_workers(graph_properties, fixture_env, world_size=world_size, backend=backend)


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_fgm_graph_properties_homogeneous_graph(world_size, backend, fixture_env):
    """
    Checks whether full graph manager's (FGM) API regarding number of nodes and edges works properly for a homogenous graph
    """
    def graph_properties(mp_dict, rank, world_size, fixture_env, **kwargs):
        import numpy as np
        from base_utils import initialize_worker, synchronize_processes, load_partition_data
            
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        if rank == 0:
            g = fixture_env.homo_graph
            mp_dict["expected_num_nodes"] = g.num_nodes()
            mp_dict["expected_num_edges"] = g.num_edges()
            mp_dict["expected_num_node_types"] = [g.num_nodes(type) for type in g.ntypes]
            mp_dict["expected_num_edge_types"] = [g.num_edges(type) for type in g.etypes]
            
        fgm, _, _ = load_partition_data(rank, HOMOGENEOUS_GRAPH_NAME, os.path.join(temp_dir, f"homogeneous_{world_size}"))
        mp_dict[f"result_{rank}_num_nodes"] = fgm.num_nodes()
        mp_dict[f"result_{rank}_num_edges"] = fgm.num_edges()
        mp_dict[f"result_{rank}_num_node_types"] = [fgm.num_nodes(type) for type in fgm.ntypes]
        mp_dict[f"result_{rank}_num_edge_types"] = [fgm.num_edges(type) for type in fgm.canonical_etypes]
        
        synchronize_processes()
        assert mp_dict["expected_num_nodes"] == sum([mp_dict[f"result_{rank}_num_nodes"] for rank in range(world_size)]), \
            "Number of nodes does not match"
        assert mp_dict["expected_num_edges"] == sum([mp_dict[f"result_{rank}_num_edges"] for rank in range(world_size)]), \
            "Number of edges does not match"
        assert np.any([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)]) == True and \
            all(np.array(mp_dict["expected_num_node_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_node_types"]) for rank in range(world_size)])), \
                "Number of nodes for specific types does not match"
        assert np.any([np.array(mp_dict[f"result_{rank}_num_edge_types"]) for rank in range(world_size)]) == True and \
            all(np.array(mp_dict["expected_num_edge_types"]) == sum([np.array(mp_dict[f"result_{rank}_num_edge_types"]) for rank in range(world_size)])), \
                "Number of edges for specific types does not match"
            
    run_workers(graph_properties, fixture_env, world_size=world_size, backend=backend)


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_mfg_graph_properties_heterogenous_graph(world_size, backend, fixture_env):
    """
    Checks whether message flow graph's (MFG) API regarding number of nodes and edges works properly for a heterogenous graph
    """
    def graph_properties(mp_dict, rank, world_size, fixture_env, **kwargs):
        from base_utils import initialize_worker, synchronize_processes, load_partition_data_mfg
            
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        blocks, _, _ = load_partition_data_mfg(rank, HETEROGENEOUS_GRAPH_NAME, os.path.join(temp_dir, f"heterogeneous_{world_size}"))
            
        for block_id, block in enumerate(blocks): 
            mp_dict[f"result_{rank}_{block_id}_num_nodes"] = block.num_nodes()
            mp_dict[f"result_{rank}_{block_id}_num_edges"] = block.num_edges()
            mp_dict[f"result_{rank}_{block_id}_num_node_types"] = [block.num_nodes(type) for type in block.ntypes]
            mp_dict[f"result_{rank}_{block_id}_num_edge_types"] = [block.num_edges(type) for type in block.canonical_etypes]
        mp_dict["number_of_blocks"] = len(blocks)
            
        synchronize_processes()
        for rank in range(world_size):
            for block_id in range(mp_dict["number_of_blocks"]):
                assert mp_dict[f"result_{rank}_{block_id}_num_nodes"] == sum(mp_dict[f"result_{rank}_{block_id}_num_node_types"]), \
                    "Sum of nodes of all types is not equal to the number of all nodes in mfg"
                assert mp_dict[f"result_{rank}_{block_id}_num_edges"] == sum(mp_dict[f"result_{rank}_{block_id}_num_edge_types"]), \
                    "Sum of edge of all types is not equal to the number of all edges in mfg"
            
    run_workers(graph_properties, fixture_env, world_size=world_size, backend=backend)
    

@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_mfg_graph_properties_homogenous_graph(world_size, backend, fixture_env):
    """
    Checks whether message flow graph's (MFG) API regarding number of nodes and edges works properly for a homogenous graph
    """
    def graph_properties(mp_dict, rank, world_size, fixture_env, **kwargs):
        from base_utils import initialize_worker, synchronize_processes, load_partition_data_mfg
            
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        blocks, _, _ = load_partition_data_mfg(rank, HOMOGENEOUS_GRAPH_NAME, os.path.join(temp_dir, f"homogeneous_{world_size}"))
        for block_id, block in enumerate(blocks): 
            mp_dict[f"result_{rank}_{block_id}_num_nodes"] = block.num_nodes()
            mp_dict[f"result_{rank}_{block_id}_num_edges"] = block.num_edges()
            mp_dict[f"result_{rank}_{block_id}_num_node_types"] = [block.num_nodes(type) for type in block.ntypes]
            mp_dict[f"result_{rank}_{block_id}_num_edge_types"] = [block.num_edges(type) for type in block.canonical_etypes]
        mp_dict["number_of_blocks"] = len(blocks)
            
        synchronize_processes()
        for rank in range(world_size):
            for block_id in range(mp_dict["number_of_blocks"]):
                assert mp_dict[f"result_{rank}_{block_id}_num_nodes"] == sum(mp_dict[f"result_{rank}_{block_id}_num_node_types"]), \
                    "Sum of nodes of all types is not equal to the number of all nodes in mfg"
                assert mp_dict[f"result_{rank}_{block_id}_num_edges"] == sum(mp_dict[f"result_{rank}_{block_id}_num_edge_types"]), \
                    "Sum of edge of all types is not equal to the number of all edges in mfg"
            
    run_workers(graph_properties, fixture_env, world_size=world_size, backend=backend)


@pytest.mark.parametrize("backend", ["ccl"])
@pytest.mark.parametrize("world_size", [2])
@sar_test
def test_fgm_local_scope(world_size, backend, fixture_env):
    """
    Test behaviour of local_scope function. Graph's srcdata/dstdata/ndata/edata
    should not be modified if the changes were made inside the "with local_scope" block.
    """
    def local_scope(mp_dict, rank, world_size, fixture_env, **kwargs):
        import torch
        from base_utils import initialize_worker, load_partition_data
            
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        fgm, _, _ = load_partition_data(rank, HETEROGENEOUS_GRAPH_NAME, os.path.join(temp_dir, f"heterogeneous_{world_size}"))
        
        fgm.edata["edge_feature"] = {etype: torch.ones(fgm.num_edges(etype), 5) for etype in fgm.canonical_etypes}
        fgm.srcdata["1st_feature"] = {srctype: torch.ones(fgm.num_src_nodes(srctype), 5) for srctype in fgm.srctypes}
        for dsttype in fgm.dsttypes:
            fgm.dstdata["2nd_feature"] = {dsttype: torch.ones(fgm.num_dst_nodes(dsttype), 10)}
        for etype in fgm.canonical_etypes:
            fgm[etype].srcdata[f"3rd_feature_{etype}"] = torch.zeros(fgm[etype].num_src_nodes(), 15)
        
        with fgm.local_scope():
            fgm.edata["edge_feature_local"] = {etype: torch.ones(fgm.num_edges(etype), 5) for etype in fgm.canonical_etypes}
            fgm.srcdata["1st_feature_local"] = {srctype: torch.ones(fgm.num_src_nodes(srctype), 5) for srctype in fgm.srctypes}
            for dsttype in fgm.dsttypes:
                fgm.dstdata["2nd_feature_local"] = {dsttype: torch.ones(fgm.num_dst_nodes(dsttype), 10)}
            for etype in fgm.canonical_etypes:
                fgm[etype].srcdata[f"3rd_feature_{etype}_local"] = torch.zeros(fgm[etype].num_src_nodes(), 15)
                
            # Asserts inside of "with fgm.local_scope():" - outisde and inside features should be available
            for ntype in fgm.ntypes:
                assert ntype in fgm.srcdata["1st_feature"].keys() and ntype in fgm.srcdata["1st_feature_local"].keys()
                assert ntype in fgm.dstdata["2nd_feature"].keys() and ntype in fgm.dstdata["2nd_feature_local"].keys()
            for etype in fgm.canonical_etypes:
                assert fgm[etype].srctypes[0] in fgm.srcdata[f"3rd_feature_{etype}"].keys() and fgm[etype].srctypes[0] in fgm.srcdata[f"3rd_feature_{etype}_local"].keys()
                assert etype in fgm.edata["edge_feature"].keys() and etype in fgm.edata["edge_feature_local"].keys()
            
        # Asserts outside of "with fgm.local_scope():" - outisde features should be available. Features defined in "with" fgm should be removed by now
        assert "edge_feature_local" not in fgm.edata.keys()
        assert "1st_feature_local" not in fgm.srcdata.keys()
        assert "2nd_feature_local" not in fgm.dstdata.keys()
        for ntype in fgm.ntypes:
            assert ntype in fgm.srcdata["1st_feature"].keys()
            assert ntype in fgm.dstdata["2nd_feature"].keys()
        for etype in fgm.canonical_etypes:
            assert fgm[etype].srctypes[0] in fgm.srcdata[f"3rd_feature_{etype}"].keys()
            assert etype in fgm.edata["edge_feature"].keys()
        
    run_workers(local_scope, fixture_env, world_size=world_size, backend=backend)
    
    
@pytest.mark.parametrize("backend", ["ccl"])
@pytest.mark.parametrize("world_size", [2])
@sar_test
def test_mfg_local_scope(world_size, backend, fixture_env):
    """
    Test behaviour of local_scope function. Graph's srcdata/dstdata/ndata/edata
    should not be modified if the changes were made inside the "with local_scope" block.
    """
    def local_scope(mp_dict, rank, world_size, fixture_env, **kwargs):
        import torch
        from base_utils import initialize_worker, load_partition_data_mfg
            
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        blocks, _, _ = load_partition_data_mfg(rank, HETEROGENEOUS_GRAPH_NAME, os.path.join(temp_dir, f"heterogeneous_{world_size}"))
        block = blocks[0]
        
        block.edata["edge_feature"] = {etype: torch.ones(block.num_edges(etype), 5) for etype in block.canonical_etypes}
        block.srcdata["1st_feature"] = {srctype: torch.ones(block.num_src_nodes(srctype), 5) for srctype in block.srctypes}
        for dsttype in block.dsttypes:
            block.dstdata["2nd_feature"] = {dsttype: torch.ones(block.num_dst_nodes(dsttype), 10)}
        for etype in block.canonical_etypes:
            block[etype].srcdata[f"3rd_feature_{etype}"] = torch.zeros(block[etype].num_src_nodes(), 15)
        
        with block.local_scope():
            block.edata["edge_feature_local"] = {etype: torch.ones(block.num_edges(etype), 5) for etype in block.canonical_etypes}
            block.srcdata["1st_feature_local"] = {srctype: torch.ones(block.num_src_nodes(srctype), 5) for srctype in block.srctypes}
            for dsttype in block.dsttypes:
                block.dstdata["2nd_feature_local"] = {dsttype: torch.ones(block.num_dst_nodes(dsttype), 10)}
            for etype in block.canonical_etypes:
                block[etype].srcdata[f"3rd_feature_{etype}_local"] = torch.zeros(block[etype].num_src_nodes(), 15)
                
            # Asserts inside of "with block.local_scope():" - outisde and inside features should be available
            for ntype in block.ntypes:
                assert ntype in block.srcdata["1st_feature"].keys() and ntype in block.srcdata["1st_feature_local"].keys()
                assert ntype in block.dstdata["2nd_feature"].keys() and ntype in block.dstdata["2nd_feature_local"].keys()
            for etype in block.canonical_etypes:
                assert block[etype].srctypes[0] in block.srcdata[f"3rd_feature_{etype}"].keys() and block[etype].srctypes[0] in block.srcdata[f"3rd_feature_{etype}_local"].keys()
                assert etype in block.edata["edge_feature"].keys() and etype in block.edata["edge_feature_local"].keys()
            
        # Asserts outside of "with block.local_scope():" - outisde features should be available. Features defined in "with" block should be removed by now
        assert "edge_feature_local" not in block.edata.keys()
        assert "1st_feature_local" not in block.srcdata.keys()
        assert "2nd_feature_local" not in block.dstdata.keys()
        for ntype in block.ntypes:
            assert ntype in block.srcdata["1st_feature"].keys()
            assert ntype in block.dstdata["2nd_feature"].keys()
        for etype in block.canonical_etypes:
            assert block[etype].srctypes[0] in block.srcdata[f"3rd_feature_{etype}"].keys()
            assert etype in block.edata["edge_feature"].keys()
                
    run_workers(local_scope, fixture_env, world_size=world_size, backend=backend)


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_degrees(world_size, backend, fixture_env):
    """
    Validate outputs of `out_degrees` and `in_degrees` functions
    for all of the nodes in a partition
    """
    def degrees(mp_dict, rank, world_size, fixture_env, **kwargs):
        import torch
        from base_utils import initialize_worker, synchronize_processes, load_partition_data
            
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        if rank == 0:
            g = fixture_env.hetero_graph
            mp_dict["expected_out_degrees"] = {canonical_etype: g.out_degrees(etype=canonical_etype) for canonical_etype in g.canonical_etypes}
            mp_dict["expected_in_degrees"] = {canonical_etype: g.in_degrees(etype=canonical_etype) for canonical_etype in g.canonical_etypes}
            
        fgm, _, _ = load_partition_data(rank, HETEROGENEOUS_GRAPH_NAME, os.path.join(temp_dir, f"heterogeneous_{world_size}"))
        mp_dict[f"result_{rank}_out_degrees"] = {canonical_etype: fgm.out_degrees(etype=canonical_etype) for canonical_etype in fgm.canonical_etypes}
        mp_dict[f"result_{rank}_in_degrees"] = {canonical_etype: fgm.in_degrees(etype=canonical_etype) for canonical_etype in fgm.canonical_etypes}
        
        synchronize_processes()
        if rank == 0:
            node_map = fixture_env.node_map[f"heterogeneous_{world_size}"]
            out_degrees_concat = {}
            in_degrees_concat = {}
            for rel in fgm.canonical_etypes:
                rel_out_degrees_concat = torch.tensor([], dtype=torch.int64)
                rel_in_degrees_concat = torch.tensor([], dtype=torch.int64)
                for idx in range(world_size):
                    rel_out_degrees_concat = torch.cat((rel_out_degrees_concat, mp_dict[f"result_{idx}_out_degrees"][rel]))
                    rel_in_degrees_concat = torch.cat((rel_in_degrees_concat, mp_dict[f"result_{idx}_in_degrees"][rel]))
                out_degrees_concat[rel] = rel_out_degrees_concat
                out_degrees_concat[rel][node_map[rel[0]]] = out_degrees_concat[rel].clone()
                in_degrees_concat[rel] = rel_in_degrees_concat
                in_degrees_concat[rel][node_map[rel[2]]] = in_degrees_concat[rel].clone()
                
            for rel, expected_out in mp_dict["expected_out_degrees"].items():
                assert torch.all(expected_out == out_degrees_concat[rel])
                
            for rel, expected_in in mp_dict["expected_in_degrees"].items():
                assert torch.all(expected_in == in_degrees_concat[rel])
        
    run_workers(degrees, fixture_env, world_size=world_size, backend=backend)
