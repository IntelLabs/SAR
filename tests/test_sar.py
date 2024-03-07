from multiprocessing_utils import *
from constants import *
import os
# Do not import DGL and SAR - these modules should be
# independently loaded inside each process


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize('world_size', [1, 2, 4, 8])
@sar_test
def test_homogeneous_fgm(world_size, backend, fixture_env):
    """
    Perform full graph inference using SAR algorithm on homogeneous graph.
    Test is comparing mean of concatenated results from all processes
    with mean of native DGL full graph inference result.
    """
    import torch
    def homogeneous_fgm(mp_dict, rank, world_size, fixture_env, **kwargs):
        import sar
        from models import GNNModel
        from base_utils import initialize_worker, load_partition_data

        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        fgm, feat, labels = load_partition_data(rank, HOMOGENEOUS_GRAPH_NAME,
                                                os.path.join(temp_dir, f"homogeneous_{world_size}"))
        model = GNNModel(feat.shape[1], feat.shape[1], labels.max()+1).to('cpu')
        sar.sync_params(model)
        model.eval()
        sar_logits = model(fgm, feat)

        mp_dict[f"result_{rank}"] = sar_logits.detach()
        if rank == 0:
            mp_dict["model"] = model
            mp_dict["graph"] = fixture_env.homo_graph
            mp_dict["node_map"] = fixture_env.node_map[f"homogeneous_{world_size}"]

    mp_dict = run_workers(homogeneous_fgm, fixture_env, world_size, backend=backend)
    
    model = mp_dict["model"]
    graph = mp_dict["graph"]
    dgl_logits = model(graph, graph.ndata['features']).detach()
    dgl_logits_mean = dgl_logits.mean(axis=1)

    sar_logits = torch.tensor([])
    for rank in range(world_size):
        sar_logits = torch.cat((sar_logits, mp_dict[f"result_{rank}"]))
    sar_logits[mp_dict["node_map"]] = sar_logits.clone()
    sar_logits_mean = sar_logits.mean(axis=1)

    assert torch.all(torch.isclose(dgl_logits_mean, sar_logits_mean, atol=1e-6, rtol=1e-6))

@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize('world_size', [1, 2, 4, 8])
@sar_test
def test_heterogeneous_fgm(world_size, backend, fixture_env):
    """
    Perform full graph inference using SAR algorithm on heterogeneous graph.
    Test is comparing mean of concatenated results from all processes
    with mean of native DGL full graph inference result.
    """
    import torch
    from models import rel_graph_embed
    def heterogeneous_fgm(mp_dict, rank, world_size, fixture_env, **kwargs):
        import sar
        from models import HeteroGNNModel, extract_embed
        from base_utils import initialize_worker, load_partition_data

        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        fgm, feats, labels = load_partition_data(rank, HETEROGENEOUS_GRAPH_NAME,
                                                os.path.join(temp_dir, f"heterogeneous_{world_size}"))
        model = HeteroGNNModel(fgm, feats.shape[1], feats.shape[1], labels.max()+1).to('cpu')
        model.eval()
        sar.sync_params(model)
        
        to_extract = {}
        node_map = fixture_env.node_map[f"heterogeneous_{world_size}"]
        for ntype in fgm.srctypes:
            if ntype == "n_type_1":
                continue
            down_lim = fgm._partition_book.partid2nids(rank, ntype).min()
            up_lim = fgm._partition_book.partid2nids(rank+1, ntype).min() if rank+1 < world_size else None
            ids = node_map[ntype][down_lim:up_lim]
            to_extract[ntype] = ids
        
        embed_layer = kwargs["embed_layer"]
        embeds = extract_embed(embed_layer, to_extract, skip_type="n_type_1")
        embeds.update({"n_type_1": feats[fgm.srcnodes("n_type_1")]})
        embeds = {k: e.to("cpu") for k, e in embeds.items()}
        
        sar_logits = model(fgm, embeds)
        sar_logits = sar_logits["n_type_1"]

        mp_dict[f"result_{rank}"] = sar_logits.detach()
        if rank == 0:
            mp_dict["model"] = model
            mp_dict["node_map"] = fixture_env.node_map[f"heterogeneous_{world_size}"]

    graph = fixture_env.hetero_graph
    max_num_nodes = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
    embed_layer = rel_graph_embed(graph, graph.ndata["features"]["n_type_1"].shape[1],
                                  num_nodes_dict=max_num_nodes,
                                  skip_type="n_type_1").to('cpu')
    mp_dict = run_workers(heterogeneous_fgm, fixture_env, world_size, backend=backend,
                          embed_layer=embed_layer)
    
    embeds = embed_layer.weight
    embeds.update({"n_type_1": graph.ndata["features"]["n_type_1"]})
    embeds = {k: e.to("cpu") for k, e in embeds.items()}
    
    model = mp_dict["model"]
    dgl_logits = model(graph, embeds)
    dgl_logits = dgl_logits["n_type_1"].detach()
    dgl_logits_mean = dgl_logits.mean(axis=1)

    sar_logits = torch.tensor([])
    for rank in range(world_size):
        sar_logits = torch.cat((sar_logits, mp_dict[f"result_{rank}"]))
    sar_logits[mp_dict["node_map"]["n_type_1"]] = sar_logits.clone()
    sar_logits_mean = sar_logits.mean(axis=1)
    
    assert torch.all(torch.isclose(dgl_logits_mean, sar_logits_mean, atol=1e-6, rtol=1e-6))


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@sar_test
def test_homogeneous_mfg(world_size, backend, fixture_env):
    """
    Perform full graph inference using SAR algorithm on homogeneous graph.
    Script is using Message Flow Graph (mfg). Test is comparing mean of
    concatenated results from all processes with mean of native DGL full
    graph inference result.
    """
    import torch
    def homogeneous_mfg(mp_dict, rank, world_size, fixture_env, **kwargs):
        import sar
        from models import GNNModel
        from base_utils import initialize_worker, load_partition_data_mfg
        
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        blocks, feat, labels = load_partition_data_mfg(rank, HOMOGENEOUS_GRAPH_NAME,
                                                       os.path.join(temp_dir, f"homogeneous_{world_size}"))
        model = GNNModel(feat.shape[1], feat.shape[1], labels.max()+1).to('cpu')
        sar.sync_params(model)
        model.eval()
        sar_logits = model(blocks, feat)

        mp_dict[f"result_{rank}"] = sar_logits.detach()
        if rank == 0:
            mp_dict["model"] = model
            mp_dict["graph"] = fixture_env.homo_graph
            mp_dict["node_map"] = fixture_env.node_map[f"homogeneous_{world_size}"]
            
    mp_dict = run_workers(homogeneous_mfg, fixture_env, world_size, backend=backend)

    model = mp_dict["model"]
    graph = mp_dict["graph"]
    dgl_logits = model(graph, graph.ndata['features']).detach()
    dgl_logits_mean = dgl_logits.mean(axis=1)

    sar_logits = torch.tensor([])
    for rank in range(world_size):
        sar_logits = torch.cat((sar_logits, mp_dict[f"result_{rank}"]))
    sar_logits[mp_dict["node_map"]] = sar_logits.clone()
    sar_logits_mean = sar_logits.mean(axis=1)

    assert torch.all(torch.isclose(dgl_logits_mean, sar_logits_mean, atol=1e-6, rtol=1e-6))


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize('world_size', [1, 2, 4, 8])
@sar_test
def test_heterogeneous_mfg(world_size, backend, fixture_env):
    """
    Perform full graph inference using SAR algorithm on heterogeneous graph.
    Script is using Message Flow Graph (mfg). Test is comparing mean of
    concatenated results from all processes with mean of native DGL full
    graph inference result.
    """
    import torch
    from models import rel_graph_embed
    def heterogeneous_mfg(mp_dict, rank, world_size, fixture_env, **kwargs):
        import sar
        from models import HeteroGNNModel, extract_embed
        from base_utils import initialize_worker, load_partition_data_mfg

        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])

        blocks, feats, labels = load_partition_data_mfg(rank, HETEROGENEOUS_GRAPH_NAME,
                                                        os.path.join(temp_dir, f"heterogeneous_{world_size}"))
        model = HeteroGNNModel(blocks[0], feats.shape[1], feats.shape[1], labels.max()+1).to('cpu')
        model.eval()
        sar.sync_params(model)
        
        to_extract = {}
        node_map = fixture_env.node_map[f"heterogeneous_{world_size}"]
        for ntype in blocks[0].srctypes:
            if ntype == "n_type_1":
                continue
            down_lim = blocks[0]._partition_book.partid2nids(rank, ntype).min()
            up_lim = blocks[0]._partition_book.partid2nids(rank+1, ntype).min() if rank+1 < world_size else None
            ids = node_map[ntype][down_lim:up_lim]
            to_extract[ntype] = ids[blocks[0].srcnodes(ntype)]
        
        embed_layer = kwargs["embed_layer"]
        embeds = extract_embed(embed_layer, to_extract, skip_type="n_type_1")
        embeds.update({"n_type_1": feats[blocks[0].srcnodes("n_type_1")]})
        embeds = {k: e.to("cpu") for k, e in embeds.items()}
        
        sar_logits = model(blocks, embeds)
        sar_logits = sar_logits["n_type_1"]

        mp_dict[f"result_{rank}"] = sar_logits.detach()
        if rank == 0:
            mp_dict["model"] = model
            mp_dict["node_map"] = fixture_env.node_map[f"heterogeneous_{world_size}"]

    graph = fixture_env.hetero_graph
    max_num_nodes = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
    embed_layer = rel_graph_embed(graph, graph.ndata["features"]["n_type_1"].shape[1],
                                  num_nodes_dict=max_num_nodes,
                                  skip_type="n_type_1").to('cpu')
    mp_dict = run_workers(heterogeneous_mfg, fixture_env, world_size, backend=backend,
                          embed_layer=embed_layer)

    embeds = embed_layer.weight
    embeds.update({"n_type_1": graph.ndata["features"]["n_type_1"]})
    embeds = {k: e.to("cpu") for k, e in embeds.items()}
    
    model = mp_dict["model"]
    dgl_logits = model(graph, embeds)
    dgl_logits = dgl_logits["n_type_1"].detach()
    dgl_logits_mean = dgl_logits.mean(axis=1)

    sar_logits = torch.tensor([])
    for rank in range(world_size):
        sar_logits = torch.cat((sar_logits, mp_dict[f"result_{rank}"]))
    sar_logits[mp_dict["node_map"]["n_type_1"]] = sar_logits.clone()
    sar_logits_mean = sar_logits.mean(axis=1)
    
    assert torch.all(torch.isclose(dgl_logits_mean, sar_logits_mean, atol=1e-6, rtol=1e-6))


@pytest.mark.parametrize("backend", ["ccl"])
@pytest.mark.parametrize('world_size', [1])
@sar_test
def test_convert_dist_graph(world_size, backend, fixture_env):
    """
    Create DGL's DistGraph object with random graph partitioned into
    one part (only way to test DistGraph locally). Then perform converting
    DistGraph into SAR GraphShardManager and check relevant properties.
    """
    def convert_dist_graph(mp_dict, rank, world_size, fixture_env, **kwargs):
        import dgl
        import sar
        from base_utils import initialize_worker

        temp_dir = fixture_env.temp_dir
        partition_file = os.path.join(temp_dir, f'homogeneous_{world_size}', f"{HOMOGENEOUS_GRAPH_NAME}.json")
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        dgl.distributed.initialize("kv_ip_config.txt")
        dist_g = dgl.distributed.DistGraph(
            HOMOGENEOUS_GRAPH_NAME, part_config=partition_file)

        sar_g = sar.convert_dist_graph(dist_g)
        assert len(sar_g.graph_shard_managers[0].graph_shards) == dist_g.get_partition_book().num_partitions()
        assert dist_g.num_edges() == sar_g.num_edges()
        assert dist_g.num_nodes() == sar_g.num_nodes()
        assert dist_g.ntypes == sar_g.ntypes
        assert dist_g.etypes == sar_g.etypes

    run_workers(convert_dist_graph, fixture_env, world_size, backend=backend)
