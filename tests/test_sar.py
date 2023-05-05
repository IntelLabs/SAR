from utils import *
import os
import scipy
import tempfile
import logging
import numpy as np
# Do not import DGL and SAR - these modules should be independent in each process


    
def sar_process(mp_dict, rank, world_size, tmp_dir):
    """
    This function should be an entry point to the 'independent' process.
    It has to simulate behaviour of SAR which will be spawned across different
    machines independently from other instances. Each process have individual memory space
    so it is suitable environment for testing SAR
    """
    import dgl
    import torch
    import sar
    from models import GNNModel

    try:
        if rank == 0:
            # partitioning takes place offline, however
            # for testing random graph is needed - only master node should do this
            # random graph partitions will be then placed in temporary directory
            graph = dgl.rand_graph(1000, 2500)
            graph = dgl.add_self_loop(graph)
            graph.ndata.clear()
            graph.ndata['features'] = torch.rand((graph.num_nodes(), 1))

            dgl.distributed.partition_graph(
                graph,
                'random_graph',
                2,
                tmp_dir,
                num_hops=1,
                balance_edges=True)

        part_file = os.path.join(tmp_dir, 'random_graph.json')
        ip_file = os.path.join(tmp_dir, 'ip_file')

        master_ip_address = sar.nfs_ip_init(rank, ip_file)

        sar.initialize_comms(rank,
                             world_size,
                             master_ip_address,
                             'ccl')

        torch.distributed.barrier() # wait for rank 0 to finish graph creation

        partition_data = sar.load_dgl_partition_data(
            part_file, rank, 'cpu')

        full_graph_manager = sar.construct_full_graph(partition_data).to('cpu')
        features = sar.suffix_key_lookup(partition_data.node_features, 'features')
        del partition_data

        model = GNNModel(features.size(1), 32, features.size(1)).to('cpu')
        sar.sync_params(model)

        logits = model(full_graph_manager, features)

        # put calculated results in multiprocessing dictionary
        mp_dict[f"result_{rank}"] = logits.detach()


        if rank == 0:
            # only rank 0 is runned within parent process
            # return used model and generated graph to caller
            return model, graph

    except Exception as e:
        mp_dict['traceback'] = str(traceback.format_exc())
        mp_dict['exception'] = e
        return None, None


def test_sar_full_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = mp.Manager()
        mp_dict = manager.dict()

        p = mp.Process(target=sar_process, args=(mp_dict, 1, 2, tmpdir))
        p.daemon = True
        p.start()

        model, graph = sar_process(mp_dict, 0, 2, tmpdir)
        p.join()

        if 'exception' in mp_dict:
            handle_mp_exception(mp_dict)

        out = model(graph, graph.ndata['features']).numpy()

        # compare mean of all values instead of each node feature individually
        # TODO: reorder SAR calculated logits to original NID mapping 
        full_graph_mean = out.mean()

        r0_logits = mp_dict["result_0"].numpy()
        r1_logits = mp_dict["result_1"].numpy()
        sar_logits_mean = np.concatenate((r0_logits, r1_logits)).mean()

        rtol = sar_logits_mean / 1000
        assert full_graph_mean == pytest.approx(sar_logits_mean, rtol)