from copy import deepcopy
import traceback
from multiprocessing_utils import *
# Do not import DGL and SAR - these modules should be
# independently loaded inside each process


@pytest.mark.parametrize('world_size', [2, 4, 8])
@sar_test
def test_sync_params(world_size):
    """
    Checks whether model's parameters are the same across all
    workers after calling sync_params function. Parameters of worker 0
    should be copied to all workers, so its parameters before and after
    sync_params should be the same
    """
    import torch
    def sync_params(mp_dict, rank, world_size, tmp_dir):
        import sar
        from tests.base_utils import initialize_worker
        from models import GNNModel
        try:
            initialize_worker(rank, world_size, tmp_dir)
            model = GNNModel(16, 4)
            if rank == 0:   
                mp_dict[f"result_{rank}"] = deepcopy(model.state_dict())
            sar.sync_params(model)
            if rank != 0:
                mp_dict[f"result_{rank}"] = model.state_dict()
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
        
    mp_dict = run_workers(sync_params, world_size)
    for rank in range(1, world_size):
        for key in mp_dict[f"result_0"].keys():
            assert torch.all(torch.eq(mp_dict[f"result_0"][key], mp_dict[f"result_{rank}"][key]))
        
    
@pytest.mark.parametrize('world_size', [2, 4, 8])
@sar_test
def test_gather_grads(world_size):
    """
    Checks whether parameter's gradients are the same across all
    workers after calling gather_grads function 
    """
    import torch
    def gather_grads(mp_dict, rank, world_size, tmp_dir):
        import sar
        import dgl
        import torch.nn.functional as F
        from models import GNNModel
        from base_utils import initialize_worker, get_random_graph, synchronize_processes,\
            load_partition_data
        try:
            initialize_worker(rank, world_size, tmp_dir)
            graph_name = 'dummy_graph'
            if rank == 0:
                g = get_random_graph()
                dgl.distributed.partition_graph(g, graph_name, world_size,
                                        tmp_dir, num_hops=1,
                                        balance_edges=True)
            synchronize_processes()
            fgm, feat, labels = load_partition_data(rank, graph_name, tmp_dir)
            model = GNNModel(feat.shape[1], labels.max()+1)
            sar.sync_params(model)
            sar_logits = model(fgm, feat)
            sar_loss = F.cross_entropy(sar_logits, labels)
            sar_loss.backward()
            sar.gather_grads(model)
            mp_dict[f"result_{rank}"] = [torch.tensor(x.grad) for x in model.parameters()]
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
        
    mp_dict = run_workers(gather_grads, world_size)
    for rank in range(1, world_size):
        for i in range(len(mp_dict["result_0"])):
            assert torch.all(torch.eq(mp_dict["result_0"][i], mp_dict[f"result_{rank}"][i]))
