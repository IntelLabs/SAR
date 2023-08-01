from copy import deepcopy
import traceback
from multiprocessing_utils import *
# Do not import DGL and SAR - these modules should be
# independently loaded inside each process


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize('world_size', [2, 4, 8])
@sar_test
def test_sync_params(world_size, backend):
    """
    Checks whether model's parameters are the same across all
    workers after calling sync_params function. Parameters of worker 0
    should be copied to all workers, so its parameters before and after
    sync_params should be the same
    """
    import torch
    def sync_params(mp_dict, rank, world_size, tmp_dir, **kwargs):
        import sar
        from tests.base_utils import initialize_worker
        from models import GNNModel
        try:
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
            model = GNNModel(16, 4)
            if rank == 0:   
                mp_dict[f"result_{rank}"] = deepcopy(model.state_dict())
            sar.sync_params(model)
            if rank != 0:
                mp_dict[f"result_{rank}"] = model.state_dict()
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
        
    mp_dict = run_workers(sync_params, world_size, backend=backend)
    for rank in range(1, world_size):
        for key in mp_dict[f"result_0"].keys():
            assert torch.all(torch.eq(mp_dict[f"result_0"][key], mp_dict[f"result_{rank}"][key]))
        
    
@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize('world_size', [2, 4, 8])
@sar_test
def test_gather_grads(world_size, backend):
    """
    Checks whether parameter's gradients are the same across all
    workers after calling gather_grads function 
    """
    import torch
    def gather_grads(mp_dict, rank, world_size, tmp_dir, **kwargs):
        import sar
        import dgl
        import torch.nn.functional as F
        from models import GNNModel
        from base_utils import initialize_worker, get_random_graph, synchronize_processes,\
            load_partition_data
        try:
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
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
        
    mp_dict = run_workers(gather_grads, world_size, backend=backend)
    for rank in range(1, world_size):
        for i in range(len(mp_dict["result_0"])):
            assert torch.all(torch.eq(mp_dict["result_0"][i], mp_dict[f"result_{rank}"][i]))


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_all_to_all(world_size, backend):
    """
    Checks whether all_to_all operation works as expected. Test is
    designed is such a way, that after calling all_to_all, each worker
    should receive a list of tensors with values equal to their rank
    """
    import torch
    def all_to_all(mp_dict, rank, world_size, tmp_dir, **kwargs):
        import sar
        from base_utils import initialize_worker
        try:
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
            send_tensors_list = [torch.tensor([x] * world_size) for x in range(world_size)]
            recv_tensors_list = [torch.tensor([-1] * world_size) for _ in range(world_size)]
            sar.comm.all_to_all(recv_tensors_list, send_tensors_list)
            mp_dict[f"result_{rank}"] = recv_tensors_list
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
    
    mp_dict = run_workers(all_to_all, world_size, backend=backend)
    for rank in range(world_size):
        for tensor in mp_dict[f"result_{rank}"]:
            assert torch.all(torch.eq(tensor, torch.tensor([rank] * world_size)))


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_exchange_single_tensor(world_size, backend):
    def exchange_single_tensor(mp_dict, rank, world_size, tmp_dir, **kwargs):
        import torch
        import sar
        from base_utils import initialize_worker
        try:
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
            send_idx = rank
            recv_idx = rank
            for _ in range(world_size):
                send_tensor = torch.tensor([send_idx] * world_size)
                recv_tensor = torch.tensor([-1] * world_size)
                sar.comm.exchange_single_tensor(recv_idx, send_idx, recv_tensor, send_tensor)
                assert torch.all(torch.eq(recv_tensor, torch.tensor([rank] * world_size)))
                send_idx = (send_idx + 1) % world_size
                recv_idx = (recv_idx - 1) % world_size
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
        
    mp_dict = run_workers(exchange_single_tensor, world_size, backend=backend)
