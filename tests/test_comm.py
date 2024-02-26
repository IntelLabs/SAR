import os
from copy import deepcopy
from multiprocessing_utils import *
from constants import *
# Do not import DGL and SAR - these modules should be
# independently loaded inside each process


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize('world_size', [2, 4, 8])
@sar_test
def test_sync_params(world_size, backend, fixture_env):
    """
    Checks whether model's parameters are the same across all
    workers after calling sync_params function. Parameters of worker 0
    should be copied to all workers, so its parameters before and after
    sync_params should be the same
    """
    def sync_params(mp_dict, rank, world_size, fixture_env, **kwargs):
        import torch
        import sar
        from base_utils import initialize_worker, synchronize_processes
        from models import GNNModel
        
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        model = GNNModel(16, 8, 4)
        if rank == 0:   
            mp_dict[f"result_{rank}"] = deepcopy(model.state_dict())
        sar.sync_params(model)
        if rank != 0:
            mp_dict[f"result_{rank}"] = model.state_dict()

        synchronize_processes()
        for rank in range(1, world_size):
            for key in mp_dict[f"result_0"].keys():
                assert torch.all(torch.eq(mp_dict[f"result_0"][key], mp_dict[f"result_{rank}"][key]))
        
    run_workers(sync_params, fixture_env, world_size, backend=backend)
    
    
@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize('world_size', [2, 4, 8])
@sar_test
def test_gather_grads(world_size, backend, fixture_env):
    """
    Checks whether parameter's gradients are the same across all
    workers after calling gather_grads function 
    """
    def gather_grads(mp_dict, rank, world_size, fixture_env, **kwargs):
        import torch
        import sar
        import torch.nn.functional as F
        from models import GNNModel
        from base_utils import initialize_worker, synchronize_processes, load_partition_data
                 
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        fgm, feat, labels = load_partition_data(rank, HOMOGENEOUS_GRAPH_NAME,
                                                os.path.join(temp_dir, f"homogeneous_{world_size}"))
        model = GNNModel(feat.shape[1], feat.shape[1], labels.max()+1)
        sar.sync_params(model)
        sar_logits = model(fgm, feat)
        sar_loss = F.cross_entropy(sar_logits, labels)
        sar_loss.backward()
        sar.gather_grads(model)
        mp_dict[f"result_{rank}"] = [torch.tensor(x.grad) for x in model.parameters()]
        
        synchronize_processes()
        for rank in range(1, world_size):
            for i in range(len(mp_dict["result_0"])):
                assert torch.all(torch.eq(mp_dict["result_0"][i], mp_dict[f"result_{rank}"][i]))
        
    run_workers(gather_grads, fixture_env, world_size, backend=backend)


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_all_to_all(world_size, backend, fixture_env):
    """
    Checks whether all_to_all operation works as expected. Test is
    designed is such a way, that after calling all_to_all, each worker
    should receive a list of tensors with values equal to their rank
    """
    def all_to_all(mp_dict, rank, world_size, fixture_env, **kwargs):
        import torch
        import sar
        from base_utils import initialize_worker, synchronize_processes
        
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        send_tensors_list = [torch.tensor([x] * world_size) for x in range(world_size)]
        recv_tensors_list = [torch.tensor([-1] * world_size) for _ in range(world_size)]
        sar.comm.all_to_all(recv_tensors_list, send_tensors_list)
        mp_dict[f"result_{rank}"] = recv_tensors_list
    
        synchronize_processes()
        for rank in range(world_size):
            for tensor in mp_dict[f"result_{rank}"]:
                assert torch.all(torch.eq(tensor, torch.tensor([rank] * world_size)))

    run_workers(all_to_all, fixture_env, world_size, backend=backend)
    

@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_exchange_single_tensor(world_size, backend, fixture_env):
    """
    Checks whether exchange_single_tensor operation works as expected. Test is
    designed is such a way, that after calling exchange_single_tensor between two machines,
    machine should recive a tensor with values equal to their rank
    """
    def exchange_single_tensor(mp_dict, rank, world_size, fixture_env, **kwargs):
        import torch
        import sar
        from base_utils import initialize_worker, synchronize_processes
        
        temp_dir = fixture_env.temp_dir
        initialize_worker(rank, world_size, temp_dir, backend=kwargs["backend"])
        send_idx = rank
        recv_idx = rank
        results = []
        for _ in range(world_size):
            send_tensor = torch.tensor([send_idx] * world_size)
            recv_tensor = torch.tensor([-1] * world_size)
            sar.comm.exchange_single_tensor(recv_idx, send_idx, recv_tensor, send_tensor)
            results.append(recv_tensor)
            send_idx = (send_idx + 1) % world_size
            recv_idx = (recv_idx - 1) % world_size
        mp_dict[f"result_{rank}"] = results
        
        synchronize_processes()
        for recv_tensor in mp_dict[f"result_{rank}"]:
            assert torch.all(torch.eq(recv_tensor, torch.tensor([rank] * world_size)))
        
    run_workers(exchange_single_tensor, fixture_env, world_size, backend=backend)
