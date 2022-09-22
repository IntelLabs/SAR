# Copyright (c) 2022 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Union, Dict
from argparse import ArgumentParser
import os
import logging
import psutil
import time
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import dgl  # type: ignore
from dgl.heterograph import DGLBlock  # type: ignore


import sar


parser = ArgumentParser(
    description="GNN training on node classification tasks in homogeneous graphs")


parser.add_argument(
    "--partitioning-json-file",
    type=str,
    default="",
    help="Path to the .json file containing partitioning information "
)

parser.add_argument('--ip-file', default='./ip_file', type=str,
                    help='File with ip-address. Worker 0 creates this file and all others read it ')


parser.add_argument('--backend', default='nccl', type=str, choices=['ccl', 'nccl', 'mpi'],
                    help='Communication backend to use '
                    )

parser.add_argument(
    "--cpu-run", action="store_true",
    help="Run on CPUs if set, otherwise run on GPUs "
)

parser.add_argument(
    "--precompute-batches", action="store_true",
    help="Precompute the batches "
)


parser.add_argument(
    "--optimized-batches-cache",
    type=str,
    default="",
    help="Prefix of the files used to store precomputed batches "
)


parser.add_argument('--train-iters', default=100, type=int,
                    help='number of training iterations ')

parser.add_argument(
    "--lr",
    type=float,
    default=1e-2,
    help="learning rate"
)


parser.add_argument('--rank', default=0, type=int,
                    help='Rank of the current worker ')

parser.add_argument('--world-size', default=2, type=int,
                    help='Number of workers ')

parser.add_argument('--hidden-layer-dim', default=256, type=int,
                    help='Dimension of GNN hidden layer')

parser.add_argument('--batch-size', default=5000, type=int,
                    help='per worker batch size ')

parser.add_argument('--num-workers', default=0, type=int,
                    help='number of dataloader workers ')

parser.add_argument('--fanout', nargs="+", type=int,
                    help='fanouts for sampling ')

parser.add_argument('--max-collective-size', default=0, type=int,
                    help='The maximum allowed size of the data in a collective. \
If a collective would communicate more than this maximum, it is split into multiple collectives.\
Collective calls with large data may cause instabilities in some communication backends  ')


class GNNModel(nn.Module):
    def __init__(self,  in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self.convs = nn.ModuleList([
            # pylint: disable=no-member
            dgl.nn.SAGEConv(in_dim, hidden_dim, aggregator_type='mean'),
            # pylint: disable=no-member
            dgl.nn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
            # pylint: disable=no-member
            dgl.nn.SAGEConv(hidden_dim, out_dim, aggregator_type='mean'),
        ])

    def forward(self,  blocks: List[Union[DGLBlock, sar.GraphShardManager]], features: torch.Tensor):
        for idx, conv in enumerate(self.convs):
            features = conv(blocks[idx], features)
            if idx < len(self.convs) - 1:
                features = F.relu(features, inplace=True)

        return features


def main():
    # psutil.Process().cpu_affinity([8])
    args = parser.parse_args()
    print('args', args)

    use_gpu = torch.cuda.is_available() and not args.cpu_run
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    if args.rank == -1:
        # Try to infer the worker's rank from environment variables
        # created by mpirun or similar MPI launchers
        args.rank = int(os.environ.get("PMI_RANK", -1))
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])

    # os.environ.putenv("GOMP_CPU_AFFINITY", "10,11")
    # os.environ.putenv("OMP_NUM_THREADS", "16")
    # Obtain the ip address of the master through the network file system
    master_ip_address = sar.nfs_ip_init(args.rank, args.ip_file)
    sar.initialize_comms(args.rank,
                         args.world_size, master_ip_address,
                         args.backend)

    # Load DGL partition data
    partition_data = sar.load_dgl_partition_data(
        args.partitioning_json_file, args.rank, torch.device('cpu'))

    # Obtain train,validation, and test masks
    # These are stored as node features. Partitioning may prepend
    # the node type to the mask names. So we use the convenience function
    # suffix_key_lookup to look up the mask name while ignoring the
    # arbitrary node type
    masks = {}
    for mask_name, indices_name in zip(['train_mask', 'val_mask', 'test_mask'],
                                       ['train_indices', 'val_indices', 'test_indices']):
        boolean_mask = sar.suffix_key_lookup(partition_data.node_features,
                                             mask_name)
        masks[indices_name] = boolean_mask.nonzero(
            as_tuple=False).view(-1).to(device)
        print(f'mask {indices_name} : {masks[indices_name]} ')
    labels = sar.suffix_key_lookup(partition_data.node_features,
                                   'labels').long().to(device)

    # Obtain the number of classes by finding the max label across all workers
    num_labels = labels.max() + 1
    sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX,
                        move_to_comm_device=True)
    num_labels = num_labels.item()

    features = sar.suffix_key_lookup(
        partition_data.node_features, 'features')  # keep features always on CPU
    full_graph_manager = sar.construct_full_graph(
        partition_data)  # Keep full graph on CPU

    node_ranges = partition_data.node_ranges
    del partition_data

    gnn_model = GNNModel(features.size(1),
                         args.hidden_layer_dim,
                         num_labels).to(device)

    # gnn_model_cpu will be used for inference
    if use_gpu:
        gnn_model_cpu = GNNModel(features.size(1),
                                 args.hidden_layer_dim,
                                 num_labels)
    else:
        gnn_model_cpu = gnn_model
    print('model', gnn_model)

    # Synchronize the model parmeters across all workers
    sar.sync_params(gnn_model)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr)

    neighbor_sampler = sar.core.sampling.DistNeighborSampler([15, 10, 5],
                                                             input_node_features={
                                                                 'features': features},
                                                             output_node_features={
                                                                 'labels': labels},
                                                             output_device=device
                                                             )

    train_nodes = masks['train_indices'] + node_ranges[sar.rank()][0]

    dataloader = sar.core.sampling.DataLoader(
        full_graph_manager,
        train_nodes,
        neighbor_sampler,
        args.batch_size,
        shuffle=True,
        precompute_optimized_batches=args.precompute_batches,
        optimized_batches_cache=(
            args.optimized_batches_cache if args.optimized_batches_cache else None),
        num_workers=args.num_workers)

    print('sampling graph edata', full_graph_manager.sampling_graph.edata)

    for k in list(masks.keys()):
        masks[k] = masks[k].to(device)

    for train_iter_idx in range(args.train_iters):
        total_loss = 0
        gnn_model.train()
        sar.Config.max_collective_size = 0

        train_t1 = dataloader_t1 = time.time()
        n_total = n_correct = 0
        pure_training_time = 0
        for block_idx, blocks in enumerate(dataloader):
            start_t1 = time.time()
            loading_time = time.time() - dataloader_t1
            print(f'in block {block_idx} : {blocks}. Loaded in {loading_time}')
            # print('block edata', [block.edata[dgl.EID] for block in blocks])
            blocks = [b.to(device) for b in blocks]
            block_features = blocks[0].srcdata['features']
            block_labels = blocks[-1].dstdata['labels']
            logits = gnn_model(blocks, block_features)

            loss = F.cross_entropy(logits, block_labels, reduction='mean')
            n_correct += (logits.argmax(1) ==
                          block_labels).float().sum()
            n_total += len(block_labels)

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # Do not forget to gather the parameter gradients from all workers
            tg = time.time()
            sar.gather_grads(gnn_model)
            print('gather grad time', time.time() - tg)
            optimizer.step()
            dataloader_t1 = time.time()
            pure_training_time += (time.time() - start_t1)

        train_time = time.time() - train_t1
        print('train time', train_time, flush=True)
        print('pure train time', pure_training_time, flush=True)

        print('loss', total_loss, flush=True)
        print('accuracy ', n_correct/n_total, flush=True)

        # Full graph inference is done on CPUs using sequential
        # aggregation and re-materialization
        gnn_model_cpu.eval()
        if gnn_model_cpu is not gnn_model:
            gnn_model_cpu.load_state_dict(gnn_model.state_dict())
        sar.Config.max_collective_size = args.max_collective_size
        with torch.no_grad():
            # Calculate accuracy for train/validation/test
            logits = gnn_model_cpu(
                [full_graph_manager] * 3, features)
            results = []
            for indices_name in ['train_indices', 'val_indices', 'test_indices']:
                n_correct = (logits[masks[indices_name]].argmax(1) ==
                             labels[masks[indices_name]].cpu()).float().sum()
                results.extend([n_correct, masks[indices_name].numel()])

            acc_vec = torch.FloatTensor(results)
            # Sum the n_correct, and number of mask elements across all workers
            sar.comm.all_reduce(acc_vec, op=dist.ReduceOp.SUM,
                                move_to_comm_device=True)
            (train_acc, val_acc, test_acc) = (acc_vec[0] / acc_vec[1],
                                              acc_vec[2] / acc_vec[3],
                                              acc_vec[4] / acc_vec[5])

            result_message = (
                f"iteration [{train_iter_idx}/{args.train_iters}] | "
            )
            result_message += ', '.join([
                f"Accuracy: "
                f"train={train_acc:.4f} "
                f"valid={val_acc:.4f} "
                f"test={test_acc:.4f} "
                f" | train time = {train_time} "
                f" |"
            ])
            print(result_message, flush=True)


if __name__ == '__main__':
    main()
