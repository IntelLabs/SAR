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

from argparse import ArgumentParser
from model import EntityClassify, rel_graph_embed, extract_embed
import time
import itertools
import torch
import torch.nn.functional as F
import torch.distributed as dist
import dgl  # type: ignore

import sar


parser = ArgumentParser(
    description="GNN training on node classification tasks in heterogenous graphs (MFG)")


parser.add_argument("--partitioning-json-file", type=str, default="",
    help="Path to the .json file containing partitioning information")

parser.add_argument("--ip-file", default="./ip_file", type=str, 
                    help="File with ip-address. Worker 0 creates this file and all others read it")

parser.add_argument("--backend", default="nccl", type=str, choices=["ccl", "nccl", "mpi", "gloo"],
                    help="Communication backend to use")

parser.add_argument("--cpu-run", action="store_true",
                    help="Run on CPUs if set, otherwise run on GPUs")

parser.add_argument("--train-iters", default=60, type=int,
                    help="number of training iterations")

parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")

parser.add_argument("--rank", default=0, type=int,
                    help="Rank of the current worker")

parser.add_argument("--world-size", default=2, type=int,
                    help="Number of workers")

parser.add_argument("--features-dim", default=128, type=int,
                    help="Dimension of the node features")


def main():
    args = parser.parse_args()
    print('args', args)

    # Patch DGL's attention-based layers and RelGraphConv to support distributed graphs
    sar.patch_dgl()
    
    use_gpu = torch.cuda.is_available() and not args.cpu_run
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Obtain the ip address of the master through the network file system
    master_ip_address = sar.nfs_ip_init(args.rank, args.ip_file)
    sar.initialize_comms(args.rank,
                         args.world_size, master_ip_address,
                         args.backend)

    # Load DGL partition data
    partition_data = sar.load_dgl_partition_data(
        args.partitioning_json_file, args.rank, device)
    partition_data.node_features["paper/features"] = partition_data.node_features["paper/features"].float()

    # Obtain train,validation, and test masks
    # These are stored as node features. Partitioning may prepend
    # the node type to the mask names. So we use the convenience function
    # suffix_key_lookup to look up the mask name while ignoring the
    # arbitrary node type
    #The train/val/test masks are only defined for nodes with type 'paper'.
    #We set the ``expand_to_all`` flag  to expand the mask to all nodes in the
    #graph (mask will be filled with zeros). We use the expand_all option when
    #loading other node-type specific tensors such as features and labels
    
    bool_masks = {}
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        local_mask = sar.suffix_key_lookup(partition_data.node_features,
                                           mask_name,
                                           expand_to_all = False,
                                           type_list = partition_data.node_type_names)
        bool_masks[mask_name] = local_mask.bool()

    labels = sar.suffix_key_lookup(partition_data.node_features,
                                   'labels',
                                   expand_to_all = False,
                                   type_list = partition_data.node_type_names).long().to(device)

    # Obtain the number of classes by finding the max label across all workers
    num_labels = labels.max() + 1
    sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True)
    num_labels = num_labels.item()
    
    features = sar.suffix_key_lookup(partition_data.node_features,
                                     'features',
                                     type_list = partition_data.node_type_names                                     
                                     ).to(device)
    
    full_graph_manager = sar.construct_full_graph(partition_data).to(device)
    
    max_num_nodes = {}
    for ntype in partition_data.partition_book.ntypes:
        nodes = full_graph_manager.srcnodes(ntype)
        nodes_max = nodes.max()
        max_num_nodes[ntype] = nodes_max + 1
    
    embed_layer = rel_graph_embed(full_graph_manager, args.features_dim, max_num_nodes).to(device)
    gnn_model = EntityClassify(full_graph_manager, args.features_dim, num_labels).to(device)
    
    print('model', gnn_model)
    embed_layer.reset_parameters()
    gnn_model.reset_parameters()

    # Synchronize the model parmeters across all workers
    sar.sync_params(gnn_model)

    # Obtain the number of labeled nodes in the training
    # This will be needed to properly obtain a cross entropy loss
    # normalized by the number of training examples
    n_train_points = torch.LongTensor([bool_masks["train_mask"].sum().item()])
    sar.comm.all_reduce(n_train_points, op=dist.ReduceOp.SUM, move_to_comm_device=True)
    n_train_points = n_train_points.item()

    all_params = itertools.chain(
            gnn_model.parameters(), embed_layer.parameters()
        )
    optimizer = torch.optim.Adam(all_params, lr=args.lr)
    
    for train_iter_idx in range(args.train_iters):
        # Train
        t_1 = time.time()
        gnn_model.train()
        
        embeds = extract_embed(embed_layer, {ntype: full_graph_manager.srcnodes(ntype) for ntype in full_graph_manager.srctypes})
        embeds.update({"paper": features[full_graph_manager.srcnodes("paper")]})
        embeds = {k: e.to(device) for k, e in embeds.items()}
    
        logits = gnn_model(full_graph_manager, embeds)
        logits = logits["paper"].log_softmax(dim=-1)
        train_mask = bool_masks["train_mask"]
        loss = F.nll_loss(logits[train_mask], labels[train_mask], reduction="sum") / n_train_points

        optimizer.zero_grad()
        loss.backward()
        # Do not forget to gather the parameter gradients from all workers
        sar.gather_grads(gnn_model)
        optimizer.step()
        train_time = time.time() - t_1

        # Calculate accuracy for train/validation/test
        results = []
        gnn_model.eval()
        with torch.no_grad():
            embeds = extract_embed(embed_layer, {ntype: full_graph_manager.srcnodes(ntype) for ntype in full_graph_manager.srctypes})
            embeds.update({"paper": features[full_graph_manager.srcnodes("paper")]})
            embeds = {k: e.to(device) for k, e in embeds.items()}
        
            logits = gnn_model(full_graph_manager, embeds)
            logits = logits["paper"].log_softmax(dim=-1)
            
            for mask_name in ['train_mask', 'val_mask', 'test_mask']:
                masked_nodes = bool_masks[mask_name]
                if masked_nodes.sum() > 0:
                    active_logits = logits[masked_nodes]
                    active_labels = labels[masked_nodes]
                    loss = F.nll_loss(active_logits, active_labels, reduction="sum")
                    n_correct = (active_logits.argmax(1) == active_labels).float().sum()
                    results.extend([loss.item(), n_correct.item(), masked_nodes.sum().item()])
                else:
                    results.extend([0.0, 0.0, 0.0])
            
        loss_acc_vec = torch.FloatTensor(results)
        # Sum the n_correct, and number of mask elements across all workers
        sar.comm.all_reduce(loss_acc_vec, op=dist.ReduceOp.SUM, move_to_comm_device=True)
        (train_loss, train_acc, val_loss, val_acc, test_loss, test_acc) = \
            (loss_acc_vec[0] / loss_acc_vec[2],
            loss_acc_vec[1] / loss_acc_vec[2],
            loss_acc_vec[3] / loss_acc_vec[5],
            loss_acc_vec[4] / loss_acc_vec[5],
            loss_acc_vec[6] / loss_acc_vec[8],
            loss_acc_vec[7] / loss_acc_vec[8])

        result_message = (
            f"iteration [{train_iter_idx}/{args.train_iters}] | "
        )
        result_message += ', '.join([
            f"train loss={train_loss:.4f}, "
            f"Accuracy: "
            f"train={100 * train_acc:.4f} "
            f"valid={100 * val_acc:.4f} "
            f"test={100 * test_acc:.4f} "
            f" | train time = {train_time} "
            f" |"
        ])
        print(result_message, flush=True)


if __name__ == '__main__':
    main()
