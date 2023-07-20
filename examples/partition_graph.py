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
import dgl  # type:ignore
import torch
from ogb.nodeproppred import DglNodePropPredDataset  # type:ignore
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
)

SUPPORTED_DATASETS = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-products": DglNodePropPredDataset,
    "ogbn-arxiv": DglNodePropPredDataset,
    "ogbn-mag": DglNodePropPredDataset,
}

parser = ArgumentParser(description="Graph partitioning for ogbn-arxiv and ogbn-products")

parser.add_argument("--dataset-root", type=str, default="./datasets/",
                    help="The OGB datasets folder")

parser.add_argument("--dataset-name", type=str, default="ogbn-arxiv",
                    choices=["ogbn-arxiv", "ogbn-products", "ogbn-mag",
                             "cora", "citeseer", "pubmed"],
                    help="Dataset name. ogbn-arxiv or ogbn-products")

parser.add_argument("--partition-out-path", type=str, default="./partition_data/",
                    help="Path to the output directory for the partition data")

parser.add_argument("--num-partitions", type=int, default=2,
                    help="Number of graph partitions to generate")

def get_dataset(args):
    dataset_name = args.dataset_name
    if dataset_name in ["cora", "citeseer", "pubmed"]:
        return SUPPORTED_DATASETS[dataset_name](args.dataset_root)
    else:
        return SUPPORTED_DATASETS[dataset_name](dataset_name, args.dataset_root)

def prepare_features(args, dataset, graph):
    if args.dataset_name in ["cora", "citeseer", "pubmed"]:
        assert all([x in graph.ndata.keys() for x in ["train_mask", "val_mask", "test_mask"]])
        return

    split_idx = dataset.get_idx_split()
    ntype = "paper" if args.dataset_name == "ogbn-mag" else None

    def idx_to_mask(idx_tensor):
        mask = torch.BoolTensor(graph.number_of_nodes(ntype)).fill_(False)
        if ntype:
            mask[idx_tensor[ntype]] = True
        else:
            mask[idx_tensor] = True
        return mask

    train_mask, val_mask, test_mask = map(
        idx_to_mask, [split_idx["train"], split_idx["valid"], split_idx["test"]])

    if "feat" in graph.ndata.keys():
        features = graph.ndata["feat"]
    else:
        features = graph.ndata["features"]

    graph.ndata.clear()

    labels = dataset[0][1]
    if ntype:
        features = features[ntype]
        labels = labels[ntype]
    labels = labels.view(-1)

    for name, val in zip(["train_mask", "val_mask", "test_mask", "labels", "features"],
                         [train_mask, val_mask, test_mask, labels, features]):
        graph.ndata[name] = {ntype: val} if ntype else val

def main():
    args = parser.parse_args()
    dataset = get_dataset(args)
    dataset_name = args.dataset_name
    if dataset_name.startswith("ogbn"):
        graph = dataset[0][0]
    else:
        graph = dataset[0]

    if dataset_name != "ogbn-mag":
        graph = dgl.remove_self_loop(graph)
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)

    prepare_features(args, dataset, graph)
    balance_ntypes = graph.ndata["train_mask"] \
                        if dataset_name in ["ogbn-products", "ogbn-arxiv"] else None
    dgl.distributed.partition_graph(
        graph, args.dataset_name,
        args.num_partitions,
        args.partition_out_path,
        num_hops=1,
        balance_ntypes=balance_ntypes,
        balance_edges=True)


if __name__ == "__main__":
    main()
