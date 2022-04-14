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


parser = ArgumentParser(description="Graph partitioning for ogbn-mag")

parser.add_argument(
    "--dataset-root",
    type=str,
    default="./datasets/",
    help="The OGB datasets folder "
)

parser.add_argument(
    "--partition-out-path",
    type=str,
    default="./partition_data/",
    help="Path to the output directory for the partition data "
)


parser.add_argument(
    '--num-partitions',
    default=2,
    type=int,
    help='Number of graph partitions to generate')


def main():
    args = parser.parse_args()
    dataset = DglNodePropPredDataset(name='ogbn-mag',
                                     root=args.dataset_root)
    graph = dataset[0][0]
    labels = dataset[0][1]['paper'].view(-1)
    split_idx = dataset.get_idx_split()

    def idx_to_mask(idx_tensor):
        mask = torch.BoolTensor(graph.number_of_nodes('paper')).fill_(False)
        mask[idx_tensor] = True
        return mask
    train_mask, val_mask, test_mask = map(
        idx_to_mask, [split_idx['train']['paper'], split_idx['valid']['paper'], split_idx['test']['paper']])
    features = graph.ndata['feat']['paper']
    for name, val in zip(['train_mask', 'val_mask', 'test_mask', 'labels', 'features'],
                         [train_mask, val_mask, test_mask, labels, features]):
        graph.ndata[name] = {'paper': val}

    dgl.distributed.partition_graph(
        graph, 'ogbn-mag',
        args.num_partitions,
        args.partition_out_path,
        num_hops=1,
        reshuffle=True,
        balance_edges=True)


if __name__ == '__main__':
    main()
