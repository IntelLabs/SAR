from argparse import ArgumentParser
import time
import json
import copy
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import dgl  # type: ignore
from dgl import function as fn # type: ignore
from dgl.heterograph import DGLBlock  # type: ignore
from ogb.nodeproppred import DglNodePropPredDataset

import sar
from sar.distributed_bn import MeanOp, VarOp, DistributedBN1D


parser = ArgumentParser(description="CorrectAndSmooth example")

parser.add_argument('--partitioning-json-file', default='', type=str,
                    help='Path to the .json file containing partitioning information'
)
parser.add_argument('--ip-file', default='./ip_file', type=str,
                    help='File with ip-address. Worker 0 creates this file and all others read it'
                    )
parser.add_argument('--backend', default='nccl', type=str, choices=['ccl', 'nccl', 'mpi'],
                    help='Communication backend to use'
                    )
parser.add_argument('--cpu-run', action='store_true',
                    help='Run on CPUs if set, otherwise run on GPUs'
                    )
parser.add_argument('--rank', default=0, type=int,
                    help='Rank of the current worker '
                    )
parser.add_argument('--world-size', default=2, type=int,
                    help='Number of workers '
                    )
parser.add_argument('--model', default="mlp", type=str, choices=['mlp', 'linear'],
                    help='Model type'
                    )
parser.add_argument('--num-layers', default=3, type=int,
                    help='Number of layers in the model'
                    )
parser.add_argument('--hidden-layer-dim', default=256, type=int,
                    help='Dimension of GNN hidden layer'
                    )
parser.add_argument('--dropout', default=0.4, type=float,
                    help='Dropout rate for layers in the model'
                    )
parser.add_argument('--lr', default=1e-2, type=float,
                    help='learning rate'
                    )
parser.add_argument('--epochs', default=300, type=int,
                    help='Number of training epochs'
                    )
parser.add_argument('--num-correction-layers', default=50, type=int,
                    help='The number of correct propagations'
                    )
parser.add_argument('--correction-alpha', default=0.979, type=float,
                    help='The coefficient of correction'
                    )
parser.add_argument('--correction-adj', default="DAD", type=str,
                    help='DAD: D^-0.5 * A * D^-0.5 | DA: D^-1 * A | AD: A * D^-1'
                    )
parser.add_argument('--num-smoothing-layers', default=50, type=int,
                    help='The number of smooth propagations'
                    )
parser.add_argument('--smoothing-alpha', default=0.756, type=float,
                    help='The coefficient of smoothing'
                    )
parser.add_argument('--smoothing-adj', default="DAD", type=str,
                    help='DAD: D^-0.5 * A * D^-0.5 | DA: D^-1 * A | AD: A * D^-1'
                    )
parser.add_argument('--autoscale', action="store_true",
                    help='Automatically determine the scaling factor for "sigma"'
                    )
parser.add_argument('--scale', default=20.0, type=float,
                    help='The scaling factor for "sigma", in case autoscale is set to False'
                    )


class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0.0):
        super(MLP, self).__init__()
        assert num_layers >= 2

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linears.append(nn.Linear(in_dim, hid_dim))
        self.bns.append(DistributedBN1D(hid_dim))

        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hid_dim, hid_dim))
            self.bns.append(DistributedBN1D(hid_dim))

        self.linears.append(nn.Linear(hid_dim, out_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()

    def forward(self, x):
        for linear, bn in zip(self.linears[:-1], self.bns):
            x = linear(x)
            x = F.relu(x, inplace=True)
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        return F.log_softmax(x, dim=-1)


class LabelPropagation(nn.Module):
    r"""

    Description
    -----------
    Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

    where unlabeled data is inferred by labeled data via propagation.

    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
        adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
    """

    def __init__(self, num_layers, alpha, adj="DAD"):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
        self.adj = adj

    @torch.no_grad()
    def forward(
        self, g, labels, mask=None, post_step=lambda y: y.clamp_(0.0, 1.0)
    ):
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)

            y = labels
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]

            last = (1 - self.alpha) * y
            degs = g.in_degrees().float().clamp(min=1)
            norm = (
                torch.pow(degs, -0.5 if self.adj == "DAD" else -1)
                .to(labels.device)
                .unsqueeze(1)
            )

            for _ in range(self.num_layers):
                # Assume the graphs to be undirected
                if self.adj in ["DAD", "AD"]:
                    y = norm * y

                g.srcdata["h"] = y
                g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                y = self.alpha * g.dstdata["h"]

                if self.adj in ["DAD", "DA"]:
                    y = y * norm

                y = post_step(last + y)

            return y


class CorrectAndSmooth(nn.Module):
    r"""

    Description
    -----------
    Introduced in `Combining Label Propagation and Simple Models Out-performs Graph Neural Networks <https://arxiv.org/abs/2010.13993>`_

    Parameters
    ----------
        num_correction_layers: int
            The number of correct propagations.
        correction_alpha: float
            The coefficient of correction.
        correction_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        num_smoothing_layers: int
            The number of smooth propagations.
        smoothing_alpha: float
            The coefficient of smoothing.
        smoothing_adj: str
            'DAD': D^-0.5 * A * D^-0.5
            'DA': D^-1 * A
            'AD': A * D^-1
        autoscale: bool, optional
            If set to True, will automatically determine the scaling factor :math:`\sigma`. Default is True.
        scale: float, optional
            The scaling factor :math:`\sigma`, in case :obj:`autoscale = False`. Default is 1.
    """

    def __init__(
        self,
        num_correction_layers,
        correction_alpha,
        correction_adj,
        num_smoothing_layers,
        smoothing_alpha,
        smoothing_adj,
        autoscale=True,
        scale=1.0,
    ):
        super(CorrectAndSmooth, self).__init__()

        self.autoscale = autoscale
        self.scale = scale

        self.prop1 = LabelPropagation(
            num_correction_layers, correction_alpha, correction_adj
        )
        self.prop2 = LabelPropagation(
            num_smoothing_layers, smoothing_alpha, smoothing_adj
        )

    def correct(self, g, y_soft, y_true, mask):
        with g.local_scope():
            assert abs(float(y_soft.sum()) / y_soft.size(0) - 1.0) < 1e-2
            numel = (
                int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            )
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(
                    y_soft.dtype
                )

            error = torch.zeros_like(y_soft)
            error[mask] = y_true - y_soft[mask]

            if self.autoscale:
                smoothed_error = self.prop1(
                    g, error, post_step=lambda x: x.clamp_(-1.0, 1.0)
                )
                sigma = error[mask].abs().sum() / numel
                scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
                scale[scale.isinf() | (scale > 1000)] = 1.0

                result = y_soft + scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result
            else:

                def fix_input(x):
                    x[mask] = error[mask]
                    return x

                smoothed_error = self.prop1(g, error, post_step=fix_input)

                result = y_soft + self.scale * smoothed_error
                result[result.isnan()] = y_soft[result.isnan()]
                return result

    def smooth(self, g, y_soft, y_true, mask):
        with g.local_scope():
            numel = (
                int(mask.sum()) if mask.dtype == torch.bool else mask.size(0)
            )
            assert y_true.size(0) == numel

            if y_true.dtype == torch.long:
                y_true = F.one_hot(y_true.view(-1), y_soft.size(-1)).to(
                    y_soft.dtype
                )

            y_soft[mask] = y_true
            return self.prop2(g, y_soft)


def evaluate(logits, labels, masks):
    """
    Calculating accuracy metric over train, validation and test indices (in a distributed way).

    :param logits: Predictions of the model
    :type logits: Tensor
    :param labels: Ground truth labels
    :type labels: Tensor
    :param masks: Dictionary of Tensors, that contain indices for train, validation and test sets
    :type masks: Dictionary
    
    :returns: Tuple of accuracy metrics: train, validation, test
    """
    results = []
    for indices_name in ['train_indices', 'val_indices', 'test_indices']:
        n_correct = (logits[masks[indices_name]].argmax(1) ==
                        labels[masks[indices_name]]).float().sum()
        results.extend([n_correct, masks[indices_name].numel()])

    acc_vec = torch.FloatTensor(results)
    # Sum the n_correct, and number of mask elements across all workers
    sar.comm.all_reduce(acc_vec, op=dist.ReduceOp.SUM, move_to_comm_device=True)
    (train_acc, val_acc, test_acc) =  \
        (acc_vec[0] / acc_vec[1],
         acc_vec[2] / acc_vec[3],
         acc_vec[4] / acc_vec[5])
    return train_acc, val_acc, test_acc


def data_normalization(features, eps=1.0e-5):
    """
    Perform features normzalization by subtracting theur means and dividing them by their standard deviations.
    Each position in features vector is normzalized independently. To calculate means and stds over whole
    dataset, workers must communicate with each other.

    :param features: Dataset's features
    :type features: Tensor
    :param eps: a value added to the variance for numerical stability 
    :type eps: float
    
    :returns: Normalized Tensor of features
    """
    mean = MeanOp.apply(features)
    var = VarOp.apply(features)
    std = torch.sqrt(var - mean**2 + eps)
    features = (features - mean) / std
    return features


def main():
    args = parser.parse_args()
    print(args)
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

    labels = sar.suffix_key_lookup(partition_data.node_features,
                                   'labels').long().to(device)

    # Obtain the number of classes by finding the max label across all workers
    num_labels = labels.max() + 1
    sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True)
    num_labels = num_labels.item() 

    features = sar.suffix_key_lookup(partition_data.node_features, 'features').to(device)
    full_graph_manager = sar.construct_full_graph(partition_data).to(device)

    if "ogbn-arxiv" in args.partitioning_json_file:
        features = data_normalization(features, args.world_size)
        
    # We do not need the partition data anymore
    del partition_data

    # load model
    if args.model == "mlp":
        model = MLP(
            features.size(1), args.hidden_layer_dim, num_labels, args.num_layers, args.dropout
        ).to(device)
    elif args.model == "linear":
        model = MLPLinear(features.size(1), num_labels).to(device)
    else:
        raise NotImplementedError(f"Model {args.model} is not supported.")

    print('model', model)
    sar.sync_params(model)
    
    # Obtain the number of labeled nodes in the training
    # This will be needed to properly obtain a cross entropy loss
    # normalized by the number of training examples
    n_train_points = torch.LongTensor([masks['train_indices'].numel()])
    sar.comm.all_reduce(n_train_points, op=dist.ReduceOp.SUM, move_to_comm_device=True)
    n_train_points = n_train_points.item()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0
    best_model = copy.deepcopy(model)

    # training
    print("---------- Training ----------")
    for epoch in range(args.epochs):
        t_1 = time.time()
        model.train()
        
        logits = model(features)
        loss = F.nll_loss(logits[masks['train_indices']],
                            labels[masks['train_indices']], reduction='sum')/n_train_points
        optimizer.zero_grad()
        loss.backward()
        # Do not forget to gather the parameter gradients from all workers
        sar.gather_grads(model)
        optimizer.step()
        train_time = time.time() - t_1

        model.eval()
        with torch.no_grad():
            logits = model(features)
            train_acc, val_acc, _ = evaluate(logits, labels, masks)
        
            result_message = (
                f"iteration [{epoch}/{args.epochs}] | "
                )
            result_message += ', '.join([
                f"train loss={loss:.4f}, "
                f"Accuracy: "
                f"train={train_acc:.4f} "
                f"valid={val_acc:.4f} "
                f" | train time = {train_time} "
                f" |"
                ])
            print(result_message, flush=True)

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(model)

    # testing & saving model
    print("---------- Testing ----------")
    best_model.eval()
    logits = best_model(features)
    _, _, test_acc = evaluate(logits, labels, masks)
    print(f"Test acc: {test_acc:.4f}")
    
    print("---------- Correct & Smoothing ----------")
    y_soft = model(features).exp()

    cs = CorrectAndSmooth(
        num_correction_layers=args.num_correction_layers,
        correction_alpha=args.correction_alpha,
        correction_adj=args.correction_adj,
        num_smoothing_layers=args.num_smoothing_layers,
        smoothing_alpha=args.smoothing_alpha,
        smoothing_adj=args.smoothing_adj,
        autoscale=args.autoscale,
        scale=args.scale,
    )

    y_soft = cs.correct(full_graph_manager, y_soft, labels[masks['train_indices']], masks['train_indices'])
    y_soft = cs.smooth(full_graph_manager, y_soft, labels[masks['train_indices']], masks['train_indices'])    
    _, _, test_acc = evaluate(y_soft, labels, masks)
    print(f"Test acc: {test_acc:.4f}")


if __name__ == '__main__':
    main()
