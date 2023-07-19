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

from typing import Optional
import torch
from torch import Tensor
import torch.distributed as dist
from torch import nn
from torch.nn import Parameter
from torch.nn import init
from .comm import all_reduce, comm_device, is_initialized


class DistributedBN1D(nn.Module):
    """Distributed Batch normalization layer

    Normalizes a 2D feature tensor using the global mean and standard deviation calculated across all workers.


    :param n_feats: The second dimension (feature dimension) in the 2D input tensor
    :type n_feats: int
    :param eps:  a value added to the variance for numerical stability 
    :type eps: float
    :param affine: When ``True``, the module will use learnable affine parameter
    :type affine: bool
    :param distributed: Boolean speficying whether to run in distributed mode where normalizing\
    statistics are calculated across all workers, or local mode where the normalizing statistics\
    are calculated using only the local input feature tensor. If not specified, it will be set to\
    ``True`` if the user has called :func:`sar.initialize_comms`, and ``False`` otherwise
    :type distributed: Optional[bool]

    """
    def __init__(self, n_feats: int, eps: float = 1.0e-5, affine: bool = True, distributed: Optional[bool] = None):
        super().__init__()
        self.n_feats = n_feats
        self.weight: Optional[Parameter]
        self.bias: Optional[Parameter]
        self.affine = affine
        if affine:
            self.weight = Parameter(torch.ones(n_feats))
            self.bias = Parameter(torch.zeros(n_feats))
        else:
            self.weight = None
            self.bias = None

        self.eps = eps

        if distributed is None:
            self.distributed = is_initialized()
        else:
            self.distributed = distributed

    def forward(self, inp):
        '''
        forward implementation of DistributedBN1D
        '''
        assert inp.ndim == 2, 'distributedBN1D must have a 2D input'
        if self.distributed:
            mean, var = mean_op(inp), var_op(inp)
            std = torch.sqrt(var - mean**2 + self.eps)
        else:
            mean = inp.mean(0)
            std = inp.std(0)
        normalized_x = (inp - mean.unsqueeze(0)) / std.unsqueeze(0)

        if self.weight is not None and self.bias is not None:
            result = normalized_x * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        else:
            result = normalized_x
        return result
    
    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)



class MeanOp(torch.autograd.Function):  # pylint: disable = abstract-method
    @staticmethod
    # pylint: disable = arguments-differ
    def forward(ctx, x):
        own_sum = torch.empty(x.size(1)+1, device=comm_device())
        own_sum[:-1] = x.sum(0).data.to(comm_device())
        own_sum[-1] = x.size(0)
        all_reduce(own_sum, op=dist.ReduceOp.SUM,move_to_comm_device = True)
        mean = (own_sum[:-1]/own_sum[-1]).to(x.device)
        ctx.n_points = torch.round(own_sum[-1]).long().item()
        ctx.inp_size = x.size(0)
        return mean

    @staticmethod
    # pylint: disable = arguments-differ
    def backward(ctx, grad):
        grad_comm = grad.to(comm_device())
        all_reduce(grad_comm, op=dist.ReduceOp.SUM,move_to_comm_device = True)
        return grad_comm.repeat(ctx.inp_size, 1).to(grad.device) / ctx.n_points


class VarOp(torch.autograd.Function):  # pylint: disable = abstract-method
    @staticmethod
    # pylint: disable = arguments-differ
    def forward(ctx, features):
        own_sum = torch.empty(features.size(1)+1, device=comm_device())
        own_sum[:-1] = (features**2).sum(0).data.to(comm_device())
        own_sum[-1] = features.size(0)
        all_reduce(own_sum, op=dist.ReduceOp.SUM,move_to_comm_device = True)
        variance = (own_sum[:-1]/own_sum[-1]).to(features.device)

        ctx.n_points = torch.round(own_sum[-1]).long().item()
        ctx.save_for_backward(features)
        return variance

    @staticmethod
    # pylint: disable = arguments-differ
    def backward(ctx, grad):
        features,  = ctx.saved_tensors
        grad_comm = grad.to(comm_device())
        all_reduce(grad_comm, op=dist.ReduceOp.SUM,move_to_comm_device = True)
        return (grad_comm.to(grad.device).unsqueeze(0) * 2 * features) / ctx.n_points


mean_op = MeanOp.apply
var_op = VarOp.apply
