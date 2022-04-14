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

import torch
import dgl.function as fn  # type: ignore
import dgl  # type: ignore
from . import GraphShardManager


def edge_softmax(graph: GraphShardManager,
                 logits: torch.Tensor, eids=dgl.ALL, norm_by: str = 'dst') -> torch.Tensor:
    """
    Implements a similar functionality as DGL's ``dgl.nn.edge_softmax`` on distributed graphs.

    Only supports a subset of the possible argument values.

    :param graph: The distributed graph 
    :type graph: GraphShardManager
    :param logits: The edge logits. The size of the first dimension should be the same as the number of edges in the ``graph`` argument
    :type logits: torch.Tensor
    :param eids:  must be ``dgl.ALL``
    :type eids: 
    :param norm_by:  must be ``'dst'``
    :type norm_by: str
    :returns: A tensor with the same size as logits contaning the softmax-normalized logits

    """

    assert eids == dgl.ALL, \
        'edge_softmax on GraphShardManager only supported when eids==dgl.ALL'

    assert norm_by == 'dst', \
        'edge_softmax on GraphShardManager only supported when norm_by==dst'

    with graph.local_scope():
        graph.edata['logits'] = logits
        with torch.no_grad():
            graph.update_all(fn.copy_e('logits', 'temp'),
                             fn.max('temp', 'max_logits'))  # pylint: disable=no-member

        graph.apply_edges(
            fn.e_sub_v('logits', 'max_logits', 'adjusted_logits'))  # pylint: disable=no-member

        graph.edata['exp_logits'] = torch.exp(graph.edata.pop('adjusted_logits'))

        graph.update_all(fn.copy_e('exp_logits', 'temp'),
                         fn.sum('temp', 'normalization'))  # pylint: disable=no-member

        graph.apply_edges(
            fn.e_div_v('exp_logits', 'normalization', 'sm_output'))  # pylint: disable=no-member

        sm_output = graph.edata.pop('sm_output')

    return sm_output
