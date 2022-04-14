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

from typing import Union, List, Optional
import dgl  # type: ignore
from .edge_softmax import edge_softmax
from .core import GraphShardManager

from . import message_has_parameters



# patch edge_softmax in dgl's nn modules

class RelGraphConv(dgl.nn.pytorch.conv.RelGraphConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @message_has_parameters(lambda self: tuple(self.linear_r.parameters()))
    def message(self, edges):
        return super().message(edges)


def patched_edge_softmax(graph, *args, **kwargs):
    if isinstance(graph, GraphShardManager):
        return edge_softmax(graph, *args, **kwargs)

    return dgl.nn.edge_softmax(graph, *args, **kwargs)  # pylint: disable=no-member


def patch_dgl():
    """Patches DGL so that attention layers (``gatconv``, ``dotgatconv``,
    ``agnngatconv``) use a different ``edge_softmax`` function
    that supports :class:`sar.core.GraphShardManager`. Also modifies DGL's
    ``RelGraphConv`` to add a decorator to its ``message`` function to tell
    SAR how to find the parameters used to create edge messages.

    """
    dgl.nn.pytorch.conv.gatconv.edge_softmax = patched_edge_softmax
    dgl.nn.pytorch.conv.dotgatconv.edge_softmax = patched_edge_softmax
    dgl.nn.pytorch.conv.agnnconv.edge_softmax = patched_edge_softmax

    dgl.nn.pytorch.conv.RelGraphConv = RelGraphConv
    dgl.nn.RelGraphConv = RelGraphConv
