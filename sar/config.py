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


class Config(object):
    '''
    General configuration for the SAR library. 


    .. py:attribute:: disable_sr : bool

        Disables sequential re-materialization of the computational graph during the backward pass.\
    The computational graph is constructed normally during the forward pass. default : False



    .. py:attribute:: max_collective_size : int

        Limits the maximum size of data in torch.distributed.all_to_all collective calls.  If non-zero,\
    the sar.comms.all_to_all wrapper method will break down the collective call into multiple torch.distributed.all_to_all\
    calls so that the size of the data in each call is below max_collective_size. default : 0

    .. py:attribute:: pipeline_depth : int

        Sets the communication pipeline depth when doing sequential aggregation or sequential re-materialization.\
    In a separate thread, SAR will pre-fetch up to ``pipeline_depth`` remote partitions into a data queue that will then\
    be processed by the compute thread. Higher values will increase memory consumption but may hide\
    communication latency. default : 1


    '''

    disable_sr: bool = False
    max_collective_size: int = 0
    pipeline_depth: int = 1
