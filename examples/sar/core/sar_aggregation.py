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

from typing import Tuple, List,  Optional, Dict, TYPE_CHECKING, Callable, Any
import inspect
import functools
import time
import logging
import threading
import torch
from torch.autograd import profiler
import dgl  # type:ignore
import dgl.function as fn  # type: ignore


from torch import Tensor

from ..config import Config
from ..comm import exchange_single_tensor,  rank,  comm_thread, world_size
from ..common_tuples import AggregationData, TensorPlace, ShardInfo

if TYPE_CHECKING:
    from .graphshard import GraphShardManager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)


def message_has_parameters(param_foo: Callable[[Any], Tuple[Tensor, ...]]):
    """A decorator for message functions that use learnable parameters. 

    You must use this decorator to tell SAR about the parameters that the message function is using to
    ensure these parameters get the correct gradients. The decorator has one parameter which is a
    callable returning a tuple containing the parameters of the message function. If the message function
    is an instance method, the callable will receive the instance as its first argument, otherwise it receives
    None. Example:
    ::

        from torch import nn
        from sar import message_has_parameters
        import dgl.function as fn  # type: ignore


        class ParameterizedAggregation(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.message_transformation = nn.Linear(dim, dim)

            @message_has_parameters(lambda self: tuple(self.message_transformation.parameters()))
            def message(self, edges):
                m = self.message_transformation(edges.src['h'])
                return {'m': m}

            def forward(grap, features):
                with graph.local_scope():
                    graph.srcdata['h'] = features
                    graph.update_all(self.message, fn.sum('m', 'result'))
                    result = graph.dstdata['result']
                return result

    ..

    :param param_foo: A callable returning a Tuple of the parameters used by the message function
    :type param_foo: Callable[[Any], Tuple[Tensor, ...]]
    :returns: The decorated message function

    """
    def func_decorator(func):
        arg_spec = inspect.getfullargspec(func)
        if 'sar_shard_info' in arg_spec.args + arg_spec.kwonlyargs:
            @functools.wraps(func)
            def decorated_func(*args, __get_params__=False, sar_shard_info=None, **kwargs):
                if __get_params__:
                    if len(args) > 0:
                        return decorated_func.param_foo(args[0])
                    return decorated_func.param_foo(None)
                return func(*args, sar_shard_info=sar_shard_info, **kwargs)
        else:
            @functools.wraps(func)
            def decorated_func(*args, __get_params__=False, **kwargs):
                if __get_params__:
                    if len(args) > 0:
                        return decorated_func.param_foo(args[0])
                    return decorated_func.param_foo(None)
                return func(*args,  **kwargs)
        decorated_func.param_foo = param_foo
        return decorated_func
    return func_decorator


class BackwardManager:
    def __init__(self, n_grads_total: int):
        super().__init__()
        self.n_grads_remaining = n_grads_total
        self.backward_event = threading.Event()
        self.grad_dict: Dict[str, Tensor] = {}

    def initialize_grad_dict(self, tensor_name: str, tensor_template: Tensor):
        if tensor_name not in self.grad_dict:
            self.grad_dict[tensor_name] = torch.zeros_like(tensor_template)

    def remote_grad_sent(self):
        self.n_grads_remaining -= 1
        logger.debug('remote grad sent : %d', self.n_grads_remaining)
        assert self.n_grads_remaining >= 0, \
            ' More grads received than expected  '

        if self.n_grads_remaining == 0:
            self.backward_event.set()

    def wait_for_all_grads(self):
        if self.n_grads_remaining == 0:
            return

        self.backward_event.wait()

    def update_grad(self, tensor_name: str, tensor: Tensor, indices: Tensor):
        self.grad_dict[tensor_name][indices] += tensor


def exchange_grads(send_grad: Tensor,
                   graph_shard_manager: "GraphShardManager",
                   backward_manager: BackwardManager,
                   tensor_name: str,
                   send_idx: int) -> None:
    """Exchanges a single gradient tensor with a remote host

    :param send_grad: The gradient to send to remote host
    :type send_grad: Tensor
    :param send_idx:  The index of the receiving host
    :type send_idx: int
    :returns: None

    """
    with torch.no_grad():
        comm_round = send_idx - rank()
        recv_idx = (rank() - comm_round) % world_size()

        indices_required_from_me = graph_shard_manager.indices_required_from_me[recv_idx]
        recv_grad = send_grad.new(indices_required_from_me.size(0),
                                  *send_grad.size()[1:]).zero_()

        exchange_single_tensor(recv_idx, send_idx, recv_grad, send_grad)
        backward_manager.update_grad(
            tensor_name, recv_grad, indices_required_from_me)
        backward_manager.remote_grad_sent()


def grad_hook(grad: Tensor, graph_shard_manager: "GraphShardManager",
              backward_manager: BackwardManager,
              tensor_name: str, remote_idx: int) -> None:

    comm_thread.submit_task(task_id=f'grad_{remote_idx}', task=functools.partial(
        exchange_grads, send_grad=grad, graph_shard_manager=graph_shard_manager,
        backward_manager=backward_manager, tensor_name=tensor_name,
        send_idx=remote_idx))


def exchange_features(graph_shard_manager: "GraphShardManager",
                      backward_manager: Optional[BackwardManager],
                      all_input_names: List[Tuple[TensorPlace, str]],
                      detached_input_tensors: Tuple[Tensor, ...],
                      comm_round: int, grad_enabled: bool) -> Tuple[int, Dict[str, Tensor]]:

    recv_idx = (rank() - comm_round) % world_size()
    send_idx = (rank() + comm_round) % world_size()
    t_1 = time.time()
    result_dict: Dict[str, Tensor] = {}

    indices_required_from_me = graph_shard_manager.indices_required_from_me[send_idx]
    n_recv_nodes = graph_shard_manager.graph_shards[recv_idx].unique_src_nodes.size(
        0)
    for tensor_idx, (tensor_place, tensor_name) in enumerate(all_input_names):
        if tensor_place == TensorPlace.SRC:  # Only fetch src tensors
            send_features = detached_input_tensors[tensor_idx][indices_required_from_me].detach(
            )
            recv_features = send_features.new(
                n_recv_nodes, *send_features.size()[1:])
            exchange_single_tensor(recv_idx, send_idx, recv_features,
                                   send_features)

            logger.debug('recv features %s', recv_features.size())
            if grad_enabled and detached_input_tensors[tensor_idx].requires_grad:
                assert backward_manager is not None
                recv_features.requires_grad_()
                recv_features.register_hook(functools.partial(grad_hook,
                                                              graph_shard_manager=graph_shard_manager,
                                                              backward_manager=backward_manager,
                                                              tensor_name=tensor_name,
                                                              remote_idx=recv_idx))
                backward_manager.initialize_grad_dict(
                    tensor_name, detached_input_tensors[tensor_idx])
            result_dict[tensor_name] = recv_features

    worker_idx = rank()
    logger.debug(f'{worker_idx} : exchange features {recv_idx},{send_idx} in %s',
                 time.time() - t_1)

    return recv_idx, result_dict


def direct_backward(aggregation_data: AggregationData,
                    backward_manager: BackwardManager,
                    detached_input_tensors: Tuple[Tensor, ...],
                    comm_round,
                    output_grad: Tensor
                    ):

    recv_idx = (rank() - comm_round) % world_size()

    graph_shard = aggregation_data.graph_shard_manager.graph_shards[recv_idx]
    tgt_node_indices = graph_shard.unique_tgt_nodes - graph_shard.tgt_range[0]

    for tensor_idx, (tensor_place, tensor_name) in enumerate(aggregation_data.all_input_names):
        if tensor_place == TensorPlace.SRC:
            g_rev = graph_shard.graph_reverse
            with g_rev.local_scope():
                g_rev.srcdata['out_grad'] = output_grad[tgt_node_indices]
                g_rev.update_all(fn.copy_u('out_grad', 'tgt_'),
                                 fn.sum('tgt_', 'sum_grad'))  # pylint: disable=no-member
                src_grad = g_rev.dstdata['sum_grad']
            backward_manager.initialize_grad_dict(
                tensor_name, detached_input_tensors[tensor_idx])

            grad_hook(src_grad, aggregation_data.graph_shard_manager,
                      backward_manager, tensor_name, recv_idx)


def add_shard_info_to_func(func, shard_info: Optional[ShardInfo]):
    if callable(func):
        arg_spec = inspect.getfullargspec(func)
        args = arg_spec.args + arg_spec.kwonlyargs
        logger.debug('looking for signature in function %s', func)
        logger.debug('found args %s', args)
        if 'sar_shard_info' in args:
            logger.debug('found sar shard info in function signature')
            return functools.partial(func, sar_shard_info=shard_info)
    return func


def shard_aggregation_node(aggregation_data: AggregationData,
                           recv_idx: int,
                           recv_dict: Dict[str, Tensor],
                           detached_input_tensors: Tuple[Tensor, ...],
                           accumulator: Optional[Tensor],
                           output_grad: Optional[Tensor],
                           agg_result: Optional[Tensor]) -> Optional[Tensor]:

    graph_shard = aggregation_data.graph_shard_manager.graph_shards[recv_idx]
    active_graph = graph_shard.graph
    dst_node_indices = graph_shard.unique_tgt_nodes - graph_shard.tgt_range[0]

    shard_info = graph_shard.shard_info
    assert shard_info is not None
    start_edge, end_edge = shard_info.edge_range

    with active_graph.local_scope():
        for tensor_idx, (tensor_place, tensor_name) in enumerate(aggregation_data.all_input_names):
            if tensor_place == TensorPlace.SRC:
                active_graph.srcdata[tensor_name] = recv_dict[tensor_name]
            elif tensor_place == TensorPlace.DST:
                active_graph.dstdata[tensor_name] = detached_input_tensors[tensor_idx][dst_node_indices]
            elif tensor_place == TensorPlace.EDGE:
                active_graph.edata[tensor_name] = detached_input_tensors[tensor_idx][start_edge: end_edge]
        with profiler.record_function("SUB_AGGREGATION"):
            active_graph.update_all(add_shard_info_to_func(aggregation_data.message_func, graph_shard.shard_info),
                                    aggregation_data.reduce_func)
        result = active_graph.dstdata[aggregation_data.reduce_func.out_field]

    if output_grad is None:  # Forward pass
        if accumulator is None:
            local_node_range = graph_shard.tgt_range[1] - \
                graph_shard.tgt_range[0]
            accumulator = torch.zeros(torch.Size(
                [local_node_range]) + result.size()[1:], device=result.device)

            if aggregation_data.reduce_func.name == 'sum':
                pass
            elif aggregation_data.reduce_func.name == 'max':
                accumulator.fill_(-torch.inf)
            elif aggregation_data.reduce_func.name == 'min':
                accumulator.fill_(torch.inf)
            else:
                raise ValueError(
                    f'unknown reduction function {aggregation_data.reduce_func}')

            accumulator[dst_node_indices] = result
        else:
            if aggregation_data.reduce_func.name == 'sum':
                accumulator[dst_node_indices] += result
            elif aggregation_data.reduce_func.name == 'max':
                accumulator[dst_node_indices] = torch.max(
                    accumulator[dst_node_indices], result)
            elif aggregation_data.reduce_func.name == 'min':
                accumulator[dst_node_indices] = torch.min(
                    accumulator[dst_node_indices], result)
            else:
                raise ValueError('unknown reduce function',
                                 aggregation_data.reduce_func)

    else:  # Backward pass
        assert agg_result is not None
        assert torch.is_grad_enabled()
        assert output_grad.size(0) == (
            graph_shard.tgt_range[1] - graph_shard.tgt_range[0])

        if aggregation_data.reduce_func.name == 'sum':
            result.backward(output_grad[dst_node_indices])
        elif aggregation_data.reduce_func.name in ['min', 'max']:
            active_positions_in_shard_output = (
                agg_result[dst_node_indices] == result)
            result[active_positions_in_shard_output].backward(
                output_grad[dst_node_indices][active_positions_in_shard_output])
        else:
            raise ValueError(
                f'unknown reduction function {aggregation_data.reduce_func}')

    return accumulator


def shard_aggregation_edge(aggregation_data: AggregationData,
                           recv_idx: int,
                           recv_dict: Dict[str, Tensor],
                           detached_input_tensors: Tuple[Tensor, ...],
                           accumulator: Optional[Tensor],
                           output_grad: Optional[Tensor],
                           agg_result: Optional[Tensor]) -> Optional[Tensor]:

    graph_shard = aggregation_data.graph_shard_manager.graph_shards[recv_idx]
    active_graph = graph_shard.graph
    dst_node_indices = graph_shard.unique_tgt_nodes - graph_shard.tgt_range[0]

    shard_info = graph_shard.shard_info
    assert shard_info is not None
    start_edge, end_edge = shard_info.edge_range
    logger.debug(
        f'edge aggregation with recv idex {recv_idx} and start edge {start_edge} and end edge {end_edge}')

    with active_graph.local_scope():
        for tensor_idx, (tensor_place, tensor_name) in enumerate(aggregation_data.all_input_names):
            if tensor_place == TensorPlace.SRC:
                active_graph.srcdata[tensor_name] = recv_dict[tensor_name]
            elif tensor_place == TensorPlace.DST:
                active_graph.dstdata[tensor_name] = detached_input_tensors[tensor_idx][dst_node_indices]
            elif tensor_place == TensorPlace.EDGE:
                active_graph.edata[tensor_name] = detached_input_tensors[tensor_idx][start_edge:end_edge]

        active_graph.apply_edges(add_shard_info_to_func(aggregation_data.message_func, graph_shard.shard_info),
                                 etype=aggregation_data.etype)
        result = active_graph.edata[aggregation_data.message_func.out_field]

    if output_grad is None:  # Forward pass
        if accumulator is None:
            n_edges_in_partition = aggregation_data.graph_shard_manager.number_of_edges()
            accumulator = torch.empty(torch.Size(
                [n_edges_in_partition]) + result.size()[1:], device=result.device)

        accumulator[start_edge:end_edge] = result

    else:  # Backward pass
        assert agg_result is not None
        assert torch.is_grad_enabled()
        result.backward(output_grad[start_edge:end_edge])

    return accumulator


def do_aggregation(aggregation_data: AggregationData,
                   backward_manager: Optional[BackwardManager],
                   output_grad: Optional[Tensor],
                   agg_result: Optional[Tensor],
                   detached_input_tensors: Tuple[Tensor, ...]) -> Optional[Tensor]:

    logger.debug('grad enabled in do_aggregation: %s', torch.is_grad_enabled())

    backward_pass = output_grad is not None and agg_result is not None
    if not backward_pass:
        assert output_grad is None and agg_result is None

    require_inputs_for_backward = not((isinstance(aggregation_data.message_func,
                                                  dgl.function.message.CopyMessageFunction)
                                       and aggregation_data.message_func.name in ['copy_u', 'copy_src']
                                       and (aggregation_data.reduce_func is None or aggregation_data.reduce_func.name == 'sum')))

    # True if this is an apply_edges call, False if it is an update_all call
    edge_update = (aggregation_data.reduce_func is None)

    logger.debug('requires input for backward in do_aggregation : %s',
                 require_inputs_for_backward)
    pipeline_stage = 0
    accumulator: Optional[Tensor] = None
    for comm_round in range(world_size()):
        if require_inputs_for_backward or not backward_pass:
            if aggregation_data.remote_data:
                while pipeline_stage <= comm_round + Config.pipeline_depth \
                        and pipeline_stage < world_size():

                    comm_thread.submit_task(
                        f"exch_feats,  comm_round = {pipeline_stage}",
                        functools.partial(exchange_features,
                                          graph_shard_manager=aggregation_data.graph_shard_manager,
                                          backward_manager=backward_manager,
                                          all_input_names=aggregation_data.all_input_names,
                                          detached_input_tensors=detached_input_tensors,
                                          comm_round=pipeline_stage, grad_enabled=torch.is_grad_enabled())
                    )
                    pipeline_stage += 1

                with profiler.record_function("COMM_FETCH"):
                    recv_idx, recv_dict = comm_thread.get_result()
            else:
                recv_idx = (rank() - comm_round) % world_size()
                recv_dict = {}

            with profiler.record_function("AGGREGATION"):
                if edge_update:
                    accumulator = shard_aggregation_edge(aggregation_data,
                                                         recv_idx,
                                                         recv_dict,
                                                         detached_input_tensors,
                                                         accumulator,
                                                         output_grad,
                                                         agg_result)
                else:
                    accumulator = shard_aggregation_node(aggregation_data,
                                                         recv_idx,
                                                         recv_dict,
                                                         detached_input_tensors,
                                                         accumulator,
                                                         output_grad,
                                                         agg_result)

        else:  # Backward pass where we do not need to fetch remote features
            assert output_grad is not None and backward_manager is not None
            direct_backward(aggregation_data,
                            backward_manager,
                            detached_input_tensors,
                            comm_round,
                            output_grad)

    return accumulator


class SAROp(torch.autograd.Function):  # pylint: disable = abstract-method
    @ staticmethod
    # pylint: disable = arguments-differ,unused-argument
    def forward(ctx, aggregation_data: AggregationData,  # type: ignore
                *all_input_tensors: Tensor) -> Tensor:  # type: ignore

        logger.debug('aggregation_data %s', aggregation_data)

        # Do not pass the parameter tensors to aggregation routines. They
        # will be implicitly used by the message function
        if aggregation_data.n_params:
            all_input_tensors = all_input_tensors[:-aggregation_data.n_params]
        if not aggregation_data.grad_enabled:
            agg_result = do_aggregation(
                aggregation_data, None, None, None, all_input_tensors)
            assert agg_result is not None
            return agg_result

        ctx.aggregation_data = aggregation_data
        if aggregation_data.remote_data:
            n_remote_tensors_with_grads = 0
            for tensor_idx, (tensor_place, _) in enumerate(aggregation_data.all_input_names):
                if all_input_tensors[tensor_idx].requires_grad and tensor_place == TensorPlace.SRC:
                    n_remote_tensors_with_grads += 1
            ctx.backward_manager = BackwardManager(
                world_size() * n_remote_tensors_with_grads)
            disable_sr = Config.disable_sr
        else:
            ctx.backward_manager = BackwardManager(0)
            disable_sr = True
            logger.debug('sar disabled since there is no remote data')

        ctx.disable_sr = disable_sr

        detached_input_tensors = tuple(
            input_tensor.detach() for input_tensor in all_input_tensors)
        for orig_tensor, detached_tensor in zip(all_input_tensors, detached_input_tensors):
            detached_tensor.requires_grad_(orig_tensor.requires_grad)

        if disable_sr:
            with torch.enable_grad():
                agg_result = do_aggregation(
                    aggregation_data, ctx.backward_manager, None, None, detached_input_tensors)
                assert agg_result is not None
            ctx.save_for_backward(*(detached_input_tensors + (agg_result,)))
            agg_result = agg_result.detach()
        else:
            agg_result = do_aggregation(aggregation_data, ctx.backward_manager,
                                        None, None, detached_input_tensors)
            assert agg_result is not None
            ctx.save_for_backward(*(detached_input_tensors + (agg_result,)))

        return agg_result

    @ staticmethod
    # pylint: disable = arguments-differ
    # type: ignore
    def backward(ctx, output_grad) -> Tuple[Optional[Tensor], ...]:
        logger.debug('backward aggregation data %s', ctx.aggregation_data)
        aggregation_data = ctx.aggregation_data
        backward_manager = ctx.backward_manager
        saved_tensors = ctx.saved_tensors
        detached_input_tensors, agg_result = saved_tensors[:-
                                                           1], saved_tensors[-1]
        with torch.enable_grad():
            if ctx.disable_sr:
                agg_result.backward(output_grad)
                logger.debug('backward no sequential rematerialization')
            else:
                do_aggregation(aggregation_data, backward_manager, output_grad,
                               agg_result, detached_input_tensors)
                logger.debug('backward with successive rematerialization')

        t1 = time.time()
        backward_manager.wait_for_all_grads()
        logger.debug('backward event wait done in %s', time.time() - t1)
        input_grads = []
        for tensor_idx, (tensor_place, tensor_name) in enumerate(aggregation_data.all_input_names):
            if tensor_place == TensorPlace.SRC:
                if detached_input_tensors[tensor_idx].requires_grad:
                    input_grads.append(backward_manager.grad_dict[tensor_name])
                else:
                    input_grads.append(None)
            else:
                input_grads.append(detached_input_tensors[tensor_idx].grad)

        return (None,) + tuple(input_grads) + (None,) * aggregation_data.n_params


sar_op = SAROp.apply
