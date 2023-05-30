.. _shards:


Distributed Graph Representation
==================================================================
SAR represents the full graph as :math:`N^2` graph shards where :math:`N` is the number of workers/partitions. Each graph shard represents the edges from one partition to another and the features associated with these edges. :class:`sar.core.GraphShard` represents a single graph shard. Each worker stores the :math:`N` graph shards containing the incoming edges for the nodes in the worker's partition. These :math:`N` graph shards are managed by the :class:`sar.core.GraphShardManager` class.  :class:`sar.core.GraphShardManager`  implements a distributed version of ``update_all`` and ``apply_edges`` which are the main methods used by GNNs to create and exchange messages in the graph. :class:`sar.core.GraphShardManager` implements ``update_all`` and ``apply_edges`` in a sequential manner by iterating through the :math:`N` graph shards to sequentially create and aggregate messages from each partition into the local partition.

The :meth:`sar.core.GraphShardManager.get_full_partition_graph` method can be used to combine the worker's :math:`N` graph shards into one monolithic graph object that represents all the incoming edges for nodes in the local partition. It returns a :class:`sar.core.DistributedBlock` object. The implementation of ``update_all`` and ``apply_edges`` in :class:`sar.core.DistributedBlock` is not sequential. Instead. It fetches all remote features in one step and aggregates all incoming messages to the local partition in one step.

In the distributed implementation of the sequential backward pass in  ``update_all`` and ``apply_edges`` in  :class:`sar.core.GraphShardManager`, it is not possible for SAR to automatically detect if the message function uses any learnable parameters, and thus SAR will not be able to backpropagate gradients to these message parameters. To tell SAR that a message function is parameterized, use the :func:`sar.core.message_has_parameters` decorator to decorate message functions that use learnable parameters.


.. _shard-limitations:

Limitations of the distributed graph objects
------------------------------------------------------------------------------------
Keep in mind that the distributed graph class :class:`sar.core.GraphShardManager` does not implement all the functionality of DGL's native graph class. For example, it does not impelement the ``successors`` and ``predecessors`` methods. It supports primarily the methods of DGL's native graphs that are relevant to GNNs such as ``update_all``, ``apply_edges``, and ``local_scope``.  It also supports setting graph node and edge features through the dictionaries ``srcdata``, ``dstdata``, and ``edata``. To remain compatible with DGLGraph :class:`sar.core.GraphShardManager` provides also access to the ``ndata`` member, which works as alias to ``srcdata``, however it is not accessible when working with MFGs.

:class:`sar.core.GraphShardManager` also supports the ``in_degrees`` and ``out_degrees`` members and supports querying the number of nodes and edges in the graph. 

The ``update_all`` method in :class:`sar.core.GraphShardManager` only supports the 4 standard reduce functions in dgl: ``max``, ``min``, ``sum``, and ``mean``. The reason behind this is that SAR runs a sequential reduction of messages and therefore requires that :math:`reduce(msg_1,msg_2,msg_3) = reduce(msg_1,reduce(msg_2,msg_3))`. 
       
.. currentmodule:: sar.core


Relevant classes methods
---------------------------------------------------------------------------
                   
.. autosummary::
   :toctree: Graph Shard classes
   :template: graphshardmanager
	     
   GraphShard	   
   GraphShardManager	 
   DistributedBlock
   message_has_parameters
