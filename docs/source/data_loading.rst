.. _data-loading:


Data loading and graph construction
==========================================================
After partitioning the graph using DGL's `partition_graph <https://docs.dgl.ai/en/0.6.x/generated/dgl.distributed.partition.partition_graph.html>`_ function, SAR can load the graph data using :func:`sar.load_dgl_partition_data`. This yields a :class:`sar.common_tuples.PartitionData` object. The ``PartitionData`` object can then be used to construct various types of graph-like objects that can be passed to GNN models. The graph construction options are described below: 

.. contents:: :local:
    :depth: 2


Constructing the full graph for sequential aggregation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Construct a single distributed graph object of type :class:`sar.GraphShardManager`::

    shard_manager = sar.construct_full_graph(partition_data)

..

The ``GraphShardManager`` object encapsulates N DGL graph objects (where N is the number of workers). Each graph object represents the edges incoming from one partition (including the local partition). ``GraphShardManager`` implements the ``update_all`` and ``apply_edges`` methods in addition to several other methods from the standard  ``dgl.heterograph.DGLheterograph`` API.  The ``update_all`` and ``apply_edges`` methods implement the sequential aggregation and rematerialization scheme to realize the distributed forward and backward passes. ``GraphShardManager`` can usually be passed to GNN layers instead of ``dgl.heterograph.DGLheterograph``.

Constructing Message Flow Graphs (MFGs) for sequential aggregation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In node classification tasks, gradients only backpropagate from the labeled nodes. DGL uses the concept of message flow graphs to construct layer-specific bi-partite graphs that update only a subset of nodes in each layer. These are the nodes  that will ultimately affect the output, assuming each node only aggregates messages from its neighbors in every layer.

If training a K-layer GNN on a node classification tasks, you can construct K distributed graph objects that reflect the message flow graphs at each layer using :class:`sar.construct_mfgs`:
::

    class GNNModel(nn.Module):
        def __init__(n_layers: int):
            super().__init__()
            self.convs = nn.ModuleList([
                dgl.nn.SAGEConv(100, 100)
                for _ in range(n_layers)
            ])

        def forward(blocks: List[sar.GraphShardManager], features: torch.Tensor):
            for idx in range(len(self.convs)):
                features = self.convs[idx](blocks[idx], features)
            return features

    K = 3 # number of layers
    gnn_model = GNNModel(K)
    train_blocks = sar.construct_mfgs(partition_data,
                                      global_indices_of_labeled_nodes_in_partition,
                                      K)
    model_out = gnn_model(train_blocks, local_node_features)

..

Using message flow graphs at each layer can substantially lower run-time and memory consumption in node classification tasks with few labeled nodes. 


Constructing full graph or MFGs for one-shot aggregation 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As described in the the :ref:`training modes <sar-modes>`, SAR supports doing one-shot distributed aggregation (mode 3). To run in this mode, you should extract the full partition graph from the :class:`sar.GraphShardManager` object and use that during training. When using the full graph:
::

    shard_manager = sar.construct_full_graph(partition_data)
    one_shot_graph = shard_manager.get_full_partition_graph()
    del shard_manager
    ## Use one_shot_graph from now on. 

..

When using MFGs:
::

    train_blocks = sar.construct_mfgs(partition_data,
                                      global_indices_of_labeled_nodes_in_partition,
                                      n_layers)
    one_shot_blocks = [block.get_full_partition_graph() for block in train_blocks]
    del train_blocks
    ## Use one_shot_blocks from now on

..
 
    
   
Relevant methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: sar


.. autosummary::
   :toctree: Data loading and graph construction
	     

   load_dgl_partition_data	      
   construct_full_graph
   construct_mfgs   

	     
