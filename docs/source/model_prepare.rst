.. _model-prepare:


Preparing your GNN model for SAR
=========================================================

The basic graph object in SAR is :class:`sar.core.GraphShardManager`. It can  typically be used as a drop-in replacement for DGL's native graph object and provided as the input graph to most GNN layers. See :ref:`the distributed graph limitations section<shard-limitations>` for some important limitations of this approach. There are situations where you need to modify your layer to accomodate :class:`sar.core.GraphShardManager` or to modify your GNN network to take into account the distributed nature of the training. Three such situations are outlined here:

Edge softmax
------------------------------------------------------------------------------------
DGL's ``edge_softmax`` function expects a native DGL graph object and will not work with a :class:`sar.core.GraphShardManager` object. Instead, you must use SAR's implementation :func:`sar.edge_softmax` which accepts a :class:`sar.core.GraphShardManager` object. DGL's attention based GNN layers make use of DGL's ``edge_softmax`` function. One solution to be able to use these layers with SAR  is to monkey-patch them as shown below: 
::
   
    import dgl
    import sar
    def patched_edge_softmax(graph, *args, **kwargs):
        if isinstance(graph, sar.GraphShardManager):
            return sar.edge_softmax(graph, *args, **kwargs)

        return dgl.nn.edge_softmax(graph, *args, **kwargs)  # pylint: disable=no-member


    dgl.nn.pytorch.conv.gatconv.edge_softmax = patched_edge_softmax
    dgl.nn.pytorch.conv.dotgatconv.edge_softmax = patched_edge_softmax
    dgl.nn.pytorch.conv.agnnconv.edge_softmax = patched_edge_softmax
   
..

``patched_edge_softmax`` dispatches to either DGL's or SAR's implementation depending on the type of the input graph. SAR has the conveninece function :func:`sar.patch_dgl` that runs the above code to patch DGL's attention-based GNN layers.

Parameterized message functions
-----------------------------------------------------------------------------------

SAR's sequential rematerialization of the computational graph during the backward pass must be aware of any learnable parameters used to create the edge messages. SAR needs to know of these parameters so that it can correctly backpropagate gradients to them. There is no easy way for SAR  to automatically detect the learnable parameters used by the message function. It is thus up to the user to use the :func:`sar.core.message_has_parameters` to tell SAR about these parameters. For example, DGL's ``RelGraphConv`` layer uses a message function with learnable parameters.  To avoid the need to modify the original code of ``RelGraphconv``, we can subclass it as follows to provide the necessary decorator for the message function, and then use the subclass in the GNN model:
::
   
    import dgl
    import sar
   
    class RelGraphConv_sar(dgl.nn.pytorch.conv.RelGraphConv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @sar.message_has_parameters(lambda self: tuple(self.linear_r.parameters()))
        def message(self, edges):
            return super().message(edges)

..            

SAR has the conveninece function :func:`sar.patch_dgl` that defines a new ``RelGraphConv`` layer as described in the code above and uses it to replace DGL's ``RelGraphConv`` layer. 
   

Batch normalization
-----------------------------------------------------------------------------------
The batch normalization layers in PyTorch such as ``torch.nn.BatchNorm1d`` will normalize the GNN node features using statistics obtained only from the node features in the local partition. So the normalizing factors (mean and standard deviation) will be different in each worker, and will depend on the way the graph is partitioned. To normalize using global statistics obtained from all nodes in the graph, you can use :class:`sar.DistributedBN1D`. :class:`sar.DistributedBN1D` has a similar interface as ``torch.nn.BatchNorm1d``. For example::

  norm_layer = sar.DistributedBN1D(out_dim, affine=True)
  ..
  #Will normalize the features of the nodes in the partition
  #by the global node statistics (mean and standard deviation)
  normalized_activations = norm_layer(partition_node_features)

..

Relevant methods
---------------------------------------------------------------------------

.. autosummary::
   :toctree: Adapting GNNs to SAR
   :template: graphshardmanager

   sar.core.message_has_parameters
   sar.edge_softmax
   sar.DistributedBN1D
   sar.patch_dgl
