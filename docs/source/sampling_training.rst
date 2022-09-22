.. _sampling:


Distributed Sampling-based training
==================================================================
In addition to distributed full-batch training, the SAR library also supports distributed sampling-based training. The main difference between SAR's distributed sampling-based component and DistDGL is that SAR uses collective communication primitives such as ``all_to_all``  during  the distributed mini-batch generation steps, while DistDGL uses point-to-point communication. One common use case in GNN training is to use sampling-based training followed by full-batch inference. Since SAR supports sampling-based as well as full-batch training and inference, this use case is particularly easy to implement. The same GNN model can be used for both full-batch and sampling-based runs. A simple 3-layer GraphSage model:

::

    class GNNModel(nn.Module):
        def __init__(self,  in_dim: int, hidden_dim: int, out_dim: int):
            super().__init__()

            self.convs = nn.ModuleList([
                dgl.nn.SAGEConv(in_dim, hidden_dim, aggregator_type='mean'),
                dgl.nn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
                dgl.nn.SAGEConv(hidden_dim, out_dim, aggregator_type='mean'),
            ])

        def forward(self,  blocks: List[Union[DGLBlock, sar.GraphShardManager]], features: torch.Tensor):
            for idx, conv in enumerate(self.convs):
                features = conv(blocks[idx], features)
                if idx < len(self.convs) - 1:
                    features = F.relu(features, inplace=True)

            return features
		
..		

Since :class:`sar.core.GraphShardManager` can be used as a drop-in replacement for DGL's native graph objects,  we can use a standard DGL model and either pass it the sampled ``DGLBlock``s or the full distributed graph.

As in full-batch training, we first load the DGL-generated partition data, and construct the full distributed graph. We then define the sampling strategy and dataloader. We use SAR's :class:`sar.DistNeighborSampler` and  :func:`sar.DataLoader` to define the sampling strategy and the distributed dataloader, respectively.

::

   partition_data = sar.load_dgl_partition_data(
        args.partitioning_json_file, args.rank, torch.device('cpu'))

   full_graph_manager = sar.construct_full_graph(
        partition_data)  # Keep full graph on CPU
        
   
   neighbor_sampler = sar.DistNeighborSampler(
   [15, 10, 5], #Fanout for every layer
   input_node_features={'features': features}, #Input features to add to srcdata of first layer's sampled block
   output_node_features={'labels': labels} #Output features to add to dstdata of last layer's sampled block
   )

   dataloader = sar.DataLoader(
        full_graph_manager, #Distributed graph
        train_nodes, #Global indices of nodes that will form the root of the sampled graphs. In node classification, these are the labeled nodes
        neighbor_sampler, #Distributed sampler
        batch_size)

..		

A typical training loop is shown below.

::

    gnn_model = construct_GNN_model(...)
    optimizer = torch.optim.Adam(gnn_model.parameters(),..)
    sar.sync_params(gnn_model)


    for epoch in range(n_epochs):
        model.train()
        for blocks in dataloader:
            block_features = blocks[0].srcdata['features']
            block_labels = blocks[-1].dstdata['labels']
            logits = gnn_model(blocks, block_features)

            output = gnn_model(blocks)
            loss = calculate_loss(output, block_labels)
            optimizer.zero_grad()
            loss.backward()
            sar.gather_grads(gnn_model)
            optimizer.step()

        # inference
        model.eval()
        with torch.no_grad():
            logits = gnn_model_cpu([full_graph_manager] * n_layers, features)
            calculate_loss_accuracy(logits, full_graph_labels)

..

Note that we obtain instances of standard ``DGLBlock`` from the distributed dataloader every training iteration. After every epoch, we run distributed full-graph inference using the :class:`sar.core.GraphShardManager`. We use the same ``GraphShardManager`` object at each layer. Alternatively, as described in the :ref:`data loading section<data-loading>`, we can construct layer-specific distributed message flow graphs (MFGs) to avoid computing redundant node features at each layer. Redundant node features are the node features that do not contribute to the output at the labeled nodes. 



Relevant classes and methods
---------------------------------------------------------------------------

.. currentmodule:: sar

.. autosummary::
   :toctree: Graph Shard classes
   :template: graphshardmanager
	     
   GraphShardManager	 
   DataLoader
   DistNeighborSampler
   
