.. _quick-start:

Quick start guide
===============================
Follow the following steps to enable distributed full-batch training in your DGL code:

.. contents::
    :depth: 2
    :local:
    :backlinks: top

Partition the graph
----------------------------------
Partition the graph using DGL's `partition_graph <https://docs.dgl.ai/en/0.6.x/generated/dgl.distributed.partition.partition_graph.html>`_ function. See `here <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/dist/partition_graph.py>`_ for an example. The number of partitions should be the same as the number of training machines/workers that will be used. SAR requires consecutive node indices in each partition, and requires that the partition information include the one-hop neighborhoods of all nodes in the partition. Setting ``num_hops = 1`` and ``reshuffle = True`` in the call to ``partition_graph`` takes care of these requirements. ``partition_graph`` yields a directory structure with the partition information and a .json file ``graph_name.json``.


An example of partitioning the ogbn-arxiv graph in two parts: ::
  
    import dgl
    import torch
    from ogb.nodeproppred import DglNodePropPredDataset

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    graph = dataset[0][0]
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    labels = dataset[0][1].view(-1)
    split_idx = dataset.get_idx_split()


    def _idx_to_mask(idx_tensor):
        mask = torch.BoolTensor(graph.number_of_nodes()).fill_(False)
        mask[idx_tensor] = True
        return mask


    train_mask, val_mask, test_mask = map(
        _idx_to_mask, [split_idx['train'], split_idx['valid'], split_idx['test']])
    features = graph.ndata['feat']
    graph.ndata.clear()
    for name, val in zip(['train_mask', 'val_mask', 'test_mask', 'labels', 'features'],
                         [train_mask, val_mask, test_mask, labels, features]):
        graph.ndata[name] = val

    dgl.distributed.partition_graph(
        graph, 'arxiv', 2, './test_partition_data/', num_hops=1, reshuffle=True)
	
..

Note that we add the labels, and the train/test/validation masks as node features so that they get split into multiple parts alongside the graph.


Initialize communication
----------------------------------
SAR uses the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package to handle all communication. See the :ref:`Communication Guide <comm-guide>`  for more information on the communication routines. We require the IP address of the master worker/machine (the machine with rank 0) to initialize the ``torch.distributed`` package. In an environment with a networked file system where all workers/machines share a common file system, we can communicate the master's IP address through the file system. In that case, use :func:`sar.nfs_ip_init` to obtain the master ip address.

Initialize the communication through a call to :func:`sar.initialize_comms` , specifying the current worker index, the total number of workers (which should be the same as the number of partitions from step 1), the master's IP address, and the communication device. The later is the device on which SAR should place the tensors before sending them through the communication backend.   For example: ::

  if backend_name == 'nccl':
      comm_device = torch.device('cuda')
  else:
      comm_device = torch.device('cpu')
  master_ip_address = sar.nfs_ip_init(rank,path_to_ip_file)
  sar.initialize_comms(rank,world_size, master_ip_address,backend_name,comm_device)
  
.. 

``backend_name`` can be ``nccl``, ``ccl``, or ``mpi``.



Load partition data and construct graph
-----------------------------------------------------------------
Use :func:`sar.load_dgl_partition_data` to load one graph partition from DGL's partition data in each worker. :func:`sar.load_dgl_partition_data` returns a :class:`sar.common_tuples.PartitionData` object that contains all the information about the partition.

There are several ways to construct a distributed graph-like object from ``PartitionData``. See :ref:`constructing distributed graphs <data-loading>` for more details. Here we will use the simplest method:  :func:`sar.construct_full_graph` which returns a :class:`sar.GraphShardManager` object which implements many of the GNN-related functionality of DGL's native graph objects. ``GraphShardManager`` can thus be used as a drop-in replacement for DGL's native graphs.

Putting it all together:
::
   
    partition_data = sar.load_dgl_partition_data(
        json_file_path, #Path to .json file created by DGL's partition_graph
        rank, #Worker rank
        device #Device to place the partition data (CPU or GPU)
    )
    shard_manager = sar.construct_full_graph(partition_data)
    
.. 


Synchronize parameters and gradients
---------------------------------------------------------------------------
In a distributed setting, each worker will construct the GNN model. Before training, we should synchronize the model parameters across all workers. :func:`sar.sync_params` is a convenience function that does just that. At the end of every training iteration, each worker needs to gather and sum the parameter gradients from all other workers before making the parameter update. This can be done using :func:`sar.gather_grads`.

Model initialization and the training loop follow the following recipe: ::

  gnn_model = construct_GNN_model(...)
  optimizer = torch.optim.Adam(gnn_model.parameters(),..)
  sar.sync_params(gnn_model)
  for train_iter in range(n_train_iters):
     model_out = gnn_model(shard_manager,features)
     loss = calculate_loss(model_out,labels)
     optimizer.zero_grad()
     loss.backward()
     sar.gather_grads(gnn_model)
     optimizer.step()

..

For complete examples, check the examples folder in the Git repository.
