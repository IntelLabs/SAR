
[Documentation](https://intel-sandbox.github.io/libraries.ai.graph-networks.distributed-gcns) | [Examples](examples/README.md)

Sequential Aggregation and Rematerialization (SAR) is a pure Python library for distributed full-batch training of Graph Neural Networks (GNNs) on large graphs. SAR is built on top of PyTorch and DGL. SAR is particularly suited for training GNNs on large graphs as it guarantees linear memory scaling, i.e, the memory needed to store the GNN activiations in each host is guaranteed to go down linearly with the number of hosts, even for densely connected graphs.

SAR requires minimal changes to existing GNN training code. SAR directly uses the graph partitioning data created by [DGL's partitioning tools](https://docs.dgl.ai/en/0.6.x/generated/dgl.distributed.partition.partition_graph.html) and can thus be used as a drop-in replacement for DGL's distributed sampling-based training. To get started using SAR, check out [SAR's documentation](https://intel-sandbox.github.io/libraries.ai.graph-networks.distributed-gcns) and the examples under the `examples/` folder.


## Installing required packages
```shell
pip3 install -r requirements.txt
```
Python3.8 or higher is required. You also need to install [torch CCL](https://github.com/intel/torch-ccl) if you want to use Intel's OneCCL communication backend. 

## Performance on ogbn-papers100M
SAR consumes up to 2x less memory when training a 3-layer GraphSage network on ogbn-papers100M (111M nodes, 3.2B edges), and up to 4x less memory when training a 3-layer Graph Attention Network (GAT). SAR achieves near linear scaling for the peak memory requirements per worker. We use a 3-layer GraphSage network with hidden layer size of 256, and a 3-layer GAT network with hidden layer size of 128 and 4 attention heads. We use batch normalization between all layers

<img src="docs/source/images/papers_sage_memory.png" width="400">  <img src="docs/source/images/papers_gat_memory.png" width="400"> 


The run-time of SAR improves as we add more workers. At 128 workers, the epoch time is 3.8s. Each worker is a 2-socket machine with 2 Icelake processors (36 cores each). The workers are connected using Infiniband HDR (200 Gbps) links. After 100 epochs, training has converged. We use a 3-layer GraphSage network with hidden layer size of 256 and batch normalization between all layers. The training curve is the same regardless of the number of workers/partition. 

<img src="docs/source/images/papers_os_scaling.png" width="450">

<img src="docs/source/images/papers_train_full_doc.png" width="400"> 




## Cite

If you use SAR in your publication, we would appreciate it if you cite the SAR paper:
```
@article{mostafa2021sequential,
  title={Sequential Aggregation and Rematerialization: Distributed Full-batch Training of Graph Neural Networks on Large Graphs},
  author={Mostafa, Hesham},
  journal={arXiv preprint arXiv:2111.06483},
  year={2021}
}
```