## Graph partitioning

``partition_arxiv_products.py`` partitions the ogbn-arxiv and ogbn-products  graphs from the [Open Graph Benchmarks](https://ogb.stanford.edu/) using DGL's metis-based partitioning. The general technique there can be used to partition arbitrary homogeneous graphs. Note that all node-related information must be included in the graph's ``ndata`` dictionary so that they are correctly partitioned with the graph. Similarly, edge-related information must be included in the graph's ``edata`` dictionary

``partition_mag.py`` partitions the [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) heterogeneous graph. Again, all node-related information are included in the graph's ``ndata`` for the relevant node types

## Full-batch Training

The script ``train_homogeneous_graph_basic.py`` demonstrates the basic functionality of SAR. It runs distriobuted training using a 3-layer GraphSage network on a partiotioned graph. If you want to train using ``N`` workers, then you need to launch the script ``N`` times, preferably on separate machines. For example, for ``N=2``, and assuming the two workers are on the same network file system, you can launch the 2 workers using the following two commands:

```shell
python3 train_homogeneous_graph_basic.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 0 --world-size 2
python3 train_homogeneous_graph_basic.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 1 --world-size 2

```
The worker with ``rank=0`` (the master)  will write its address to the file specified by the ``--ip-file`` option and the other worker(s) will read this file and connect to the master.


The ``train_homogeneous_graph_advanced.py`` script demonstrates more advanced features of SAR such as distributed construction of Message Flow Graphs (MFGs), and the multiple training modes supported by SAR. 

The ``train_heterogeneous_graph.py`` script demonstrates training on a heterogeneous graph (ogbn-mag). The script trains a 3-layer R-GCN.


## Sampling-based Training
The script ``train_homogeneous_sampling_basic.py`` demonstrates distributed sampling-based training on SAR. It demonstrates the unique ability of  the SAR library to run distributed sampling-based training followed by memory-efficient distributed full-graph inference. The script uses a 3-layer GraphSage network. Assuming 2 machines, you can launch training on the 2 machines using the following 2 commands, where each is executed on a different machine:

```shell
python3 train_homogeneous_sampling_basic.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 0 --world-size 2
python3 train_homogeneous_sampling_basic.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 1 --world-size 2

```

## Distributed Mini-Batch Training with Full-Graph inference
The script ``train_distdgl_with_sar_inference.py`` showcases how SAR can be effectively combined with native DGL distributed training. In this particular example, the training process utilizes a sampling approach, while the evaluation phase leverages the SAR library to perform computations on the entire graph.
```shell
python /home/ubuntu/workspace/dgl/tools/launch.py \
    --workspace /home/ubuntu/workspace/SAR/examples \
    --num_trainers 1 \
    --num_samplers 2 \
    --num_servers 1 \
    --part_config partition_data/ogbn-products.json \
    --ip_config ip_config.txt \
    "/home/ubuntu/miniconda3/bin/python train_distdgl_with_sar_inference.py --graph_name ogbn-products --ip_config ip_config.txt --num_epochs 2 --batch_size 1000 --part_config partition_data/ogbn-products.json"
```

## Correct and Smooth
Example taken from [DGL implemenetation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/correct_and_smooth) of C&S. Code is adjusted to perform distributed training with SAR. For instance, you can run the example with following commands:

* **Plain MLP + C&S**
```shell
python correct_and_smooth.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 0 --world-size 2 --dropout 0.5 --correction-adj DA --smoothing-adj AD --autoscale
```

* **Plain Linear + C&S**
```shell
python correct_and_smooth.py --partitioning-json-file /path/to/partitioning/graph_name.json --ip-file /path/to/ip_file --rank 0 --world-size 2 --model linear --dropout 0.5 --epochs 1000 --correction-alpha 0.87 --smoothing-alpha 0.81 --correction-adj AD --autoscale
