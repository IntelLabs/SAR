## Hetero RGCN

Exampel script for ogbn-mag dataset.
Original script: https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py
You can find here two scripts `train_heterogeneous_graph.py` and `train_heterogeneous_graph_mfg.py`, the former is a simple full graph training and inference. The latter is a training and inference script which utilizes Message Flow Graph (MFG) - this approach is more computationally effective, because it computes embeddings only for nodes which require it, i.e. labeled nodes.

### Results
Obtained results for two partitions (ogbn-mag dataset):
- Train Acc: 77.18 ± 2.85%
- Validation Acc: 40.03 ± 0.45%
- Test Acc: 39.06 ± 0.44%
Presented results are the average accuracies obtained after running 10 experiments (1 experiment = 60 epochs). Results from each experiment (train/val/test accuracies) were not necessarily taken from the 60th epoch. The values were obtained at the moment when the validation accuraccy was the highest.
(Note: Results achieved in https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py are different, because it uses mini-batch training instead of full-graph like in SAR)


### Run command:

```
python examples/rgcn-hetero/train_heterogeneous_graph.py --partitioning-json-file partition_data/ogbn-mag.json  --ip-file ip_file --backend ccl --rank 0 --world-size 2 --train-iters 60

python examples/rgcn-hetero/train_heterogeneous_graph.py --partitioning-json-file partition_data/ogbn-mag.json  --ip-file ip_file --backend ccl --rank 1 --world-size 2 --train-iters 60
```