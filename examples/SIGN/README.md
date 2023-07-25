## SIGN: Scalable Inception Graph Neural Networks

Original script: https://github.com/dmlc/dgl/tree/master/examples/pytorch/sign

Provided `train_sign_with_sar.py` script is an example how to intergrate SAR to preprocess graph data for training.

### Results
Obtained results for two partitions:
- ogbn-products: 0.7832
- reddit: 0.9639

### Run command:

```
python train_sign_with_sar.py --partitioning-json-file partition_data/reddit.json --ip-file ip_file --backend ccl --rank 0 --world-size 2

python train_sign_with_sar.py --partitioning-json-file partition_data/reddit.json --ip-file ip_file --backend ccl --rank 1 --world-size 2
```