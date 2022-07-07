#!/bin/bash

python3 train_homogeneous_graph_advanced.py --partitioning-json-file ../partition_data/ogbn-arxiv.json         --log_dir log/fixed_channel/K=1 --train-mode one_shot_aggregation --ip-file ip_file.txt --fed_agg_round 1 --lr 1e-3             --train-iters 500 --n_kernel 16 --rank 0 --world-size 2 --backend ccl &
python3 train_homogeneous_graph_advanced.py --partitioning-json-file ../partition_data/ogbn-arxiv.json         --log_dir log/fixed_channel/K=1 --train-mode one_shot_aggregation --ip-file ip_file.txt --fed_agg_round 1 --lr 1e-3             --train-iters 500 --n_kernel 16 --rank 1 --world-size 2 --backend ccl &
