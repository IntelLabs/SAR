#!/bin/bash

python3 fed_train_homogeneous_basic.py --partitioning-json-file partition_data/ogbn-arxiv.json         --log_dir log/no_channel_dist --disable_cut_edges --ip-file ip_file.txt --fed_agg_round 1 --lr 1e-3             --train-iters 500 --n_kernel 10 --rank 0 --world-size 2 --backend ccl &
python3 fed_train_homogeneous_basic.py --partitioning-json-file partition_data/ogbn-arxiv.json         --log_dir log/no_channel_dist --disable_cut_edges --ip-file ip_file.txt --fed_agg_round 1 --lr 1e-3             --train-iters 500 --n_kernel 10 --rank 1 --world-size 2 --backend ccl &
