N_CLIENTS = 2
ROUND = 501
cmd = []
LOG_DIR = "local_global_random"

for i in range(N_CLIENTS):
    cmd += f"python3 fed_train_homogeneous_basic.py --partitioning-json-file partition_data/ogbn-arxiv.json \
        --log_dir {LOG_DIR} --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
            --train-iters 500 --n_kernel 10 --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"

with open("commands.sh", "w") as f:
    for c in cmd:
        f.writelines(c)
