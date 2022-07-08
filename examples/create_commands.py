N_CLIENTS = 20
ROUND = 1
cmd_lc = []
cmd_nc = []
cmd_rc = []
LOG_DIR_LC = f"log/fixed_channel/K={ROUND}"
LOG_DIR_NC = f"log/no_channel/K={ROUND}"
LOG_DIR_RC = f"log/full_graph/K={ROUND}"
EXP = "lc"

for i in range(N_CLIENTS):
    cmd_lc += f"python3 train_homogeneous_graph_advanced.py --partitioning-json-file ../partition_data/ogbn-arxiv.json \
        --log_dir {LOG_DIR_LC} --train-mode one_shot_aggregation --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
            --train-iters 500 --n_kernel 256 --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"
    cmd_nc += f"python3 fed_train_homogeneous_basic.py --partitioning-json-file ../partition_data/ogbn-arxiv.json \
        --log_dir {LOG_DIR_NC} --disable_cut_edges --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
            --train-iters 500 --n_kernel 16 --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"
    cmd_rc += f"python3 fed_train_homogeneous_basic.py --partitioning-json-file ../partition_data/ogbn-arxiv.json \
        --log_dir {LOG_DIR_RC} --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
            --train-iters 500 --n_kernel 16 --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"

with open("commands.sh", "w") as f:
    f.writelines("#!/bin/bash\n\n")
    if EXP == "nc":
        for c in cmd_nc:
            f.writelines(c)
    elif EXP == "lc":
        for c in cmd_lc:
            f.writelines(c)
    elif EXP == "rc":
        for c in cmd_rc:
            f.writelines(c)
