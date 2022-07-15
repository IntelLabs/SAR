N_CLIENTS = 10
ROUND = 10
cmd_lc = []
cmd_nc = []
cmd_rc = []
cmd_vrc = []
COMP_TYPE = "node"
COMP_RATIO_A = "2"
COMP_RATIO_B = "1024"
STEP = 32
TRAIN_ITER = 512

LOG_DIR_LC = f"log/fixed_channel/{COMP_TYPE}/CR={COMP_RATIO_B}/K={ROUND}"
LOG_DIR_NC = f"log/no_channel/K={ROUND}"
LOG_DIR_RC = f"log/full_graph/K={ROUND}"
LOG_DIR_VCR = f"log/varying_channel/{COMP_TYPE}/CR={TRAIN_ITER//STEP}/K={ROUND}"
EXP = "vcr"

for i in range(N_CLIENTS):
    cmd_lc += f"python3 train_homogeneous_graph_advanced.py --partitioning-json-file ../partition_data/ogbn-arxiv.json \
        --log_dir {LOG_DIR_LC} --compression_type {COMP_TYPE} --compression_ratio_b {COMP_RATIO_B} --train-mode one_shot_aggregation --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
            --train-iters {TRAIN_ITER} --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"
    cmd_nc += f"python3 fed_train_homogeneous_basic.py --partitioning-json-file ../partition_data/ogbn-arxiv.json \
        --log_dir {LOG_DIR_NC} --disable_cut_edges --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
            --train-iters {TRAIN_ITER} --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"
    cmd_rc += f"python3 fed_train_homogeneous_basic.py --partitioning-json-file ../partition_data/ogbn-arxiv.json \
        --log_dir {LOG_DIR_RC} --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
            --train-iters {TRAIN_ITER} --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"
    cmd_vrc += f"python3 train_homogeneous_graph_advanced.py --partitioning-json-file ../partition_data/ogbn-arxiv.json \
        --log_dir {LOG_DIR_VCR} --compression_type {COMP_TYPE} --compression_ratio_b {COMP_RATIO_B} \
            --compression_ratio_a {COMP_RATIO_A} --compression_step {STEP} --train-mode one_shot_aggregation --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
            --train-iters {TRAIN_ITER} --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"

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
    elif EXP == "vcr":
        for c in cmd_vrc:
            f.writelines(c)

