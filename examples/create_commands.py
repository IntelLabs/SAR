N_CLIENTS = 10
ROUND = 1
cmd_fcr = []
cmd_vcr = []
COMP_TYPE = "node"

TRAIN_ITER = 512
COMP_RATIO = 4
STEP = TRAIN_ITER // COMP_RATIO

LOG_DIR_FCR = f"log/fixed_cr/{COMP_TYPE}/CR={COMP_RATIO}/K={ROUND}"
LOG_DIR_VCR = f"log/varying_cr/{COMP_TYPE}/CR={TRAIN_ITER//STEP}/K={ROUND}"
EXP = "vcr"
TRIALS = 2
for t in range(TRIALS):
    LOG_DIR_FCR = f"log/fixed_cr/{COMP_TYPE}/CR={COMP_RATIO}/K={ROUND}/TRIAL={t}"
    LOG_DIR_VCR = f"log/varying_cr/{COMP_TYPE}/CR={TRAIN_ITER//STEP}/K={ROUND}"
    for i in range(N_CLIENTS):
        cmd_fcr += f"python3 train_homogeneous_graph_advanced.py --partitioning-json-file ../partition_data/ogbn-arxiv.json \
            --log_dir {LOG_DIR_FCR} --compression_type {COMP_TYPE} --comp_ratio {COMP_RATIO} --enable_cr --train-mode one_shot_aggregation --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
                --train-iters {TRAIN_ITER} --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"
        cmd_vcr += f"python3 train_homogeneous_graph_advanced.py --partitioning-json-file ../partition_data/ogbn-arxiv.json \
            --log_dir {LOG_DIR_VCR} --enable_cr --enable_vcr --compression_type {COMP_TYPE} --compression_step {STEP} --train-mode one_shot_aggregation --ip-file ip_file.txt --fed_agg_round {ROUND} --lr 1e-3 \
                --train-iters {TRAIN_ITER} --rank {i} --world-size {N_CLIENTS} --backend ccl &\n"
    cmd_fcr += "wait\n\n"
    cmd_vcr += "wait\n\n"


with open(f"commands.sh", "w") as f:
    f.writelines("#!/bin/bash\n\n")
    if EXP == "fcr":
        for c in cmd_fcr:
            f.writelines(c)
    elif EXP == "vcr":
        for c in cmd_vcr:
            f.writelines(c)
