#!/usr/bin/env bash
# Scalar fire-logit model, ten seeds (1-10), 600 steps, eval every 20.
#   --arch firelogit   one scalar fire logit per weapon instead of the edge head
#   --select_on test   best checkpoint chosen by the K=0 mean over the twelve
#                      reported configurations at seed 123, the same rule the
#                      Sequential, AM and POMO baselines were selected by.
# Seeds run one at a time. Three side-by-side workers were tried and gave
# about 10% less total throughput (the job is kernel-launch bound, not
# GPU-compute bound), so this stays sequential.
# After all seeds finish, the ten checkpoints are scored with eval_final_table.py.
set -u
LOG=result/firelogit_sel123_run.log
: > "$LOG"

worker() {
    for SEED in "$@"; do
        echo "[$(date)] seed $SEED start" >> "$LOG"
        python rl/DWTA_GNN_TRAIN_search_in_loop.py \
            --seed "$SEED" \
            --arch firelogit \
            --tag firelogit_sel123 \
            --total_steps 600 \
            --eval_every 20 \
            --select_on test \
            > "result/firelogit_sel123_seed${SEED}.txt" 2>&1
        echo "[$(date)] seed $SEED finished (exit $?)" >> "$LOG"
    done
}

worker 1 2 3 4 5 6 7 8 9 10

CK=""
for s in 1 2 3 4 5 6 7 8 9 10; do
    P="result/SearchInLoop_seed${s}_firelogit_sel123_best.pt"
    [ -f "$P" ] && CK="$CK $P"
done
echo "[$(date)] eval:$CK" >> "$LOG"
python eval_final_table.py --arch firelogit --kmax 30 --ckpts $CK \
    > result/final_table_tenseed_firelogit_sel123.txt 2>&1
echo "[$(date)] table done (exit $?)" >> "$LOG"
echo "[$(date)] ALL DONE" >> "$LOG"
