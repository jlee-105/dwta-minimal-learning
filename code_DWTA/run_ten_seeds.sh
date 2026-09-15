#!/usr/bin/env bash
# Ten training runs for the reported model.
#
# Checkpoint selection runs on the validation set defined in
# DWTA_GNN_TRAIN_search_in_loop.py (VAL_SEED=7717, sizes drawn from the
# training range), so no instance that appears in the reported table
# influences which checkpoint is kept. Scoring is a separate step:
# eval_final_table.py re-evaluates every saved checkpoint on the twelve
# held-out configurations at seed 123.
set -u

LOG=result/ten_seeds_run.log
: > "$LOG"

echo "[$(date)] starting ten seeds" | tee -a "$LOG"

for SEED in 1 2 3 4 5 6 7 8 9 10; do
    OUT="result/ten_seeds_seed${SEED}.txt"
    echo "[$(date)] seed $SEED -> $OUT" | tee -a "$LOG"
    python rl/DWTA_GNN_TRAIN_search_in_loop.py \
        --seed "$SEED" \
        --tag tenseed \
        --total_steps 400 \
        --eval_every 20 \
        > "$OUT" 2>&1
    STATUS=$?
    echo "[$(date)] seed $SEED finished (exit $STATUS)" | tee -a "$LOG"
done

echo "[$(date)] all seeds done" | tee -a "$LOG"
ls -1 result/SearchInLoop_seed*_tenseed_best.pt 2>/dev/null | tee -a "$LOG"
echo "[$(date)] next: eval_final_table.py --ckpts <those checkpoints>" | tee -a "$LOG"
