#!/bin/bash
# Waits for the running seed-6 job (PID passed as $1) to exit, then runs seed 7
# with the cheap in-training eval, then regenerates the cross-seed table for
# every checkpoint that exists. Sequential by design: one GPU.
set -u
SEED6_PID="$1"

echo "[$(date)] waiting for seed 6 (PID $SEED6_PID)"
while kill -0 "$SEED6_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] seed 6 finished"

echo "[$(date)] starting seed 7 (fast_eval)"
python rl/DWTA_GNN_TRAIN_search_in_loop.py \
  --total_steps 400 --eval_every 20 --k_edits 3 --k_eval 3 --fast_eval \
  --batch_size 15 --para_size 10 --seed 7 --tag fireonly \
  > result/search_in_loop_train_seed7.txt 2>&1
echo "[$(date)] seed 7 finished (exit $?)"

CKPTS=""
for s in 5 6 7; do
  P="result/SearchInLoop_seed${s}_fireonly_best.pt"
  [ -f "$P" ] && CKPTS="$CKPTS $P"
done
echo "[$(date)] checkpoints found:$CKPTS"

if [ -n "$CKPTS" ]; then
  echo "[$(date)] generating cross-seed table"
  python eval_final_table.py --ckpts $CKPTS --kmax 30 \
    > result/final_table_multiseed.txt 2>&1
  echo "[$(date)] table finished (exit $?)"
fi
echo "[$(date)] ALL DONE"
