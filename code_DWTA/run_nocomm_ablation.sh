#!/bin/bash
# Communication-layer ablation: same recipe, same seeds, same budgets, only the
# weapon-to-weapon attention removed. Sequential on one GPU.
set -u
for s in 5 6 7; do
  echo "[$(date)] seed $s (nocomm) starting"
  python rl/DWTA_GNN_TRAIN_search_in_loop.py \
    --total_steps 400 --eval_every 20 --k_edits 3 --k_eval 3 --fast_eval \
    --batch_size 15 --para_size 10 --seed $s --tag nocomm --arch nocomm \
    > result/search_in_loop_nocomm_seed$s.txt 2>&1
  echo "[$(date)] seed $s (nocomm) done (exit $?)"
done

CK=""
for s in 5 6 7; do
  P="result/SearchInLoop_seed${s}_nocomm_best.pt"
  [ -f "$P" ] && CK="$CK $P"
done
echo "[$(date)] nocomm checkpoints:$CK"
if [ -n "$CK" ]; then
  python eval_final_table.py --ckpts $CK --kmax 30 --arch nocomm \
    > result/final_table_nocomm.txt 2>&1
  echo "[$(date)] nocomm table done (exit $?)"
fi
echo "[$(date)] ABLATION DONE"
