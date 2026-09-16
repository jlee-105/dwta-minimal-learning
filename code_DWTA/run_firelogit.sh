#!/usr/bin/env bash
# Fire-logit ablation: does the policy need per-target scores at all?
#
# The reported actor scores every legal (weapon, target) edge and reads the
# firing probability off a softmax against a hold score. Those target scores
# never pick a target, since assignment is done afterwards by the greedy rule.
# This variant replaces the whole edge head with one scalar logit per weapon.
#
# Trained at 600 steps. The reported model used 400 and three of its ten runs
# were still improving at that point, so a little more headroom is given here.
# The validation trace records every twentieth step, so the 400-step point
# remains available for a budget-matched comparison.
#
# Ten seeds, matching the reported model, so the result can go straight into
# the paper rather than serving only as a probe.
set -u
LOG=result/firelogit_run.log
: > "$LOG"
for SEED in 1 2 3 4 5 6 7 8 9 10; do
    echo "[$(date)] firelogit seed $SEED" | tee -a "$LOG"
    python rl/DWTA_GNN_TRAIN_search_in_loop.py \
        --seed "$SEED" \
        --arch firelogit \
        --tag firelogit \
        --total_steps 600 \
        --eval_every 20 \
        > "result/firelogit_seed${SEED}.txt" 2>&1
    echo "[$(date)] seed $SEED finished (exit $?)" | tee -a "$LOG"
done
echo "[$(date)] done" | tee -a "$LOG"
