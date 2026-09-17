#!/usr/bin/env bash
# Queued 2026-09-17: once the six-layer AM retrain ends, re-measure inference
# time for every method with the fire-logit model and AM-L6 on an idle GPU.
set -u
until grep -q "Saved best policy" result/rl4co_am_multiscale_L6_train.log; do sleep 60; done
python measure_method_timing.py > result/timing_methods.txt 2>&1
../venv/Scripts/python.exe measure_rl4co_timing.py > result/timing_rl4co.txt 2>&1
echo "[$(date)] timing done" >> result/rl4co_am_multiscale_L6_train.log
