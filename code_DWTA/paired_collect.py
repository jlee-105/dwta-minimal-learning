"""
Per-instance objectives for the paired significance tests (reviewer 3.8).

The reported tables keep only per-configuration means, so no pair of methods
can be compared instance by instance. This script re-runs every method that
lives in this environment on the same 120 seed-123 instances (twelve
configurations, ten instances each) and writes one row per
(method, configuration, instance):

  Ours_s{1..10}_K0 / _K10   fire-logit checkpoints, same protocol as
                            eval_final_table.py (K=10 prefix of its curve)
  Greedy, Auction           heuristics, fire/hold decided by the mechanism
  Sequential                single-pass sequential decoder

AM and POMO need the RL4CO environment in ../venv and are collected by
paired_collect_rl4co.py. SCIP per-instance values already exist in
result/scip_moderate_*_600s.csv.

Output: result/paired_instances.csv
"""
import csv

import numpy as np
import torch

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_firelogit import create_gnn_actor_firelogit
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.DWTA_Simulator import Environment
from common.auction_refinement import (
    auction_round_action,
    auction_round_action_multifire_fast as auction_round_action_multifire,
)
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals
from eval_final_table import CONFIGS, N_EVAL, SEED, curve_for_instance
from measure_method_timing import make_env, SEQ_CKPT

OURS_CKPT = "result/SearchInLoop_seed%d_firelogit_sel123_best.pt"
OUT = "result/paired_instances.csv"


def normalized(env, N):
    final = env.current_target_value[:, :, 0:N].sum().item()
    original = env.original_target_value[:, :, 0:N].sum().item()
    return final / (original + 1e-8)


@torch.no_grad()
def heuristic_obj(assign_fn, ae, wtp, M, N, T):
    env = Environment(assignment_encoding=ae.clone(),
                      weapon_to_target_prob=wtp.clone(), max_time=T)
    for _ in range(T):
        action = assign_fn(env.current_target_value[:, :, 0:N],
                           env.weapon_to_target_prob[:, :, :M, :N],
                           env.mask_per_weapon[:, :, :M, :N] > 0)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    return normalized(env, N)


@torch.no_grad()
def sequential_obj(actor, ae, wtp, M, N, T):
    env = Environment(assignment_encoding=ae.clone(),
                      weapon_to_target_prob=wtp.clone(), max_time=T)
    for _ in range(T):
        for _ in range(M):
            policy, _ = actor(assignment_embedding=env.assignment_encoding,
                              prob=env.weapon_to_target_prob,
                              mask=env.mask.clone())
            idx = int(policy.reshape(1, 1, -1).argmax().item())
            if idx >= M * N:
                break
            m, n = divmod(idx, N)
            action = torch.full((1, 1, M), N, dtype=torch.long, device=DEVICE)
            action[0, 0, m] = n
            env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    return normalized(env, N)


def main():
    ours = {}
    for s in range(1, 11):
        a = create_gnn_actor_firelogit().to(DEVICE)
        a.load_state_dict(torch.load(OURS_CKPT % s, map_location=DEVICE, weights_only=False))
        a.eval()
        ours[s] = a
    seq = create_gnn_actor_sequential().to(DEVICE)
    seq.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE))
    seq.eval()

    rows = []
    for (M, N, T) in CONFIGS:
        cfg = "%dM_%dN_%dT" % (M, N, T)
        rng_i = np.random.default_rng(SEED)
        inst = [generate_moderate_temporal_instance(M, N, T, rng=rng_i) for _ in range(N_EVAL)]
        for i, (V, P, TW, A, PR, C) in enumerate(inst):
            patch_globals(M, N, T, A, PR, C)
            ae, wtp = make_env(V, P, TW, M, N, T, A)
            for s, actor in ours.items():
                c, _ = curve_for_instance(actor, ae, wtp, M, N, T, 10,
                                          np.random.default_rng(SEED + i))
                rows.append(("Ours_s%d_K0" % s, cfg, i, c[0]))
                rows.append(("Ours_s%d_K10" % s, cfg, i, c[10]))
            rows.append(("Greedy", cfg, i, heuristic_obj(auction_round_action_multifire, ae, wtp, M, N, T)))
            rows.append(("Auction", cfg, i, heuristic_obj(auction_round_action, ae, wtp, M, N, T)))
            rows.append(("Sequential", cfg, i, sequential_obj(seq, ae, wtp, M, N, T)))
        done = [r for r in rows if r[1] == cfg]
        def m(name): return np.mean([r[3] for r in done if r[0] == name])
        print("%-13s Ours_s1_K0=%.4f Greedy=%.4f Auction=%.4f Sequential=%.4f"
              % (cfg, m("Ours_s1_K0"), m("Greedy"), m("Auction"), m("Sequential")), flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "config", "instance", "objective"])
        w.writerows(rows)
    print("wrote %d rows to %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
