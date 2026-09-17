"""Per-instance wall-clock time for every method in the main results table.

All methods are timed on the SAME ten seed-123 instances per configuration
that eval_final_table.py scores, so the times in the paper's main table come
from one measurement rather than from separate runs of different scripts.

Reported: mean seconds per instance, CUDA-synchronized, after two untimed
warm-up instances per configuration drawn from a separate RNG stream.

SCIP is not timed here: it is run under a fixed 600 s budget per instance and
its time is that budget, not a measured quantity.
"""
import sys
import time

import numpy as np
import torch

sys.path.append("rl")

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_firelogit import create_gnn_actor_firelogit
from common.DWTA_GNN_sequential import create_gnn_actor_sequential
from common.DWTA_Simulator import Environment
from common.Dynamic_Instance_generation import input_generation
from common.auction_refinement import (
    auction_round_action,
    auction_round_action_multifire_fast as auction_round_action_multifire,
)
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

CONFIGS = [(5, 5, 5), (5, 7, 5), (10, 15, 5),
           (15, 15, 5), (15, 20, 5), (20, 30, 5),
           (30, 30, 10), (30, 40, 10), (40, 50, 10),
           (50, 50, 15), (50, 70, 15), (70, 100, 15)]
TIERS = ["Small"] * 3 + ["Medium"] * 3 + ["Large"] * 3 + ["Battlefield"] * 3
N_EVAL, N_WARM, SEED = 10, 2, 123

OURS_CKPT = "result/SearchInLoop_seed5_firelogit_sel123_best.pt"
SEQ_CKPT = "result/Sequential_multiscale_seed5_best_actor.pt"


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def make_env(V, P, TW, M, N, T, A):
    ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=np.asarray(P),
                               TW=TW, max_time=T, batch_size=1, alpha=1.0, amm=A)
    return ae.unsqueeze(1), wtp.unsqueeze(1)


@torch.no_grad()
def run_heuristic(assign_fn, ae, wtp, M, N, T):
    """Greedy / Auction: the mechanism decides fire-or-hold itself."""
    env = Environment(assignment_encoding=ae.clone(),
                      weapon_to_target_prob=wtp.clone(), max_time=T)
    for _ in range(T):
        action = assign_fn(env.current_target_value[:, :, 0:N],
                           env.weapon_to_target_prob[:, :, :M, :N],
                           env.mask_per_weapon[:, :, :M, :N] > 0)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()


@torch.no_grad()
def run_ours(actor, ae, wtp, M, N, T):
    """Construction pass only: encode, fire/hold, assignment. No search."""
    env = Environment(assignment_encoding=ae.clone(),
                      weapon_to_target_prob=wtp.clone(), max_time=T)
    for _ in range(T):
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob,
                          env.mask_per_weapon)
        p_noop = policy[:, :, :M, N].clamp(1e-6, 1 - 1e-6)
        legal = (env.mask_per_weapon[:, :, :M, :N] > 0).any(-1)
        fire = ((1.0 - p_noop) > 0.5) & legal
        action = auction_round_action_multifire(
            env.current_target_value[:, :, 0:N],
            env.weapon_to_target_prob[:, :, :M, :N],
            env.mask_per_weapon[:, :, :M, :N] > 0,
            must_fire=fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()


@torch.no_grad()
def run_sequential(actor, ae, wtp, M, N, T):
    """One (weapon, target) pair per network call, M calls per stage."""
    env = Environment(assignment_encoding=ae.clone(),
                      weapon_to_target_prob=wtp.clone(), max_time=T)
    for _ in range(T):
        for _ in range(M):
            policy, _ = actor(assignment_embedding=env.assignment_encoding,
                              prob=env.weapon_to_target_prob,
                              mask=env.mask.clone())
            flat = policy.reshape(1, 1, -1)
            idx = int(flat.argmax().item())
            if idx >= M * N:
                break
            m, n = divmod(idx, N)
            action = torch.full((1, 1, M), N, dtype=torch.long, device=DEVICE)
            action[0, 0, m] = n
            env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()


def time_method(fn, instances, M, N, T, warm):
    for V, P, TW, A, PR, C in warm:
        patch_globals(M, N, T, A, PR, C)
        fn(*make_env(V, P, TW, M, N, T, A), M, N, T)
    sync()
    t0 = time.perf_counter()
    for V, P, TW, A, PR, C in instances:
        patch_globals(M, N, T, A, PR, C)
        fn(*make_env(V, P, TW, M, N, T, A), M, N, T)
    sync()
    return (time.perf_counter() - t0) / len(instances)


def main():
    ours = create_gnn_actor_firelogit().to(DEVICE)
    ours.load_state_dict(torch.load(OURS_CKPT, map_location=DEVICE))
    ours.eval()

    seq = create_gnn_actor_sequential().to(DEVICE)
    seq.load_state_dict(torch.load(SEQ_CKPT, map_location=DEVICE))
    seq.eval()

    methods = [
        ("Greedy", lambda ae, wtp, M, N, T: run_heuristic(auction_round_action_multifire, ae, wtp, M, N, T)),
        ("Auction", lambda ae, wtp, M, N, T: run_heuristic(auction_round_action, ae, wtp, M, N, T)),
        ("Sequential", lambda ae, wtp, M, N, T: run_sequential(seq, ae, wtp, M, N, T)),
        ("Ours(K=0)", lambda ae, wtp, M, N, T: run_ours(ours, ae, wtp, M, N, T)),
    ]

    per_tier = {}
    print("%-14s %10s %10s %12s %11s" % ("config", "Greedy", "Auction", "Sequential", "Ours(K=0)"),
          flush=True)
    for tier, (M, N, T) in zip(TIERS, CONFIGS):
        rng = np.random.default_rng(SEED)
        inst = [generate_moderate_temporal_instance(M, N, T, rng=rng) for _ in range(N_EVAL)]
        wrng = np.random.default_rng(SEED + 9991)
        warm = [generate_moderate_temporal_instance(M, N, T, rng=wrng) for _ in range(N_WARM)]

        row = []
        for name, fn in methods:
            row.append(time_method(fn, inst, M, N, T, warm))
        per_tier.setdefault(tier, []).append(row)
        print("%-14s %10.4f %10.4f %12.4f %11.4f"
              % ("%dM_%dN_%dT" % (M, N, T), row[0], row[1], row[2], row[3]), flush=True)

    print("", flush=True)
    print("%-14s %10s %10s %12s %11s" % ("tier", "Greedy", "Auction", "Sequential", "Ours(K=0)"),
          flush=True)
    allrows = []
    for tier in ["Small", "Medium", "Large", "Battlefield"]:
        a = np.array(per_tier[tier]).mean(0)
        allrows.append(a)
        print("%-14s %10.4f %10.4f %12.4f %11.4f" % (tier, a[0], a[1], a[2], a[3]), flush=True)
    a = np.array(allrows).mean(0)
    print("%-14s %10.4f %10.4f %12.4f %11.4f" % ("MEAN", a[0], a[1], a[2], a[3]), flush=True)


if __name__ == "__main__":
    main()
