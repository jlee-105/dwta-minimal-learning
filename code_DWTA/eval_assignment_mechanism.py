"""Same trained fire/hold policy, two different target-assignment mechanisms.

The reported pipeline assigns targets with auction_round_action_multifire,
which is greedy best-marginal-value-first with a survival update after each
pick (many-to-one, no prices). The baseline table's Auction column instead
uses auction_round_action, Bertsekas price-based bidding with eviction
(one weapon per target).

Those two were never compared under a common fire/hold decision: in the
baseline table each mechanism also decided fire/hold by its own myopic rule,
so the gap there mixes assignment quality with how often each rule holds
fire. This script removes that confound. It takes the trained policy's
fire/hold schedule and replays it under each mechanism, so the only thing
that differs is how firing weapons are matched to targets.

No search (K=0), greedy decoding, the same ten seed-123 instances per
configuration as eval_final_table.py.

CAVEAT: the policy was trained with multifire in the loop, so the price
auction arm is off-distribution. Its numbers are a lower bound on what a
policy trained against it would reach.

Usage:
  python eval_assignment_mechanism.py
  python eval_assignment_mechanism.py --ckpts result/SearchInLoop_seed5_nocomm_best.pt
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.append("rl")

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN import create_gnn_actor as create_gnn_actor_nocomm
from common.DWTA_Simulator import Environment
from common.Dynamic_Instance_generation import input_generation
from common.auction_refinement import (
    auction_round_action,
    auction_round_action_multifire,
    auction_round_action_price_multifire,
)
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
EVAL_SEED = 123
EVAL_N = 10

DEFAULT_CKPTS = [
    "result/SearchInLoop_seed5_nocomm_best.pt",
    "result/SearchInLoop_seed6_nocomm_best.pt",
    "result/SearchInLoop_seed7_nocomm_best.pt",
]

MECHANISMS = {
    "multifire": auction_round_action_multifire,          # greedy marginal, many-to-one
    "price1to1": auction_round_action,                    # price auction, 1-to-1
    "priceMany": auction_round_action_price_multifire,    # price auction, many-to-one
}


@torch.no_grad()
def run_episode(actor, assign_fn, ae, wtp, nw, nt, mt):
    """One construction pass: the policy decides fire/hold at every stage and
    assign_fn matches the firing weapons to targets. Returns the normalized
    remaining value."""
    env = Environment(assignment_encoding=ae.clone(),
                      weapon_to_target_prob=wtp.clone(), max_time=mt)
    original = env.original_target_value[:, :, 0:nt].sum(2)
    for _ in range(mt):
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob,
                          env.mask_per_weapon)
        p_noop = policy[:, :, :nw, nt].clamp(1e-6, 1 - 1e-6)
        legal = (env.mask_per_weapon[:, :, :nw, :nt] > 0).any(-1)
        fire = ((1.0 - p_noop) > 0.5) & legal
        action = assign_fn(
            env.current_target_value[:, :, 0:nt],
            env.weapon_to_target_prob[:, :, :nw, :nt],
            env.mask_per_weapon[:, :, :nw, :nt] > 0,
            must_fire=fire)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    final = env.current_target_value[:, :, 0:nt].sum(2)
    return (final / (original + 1e-8)).mean().item()


@torch.no_grad()
def eval_checkpoint(path):
    actor = create_gnn_actor_nocomm().to(DEVICE)
    actor.load_state_dict(torch.load(path, map_location=DEVICE))
    actor.eval()

    per_config = {name: [] for name in MECHANISMS}
    print("=== %s ===" % path, flush=True)
    print("%-14s %10s %10s %10s" % ("config", "multifire", "price1to1", "priceMany"), flush=True)

    for (M, N, T) in EVAL_CONFIGS:
        rng_i = np.random.default_rng(EVAL_SEED)
        inst = [generate_moderate_temporal_instance(M, N, T, rng=rng_i)
                for _ in range(EVAL_N)]
        means = {}
        for name, fn in MECHANISMS.items():
            objs = []
            for V, P, TW, A, PR, C in inst:
                patch_globals(M, N, T, A, PR, C)
                ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V,
                                           prob=np.asarray(P), TW=TW, max_time=T,
                                           batch_size=1, alpha=1.0, amm=A)
                ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
                objs.append(run_episode(actor, fn, ae, wtp, M, N, T))
            means[name] = float(np.mean(objs))
            per_config[name].append(means[name])
        print("%-14s %10.4f %10.4f %10.4f"
              % ("%dM_%dN_%dT" % (M, N, T), means["multifire"], means["price1to1"],
                 means["priceMany"]), flush=True)

    overall = {name: float(np.mean(v)) for name, v in per_config.items()}
    print("%-14s %10.4f %10.4f %10.4f"
          % ("MEAN", overall["multifire"], overall["price1to1"],
             overall["priceMany"]), flush=True)
    print("", flush=True)
    return per_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS)
    args = ap.parse_args()

    t0 = time.time()
    all_runs = [eval_checkpoint(p) for p in args.ckpts]

    print("=== ACROSS-SEED SUMMARY (%d checkpoints, K=0) ===" % len(all_runs), flush=True)
    print("%-14s %18s %18s %18s" % ("config", "multifire", "price1to1", "priceMany"), flush=True)
    for i, (M, N, T) in enumerate(EVAL_CONFIGS):
        cells = []
        for name in MECHANISMS:
            vals = [r[name][i] for r in all_runs]
            cells.append("%.4f+-%.4f" % (float(np.mean(vals)), float(np.std(vals))))
        print("%-14s %18s %18s %18s" % ("(%d,%d,%d)" % (M, N, T), cells[0], cells[1], cells[2]), flush=True)

    means = {}
    for name in MECHANISMS:
        means[name] = float(np.mean([np.mean(r[name]) for r in all_runs]))
    print("%-14s %18.4f %18.4f %18.4f" % ("MEAN", means["multifire"], means["price1to1"], means["priceMany"]), flush=True)
    print("elapsed %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
