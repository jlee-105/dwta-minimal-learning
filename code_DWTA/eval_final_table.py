"""
Per-configuration table for a trained fire/hold checkpoint.

Produces the two tables the paper needs from one pass:
  (a) main table row  -- construction-only and construction+search per config
  (b) budget sweep    -- objective at K = 0, 3, 10, 30, plus the edit index
                         after which no further edit was accepted

Accepts multiple checkpoints so the multi-seed table can be produced in one
run; reports per-seed values and the across-seed mean and standard deviation,
which is what a Q1 venue will expect rather than a single-seed number.

Usage:
  python eval_final_table.py --ckpts result/SearchInLoop_seed5_fireonly_best.pt \
                                     result/SearchInLoop_seed6_fireonly_best.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append("rl")

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN import create_gnn_actor as create_gnn_actor_nocomm
from common.Dynamic_Instance_generation import input_generation
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "sil", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rl", "DWTA_GNN_TRAIN_search_in_loop.py"))
sil = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sil)

CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
TIERS = ["Small"] * 3 + ["Medium"] * 3 + ["Large"] * 3 + ["Battlefield"] * 3
K_REPORT = [0, 3, 10, 30]
N_EVAL = 10
SEED = 123


@torch.no_grad()
def curve_for_instance(actor, ae, wtp, nw, nt, mt, kmax, rng):
    """Best-so-far objective after each of kmax re-simulations, plus the
    index of the last accepted flip (0 if none were accepted)."""
    fire = sil.rollout(actor, ae, wtp, nw, nt, mt, sample=False)[1]
    n = mt * nw
    best_obj, _ = sil.simulate_fixed(ae, wtp, fire, nw, nt, mt)
    best = best_obj.mean().item()
    out = [best]
    cur = fire.clone()
    tried = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    last_accept = 0
    for k in range(1, kmax + 1):
        flat = cur.reshape(n)
        avail = ~(tried)
        if not bool(avail.any()):
            out.append(best)
            continue
        pick = int(rng.choice(avail.nonzero(as_tuple=True)[0].tolist()))
        cand = flat.clone()
        cand[pick] = ~cand[pick]
        cand = cand.reshape(1, 1, mt, nw)
        obj, _ = sil.simulate_fixed(ae, wtp, cand, nw, nt, mt)
        v = obj.mean().item()
        if v < best - 1e-9:
            cur, best, last_accept = cand, v, k
            tried = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        else:
            tried[pick] = True
        out.append(best)
    return out, last_accept


def eval_ckpt(path, kmax, arch="comm"):
    mk = create_gnn_actor_comm_sinkhorn if arch == "comm" else create_gnn_actor_nocomm
    actor = mk().to(DEVICE)
    actor.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
    actor.eval()
    per_cfg = []
    for (M, N, T) in CONFIGS:
        rng_i = np.random.default_rng(SEED)
        inst = [generate_moderate_temporal_instance(M, N, T, rng=rng_i) for _ in range(N_EVAL)]
        curves, accepts = [], []
        for i, (V, P, TW, A, PR, C) in enumerate(inst):
            patch_globals(M, N, T, A, PR, C)
            ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V,
                                       prob=np.asarray(P), TW=TW, max_time=T,
                                       batch_size=1, alpha=1.0, amm=A)
            c, la = curve_for_instance(actor, ae.unsqueeze(1), wtp.unsqueeze(1),
                                       M, N, T, kmax, np.random.default_rng(SEED + i))
            curves.append(c)
            accepts.append(la)
        per_cfg.append((np.array(curves).mean(axis=0), float(np.mean(accepts))))
        c = per_cfg[-1][0]
        print("  %-13s K0=%.4f K3=%.4f K10=%.4f K30=%.4f  lastAccept=%.1f"
              % ("%dM_%dN_%dT" % (M, N, T), c[0], c[min(3, kmax)],
                 c[min(10, kmax)], c[min(30, kmax)], per_cfg[-1][1]), flush=True)
    return per_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--arch", choices=["comm", "nocomm"], default="comm")
    args = ap.parse_args()

    all_runs = []
    for p in args.ckpts:
        if not os.path.exists(p):
            print("MISSING: %s -- skipped" % p)
            continue
        print("=== %s ===" % p, flush=True)
        all_runs.append(eval_ckpt(p, args.kmax, args.arch))

    if not all_runs:
        print("no checkpoints evaluated")
        return

    ns = len(all_runs)
    print("")
    print("=== ACROSS-SEED SUMMARY (%d seed%s) ===" % (ns, "" if ns == 1 else "s"))
    hdr = "%-11s %-13s" % ("Tier", "Config")
    for k in K_REPORT:
        hdr += " %15s" % ("K=%d" % k)
    print(hdr)
    means_by_k = {k: [] for k in K_REPORT}
    for ci, (M, N, T) in enumerate(CONFIGS):
        line = "%-11s %-13s" % (TIERS[ci], "(%d,%d,%d)" % (M, N, T))
        for k in K_REPORT:
            vals = [run[ci][0][min(k, args.kmax)] for run in all_runs]
            means_by_k[k].append(float(np.mean(vals)))
            if ns == 1:
                line += " %15.4f" % np.mean(vals)
            else:
                line += " %9.4f+-%.4f" % (np.mean(vals), np.std(vals))
        print(line)
    line = "%-11s %-13s" % ("", "MEAN")
    for k in K_REPORT:
        line += " %15.4f" % float(np.mean(means_by_k[k]))
    print(line)

    print("")
    print("Reference means from earlier experiments (same instances, same protocol):")
    print("  Sequential, single pass                      0.1432")
    print("  parallel fire/hold + auction + learned edits  0.1325  (K=10)")
    print("  sequential fire/hold + auction + learned edits 0.1373 (K=10)")


if __name__ == "__main__":
    main()
