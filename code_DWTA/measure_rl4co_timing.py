"""Per-instance wall-clock decode time for the AM and POMO baselines.

AM and POMO run inside RL4CO's own environment, so they cannot be timed by
measure_method_timing.py, which drives common/DWTA_Simulator.py directly.
This script times them on the same seed-123 moderate instances, one per
configuration, under single-path greedy decoding, which is how the paper
reports them.
"""
import sys
import time

import numpy as np
import torch

sys.path.append("rl")

from common.TORCH_OBJECTS import DEVICE
from rl4co.models.zoo.am.policy import AttentionModelPolicy

from common.rl4co_dwta_env import DWTAEnv
from common.rl4co_dwta_embeddings import (
    DWTAInitEmbedding,
    DWTADynamicEmbedding,
    DWTAContext,
)
from common.rl4co_eval import load_moderate_fixed_instances_as_td

CONFIGS = [(5, 5, 5), (5, 7, 5), (10, 15, 5),
           (15, 15, 5), (15, 20, 5), (20, 30, 5),
           (30, 30, 10), (30, 40, 10), (40, 50, 10),
           (50, 50, 15), (50, 70, 15), (70, 100, 15)]
TIERS = ["Small"] * 3 + ["Medium"] * 3 + ["Large"] * 3 + ["Battlefield"] * 3
SEED, N_EVAL = 123, 10

CKPTS = {
    "AM": "result/RL4CO_AM_multiscale_seed5_best_policy.pt",
    "POMO": "result/RL4CO_POMO_multiscale_seed5_best_policy.pt",
}


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_policy(policy, env, td):
    """One instance at a time, batch size 1.

    RL4CO would happily decode all ten instances in a single batched forward,
    but every other method in the table processes instances one by one, and a
    batched pass would credit AM and POMO with parallelism the others are not
    given. Each instance is therefore decoded on its own and the times are
    averaged."""
    policy.eval()
    n = td.batch_size[0]
    policy(env.reset(td[0:1].clone()), env=env, decode_type="greedy")  # warm-up
    sync()
    t0 = time.perf_counter()
    for i in range(n):
        policy(env.reset(td[i:i + 1].clone()), env=env, decode_type="greedy")
    sync()
    return (time.perf_counter() - t0) / n


def main():
    # AM and POMO share RL4CO's AttentionModelPolicy; the training scripts
    # differ only in encoder depth and normalization, so rebuild each with the
    # settings its own trainer used before loading the saved state dict.
    specs = {
        "AM": dict(num_encoder_layers=3, normalization="batch"),
        "POMO": dict(num_encoder_layers=6, normalization="batch"),
    }
    policies = {}
    for name, path in CKPTS.items():
        pol = AttentionModelPolicy(
            embed_dim=128,
            num_heads=8,
            env_name="dwta",
            init_embedding=DWTAInitEmbedding(128),
            dynamic_embedding=DWTADynamicEmbedding(128),
            context_embedding=DWTAContext(128),
            use_graph_context=False,
            **specs[name],
        ).to(DEVICE)
        pol.load_state_dict(torch.load(path, map_location=DEVICE))
        policies[name] = pol
        print("loaded %s from %s" % (name, path), flush=True)

    per_tier = {}
    print("%-14s %10s %10s" % ("config", "AM", "POMO"), flush=True)
    for tier, (M, N, T) in zip(TIERS, CONFIGS):
        td, _, _, _ = load_moderate_fixed_instances_as_td(
            DEVICE, M=M, N=N, T=T, seed=SEED, n=N_EVAL)
        env = DWTAEnv(generator_params=dict(num_weapon=M, num_target=N, max_time=T))
        row = [time_policy(policies[k], env, td) for k in ("AM", "POMO")]
        per_tier.setdefault(tier, []).append(row)
        print("%-14s %10.4f %10.4f" % ("%dM_%dN_%dT" % (M, N, T), row[0], row[1]), flush=True)

    print("", flush=True)
    print("%-14s %10s %10s" % ("tier", "AM", "POMO"), flush=True)
    allrows = []
    for tier in ["Small", "Medium", "Large", "Battlefield"]:
        a = np.array(per_tier[tier]).mean(0)
        allrows.append(a)
        print("%-14s %10.4f %10.4f" % (tier, a[0], a[1]), flush=True)
    a = np.array(allrows).mean(0)
    print("%-14s %10.4f %10.4f" % ("MEAN", a[0], a[1]), flush=True)


if __name__ == "__main__":
    main()
