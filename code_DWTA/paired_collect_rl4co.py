"""
Per-instance objectives for AM and POMO on the 120 seed-123 instances.
Companion to paired_collect.py; needs the RL4CO environment:

  ../venv/Scripts/python.exe paired_collect_rl4co.py

Same checkpoints, policy construction and greedy single-path decode as
measure_rl4co_timing.py and the training-time evaluation. Output:
result/paired_instances_rl4co.csv
"""
import csv

import torch

from rl4co.models.zoo.am.policy import AttentionModelPolicy

from common.rl4co_dwta_env import DWTAEnv
from common.rl4co_dwta_embeddings import DWTAContext, DWTADynamicEmbedding, DWTAInitEmbedding
from common.rl4co_eval import load_moderate_fixed_instances_as_td
from measure_rl4co_timing import CKPTS, CONFIGS, SEED, N_EVAL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPECS = {
    "AM": dict(num_encoder_layers=6, normalization="batch"),
    "POMO": dict(num_encoder_layers=6, normalization="batch"),
}
OUT = "result/paired_instances_rl4co.csv"


@torch.no_grad()
def per_instance(policy, env, td):
    init_value = td["value"].sum(-1).clone()
    out = policy(env.reset(td.clone()), env=env, decode_type="greedy")
    return ((-out["reward"]) / init_value.clamp_min(1e-8)).tolist()


def main():
    policies = {}
    for name, path in CKPTS.items():
        pol = AttentionModelPolicy(
            embed_dim=128, num_heads=8, env_name="dwta",
            init_embedding=DWTAInitEmbedding(128),
            dynamic_embedding=DWTADynamicEmbedding(128),
            context_embedding=DWTAContext(128),
            use_graph_context=False, **SPECS[name],
        ).to(DEVICE)
        pol.load_state_dict(torch.load(path, map_location=DEVICE))
        pol.eval()
        policies[name] = pol

    rows = []
    for (M, N, T) in CONFIGS:
        cfg = "%dM_%dN_%dT" % (M, N, T)
        td, _, _, _ = load_moderate_fixed_instances_as_td(DEVICE, M=M, N=N, T=T, seed=SEED, n=N_EVAL)
        env = DWTAEnv(generator_params=dict(num_weapon=M, num_target=N, max_time=T))
        means = []
        for name, pol in policies.items():
            vals = per_instance(pol, env, td)
            rows += [(name, cfg, i, v) for i, v in enumerate(vals)]
            means.append("%s=%.4f" % (name, sum(vals) / len(vals)))
        print(cfg, " ".join(means), flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "config", "instance", "objective"])
        w.writerows(rows)
    print("wrote %d rows to %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
