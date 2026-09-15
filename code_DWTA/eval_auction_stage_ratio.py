"""
Empirical per-stage quality of the many-to-one auction: given the firing set F
chosen by the trained policy at each stage, compare the auction's allocation
value f_t(S_auc) against the exact stage optimum OPT_t(F), computed by brute
force enumeration of every feasible assignment of the firing weapons to their
legal targets. Feasible only at Small tier (5 weapons: <= 7^5 combinations).

Supports the manuscript claim that the 1/2 guarantee is a loose worst-case
floor and the mechanism is near-optimal on real instances.
"""
import itertools

import numpy as np
import torch

from common.DWTA_GNN_comm import create_gnn_actor_comm
from common.TORCH_OBJECTS import DEVICE
from common.Dynamic_Instance_generation import input_generation
from common.DWTA_Simulator import Environment
from common.auction_refinement import auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from eval_tiered_benchmark import patch_globals

CONFIGS = [(5, 5, 5), (5, 7, 5), (5, 15, 5)]  # weapons fixed at 5 so OPT stays enumerable; last config probes more targets
N_EVAL = 10
SEED = 123
CKPT = "result/CommSCoPE_multiscale_seed5_best_actor.pt"


def stage_value(rv, prob, assign):
    """f_t of an assignment {weapon m -> target n}: sum_n rv_n * (1 - prod (1-P))."""
    destroyed = 0.0
    for n in set(assign.values()):
        surv = 1.0
        for m, tgt in assign.items():
            if tgt == n:
                surv *= (1.0 - prob[m, n])
        destroyed += rv[n] * (1.0 - surv)
    return destroyed


def stage_opt(rv, prob, legal, firing):
    """Exact OPT_t(F) by enumeration. Monotonicity means every firing weapon
    with a legal target is assigned in some optimum, so enumerating full
    products over legal targets suffices."""
    weapons = [m for m in firing if legal[m].any()]
    if not weapons:
        return 0.0
    choice_lists = [np.flatnonzero(legal[m]) for m in weapons]
    best = 0.0
    for combo in itertools.product(*choice_lists):
        best = max(best, stage_value(rv, prob, dict(zip(weapons, combo))))
    return best


@torch.no_grad()
def run_instance(actor, V, P, TW, nw, nt, mt, amm, prep, cost):
    patch_globals(nw, nt, mt, amm, prep, cost)
    ae, wtp = input_generation(NUM_WEAPON=nw, NUM_TARGET=nt, value=V, prob=P, TW=TW,
                               max_time=mt, batch_size=1, alpha=1.0, amm=amm)
    ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
    env = Environment(assignment_encoding=ae, weapon_to_target_prob=wtp, max_time=mt)

    ratios = []
    for _ in range(mt):
        remaining_value = env.current_target_value[:, :, 0:nt]
        prob_t = env.weapon_to_target_prob[:, :, :nw, :nt]
        legal_mask = env.mask_per_weapon[:, :, :nw, :nt] > 0
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob, env.mask_per_weapon)
        policy_choice = policy[:, :, :nw, :].argmax(dim=-1)
        must_fire = policy_choice < nt
        action = auction_round_action_multifire(remaining_value, prob_t, legal_mask,
                                                must_fire=must_fire)

        rv = remaining_value[0, 0].detach().cpu().numpy()
        pr = prob_t[0, 0].detach().cpu().numpy()
        lg = legal_mask[0, 0].detach().cpu().numpy()
        mf = must_fire[0, 0].detach().cpu().numpy()
        act = action[0, 0].detach().cpu().numpy()

        auc_assign = {m: int(act[m]) for m in range(nw) if act[m] < nt}
        f_auc = stage_value(rv, pr, auc_assign)
        opt = stage_opt(rv, pr, lg, [m for m in range(nw) if mf[m]])
        if opt > 1e-9:
            ratios.append(f_auc / opt)

        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    return ratios


if __name__ == "__main__":
    actor = create_gnn_actor_comm().to(DEVICE)
    actor.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False))
    actor.eval()

    all_ratios = []
    for M, N, T in CONFIGS:
        rng = np.random.default_rng(SEED)
        cfg_ratios = []
        for _ in range(N_EVAL):
            V, P, TW, AMM, PREP, COST = generate_moderate_temporal_instance(M, N, T, rng=rng)
            cfg_ratios.extend(run_instance(actor, V, np.asarray(P), TW, M, N, T, AMM, PREP, COST))
        all_ratios.extend(cfg_ratios)
        r = np.array(cfg_ratios)
        print(f"{M}M_{N}N_{T}T: stages={len(r)}  mean={r.mean():.4f}  min={r.min():.4f}  "
              f"share_optimal={(r > 1 - 1e-9).mean():.2%}", flush=True)

    r = np.array(all_ratios)
    print(f"\nALL: stages={len(r)}  mean={r.mean():.4f}  min={r.min():.4f}  "
          f"share_optimal={(r > 1 - 1e-9).mean():.2%}")
    print("(ratio = f_t(auction) / exact OPT_t(F); guarantee floor is 0.5)")
