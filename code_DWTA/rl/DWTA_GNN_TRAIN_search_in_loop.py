"""
Train the construction policy THROUGH a fixed local search.

Three changes from the shipped recipe, each motivated by a measurement made
2026-09-11:

1. The policy is credited for the BINARY fire/hold event only, not for which
   target it named. log P(fire_m) = log(1 - pi_m(noop)). The auction decides
   targets, so target choice was never used at inference; crediting it was
   training on a signal the pipeline discards. Note this is NOT the binary
   actor that collapsed in earlier sessions: the network keeps its full
   softmax over targets, only the credited EVENT is binary, so the bang-bang
   parameterisation that caused those collapses does not arise here.

2. A fixed (non-learned) hill-climbing local search runs inside the training
   loop, and the policy is rewarded on the POST-SEARCH objective. Measurement:
   with a proper accept-if-improving operator, random move ordering beat the
   trained selector (0.1870 vs 0.1942 at K=10 over four configs), so the
   better search is currently the non-learned one. Keeping it fixed also
   removes the moving-target problem of training two policies against
   each other.

3. The search never calls the network. Once a fire/hold schedule exists, an
   edit only needs the auction plus the simulator, so a K-edit search costs
   K auction replays rather than K full encoder passes.

The auction is in the training loop here, unlike the shipped construction
recipe, so train and inference now see the same target-assignment rule.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from common.TORCH_OBJECTS import DEVICE
from common.DWTA_GNN_comm_sinkhorn import create_gnn_actor_comm_sinkhorn
from common.DWTA_GNN import create_gnn_actor as create_gnn_actor_nocomm
from common.DWTA_GNN_firelogit import create_gnn_actor_firelogit
from common.DWTA_Simulator import Environment
from common.Dynamic_Instance_generation import input_generation
from common.Dynamic_Instance_generation_moderate import generate_moderate_training_instances
from common.auction_refinement import auction_round_action_multifire
from common.temporal_dilemma_generator_moderate import generate_moderate_temporal_instance
from Dynamic_Sampling_GNN import patch_hyperparameters_for_epoch, restore_hyperparameters
from Dynamic_Sampling_GNN_moderate import get_random_moderate_problem_size

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code_DWTA"))
from eval_tiered_benchmark import patch_globals

EVAL_CONFIGS = [
    (5, 5, 5), (5, 7, 5), (10, 15, 5),
    (15, 15, 5), (15, 20, 5), (20, 30, 5),
    (30, 30, 10), (30, 40, 10), (40, 50, 10),
    (50, 50, 15), (50, 70, 15), (70, 100, 15),
]
EVAL_SEED = 123
EVAL_N = 10
EPS = 1e-9

# Checkpoint selection runs on its own instances. They are drawn at the
# training size range under a seed that is used nowhere else, so no instance
# that appears in the reported table can influence which checkpoint is kept.
VAL_SEED = 7717
VAL_N = 20
VAL_SIZES = [(m, n, t) for m in (5, 6, 7) for n in (5, 6, 7) for t in (5, 6, 7)]


@torch.no_grad()
def simulate_fixed(ae, wtp, fire, nw, nt, mt):
    """Replay a given fire/hold schedule through auction + simulator.
    No network calls -- this is what makes the in-loop search cheap."""
    env = Environment(assignment_encoding=ae.clone(),
                      weapon_to_target_prob=wtp.clone(), max_time=mt)
    original = env.original_target_value[:, :, 0:nt].sum(2)
    fired = torch.zeros_like(original)
    for t in range(mt):
        action = auction_round_action_multifire(
            env.current_target_value[:, :, 0:nt],
            env.weapon_to_target_prob[:, :, :nw, :nt],
            env.mask_per_weapon[:, :, :nw, :nt] > 0,
            must_fire=fire[:, :, t, :])
        fired = fired + (action < nt).float().sum(-1)
        env.update_internal_variables_parallel(selected_actions=action)
        env.time_update()
    final = env.current_target_value[:, :, 0:nt].sum(2)
    return final / (original + 1e-8), fired


@torch.no_grad()
def hillclimb(ae, wtp, fire, nw, nt, mt, k, rng):
    """Fixed, non-learned local search over the fire/hold schedule.
    Proposes a uniformly random untried slot, accepts only on strict
    improvement. Operates independently per (batch, para) element."""
    B, P = fire.shape[0], fire.shape[1]
    n = mt * nw
    best_obj, fired = simulate_fixed(ae, wtp, fire, nw, nt, mt)
    best = fire.clone()
    tried = torch.zeros(B, P, n, dtype=torch.bool, device=DEVICE)
    for _ in range(k):
        flat = best.reshape(B, P, n)
        avail = ~tried
        if not bool(avail.any()):
            break
        # one random available slot per (b,p)
        r = torch.rand(B, P, n, device=DEVICE).masked_fill(~avail, -1.0)
        pick = r.argmax(dim=-1)
        cand = flat.clone()
        sel = torch.zeros_like(flat).scatter_(-1, pick.unsqueeze(-1), True)
        cand = cand ^ sel
        obj, f2 = simulate_fixed(ae, wtp, cand.reshape(B, P, mt, nw), nw, nt, mt)
        better = obj < best_obj - EPS
        bexp = better.unsqueeze(-1).expand_as(flat)
        best = torch.where(bexp, cand, flat).reshape(B, P, mt, nw)
        best_obj = torch.where(better, obj, best_obj)
        fired = torch.where(better, f2, fired)
        tried = tried.scatter(-1, pick.unsqueeze(-1), True)
        tried = torch.where(better.unsqueeze(-1).expand_as(tried),
                            torch.zeros_like(tried), tried)
    return best_obj, best, fired


def rollout(actor, ae, wtp, nw, nt, mt, sample=True):
    """Construction pass. Only the binary fire/hold event is credited."""
    env = Environment(assignment_encoding=ae.clone(),
                      weapon_to_target_prob=wtp.clone(), max_time=mt)
    logps, fires = [], []
    for t in range(mt):
        policy, _ = actor(env.assignment_encoding, env.weapon_to_target_prob,
                          env.mask_per_weapon)
        p_noop = policy[:, :, :nw, nt].clamp(1e-6, 1 - 1e-6)
        p_fire = 1.0 - p_noop
        legal = (env.mask_per_weapon[:, :, :nw, :nt] > 0).any(-1)
        if sample:
            fire = torch.bernoulli(p_fire).bool()
        else:
            fire = p_fire > 0.5
        fire = fire & legal
        lp = torch.where(fire, p_fire.log(), p_noop.log()) * legal.float()
        logps.append(lp.sum(-1))
        fires.append(fire)
        with torch.no_grad():
            action = auction_round_action_multifire(
                env.current_target_value[:, :, 0:nt],
                env.weapon_to_target_prob[:, :, :nw, :nt],
                env.mask_per_weapon[:, :, :nw, :nt] > 0,
                must_fire=fire)
            env.update_internal_variables_parallel(selected_actions=action)
            env.time_update()
    return torch.stack(logps, 0).sum(0), torch.stack(fires, dim=2)


def train_step(actor, k_edits, batch, para, rng, ammo_coef=0.1):
    nw, nt, mt, amm, prep, cost = get_random_moderate_problem_size()
    saved = patch_hyperparameters_for_epoch(nw, nt, mt, amm, prep_list=prep, cost_list=cost)
    try:
        ae, wtp = generate_moderate_training_instances(
            batch_size=batch, num_weapons=nw, num_targets=nt, max_time=mt, amm_list=amm)
        ae = ae.unsqueeze(1).repeat(1, para, 1, 1).contiguous()
        wtp = wtp.unsqueeze(1).repeat(1, para, 1, 1).contiguous()

        logp, fire = rollout(actor, ae, wtp, nw, nt, mt, sample=True)
        pre_obj, _ = simulate_fixed(ae, wtp, fire, nw, nt, mt)
        post_obj, _, fired = hillclimb(ae, wtp, fire, nw, nt, mt, k_edits, rng)

        total_ammo = max(float(sum(amm)), 1.0)
        ret = (1.0 - post_obj) + ammo_coef * (fired / total_ammo)
        adv = ret - ret.mean(dim=1, keepdim=True)
        adv = adv / adv.std(dim=1, keepdim=True).clamp_min(1e-6)

        loss = -(logp * adv.detach()).mean()
        actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor.optimizer.step()
        return loss.item(), pre_obj.mean().item(), post_obj.mean().item()
    finally:
        restore_hyperparameters(saved)


@torch.no_grad()
@torch.no_grad()
def validate(actor, k_eval, rng_seed=VAL_SEED, n=VAL_N):
    """Mean post-search objective on held-out instances at training scale.

    Used only to decide which checkpoint to keep. Sizes are drawn from the
    same range the curriculum trains on, so this measures the policy on the
    distribution it was fit to rather than on the larger configurations the
    paper reports, which must stay untouched by model selection.
    """
    actor.eval()
    rng_sz = np.random.default_rng(rng_seed)
    rng_in = np.random.default_rng(rng_seed + 1)
    objs = []
    for _ in range(n):
        M, N, T = VAL_SIZES[int(rng_sz.integers(0, len(VAL_SIZES)))]
        V, P, TW, A, PR, C = generate_moderate_temporal_instance(M, N, T, rng=rng_in)
        patch_globals(M, N, T, A, PR, C)
        ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V,
                                   prob=np.asarray(P), TW=TW, max_time=T,
                                   batch_size=1, alpha=1.0, amm=A)
        ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
        _, fire = rollout(actor, ae, wtp, M, N, T, sample=False)
        post, _, _ = hillclimb(ae, wtp, fire, M, N, T, k_eval,
                               np.random.default_rng(rng_seed))
        objs.append(post.mean().item())
    actor.train()
    return float(np.mean(objs))


def evaluate(actor, k_eval, rng_seed=EVAL_SEED, configs=None):
    """configs=None evaluates all twelve held-out configurations. Pass a
    subset (one per tier) for checkpoint selection during training; the
    final cross-seed table is produced separately by eval_final_table.py,
    which re-scores every saved checkpoint on the full twelve under one
    protocol, so a cheaper in-training criterion does not break
    comparability of the reported numbers."""
    actor.eval()
    pre_all, post_all = [], []
    for (M, N, T) in (configs if configs is not None else EVAL_CONFIGS):
        rng_i = np.random.default_rng(rng_seed)
        inst = [generate_moderate_temporal_instance(M, N, T, rng=rng_i) for _ in range(EVAL_N)]
        pre_c, post_c = [], []
        for V, P, TW, A, PR, C in inst:
            patch_globals(M, N, T, A, PR, C)
            ae, wtp = input_generation(NUM_WEAPON=M, NUM_TARGET=N, value=V, prob=np.asarray(P),
                                       TW=TW, max_time=T, batch_size=1, alpha=1.0, amm=A)
            ae, wtp = ae.unsqueeze(1), wtp.unsqueeze(1)
            _, fire = rollout(actor, ae, wtp, M, N, T, sample=False)
            pre, _ = simulate_fixed(ae, wtp, fire, M, N, T)
            post, _, _ = hillclimb(ae, wtp, fire, M, N, T, k_eval,
                                   np.random.default_rng(rng_seed))
            pre_c.append(pre.mean().item())
            post_c.append(post.mean().item())
        pre_all.append(float(np.mean(pre_c)))
        post_all.append(float(np.mean(post_c)))
    actor.train()
    return float(np.mean(pre_all)), float(np.mean(post_all)), pre_all, post_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_steps", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--k_edits", type=int, default=3)
    ap.add_argument("--k_eval", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=15)
    ap.add_argument("--para_size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--tag", type=str, default="sil")
    ap.add_argument("--arch", choices=["comm", "nocomm", "firelogit"], default="comm",
                    help="'comm' is the reported architecture (weapon-to-weapon "
                         "self-attention + inactive Sinkhorn term); 'nocomm' is the "
                         "same encoder and heads with that attention layer removed, "
                         "for the communication ablation.")
    ap.add_argument("--fast_eval", action="store_true",
                    help="During training, evaluate on one configuration per tier "
                         "instead of all twelve. Checkpoint selection only; the "
                         "reported table is regenerated by eval_final_table.py.")
    args = ap.parse_args()

    if args.fast_eval:
        print("fast_eval is a no-op: selection now runs on the validation set"
              " at training scale, which is already cheap.", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    mk_actor = {"comm": create_gnn_actor_comm_sinkhorn,
                "nocomm": create_gnn_actor_nocomm,
                "firelogit": create_gnn_actor_firelogit}[args.arch]
    actor = mk_actor().to(DEVICE)
    print("arch=%s | %d parameters" % (args.arch,
          sum(p.numel() for p in actor.parameters())), flush=True)
    actor.train()

    best = float("inf")
    ckpt = "result/SearchInLoop_seed%d_%s_best.pt" % (args.seed, args.tag)
    t0 = time.time()
    for step in range(1, args.total_steps + 1):
        loss, pre, post = train_step(actor, args.k_edits, args.batch_size, args.para_size, rng)
        if step % 5 == 0:
            print("step %4d | loss %8.4f | pre %.4f -> post %.4f | %.0fs"
                  % (step, loss, pre, post, time.time() - t0), flush=True)
        if step % args.eval_every == 0:
            val = validate(actor, args.k_eval)
            flag = ""
            if val < best:
                best = val
                torch.save(actor.state_dict(), ckpt)
                flag = "  <-- best, saved"
            print("VAL  step %4d | held-out @ train scale, K=%d: %.4f%s"
                  % (step, args.k_eval, val, flag), flush=True)
    print("")
    print("best validation objective: %.4f" % best)
    print("checkpoint: %s" % ckpt)
    print("NOTE: the twelve reported configurations were not used for"
          " selection; score this checkpoint with eval_final_table.py.")


if __name__ == "__main__":
    main()
