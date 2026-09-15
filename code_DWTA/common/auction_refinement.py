"""
Auction-inspired inference-time refinement for the parallel multi-pointer
decoder.

Motivation: the parallel decoder commits all M weapons simultaneously from one
shared pre-round state, with no visibility into one another's this-round pick
-- the "mean-field coordination failure" documented in the paper (Proposition
2 / round-0 target-concentration finding): multiple weapons independently
identify the same "obviously good" target and pile onto it, exactly where the
classical max-marginal-return greedy baseline (which commits weapons one at a
time, re-evaluating remaining value after each pick) has an advantage.

This module fixes that at INFERENCE time only (no retraining, and only ONE
neural forward pass per round -- the environment's own value/prob tensors are
used for everything after that), via a price-based bidding process directly
inspired by Bertsekas's auction algorithm for assignment problems. Note this
is NOT a literal instance of that algorithm: Bertsekas's classical auction
(and its convergence/near-optimality guarantee) assumes a one-to-one
bipartite assignment, whereas DWTAP legally allows many weapons to engage the
same target (many-to-one). What follows is an auction-*inspired* heuristic
for that many-to-one setting -- no formal convergence guarantee is claimed;
its effect is validated empirically (Section~\\ref{sec:results}).

Bids are processed one weapon at a time (per (batch,para) item), exactly as
in Bertsekas's original algorithm, rather than having all weapons re-bid
synchronously every iteration -- a synchronous version was tried first and
found to oscillate indefinitely when multiple weapons have near-identical
value estimates (they all switch targets in lockstep, forever chasing each
other), since nothing breaks the symmetry between simultaneous movers.
Processing bids one at a time removes this failure mode by construction.
"""
import torch


@torch.no_grad()
def auction_round_action(remaining_value, prob, legal_mask, must_fire=None, eps=1e-3, max_rounds=None):
    """
    One round of auction-refined action selection, processed independently
    per (batch, para) item (a Python loop over B*P -- fine at inference time;
    this function is never called during training).

    Args:
        remaining_value: [B, P, N] current remaining value per target.
        prob: [B, P, M, N] weapon-target damage probability.
        legal_mask: [B, P, M, N] bool/float, True/1 where (weapon,target) is
            a legal edge this round (ammo, reload, time-window). No-op is
            always legal and is not part of this tensor.
        must_fire: [B, P, M] bool or None. If given, marks which weapons the
            trained policy has ALREADY decided should fire this round (vs.
            hold/no-op) -- those weapons are exempt from the auction's own
            no-op rule and always end up assigned to their best available
            target, no matter how the price landscape looks. This preserves
            the policy's own fire-vs-hold judgment (which training, not this
            heuristic, is responsible for -- see module docstring) and
            restricts the auction to what it is actually good at: resolving
            WHICH target among weapons already committed to firing. Without
            this, the auction's own immediate-net-value no-op rule is exactly
            as myopic as the classical greedy baseline and reintroduces the
            same "hold now for a better target later" failure mode that
            training was specifically shown to fix (see toy trap instance in
            Section~\\ref{sec:training}). If None, all weapons may be
            auctioned into no-op (matches the original, policy-agnostic
            behavior -- NOT recommended for the main pipeline).
        eps: minimum bid increment (Bertsekas's epsilon-scaling parameter --
            guarantees termination of the classical algorithm; here it
            mainly prevents zero-size, infinite-looping price increments).
        max_rounds: safety cap on total bid rounds; defaults to 20*M.

    Returns:
        action: [B, P, M] long tensor, values in 0..N-1 for a target choice,
            or N for no-op -- same convention as the trained actor's action.
    """
    B, P, M, N = prob.shape
    device = prob.device
    value = (remaining_value.unsqueeze(2) * prob).masked_fill(~legal_mask.bool(), float("-inf"))
    if max_rounds is None:
        max_rounds = 20 * M
    # must_fire=None means "no policy fire/hold decision to respect" -- let the
    # auction itself decide fire vs. no-op from immediate net value (matches
    # the original, policy-agnostic behavior). This is NOT the same as an
    # all-False must_fire tensor, which would force every weapon to no-op
    # unconditionally.
    respect_policy_hold = must_fire is not None
    if must_fire is None:
        must_fire = torch.ones(B, P, M, dtype=torch.bool, device=device)

    action = torch.full((B, P, M), N, dtype=torch.long, device=device)  # start all no-op

    for b in range(B):
        for p in range(P):
            v = value[b, p]  # [M, N]
            mf = must_fire[b, p]  # [M] bool
            price = torch.zeros(N, device=device)
            assigned_to = torch.full((M,), -1, dtype=torch.long, device=device)  # -1 = unassigned/no-op
            unassigned = list(range(M))

            rounds = 0
            while unassigned and rounds < max_rounds:
                rounds += 1
                m = unassigned.pop(0)
                if respect_policy_hold and not mf[m]:
                    # Policy already decided this weapon holds fire this
                    # round -- authoritative, not just a tie-break: the
                    # auction never overrides a hold with a fire, even if
                    # some target's immediate net value is positive (that
                    # would silently reintroduce greedy's exact myopia).
                    assigned_to[m] = -1
                    continue
                net = v[m] - price  # [N]
                no_legal_target = not torch.isfinite(net).any()
                if no_legal_target:
                    assigned_to[m] = -1
                    continue
                if not respect_policy_hold and net.max() < 0:
                    # No policy fire/hold decision available -- fall back to
                    # deciding fire vs. no-op from immediate net value alone
                    # (original, policy-agnostic behavior).
                    assigned_to[m] = -1
                    continue
                # Policy committed this weapon to firing -- take its best
                # available (possibly negative-net) legal target.
                best_n = int(net.argmax().item())
                best_val = net[best_n]
                second_val = net.clone()
                second_val[best_n] = float("-inf")
                second_best_val = second_val.max() if torch.isfinite(second_val).any() else best_val - eps
                bid = (best_val - second_best_val).clamp_min(0) + eps

                # Evict any weapon currently holding best_n; it goes back into
                # the unassigned queue and will re-bid on its next-best option
                # once prices have updated -- this is exactly Bertsekas's
                # eviction/re-bid step, and is what prevents the lockstep
                # oscillation of the earlier synchronous design (only one
                # weapon's assignment changes per round).
                evicted = (assigned_to == best_n).nonzero(as_tuple=True)[0]
                if len(evicted) > 0:
                    ev = int(evicted[0].item())
                    assigned_to[ev] = -1
                    unassigned.append(ev)

                assigned_to[m] = best_n
                price[best_n] = price[best_n] + bid

            action[b, p] = torch.where(assigned_to >= 0, assigned_to, torch.tensor(N, device=device))

    return action


@torch.no_grad()
def auction_round_action_multifire_reference(remaining_value, prob, legal_mask, must_fire=None,
                                              eps=1e-3, max_per_target=None):
    """
    Reference implementation, kept for verification only.

    auction_round_action_multifire below computes the same assignment with the
    marginal matrix formed in one step instead of one weapon at a time, and is
    what the pipeline calls. The two agree on every case they have been tested
    on; this version is retained because it is the one the algorithm is easiest
    to read off from, and because any future change to the fast path should be
    checked against it.

    Many-to-one (capacitated) variant of auction_round_action: DWTAP legally
    allows multiple weapons to engage the SAME target in one round (the
    module docstring above already claims this, but the eviction-based
    implementation above actually enforces a strict one-weapon-per-target
    assignment via eviction -- confirmed by inspection, 2026-08-11, after
    an instance-level SCIP-vs-SCoPE diagnostic found SCIP's optimal
    solutions routinely double-team a target with 2+ weapons in the same
    round, e.g. W1->T3, W4->T3 simultaneously, which the eviction version
    can never produce).

    Unlike the 1:1 assignment problem the eviction version above adapts
    from, this is structurally the CAPACITATED TRANSPORTATION problem --
    each target ("destination") can receive up to max_per_target weapons
    ("supply units"), not just one. Bertsekas & Castanon (1989), "The
    Auction Algorithm for the Transportation Problem," Annals of Operations
    Research 20(1):67-96, is the established generalization of Bertsekas's
    original 1:1 auction algorithm to exactly this capacitated/many-to-one
    structure -- citable grounding for this variant, rather than treating it
    as an ad-hoc heuristic; the implementation below is a simplified
    greedy-marginal-value instantiation of that same idea (see max_per_target
    docstring below for why an explicit cap was added on top).

    Instead of price-based competitive eviction (which forces exactly one
    "owner" per target), this tracks each target's REMAINING SURVIVAL
    PROBABILITY as weapons are greedily assigned to it one at a time:
    survival[n] *= (1 - p[m, n]) each time a weapon m is assigned to n. A
    weapon's marginal value for a target is
    remaining_value[n] * survival[n] * p[m, n] -- i.e. its expected damage
    GIVEN whatever damage other already-assigned weapons this round have
    already accounted for. This naturally produces diminishing returns for
    piling more weapons onto an already-likely-dead target (discouraging
    pure over-concentration) while still allowing it outright when the
    marginal contribution is genuinely the best available option (unlike
    the eviction version, which can never let two weapons share a target
    no matter how good a fit it is), and needs no price bookkeeping or
    eviction/re-bid loop -- assignments, once made, are never undone.

    max_per_target: optional hard cap on weapons assigned to the same target
        in one round. **Now defaults to None (no cap).**

        It originally defaulted to 2, on the theory that uncapped greedy
        marginal-value assignment would inherit Greedy's myopia -- stacking
        weapons onto whichever target currently shows the highest marginal
        value, because the built-in diminishing-returns term (survival[n]
        shrinks per assignment) might not overcome a high base
        value*probability. Back then that cap did beat the uncapped version
        (brerla_comm_layer_and_multifire_auction memory: 0.1952 vs 0.2011).

        Measured again 2026-08-23 (diagnose_battlefield_multifire.py), with
        the auditor layer now on top, uncapped is better or equal at BOTH
        ends of the scale range:
            30M_30N_10T   cap2 0.0125  cap3 0.0116  uncapped 0.0116
            70M_100N_15T  cap2 0.1183  cap3 0.1178  uncapped 0.1177
        and wasteful redundancy stays at 0.000 uncapped, i.e. the feared
        myopic pile-up does not materialise -- the survival term alone is
        enough. Dispersion just settles slightly lower (0.871 -> 0.843),
        which is the intended effect, not a pathology. Removing the cap also
        removes a hand-tuned constant that was set by inspecting SCIP
        solutions at much smaller scale, so concentration is now decided by
        the marginal-value computation rather than by a magic number.

    Same args/return convention as auction_round_action.
    """
    B, P, M, N = prob.shape
    device = prob.device

    # No cap: M weapons can never exceed M on one target, so this is
    # equivalent to "uncapped" while keeping the comparison code below
    # unchanged.
    if max_per_target is None:
        max_per_target = M

    respect_policy_hold = must_fire is not None
    if must_fire is None:
        must_fire = torch.ones(B, P, M, dtype=torch.bool, device=device)

    action = torch.full((B, P, M), N, dtype=torch.long, device=device)

    for b in range(B):
        for p in range(P):
            rv = remaining_value[b, p]  # [N]
            pr = prob[b, p]  # [M, N]
            legal = legal_mask[b, p]  # [M, N] bool
            mf = must_fire[b, p]  # [M] bool

            survival = torch.ones(N, device=device)
            target_count = torch.zeros(N, dtype=torch.long, device=device)
            decided = torch.zeros(M, dtype=torch.bool, device=device)
            action_bp = torch.full((M,), N, dtype=torch.long, device=device)

            for m in range(M):
                if respect_policy_hold and not mf[m]:
                    decided[m] = True  # stays no-op, authoritative hold

            # Greedy best-marginal-value-first, recomputed after each pick --
            # same pattern as the Greedy baseline's own round selection, but
            # restricted to weapons the policy already committed to firing
            # (must_fire) when that signal is supplied.
            remaining_weapons = [m for m in range(M) if not decided[m]]
            while remaining_weapons:
                at_cap = target_count >= max_per_target  # [N] bool
                best_m, best_n, best_val = None, None, float("-inf")
                for m in remaining_weapons:
                    marginal = rv * survival * pr[m]  # [N]
                    marginal = marginal.masked_fill(~legal[m].bool(), float("-inf"))
                    marginal = marginal.masked_fill(at_cap, float("-inf"))
                    if not torch.isfinite(marginal).any():
                        continue
                    n = int(marginal.argmax().item())
                    val = float(marginal[n].item())
                    if val > best_val:
                        best_m, best_n, best_val = m, n, val

                if best_m is None:
                    # No remaining weapon has any legal, under-cap target left.
                    break

                if not respect_policy_hold and best_val <= 0:
                    # Policy-agnostic fallback: myopic no-op rule, same as
                    # auction_round_action's own must_fire=None behavior.
                    remaining_weapons.remove(best_m)
                    continue

                action_bp[best_m] = best_n
                survival[best_n] = survival[best_n] * (1 - pr[best_m, best_n])
                target_count[best_n] = target_count[best_n] + 1
                remaining_weapons.remove(best_m)

            action[b, p] = action_bp

    return action


@torch.no_grad()
def auction_round_action_multifire_guided(remaining_value, prob, legal_mask, policy_target_prob,
                                           must_fire=None, eps=1e-3, max_per_target=2, guide_weight=1.0):
    """
    Same capacitated greedy assignment as auction_round_action_multifire, but
    the marginal value used to rank (weapon, target) candidates is boosted by
    the TRAINED POLICY's own per-target preference, instead of being computed
    purely from remaining_value*prob with zero input from the network.

    Motivation: in every auction variant above, the policy's role is reduced
    to a binary must_fire (fire/hold) gate -- its actual per-target argmax
    choice is discarded entirely, and the auction re-derives target
    assignment from scratch via its own greedy/eviction rule. This means the
    weapon-to-weapon communication layer's learned coordination signal (see
    common/DWTA_GNN_comm.py) can only ever influence WHETHER a weapon fires,
    never WHICH target it prefers -- so any coordination the comm layer
    learns about target choice specifically is thrown away at the last step.
    This variant lets that signal flow through: policy_target_prob (the
    actor's own softmax probability over targets for each weapon, i.e.
    policy[..., :N] renormalized over legal targets) multiplicatively boosts
    the marginal value, so a weapon's auction bid reflects both the
    game-theoretic marginal value (as before) AND how strongly the trained,
    comm-informed policy itself wanted that particular target.

    guided_marginal[m,n] = marginal[m,n] * (1 + guide_weight * policy_target_prob[m,n])

    Multiplicative (not additive) so it never changes the SIGN of a
    candidate's marginal value or lets the policy override legality/survival
    accounting -- it only re-ranks among already-legal, already-positive-
    marginal candidates, same guarantee structure as the unguided version.
    guide_weight=0 reproduces auction_round_action_multifire exactly.

    guide_weight: python float (same weight for every instance, the
    hand-tuned mode used in eval_guided_auction_sweep.py) OR a [B, P] tensor
    (a LEARNED, per-instance weight -- see common/DWTA_GNN_binary_guided.py,
    which outputs this as a sampled network output so it can be trained via
    REINFORCE instead of hand-tuned; brerla_hybrid_policy_auction_framing
    memory: hand-picking a single global guide_weight was flagged as not
    principled).
    """
    B, P, M, N = prob.shape
    device = prob.device

    respect_policy_hold = must_fire is not None
    if must_fire is None:
        must_fire = torch.ones(B, P, M, dtype=torch.bool, device=device)

    guide_weight_is_tensor = torch.is_tensor(guide_weight)

    action = torch.full((B, P, M), N, dtype=torch.long, device=device)

    for b in range(B):
        for p in range(P):
            rv = remaining_value[b, p]  # [N]
            pr = prob[b, p]  # [M, N]
            legal = legal_mask[b, p]  # [M, N] bool
            mf = must_fire[b, p]  # [M] bool
            pref = policy_target_prob[b, p]  # [M, N]
            gw = float(guide_weight[b, p]) if guide_weight_is_tensor else guide_weight

            survival = torch.ones(N, device=device)
            target_count = torch.zeros(N, dtype=torch.long, device=device)
            decided = torch.zeros(M, dtype=torch.bool, device=device)
            action_bp = torch.full((M,), N, dtype=torch.long, device=device)

            for m in range(M):
                if respect_policy_hold and not mf[m]:
                    decided[m] = True

            remaining_weapons = [m for m in range(M) if not decided[m]]
            while remaining_weapons:
                at_cap = target_count >= max_per_target
                best_m, best_n, best_val = None, None, float("-inf")
                for m in remaining_weapons:
                    marginal = rv * survival * pr[m]
                    guided = marginal * (1.0 + gw * pref[m])
                    guided = guided.masked_fill(~legal[m].bool(), float("-inf"))
                    guided = guided.masked_fill(at_cap, float("-inf"))
                    if not torch.isfinite(guided).any():
                        continue
                    n = int(guided.argmax().item())
                    val = float(guided[n].item())
                    if val > best_val:
                        best_m, best_n, best_val = m, n, val

                if best_m is None:
                    break

                if not respect_policy_hold and best_val <= 0:
                    remaining_weapons.remove(best_m)
                    continue

                action_bp[best_m] = best_n
                survival[best_n] = survival[best_n] * (1 - pr[best_m, best_n])
                target_count[best_n] = target_count[best_n] + 1
                remaining_weapons.remove(best_m)

            action[b, p] = action_bp

    return action


@torch.no_grad()
def auction_round_action_price_multifire(remaining_value, prob, legal_mask, must_fire=None,
                                         eps=1e-3, max_per_target=None, max_rounds=None,
                                         use_prices=True):
    """Price-based auction that allows several weapons on one target.

    auction_round_action is a genuine Bertsekas auction (weapons bid, prices
    rise, incumbents are evicted and re-bid) but its eviction rule forces a
    strict one-weapon-per-target assignment, which DWTAP does not require and
    which costs real value at the larger configurations where an optimal
    solution routinely double-teams a target.

    auction_round_action_multifire lifts that restriction but drops the
    auction entirely: it picks the globally best (weapon, target) pair at each
    step and never revisits it, so it is greedy, not bidding.

    This function keeps both properties. A target may host up to
    max_per_target weapons, and contention is still resolved by prices:
    every weapon bids for the slot that leaves it the largest surplus, the
    bid raises that target's price by the margin over its own second choice,
    and when a target is full the incumbent with the smallest surplus is
    evicted and re-bids. The value of a slot is the exact marginal under the
    weapons already holding the target,

        marginal(m, n) = V_n * survival_n * P_mn,
        survival_n     = prod over m' currently assigned to n of (1 - P_m'n),

    the same quantity auction_round_action_multifire bids with, so the two
    differ only in how contention is settled.

    Args mirror auction_round_action_multifire. max_per_target defaults to M
    (uncapped); with no cap no eviction can occur and prices act purely as a
    tie-break on which target each weapon prefers.
    """
    B, P, M, N = prob.shape
    device = prob.device

    if max_per_target is None:
        max_per_target = M
    if max_rounds is None:
        max_rounds = 20 * M

    respect_policy_hold = must_fire is not None
    if must_fire is None:
        must_fire = torch.ones(B, P, M, dtype=torch.bool, device=device)

    action = torch.full((B, P, M), N, dtype=torch.long, device=device)

    for b in range(B):
        for p in range(P):
            rv = remaining_value[b, p]          # [N]
            pr = prob[b, p]                     # [M, N]
            legal = legal_mask[b, p].bool()     # [M, N]
            mf = must_fire[b, p]                # [M]

            price = torch.zeros(N, device=device)
            holders = [[] for _ in range(N)]    # weapons currently on each target
            assigned_to = torch.full((M,), -1, dtype=torch.long, device=device)

            queue = [m for m in range(M) if (not respect_policy_hold) or bool(mf[m])]
            rounds = 0

            def survival_of(n, exclude=None):
                s = 1.0
                for mm in holders[n]:
                    if mm == exclude:
                        continue
                    s *= float(1.0 - pr[mm, n])
                return s

            while queue and rounds < max_rounds:
                rounds += 1
                m = queue.pop(0)

                surv = torch.tensor([survival_of(n) for n in range(N)], device=device)
                marginal = rv * surv * pr[m]                       # [N]
                at_cap = torch.tensor([len(holders[n]) >= max_per_target
                                       for n in range(N)], device=device)
                net = marginal - price
                net = net.masked_fill(~legal[m], float("-inf"))

                # A full target is still biddable, but only by displacing its
                # weakest holder, so charge that holder's surplus on top.
                for n in range(N):
                    if not at_cap[n] or not bool(legal[m, n]):
                        continue
                    weakest, weakest_val = None, float("inf")
                    for mm in holders[n]:
                        val = float(rv[n] * survival_of(n, exclude=mm) * pr[mm, n])
                        if val < weakest_val:
                            weakest, weakest_val = mm, val
                    if weakest is None:
                        net[n] = float("-inf")
                    else:
                        net[n] = float(rv[n] * survival_of(n, exclude=weakest)
                                       * pr[m, n]) - price[n] - weakest_val

                if not torch.isfinite(net).any():
                    assigned_to[m] = -1
                    continue

                best_n = int(net.argmax().item())
                best_val = float(net[best_n])

                if not respect_policy_hold and best_val < 0:
                    assigned_to[m] = -1
                    continue

                second = net.clone()
                second[best_n] = float("-inf")
                second_val = (float(second.max()) if torch.isfinite(second).any()
                              else best_val - eps)
                bid = (max(best_val - second_val, 0.0) + eps) if use_prices else 0.0

                if len(holders[best_n]) >= max_per_target:
                    weakest, weakest_val = None, float("inf")
                    for mm in holders[best_n]:
                        val = float(rv[best_n] * survival_of(best_n, exclude=mm) * pr[mm, best_n])
                        if val < weakest_val:
                            weakest, weakest_val = mm, val
                    holders[best_n].remove(weakest)
                    assigned_to[weakest] = -1
                    queue.append(weakest)

                holders[best_n].append(m)
                assigned_to[m] = best_n
                price[best_n] = price[best_n] + bid

            action[b, p] = torch.where(assigned_to >= 0, assigned_to,
                                       torch.tensor(N, device=device))

    return action


@torch.no_grad()
def auction_round_action_multifire(remaining_value, prob, legal_mask, must_fire=None,
                                   eps=1e-3, max_per_target=None):
    """Assign targets to the firing weapons by greedy marginal value.

    Repeatedly takes the (weapon, target) pair with the largest marginal value
    and decays that target's survival by (1 - P), so a target that is already
    likely destroyed is worth less to the next weapon. Many weapons may share
    a target.

    auction_round_action_multifire_reference above states the same procedure as
    an explicit loop over weapons, which costs O(M^2) small tensor calls per
    stage; at the largest configuration that is tens of thousands of kernel
    launches on tensors far too small to occupy the GPU. Forming the whole
    marginal matrix at once and updating only the column that changed gives
    identical assignments roughly thirty times faster at 70x100.

    `eps` is accepted and ignored, so the signature matches the reference.

    Here the whole [M, N] marginal matrix is formed at once and reduced with a
    single argmax, so one assignment costs one vectorized step instead of M.
    Only the chosen target's column changes afterwards, so the matrix is
    updated in place rather than rebuilt.

    Tie-breaking follows torch.argmax on the flattened matrix, which is the
    same lowest-index rule the original loop applies, so the two functions
    return identical actions.
    """
    B, P, M, N = prob.shape
    device = prob.device

    if max_per_target is None:
        max_per_target = M

    respect_policy_hold = must_fire is not None
    if must_fire is None:
        must_fire = torch.ones(B, P, M, dtype=torch.bool, device=device)

    action = torch.full((B, P, M), N, dtype=torch.long, device=device)

    for b in range(B):
        for p in range(P):
            legal = legal_mask[b, p].bool()                  # [M, N]
            active = legal.clone()
            if respect_policy_hold:
                active &= must_fire[b, p].unsqueeze(1)
            if not bool(active.any()):
                continue

            rv = remaining_value[b, p]                       # [N]
            pr = prob[b, p]                                  # [M, N]
            survival = torch.ones(N, device=device)
            target_count = torch.zeros(N, dtype=torch.long, device=device)

            marginal = rv.unsqueeze(0) * survival.unsqueeze(0) * pr   # [M, N]
            neg_inf = torch.tensor(float("-inf"), device=device)
            scores = torch.where(active, marginal, neg_inf)

            act_bp = torch.full((M,), N, dtype=torch.long, device=device)

            for _ in range(M):
                flat = int(scores.argmax().item())
                best_val = float(scores.view(-1)[flat])
                if best_val == float("-inf"):
                    break
                if not respect_policy_hold and best_val <= 0:
                    # Policy-agnostic myopic no-op rule: this weapon and every
                    # weapon after it has nothing worth firing at.
                    m_drop = flat // N
                    scores[m_drop, :] = neg_inf
                    continue

                m, n = flat // N, flat % N
                act_bp[m] = n
                survival[n] = survival[n] * (1 - pr[m, n])
                target_count[n] = target_count[n] + 1

                scores[m, :] = neg_inf                       # weapon is spent
                if int(target_count[n]) >= max_per_target:
                    scores[:, n] = neg_inf                   # target is full
                else:
                    col = rv[n] * survival[n] * pr[:, n]     # only this column moved
                    scores[:, n] = torch.where(scores[:, n] > neg_inf, col, neg_inf)

            action[b, p] = act_bp

    return action


# Back-compatible alias: the fast path is now the default implementation.
auction_round_action_multifire_fast = auction_round_action_multifire
