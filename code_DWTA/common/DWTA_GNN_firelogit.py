"""Participation policy with a single fire logit per weapon.

The reported actor (common/DWTA_GNN.py) scores every legal (weapon, target)
edge and a per-weapon hold option, then reads the probability of firing off a
softmax over the two. Those target scores never choose a target: assignment is
done afterwards by the greedy marginal rule, and the edge scores enter only
through the softmax denominator, where they act as "how attractive are this
weapon's options".

If that signal is not doing real work, the head can be a single scalar per
weapon and the edge scoring machinery can go. This module is that variant,
built so the two differ in as little as possible: same encoder, same number of
message-passing layers, same hidden width, same no-op projection inputs. What
is removed is `edge_residual` and `edge_score`, and what changes is that the
per-weapon logit is passed through a sigmoid instead of competing with edge
scores in a softmax.

Parameter counts:
    reported actor      1,307,778
    this variant        1,274,113   (edge head removed)

The forward signature is kept identical to EdgeAwareGNN_ACTOR so the training
loop and every evaluation script can use it unchanged: it returns a
[batch, para, W, T+1] tensor whose last slot carries the hold probability and
whose target slots carry the fire probability spread uniformly over the legal
targets. Nothing downstream reads those target slots except through the no-op
column, so the spreading is a formality that keeps shapes compatible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.Dynamic_HYPER_PARAMETER import (
    EMBEDDING_DIM, NUM_WEAPON_FEATURES, NUM_TARGET_FEATURES, NUM_EDGE_FEATURES,
    ACTOR_LEARNING_RATE, ACTOR_WEIGHT_DECAY, BUFFER_SIZE,
)
from common.DWTA_GNN import EdgeAwareGNNLayer, ResidualBlock, ReplayMemory


class FireLogitGNN_ACTOR(nn.Module):
    """One scalar fire logit per weapon; no per-target scoring head."""

    def __init__(self, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList([EdgeAwareGNNLayer() for _ in range(num_layers)])

        # Same inputs as the reported actor's hold head: the weapon's own
        # embedding plus pooled weapon and target context. Only the meaning of
        # the output changes, from a hold score competing with edge scores to a
        # fire logit read directly.
        self.fire_proj = nn.Linear(EMBEDDING_DIM * 3, EMBEDDING_DIM)
        self.fire_residual = ResidualBlock(dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM,
                                           dropout=0.1)
        self.fire_score = nn.Linear(EMBEDDING_DIM, 1)

        self.assignment_embedding = None
        self.current_state = None
        self.replay_memory = ReplayMemory(capacity=BUFFER_SIZE)

        import torch.optim as optim
        self.optimizer = optim.Adam(self.parameters(), lr=ACTOR_LEARNING_RATE,
                                    weight_decay=ACTOR_WEIGHT_DECAY)
        self.lr_stepper = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)

    def forward(self, assignment_embedding, prob, mask):
        batch_size, para_size = assignment_embedding.shape[:2]
        if prob is None:
            from common.Dynamic_HYPER_PARAMETER import NUM_WEAPONS, NUM_TARGETS
            num_weapons, num_targets = NUM_WEAPONS, NUM_TARGETS
        else:
            num_weapons, num_targets = prob.shape[2], prob.shape[3]

        n_feat = assignment_embedding.size(-1)
        assignments = assignment_embedding[:, :, :-1, :]
        feats = assignments.view(batch_size, para_size, num_weapons, num_targets, n_feat)

        weapon_h = feats[:, :, :, 0, :NUM_WEAPON_FEATURES]
        target_h = feats[:, :, 0, :, NUM_WEAPON_FEATURES:NUM_WEAPON_FEATURES + NUM_TARGET_FEATURES]
        edge_h = feats[:, :, :, :, -NUM_EDGE_FEATURES:]

        for layer in self.gnn_layers:
            weapon_h, target_h, edge_h = layer(weapon_h, target_h, edge_h)

        global_weapon = weapon_h.mean(dim=2)
        global_target = target_h.mean(dim=2)
        ctx = torch.cat([global_weapon, global_target], dim=-1)
        ctx = ctx.unsqueeze(2).expand(-1, -1, num_weapons, -1)

        h = self.fire_proj(torch.cat([weapon_h, ctx], dim=-1))
        h = self.fire_residual(h)
        fire_logit = self.fire_score(h).squeeze(-1)                 # [B, P, W]
        p_fire = torch.sigmoid(fire_logit).clamp(1e-6, 1 - 1e-6)

        # A weapon with no legal target cannot fire whatever the logit says.
        legal_edges = mask[:, :, :num_weapons, :num_targets].bool()
        has_target = legal_edges.any(-1)
        p_fire = torch.where(has_target, p_fire, torch.zeros_like(p_fire))

        # Repackage as [B, P, W, T+1] so callers that expect the reported
        # actor's output shape keep working. Fire mass is spread uniformly over
        # a weapon's legal targets; only the no-op column is ever read.
        n_legal = legal_edges.sum(-1, keepdim=True).clamp_min(1)
        per_target = (p_fire.unsqueeze(-1) / n_legal) * legal_edges.float()
        policy = torch.cat([per_target, (1.0 - p_fire).unsqueeze(-1)], dim=-1)

        self.assignment_embedding = torch.zeros(
            batch_size, para_size, num_weapons * num_targets + 1, EMBEDDING_DIM,
            device=policy.device)
        self.current_state = self.assignment_embedding
        return policy, self.assignment_embedding


def create_gnn_actor_firelogit(num_layers=3):
    return FireLogitGNN_ACTOR(num_layers=num_layers)
