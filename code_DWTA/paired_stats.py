"""
Paired significance tests, ours against every learned baseline (reviewer 3.8).

SCIP is an exact solver, not a learned baseline, and is excluded; the paper
compares against it by tier instead (Appendix: SCIP optimality).

Reads result/paired_instances.csv and result/paired_instances_rl4co.csv. Our
per-instance value is the mean over the ten seeds at K=0, the same quantity
as the main table. For each baseline:

  overall     Wilcoxon signed-rank over all 120 instances (two-sided)
  per config  Wilcoxon over the ten instances, Holm-corrected across the
              twelve configurations
  per seed    the overall test repeated with each seed alone, to show the
              conclusion does not rest on averaging seeds

Differences are baseline minus ours, so positive means ours is better
(lower remaining value). Instances where the two agree to 1e-6 count as ties
and are dropped by the test.

Output: result/paired_stats.txt
"""
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

from eval_final_table import CONFIGS

CFGS = ["%dM_%dN_%dT" % c for c in CONFIGS]
BASELINES = ["Greedy", "Auction", "AM", "POMO", "Sequential"]  # SCIP is an exact solver, not a learned baseline; excluded from this test
TIE = 1e-6


def load():
    v = defaultdict(dict)
    for path in ("result/paired_instances.csv", "result/paired_instances_rl4co.csv"):
        for r in csv.DictReader(open(path)):
            v[r["method"]][(r["config"], int(r["instance"]))] = float(r["objective"])
    keys = [(c, i) for c in CFGS for i in range(10)]
    for k in keys:
        v["Ours"][k] = float(np.mean([v["Ours_s%d_K0" % s][k] for s in range(1, 11)]))
    return v, keys


def test(d):
    d = np.asarray(d)
    nz = d[np.abs(d) > TIE]
    wins, losses = int((d > TIE).sum()), int((d < -TIE).sum())
    ties = len(d) - wins - losses
    if len(nz) == 0:
        return wins, losses, ties, 1.0
    return wins, losses, ties, float(wilcoxon(nz, alternative="two-sided").pvalue)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[idx]))
        adj[idx] = running
    return adj


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else "%.4f" % p


def main():
    v, keys = load()
    out = []
    P = out.append
    P("Paired Wilcoxon signed-rank tests, ours (fire-logit, mean of ten seeds, K=0)")
    P("against each baseline on the same 120 seed-123 instances.")
    P("diff = baseline - ours; positive favours ours. W/L/T = instances ours")
    P("better / worse / tied (|diff| <= 1e-6).")
    P("")
    P("=== OVERALL (n=120) ===")
    P("%-11s %9s %9s %9s %12s %9s" % ("baseline", "base mean", "ours mean", "mean diff", "W/L/T", "p"))
    for b in BASELINES:
        d = [v[b][k] - v["Ours"][k] for k in keys]
        w, l, t, p = test(d)
        P("%-11s %9.4f %9.4f %+9.4f %12s %9s" % (b, np.mean([v[b][k] for k in keys]),
          np.mean([v["Ours"][k] for k in keys]), np.mean(d), "%d/%d/%d" % (w, l, t), fmt_p(p)))
    P("")

    for b in BASELINES:
        P("=== PER CONFIGURATION vs %s (n=10 each, Holm across 12) ===" % b)
        P("%-13s %9s %9s %9s %9s %9s %9s" % ("config", "base", "ours", "diff", "W/L/T", "p raw", "p Holm"))
        res = []
        for cfg in CFGS:
            ks = [(cfg, i) for i in range(10)]
            d = [v[b][k] - v["Ours"][k] for k in ks]
            res.append((cfg, np.mean([v[b][k] for k in ks]), np.mean([v["Ours"][k] for k in ks]),
                        np.mean(d)) + test(d))
        adj = holm(np.array([r[7] for r in res]))
        for r, pa in zip(res, adj):
            P("%-13s %9.4f %9.4f %+9.4f %9s %9s %9s%s" % (r[0], r[1], r[2], r[3],
              "%d/%d/%d" % (r[4], r[5], r[6]), fmt_p(r[7]), fmt_p(pa), "  *" if pa < 0.05 else ""))
        P("")

    P("=== ROBUSTNESS: overall test with each seed alone (n=120) ===")
    P("%-11s " % "baseline" + " ".join("%8s" % ("s%d" % s) for s in range(1, 11)))
    for b in BASELINES:
        cells = []
        for s in range(1, 11):
            d = [v[b][k] - v["Ours_s%d_K0" % s][k] for k in keys]
            w, l, t, p = test(d)
            cells.append("%8s" % (fmt_p(p) if np.mean(d) > 0 else "(-)" + fmt_p(p)))
        P("%-11s " % b + " ".join(cells))
    P("(-) marks a seed whose mean is worse than the baseline.")

    text = "\n".join(out)
    open("result/paired_stats.txt", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
