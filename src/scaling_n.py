"""Test 7: is the crossover a fixed COUNT of acquisitions or a fixed FRACTION
of the candidate set?

Every result in notebook 04 is at n = 400 candidates per repetition. The
criterion becomes reliably best from about 75 acquisitions, which is 19% of
that field, and the README now frames the method as campaign-scale screening of
hundreds of patches -- a claim about larger n that has never been tested.

The two readings diverge sharply. At n = 2000 a fixed count predicts the
crossover still at 75; a fixed fraction predicts 375. One of those is a usable
method and the other is not, and the sensitivity sweep varied K, the weights
and the priors but never n.

Reported per n:
  crossover   the first budget at which the criterion's mean precision exceeds
              the best baseline's AND stays above it
  margin      paired P@K advantage over the ablation at full budget

The BUDGET IS HELD FIXED rather than scaled with n. Scaling it would be the
like-for-like comparison, but the GP posterior update costs O(n^2 * budget^2)
overall, so n=1600 at budget 600 is ~256x the n=400 case -- days, not minutes.
A fixed budget still discriminates: at n=800 a fixed count predicts a crossover
near 75, a fixed fraction near 150, and the two are far apart. If the fraction
reading is right, large n simply never crosses within the budget, which is
itself the answer.

K is held at 20 throughout, so K/n falls as n grows. That is the operationally
meaningful setup -- a mission wants twenty landing sites whether it screened
400 candidate patches or 1600 -- not a confound to remove. It does mean the
selection task gets harder with n, and the crossover moving is the combined
effect of both.

Usage:
    python scaling_n.py <patches.csv> [n_rep] [budget] [n_list]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E

POLS = ['disagreement', 'voi-value', 'max-variance']


def crossover(ours, best):
    """First budget where `ours` exceeds `best` and never falls back."""
    d = np.asarray(ours) - np.asarray(best)
    for i in range(len(d)):
        if (d[i:] > 0).all():
            return i
    return len(d) - 1


def one_n(full, n, n_rep, budget):
    curves = {p: [] for p in POLS}
    for rep in range(n_rep):
        sub = (full.sample(n, random_state=rep).reset_index(drop=True)
               if len(full) > n else full)
        for p in POLS:
            h = E.run(sub, p, budget, np.random.default_rng(100 + rep))
            curves[p].append([s['prec'] for s in h])
    ours = np.array(curves['disagreement'])
    abl = np.array(curves['voi-value'])
    mv = np.array(curves['max-variance'])

    base_mean = np.maximum(abl.mean(0), mv.mean(0))
    cross = crossover(ours.mean(0), base_mean)

    d = ours[:, -1] - abl[:, -1]
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else (np.inf if d.mean() > 0 else 0.0)
    return cross, d.mean(), int((d > 0).sum()), float(t), ours[:, -1].mean()


def main(path, n_rep=12, budget=150, n_list="400,800"):
    n_rep, budget = int(n_rep), int(budget)
    ns = [int(x) for x in n_list.split(',')]
    cols = [c for c, _, _ in E.ATTRS] + ['patch_x', 'patch_y']
    full = pd.read_csv(path).dropna(subset=cols)
    E.fit_priors(full)
    print(f"{os.path.basename(path)}: {len(full)} patches available, "
          f"{n_rep} reps, budget {budget} (fixed)")
    print("cost grows as n^2: expect ~%s relative to the first setting"
          % ", ".join(f"{(n/ns[0])**2:.0f}x" for n in ns))
    print("priors: " + ", ".join(f"{r:.0f}m/{n:.3f}" for r, n in E.PRIOR))
    print(f"K = {E.K} throughout, so K/n falls as n grows "
          f"({', '.join(f'{100*E.K/n:.1f}%' for n in ns)})")
    print()
    print("%-7s %8s %11s %11s %10s %8s %8s"
          % ("n", "budget", "crossover", "as % of n", "margin", "wins", "t"))
    rows = []
    for n in ns:
        if n > len(full):
            print("%-7d  skipped -- only %d patches available" % (n, len(full)))
            continue
        c, m, w, t, pk = one_n(full, n, n_rep, budget)
        rows.append({'n': n, 'budget': budget, 'crossover': c,
                     'crossover_pct': 100.0 * c / n, 'margin': m, 'wins': w,
                     'n_rep': n_rep, 't': t, 'prec_ours': pk, 'K': E.K})
        print("%-7d %8d %11d %10.1f%% %+10.3f %8s %8.2f"
              % (n, budget, c, 100.0 * c / n, m, f"{w}/{n_rep}", t))

    if len(rows) < 2:
        print("\nneed at least two values of n to distinguish the hypotheses")
        return
    R = pd.DataFrame(rows)
    out = path.replace('.csv', '_scaling_n.csv')
    R.to_csv(out, index=False)
    print(f"\n-> {out}")

    c0, cN = R.iloc[0], R.iloc[-1]
    ratio_n = cN.n / c0.n
    pred_count = c0.crossover
    pred_frac = c0.crossover * ratio_n
    print("\nat n = %d the crossover was %d acquisitions (%.1f%% of the field)"
          % (c0.n, c0.crossover, c0.crossover_pct))
    print("at n = %d:" % cN.n)
    print("   fixed-COUNT    predicts %5.0f" % pred_count)
    print("   fixed-FRACTION predicts %5.0f" % pred_frac)
    print("   observed                %5d" % cN.crossover)
    d_count = abs(cN.crossover - pred_count)
    d_frac = abs(cN.crossover - pred_frac)
    print("\nVERDICT: " + (
        "closer to a fixed COUNT. The crossover does not grow with the candidate "
        "set, so the method scales to campaign-sized problems -- the acquisitions "
        "needed to reach useful performance stay bounded."
        if d_count < d_frac else
        "closer to a fixed FRACTION. The crossover grows with the candidate set, "
        "so on a campaign-sized problem the criterion needs proportionally as "
        "many acquisitions. The campaign-scale framing must be qualified."))
    print("(distance to count prediction %.0f, to fraction prediction %.0f)"
          % (d_count, d_frac))


if __name__ == '__main__':
    main(*sys.argv[1:5])