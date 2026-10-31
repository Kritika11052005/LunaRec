"""Test 10: does the technical effect survive the choice of shortlist length?

Every headline figure in notebook 04 is measured at K = 20, and K is the one
axis the sensitivity sweep found a real dependence on: the ablation margin runs
0.440 at K = 5 down to 0.047 at K = 50 (14/20, t = 2.26).

The 34 % acquisition saving has never been measured at any other K. Test 9
already showed that figure was strongly dependent on a choice we assumed was
neutral (the target), so assuming it is K-independent is the same mistake with
a different variable.

Reported per K:
  margin      paired P@K advantage of the criterion over the ablation
  saving      acquisitions the criterion needs to reach the best baseline's
              FULL-BUDGET score, as a percentage of that budget -- the
              technical-effect figure, computed per repetition so it carries a
              confidence interval

Only three policies are run (criterion, ablation, max-variance), because those
are what the two quantities need; random and coverage would add a third of the
runtime for nothing.

Usage:
    python effect_vs_k.py <patches.csv> [n_rep] [budget] [k_list]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E

POLS = ['disagreement', 'voi-value', 'max-variance']


def one_k(df, k, n_rep, budget):
    E.K = k
    curves = {p: [] for p in POLS}
    for rep in range(n_rep):
        sub = (df.sample(min(400, len(df)), random_state=rep).reset_index(drop=True)
               if len(df) > 400 else df)
        for p in POLS:
            h = E.run(sub, p, budget, np.random.default_rng(100 + rep))
            curves[p].append([s['prec'] for s in h])

    ours = np.array(curves['disagreement'])
    abl = np.array(curves['voi-value'])
    mv = np.array(curves['max-variance'])

    d = ours[:, -1] - abl[:, -1]
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else (np.inf if d.mean() > 0 else 0.0)

    # best baseline per rep, then acquisitions for the criterion to match it
    saved = []
    for i in range(n_rep):
        target = max(abl[i, -1], mv[i, -1])
        c = ours[i]
        hit = next((j for j, v in enumerate(c) if v >= target and (c[j:] >= target).all()),
                   len(c) - 1)
        saved.append(100.0 * (budget - hit) / budget)
    saved = np.array(saved)
    ci = 1.96 * saved.std(ddof=1) / np.sqrt(n_rep) if n_rep > 1 else 0.0
    return (d.mean(), int((d > 0).sum()), float(t),
            saved.mean(), ci, int((saved > 0).sum()),
            ours[:, -1].mean(), max(abl[:, -1].mean(), mv[:, -1].mean()))


def main(path, n_rep=15, budget=150, k_list="5,10,20,35,50"):
    n_rep, budget = int(n_rep), int(budget)
    ks = [int(x) for x in k_list.split(',')]
    cols = [c for c, _, _ in E.ATTRS] + ['patch_x', 'patch_y']
    df = pd.read_csv(path).dropna(subset=cols)
    base_k = E.K
    E.fit_priors(df)
    print(f"{os.path.basename(path)}: {len(df)} patches, {n_rep} reps, budget {budget}")
    print("priors: " + ", ".join(f"{r:.0f}m/{n:.3f}" for r, n in E.PRIOR))
    print()
    print("%-5s %9s %8s %8s %12s %9s %10s %10s"
          % ("K", "margin", "wins", "t", "saved %", "+/- 95%", "P@K ours", "P@K base"))
    rows = []
    for k in ks:
        m, w, t, sv, ci, nsv, pk_o, pk_b = one_k(df, k, n_rep, budget)
        rows.append({'K': k, 'margin': m, 'wins': w, 'n_rep': n_rep, 't': t,
                     'saved_pct': sv, 'ci95': ci, 'reps_saving': nsv,
                     'prec_ours': pk_o, 'prec_best_base': pk_b, 'budget': budget})
        print("%-5d %+9.3f %8s %8.2f %11.0f%% %9.1f %10.3f %10.3f"
              % (k, m, f"{w}/{n_rep}", t, sv, ci, pk_o, pk_b))
    E.K = base_k

    out = path.replace('.csv', '_effect_vs_k.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n-> {out}")

    R = pd.DataFrame(rows)
    at20 = R[R.K == 20]
    hi = R[R.K == max(ks)]
    print("\nmargin  : %.3f at K=%d  ->  %.3f at K=%d"
          % (R.iloc[0].margin, ks[0], R.iloc[-1].margin, ks[-1]))
    print("saving  : %.0f%% at K=%d  ->  %.0f%% at K=%d"
          % (R.iloc[0].saved_pct, ks[0], R.iloc[-1].saved_pct, ks[-1]))
    if len(at20):
        print("at K=20 (the headline): margin %.3f, saving %.0f%%"
              % (at20.iloc[0].margin, at20.iloc[0].saved_pct))
    ok = bool((hi.saved_pct > 0).all() and (hi.saved_pct - hi.ci95 > 0).all())
    print("\nVERDICT: " + (
        "the acquisition saving survives to K=%d (%.0f%% +/- %.0f, CI excludes zero). "
        "The technical effect is not specific to K=20."
        % (max(ks), hi.iloc[0].saved_pct, hi.iloc[0].ci95) if ok else
        "the saving does NOT survive at K=%d (%.0f%% +/- %.0f). The technical "
        "effect is K-dependent and every claim must state K=20."
        % (max(ks), hi.iloc[0].saved_pct, hi.iloc[0].ci95)))


if __name__ == '__main__':
    main(*sys.argv[1:5])