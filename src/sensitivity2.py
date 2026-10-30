"""
Does the result survive the choices nobody measured?

Three settings in evaluate.py are modelling choices rather than measurements:

  W_SPREAD = 0.07   how uncertain the stakeholder weights are
  K        = 20     how long the shortlist is
  PRIOR            the fitted GP range and nugget per attribute

W_SPREAD matters most. The criterion acts on DISAGREEMENT between
stakeholders, and this constant sets how much disagreement exists. Near zero
the stakeholders agree, so a disagreement-targeting rule has nothing to target
and should collapse toward its own ablation. Very large and every shortlist is
uncertain, so the signal may drown. If the advantage appears only in a narrow
band around 0.07, the finding is about 0.07.

There is no data to fit it from -- no elicited ISRO preferences exist -- so it
cannot be resolved the way the variogram resolved the spatial prior. It can
only be surveyed, which is what this does.

The reported quantity is the paired margin in Precision@K over the ABLATION
(voi-value), because that is the comparison the inventive step rests on: same
spread, same spatial propagation, blind to shortlist structure.

Usage:
    python sensitivity.py <patches.csv> [n_rep] [budget]
"""

import os
import sys

import numpy as np
import pandas as pd

# Before importing the local modules: isolated interpreters (python -I) drop
# the script's own directory from sys.path, so a plain import fails.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as E


def margin(df, n_rep, budget):
    """Paired P@K margin of the criterion over the ablation, across reps."""
    diffs = []
    for rep in range(n_rep):
        sub = (df.sample(min(400, len(df)), random_state=rep).reset_index(drop=True)
               if len(df) > 400 else df)
        got = {}
        for p in ('voi-value', 'disagreement'):
            h = E.run(sub, p, budget, np.random.default_rng(100 + rep))
            got[p] = h[-1]['prec']
        diffs.append(got['disagreement'] - got['voi-value'])
    d = np.array(diffs)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else np.inf
    return d.mean(), int((d > 0).sum()), len(d), t


def main(path, n_rep=12, budget=60):
    n_rep, budget = int(n_rep), int(budget)
    cols = [c for c, _, _ in E.ATTRS] + ['patch_x', 'patch_y']
    df = pd.read_csv(path).dropna(subset=cols)

    # Fit the priors for this scene exactly as evaluate.main does, so the sweep
    # runs against the same model as the headline result.
    try:
        fits = E.fit_priors(df)
        print(f"{path}: {len(df)} patches, budget={budget}, "
              f"{n_rep} reps per setting")
        for rng_, nug, r2, label in fits:
            print(f"  {label:<13} range {rng_:6.0f} m  nugget {nug:.3f}  "
                  f"variogram R2 {r2:.3f}")
    except Exception as e:
        print(f"(variogram fit unavailable: {e}; using placeholder priors)")
    print("\nmargin = Precision@K(criterion) - Precision@K(ablation), paired\n")

    base_spread, base_k, base_prior = E.W_SPREAD, E.K, list(E.PRIOR)

    print("1. stakeholder weight spread  (the unmeasured constant)")
    print('%-12s %10s %8s %10s' % ('W_SPREAD', 'margin', 'wins', 't'))
    for s in (0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.25):
        E.W_SPREAD = s
        m, w, n, t = margin(df, n_rep, budget)
        tag = '   <- current' if abs(s - base_spread) < 1e-9 else ''
        print('%-12.2f %10.3f %8s %10.2f%s' % (s, m, f"{w}/{n}", t, tag))
    E.W_SPREAD = base_spread

    print("\n2. shortlist length")
    print('%-12s %10s %8s %10s' % ('K', 'margin', 'wins', 't'))
    for k_ in (5, 10, 20, 30, 50):
        E.K = k_
        m, w, n, t = margin(df, n_rep, budget)
        tag = '   <- current' if k_ == base_k else ''
        print('%-12d %10.3f %8s %10.2f%s' % (k_, m, f"{w}/{n}", t, tag))
    E.K = base_k

    print("\n3. prior misspecification  (how wrong can the covariance be?)")
    print("   all three attribute ranges scaled together")
    print('%-12s %10s %8s %10s' % ('range x', 'margin', 'wins', 't'))
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        # PRIOR is a list of (range_m, nugget) per attribute. Scaling every
        # range together asks what happens when the spatial model is wrong in
        # the same direction everywhere, which is how a misfitted variogram
        # would actually fail -- not one attribute at a time.
        E.PRIOR = [(r * mult, n) for r, n in base_prior]
        m, w, n_, t = margin(df, n_rep, budget)
        tag = '   <- fitted' if mult == 1.0 else ''
        print('%-12.2f %10.3f %8s %10.2f%s' % (mult, m, f"{w}/{n_}", t, tag))
    E.PRIOR = base_prior

    print("\n4. nugget misspecification  (the parameter that is NOT identified)")
    print("   all three attribute nuggets overridden to the same value")
    print('%-12s %10s %8s %10s' % ('nugget', 'margin', 'wins', 't'))
    # The nugget is variance at ZERO separation, and the patch grid is 200 m --
    # so there is no data below 200 m and the parameter is extrapolation, not
    # measurement. Three estimators give three answers on the same scenes:
    #
    #   weighted-LS variogram (old weighting)  0.44 - 0.52
    #   Cressie-weighted variogram             0.24 - 0.33
    #   marginal likelihood                    0.00 - 0.02
    #
    # An absolute override rather than a multiplier, because the fitted
    # illumination nugget is 0.000 and scaling it leaves it at zero -- which
    # would silently exclude the attribute from the sweep.
    for nug in (0.0, 0.05, 0.15, 0.30, 0.45, 0.60):
        E.PRIOR = [(r, nug) for r, _ in base_prior]
        m, w, n_, t = margin(df, n_rep, budget)
        tag = '   <- fitted (Cressie)' if abs(nug - 0.30) < 1e-9 else ''
        print('%-12.2f %10.3f %8s %10.2f%s' % (nug, m, f"{w}/{n_}", t, tag))
    E.PRIOR = base_prior

    print("\nreading this: a margin that stays positive with high win counts")
    print("across all four sweeps means the advantage is a property of the")
    print("criterion. The nugget sweep matters most -- that parameter is not")
    print("identified by 200 m-spaced data, so the result must hold across the")
    print("whole span the estimators disagree over, not just at a fitted value.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
