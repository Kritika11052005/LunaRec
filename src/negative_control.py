"""Negative control: does the criterion's advantage survive the destruction of
the thing it is supposed to exploit?

The criterion propagates each candidate observation through a spatial coupling
term. If it still beat its ablation on a field with NO spatial structure, the
margin would be an artefact of the scoring machinery rather than evidence about
the Moon, and every result in notebook 04 would be suspect.

The shuffle applies ONE permutation to all three attribute columns, holding
patch_x/patch_y fixed. That destroys the spatial arrangement while preserving
exactly:
  - each attribute's marginal distribution (the same values, reordered)
  - the correlation BETWEEN attributes at a patch (dn_std ~ shadow_local)
  - the number of patches, the geometry, the budget, K, the stakeholder weights

So the only thing removed is the spatial signal. Priors are re-fitted on the
shuffled field, as they are on the real one -- fitting them on the real field
and applying them to noise would test nothing.

Expectation, stated before running: with no spatial structure, observing a
patch tells you nothing about any other, so every unobserved patch keeps its
prior variance and an inclusion probability near K/n. Boundary uncertainty is
then flat across candidates, both policies degenerate toward random tie-breaks,
and the margin should be indistinguishable from zero.

Usage:
    python negative_control.py <patches.csv> [n_rep] [budget] [n_shuffle]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from targeted_aniso import margin


def shuffle_field(df, seed):
    """One permutation across all attribute columns; coordinates untouched."""
    cols = [c for c, _, _ in E.ATTRS]
    out = df.copy()
    perm = np.random.default_rng(seed).permutation(len(df))
    out[cols] = df[cols].values[perm]
    return out.reset_index(drop=True)


def main(path, n_rep=20, budget=150, n_shuffle=3):
    n_rep, budget, n_shuffle = int(n_rep), int(budget), int(n_shuffle)
    cols = [c for c, _, _ in E.ATTRS] + ['patch_x', 'patch_y']
    full = pd.read_csv(path).dropna(subset=cols)
    print(f"{os.path.basename(path)}: {len(full)} patches, {n_rep} reps, "
          f"budget {budget}, {n_shuffle} shuffles")

    fits = E.fit_priors(full)
    real_prior = list(E.PRIOR)
    print("\nREAL field")
    print("  priors: " + ", ".join(f"{l} {r:.0f} m/{n:.3f}" for r, n, _, l in fits))
    m, w, n_, t = margin(full, n_rep, budget)
    print("  margin vs ablation: %+.3f  (%d/%d, t = %.2f)" % (m, w, n_, t))

    print("\nSHUFFLED field -- spatial structure destroyed, marginals preserved")
    res = []
    for s in range(n_shuffle):
        sh = shuffle_field(full, 1000 + s)
        f2 = E.fit_priors(sh)
        m2, w2, n2, t2 = margin(sh, n_rep, budget)
        res.append((m2, t2))
        print("  shuffle %d  priors %s" % (s, ", ".join(f"{r:.0f}/{n:.2f}" for r, n, _, _ in f2)))
        print("             margin %+.3f  (%d/%d, t = %.2f)" % (m2, w2, n2, t2))
    E.PRIOR = real_prior

    mm = np.array([r[0] for r in res]); tt = np.array([r[1] for r in res])
    print("\nreal      %+.3f" % m)
    print("shuffled  %+.3f  (mean of %d, range %+.3f to %+.3f, max |t| %.2f)"
          % (mm.mean(), n_shuffle, mm.min(), mm.max(), np.abs(tt).max()))
    if m:
        print("shuffled margin is %.0f%% of the real one" % (100.0 * mm.mean() / m))
    # The question is ONLY whether a significantly POSITIVE margin survives the
    # shuffle. A negative or null margin is the expected outcome, not a failure,
    # and a ratio against the real margin is unstable when that margin is small.
    worst_t = tt.max()
    ok = worst_t < 2.0
    print("\nlargest POSITIVE t across shuffles: %+.2f  (threshold 2.00)" % worst_t)
    print("VERDICT: " + ("PASS -- no advantage survives on structureless data, so the "
                         "margin on the real field is evidence about the field rather "
                         "than about the scoring machinery."
                         if ok else
                         "FAIL -- a significant margin survives the shuffle. Part of the "
                         "advantage comes from the harness; investigate before claiming it."))


if __name__ == '__main__':
    main(*sys.argv[1:5])