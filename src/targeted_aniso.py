"""Targeted anisotropy sensitivity: does the MEASURED anisotropy hurt?

sensitivity.py sweeps range by a common factor applied to all three attributes
at once. That is not the situation the data shows. The field is anisotropic by
a DIFFERENT amount per attribute, and the isotropic GP uses one range per
attribute, so the real misspecification is: attribute a's range is wrong by
short_a/iso_a in one direction and long_a/iso_a in the other.

This sets each attribute's range to its own measured short axis, then to its
own measured long axis, and reports the margin against the ablation in each
case. If the margin survives both, the measured anisotropy is covered and the
isotropic model is adequate; sensitivity.py's global 4x case is then a
conservative upper bound rather than an estimate.

Usage:
    python targeted_aniso.py <patches.csv> <short_h,short_i,short_t> \
                             <long_h,long_i,long_t> [n_rep] [budget]

Ratios are per-attribute multipliers on the FITTED range, in ATTRS order
(hazard, illumination, texture).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E


def margin(df, n_rep, budget, seed0=100):
    """Paired margin of 'disagreement' over the ablation, on identical subsamples."""
    ours, abl = [], []
    for rep in range(n_rep):
        sub = (df.sample(min(400, len(df)), random_state=rep).reset_index(drop=True)
               if len(df) > 400 else df)
        for pol, acc in (('disagreement', ours), ('voi-value', abl)):
            h = E.run(sub, pol, budget, np.random.default_rng(seed0 + rep))
            acc.append(h[-1]['prec'])
    d = np.array(ours) - np.array(abl)
    t = (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else np.inf
    return float(d.mean()), int((d > 0).sum()), len(d), float(t)


def main(path, short_s, long_s, n_rep=20, budget=150, only=None):
    n_rep, budget = int(n_rep), int(budget)
    short = [float(x) for x in short_s.split(',')]
    long_ = [float(x) for x in long_s.split(',')]
    assert len(short) == len(long_) == len(E.ATTRS), "need one ratio per attribute"

    cols = [c for c, _, _ in E.ATTRS] + ['patch_x', 'patch_y']
    full = pd.read_csv(path).dropna(subset=cols)
    E.fit_priors(full)
    base = list(E.PRIOR)
    print(f"{os.path.basename(path)}: {len(full)} patches, {n_rep} reps, budget {budget}")
    print("fitted (isotropic) priors:",
          ", ".join(f"{l} {r:.0f} m/{n:.3f}" for (r, n), (_, _, l) in zip(base, E.ATTRS)))
    print()
    print("%-26s %10s %8s %9s %s" % ("setting", "margin", "wins", "t", "ranges (m)"))

    SETTINGS = (("fit",   "fitted (isotropic)",      [1.0] * len(base)),
                ("short", "each attr -> SHORT axis", short),
                ("long",  "each attr -> LONG axis",  long_))
    want = set(only.split(",")) if only else {"fit", "short", "long"}
    out = {}
    for key, name, mult in SETTINGS:
        if key not in want:
            continue
        E.PRIOR = [(r * m, n) for (r, n), m in zip(base, mult)]
        m_, w_, n_, t_ = margin(full, n_rep, budget)
        out[name] = (m_, w_, t_)
        print("%-26s %10.3f %8s %9.2f  %s"
              % (name, m_, f"{w_}/{n_}", t_,
                 " ".join(f"{r:.0f}" for r, _ in E.PRIOR)))
    E.PRIOR = base

    if len(out) < 3:
        print("\n(partial run: %s) -- combine with the rows already recorded"
              % ", ".join(sorted(want)))
        return
    base_m = out["fitted (isotropic)"][0]
    worst = min(out["each attr -> SHORT axis"][0], out["each attr -> LONG axis"][0])
    worst_t = min(out["each attr -> SHORT axis"][2], out["each attr -> LONG axis"][2])
    print(f"\nmargin at fitted settings : {base_m:+.3f}")
    print(f"worst case under measured anisotropy : {worst:+.3f} (t = {worst_t:.2f})")
    print(f"retained: {100.0 * worst / base_m:.0f}% of the fitted-setting margin"
          if base_m > 0 else "")
    ok = worst > 0 and worst_t > 2.0
    print("\nVERDICT: " + ("measured anisotropy is COVERED -- the isotropic GP is adequate, "
                           "and the global 4x sweep case is a conservative upper bound"
                           if ok else
                           "measured anisotropy DEGRADES the result -- state it as a limitation "
                           "and consider an anisotropic covariance"))


if __name__ == '__main__':
    main(*sys.argv[1:7])