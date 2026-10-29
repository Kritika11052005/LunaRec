"""
Fit the spatial covariance of the hazard label, per scene.

evaluate.py currently hardcodes RANGE_M = 1100 and NUGGET = 0.10. Those came
from one early fit in cluster B and have been carried, unexamined, through
every result since. They are the GP's entire prior: the range sets how far an
observation informs its neighbours, and the nugget sets how much of the signal
is unlearnable noise. If they are wrong, every policy is working from a wrong
model of the ground -- and the criterion, which propagates information through
the covariance, would be the policy most affected.

So this is not a tidying step. It converts the single largest assumption in
the method into a measurement, per scene, from the data itself.

Model (spherical, matching spherical_cov in evaluate.py):

    gamma(h) = n + (s - n) * [1.5 (h/a) - 0.5 (h/a)^3]   for h < a
             = s                                          for h >= a

  n = nugget     variance at zero separation: measurement noise plus any
                 structure finer than the 200 m patch grid
  s = sill       total variance
  a = range      separation beyond which patches are uncorrelated

The label is standardised first, so s ~ 1 and n is directly the fraction
evaluate.py calls NUGGET.

Usage:
    python fit_variogram.py <patches.csv> [more.csv ...]
"""

import sys

import numpy as np
import pandas as pd

MAX_LAG_M = 4000.0     # fit well beyond the expected range so the sill is
                       # pinned by data rather than by extrapolation
N_BINS = 40
MAX_PAIRS_N = 1400     # subsample patches if needed; pair count is O(n^2)


def empirical_variogram(xy, z, max_lag=MAX_LAG_M, n_bins=N_BINS):
    """Binned semivariance: 0.5 * mean squared difference, by separation."""
    d = np.hypot(xy[:, 0][:, None] - xy[:, 0], xy[:, 1][:, None] - xy[:, 1])
    g = 0.5 * (z[:, None] - z[None, :]) ** 2
    iu = np.triu_indices(len(z), k=1)
    d, g = d[iu], g[iu]
    keep = d <= max_lag
    d, g = d[keep], g[keep]

    edges = np.linspace(0, max_lag, n_bins + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    centres, gamma, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 30:
            centres.append(d[m].mean())
            gamma.append(g[m].mean())
            counts.append(int(m.sum()))
    return np.array(centres), np.array(gamma), np.array(counts)


def spherical_model(h, n, s, a):
    out = np.full_like(h, s, dtype=float)
    inside = h < a
    r = h[inside] / a
    out[inside] = n + (s - n) * (1.5 * r - 0.5 * r ** 3)
    return out


def fit(centres, gamma, counts):
    """Weighted least squares; weights = pair counts, so well-populated short
    lags (which determine the nugget and range) are not outvoted by the many
    noisy long-lag bins."""
    from scipy.optimize import curve_fit
    s0 = float(np.median(gamma[-5:])) if len(gamma) >= 5 else float(gamma.max())
    p0 = [0.1 * s0, s0, 1000.0]
    bounds = ([0.0, 1e-6, 50.0], [s0 * 1.5 if s0 > 0 else 1.0, s0 * 3, MAX_LAG_M])
    popt, _ = curve_fit(spherical_model, centres, gamma, p0=p0,
                        sigma=1.0 / np.sqrt(counts), bounds=bounds, maxfev=20000)
    pred = spherical_model(centres, *popt)
    ss_res = float(((gamma - pred) ** 2).sum())
    ss_tot = float(((gamma - gamma.mean()) ** 2).sum())
    return popt, 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def main(*paths):
    rows = []
    for path in paths:
        df = pd.read_csv(path).dropna(subset=['shadow_local', 'patch_x', 'patch_y'])
        if len(df) > MAX_PAIRS_N:
            df = df.sample(MAX_PAIRS_N, random_state=0)
        xy = df[['patch_x', 'patch_y']].values
        raw = df['shadow_local'].values
        z = (raw - raw.mean()) / raw.std()          # same standardisation as evaluate.py

        c, g, k = empirical_variogram(xy, z)
        (n, s, a), r2 = fit(c, g, k)
        nug_frac = n / s if s > 0 else np.nan
        rows.append((path, len(df), a, nug_frac, s, r2))

        print(f"{path}  ({len(df)} patches used)")
        print(f"  range   {a:8.0f} m     <- evaluate.py RANGE_M")
        print(f"  nugget  {nug_frac:8.3f}       <- evaluate.py NUGGET (as a fraction of sill)")
        print(f"  sill    {s:8.3f}       (should be ~1 on standardised data)")
        print(f"  fit R2  {r2:8.3f}")
        print("  lag(m)   gamma   pairs")
        for i in range(0, len(c), max(1, len(c) // 8)):
            print(f"  {c[i]:6.0f}  {g[i]:6.3f}  {k[i]:7d}")
        print()

    if len(rows) > 1:
        A = np.array([r[2] for r in rows])
        N = np.array([r[3] for r in rows])
        print("across scenes")
        print(f"  range   mean {A.mean():7.0f} m   spread {A.min():.0f}-{A.max():.0f}")
        print(f"  nugget  mean {N.mean():7.3f}     spread {N.min():.3f}-{N.max():.3f}")
        print()
        print(f"  Currently hardcoded: RANGE_M = 1100, NUGGET = 0.10")
        print(f"  A wide spread across scenes means one global constant cannot")
        print(f"  be right for all of them, and the parameters belong per scene.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
