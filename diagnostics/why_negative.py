"""
Why is DEM-predicted shadow mildly ANTI-correlated with the observed label?

r = -0.045 on shadow_local and -0.117 on shadow_frac, n = 1930. That is far
outside chance, so "the DEM is uninformative" does not fit: an uninformative
predictor lands at 0.000, not reliably below it.

Hypothesis (H1), which would SUPPORT the complementary-measurement thesis:
  The DEM predicts MACRO shadows -- crater rims throwing hundreds of metres.
  shadow_local measures MICRO shadows against a 15 m background -- boulders.
  Crater floors are smooth and boulder-poor, so patches the DEM says are
  shadowed should have LOW roughness, and the anti-correlation is a roughness
  effect rather than anything about shadows.

Hypothesis (H2), which would be a BUG:
  Shadows are being placed on the wrong side of obstacles -- a residual
  geometry error -- so predicted shadow lands on sunlit ground and vice versa.

These make opposite predictions, which is what makes the test worth running:

  H1: predicted shadow correlates NEGATIVELY with roughness, and once
      roughness is partialled out the shadow anti-correlation COLLAPSES
      toward zero. The DEM is tracking smoothness, not misplaced shadow.

  H2: the anti-correlation SURVIVES partialling out roughness, because a
      flipped shadow field is still about shadow. H2 also predicts that
      flipping the azimuth by 180 deg turns r positive; H1 predicts it does
      not help, because there is no real shadow signal to recover.

Usage:
    python why_negative.py <patches_*_rt.csv>
"""

import sys

import numpy as np
import pandas as pd


def pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())


def partial(x, y, z):
    """Correlation of x and y with the linear effect of z removed from both."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def main(path):
    df = pd.read_csv(path)
    if 'shadow_pred_dem' not in df:
        sys.exit("no shadow_pred_dem column -- run raytrace_shadow.py first")

    pred = df['shadow_pred_dem'].values
    print(f"{path}: {len(df)} patches\n")

    print("step 1 -- what does predicted shadow actually track?")
    for col in ('roughness', 'slope_mean', 'slope_p95', 'dn_mean'):
        if col in df:
            r, n = pearson(pred, df[col].values)
            print(f"  pred_shadow vs {col:<12} r = {r:+.3f}  (n={n})")
    print("  H1 expects a clear NEGATIVE correlation with roughness:")
    print("  DEM-shadowed patches are smooth crater floors.\n")

    print("step 2 -- does the anti-correlation survive removing roughness?")
    for col in ('shadow_local', 'shadow_frac'):
        if col in df:
            raw, n = pearson(pred, df[col].values)
            par, _ = partial(pred, df[col].values, df['roughness'].values)
            shrink = 100 * (1 - abs(par) / abs(raw)) if raw else np.nan
            print(f"  {col:<13} raw r = {raw:+.3f}   "
                  f"partialling out roughness -> {par:+.3f}   "
                  f"({shrink:+.0f}% of the effect gone)")

    print("\n  collapses toward 0  -> H1: the DEM tracks smoothness. The "
          "OHRC label\n"
          "                         measures something the DEM does not "
          "contain, which\n"
          "                         is the premise LunaRec needs.")
    print("  survives            -> H2: residual geometry error. Re-run "
          "raytrace_shadow\n"
          "                         with azimuth + 180 as a diagnostic; if r "
          "flips positive,\n"
          "                         the shadow field is mirrored and the "
          "march needs fixing.")

    if 'roughness' in df:
        print("\nstep 3 -- sanity on the physical story")
        q = pd.qcut(pd.Series(pred).rank(method='first'), 4,
                    labels=['least shadowed', 'q2', 'q3', 'most shadowed'])
        g = df.groupby(q, observed=True).agg(
            roughness=('roughness', 'mean'),
            slope=('slope_mean', 'mean'),
            shadow_local=('shadow_local', 'mean'),
            n=('roughness', 'size'))
        print(g.to_string())
        print("\n  H1 expects roughness to FALL monotonically from least to "
              "most DEM-shadowed.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
