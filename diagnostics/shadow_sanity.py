"""
Are the DEM-predicted shadows dark in the image?

This is a falsification test, not an exploration. A cast shadow is ground the
sun cannot reach. Whatever the DEM does or does not know about boulders, a
patch it says is 100% shadowed MUST be dark in OHRC. There is no physical
arrangement in which it is not.

why_negative.py returned:
    pred_shadow vs dn_mean   r = +0.078
which is the wrong sign. Predicted-shadow patches are marginally BRIGHTER than
average. That is the signature of a shadow field in the wrong place.

The distinction this settles:
  (A) the DEM is correctly placed but blind to boulder-scale shadow
      -> the premise LunaRec rests on; r ~ 0 against shadow_local is a real
         finding and the architecture pivot stands.
  (B) the DEM shadow field is misplaced
      -> every DEM-vs-label correlation computed so far is meaningless,
         including the one the pivot was built on. Not a disaster, but it has
         to be known before anything is written down.

Under (A), fully-predicted-shadow patches are DARK: their dn_mean sits far
below the scene median, near the dark threshold.
Under (B), they look like everything else.

A note on what counts as dark: lunar shadows are not black. Scattered light
from surrounding slopes fills them, so expect low DN rather than zero. The
comparison is against the scene's own distribution, not against an absolute.

Usage:
    python shadow_sanity.py <patches_*_rt.csv>
"""

import sys

import numpy as np
import pandas as pd


def main(path):
    df = pd.read_csv(path)
    for col in ('shadow_pred_dem', 'dn_mean'):
        if col not in df:
            sys.exit(f"missing column {col} -- run raytrace_shadow.py first")

    pred = df['shadow_pred_dem'].values
    dn = df['dn_mean'].values
    ok = np.isfinite(pred) & np.isfinite(dn)
    pred, dn, df = pred[ok], dn[ok], df[ok].reset_index(drop=True)

    print(f"{path}: {len(df)} patches with both columns\n")
    print(f"scene brightness: median {np.median(dn):.1f}  "
          f"p5 {np.percentile(dn, 5):.1f}  p95 {np.percentile(dn, 95):.1f}\n")

    print("brightness by how much shadow the DEM predicts")
    print("%-22s %7s %10s %10s %10s" % ('predicted shadow', 'n', 'dn mean',
                                        'dn median', 'dn p10'))
    bins = [(-0.01, 0.001, 'none (0%)'),
            (0.001, 0.10, '0-10%'),
            (0.10, 0.50, '10-50%'),
            (0.50, 0.99, '50-99%'),
            (0.99, 1.01, 'full (>=99%)')]
    rows = []
    for lo, hi, name in bins:
        m = (pred > lo) & (pred <= hi)
        if m.sum() == 0:
            print("%-22s %7d %10s %10s %10s" % (name, 0, '-', '-', '-'))
            continue
        rows.append((name, m.sum(), dn[m].mean()))
        print("%-22s %7d %10.1f %10.1f %10.1f"
              % (name, m.sum(), dn[m].mean(), np.median(dn[m]),
                 np.percentile(dn[m], 10)))

    full = pred >= 0.99
    none = pred <= 0.001
    print()
    if full.sum() >= 20 and none.sum() >= 20:
        a, b = dn[full].mean(), dn[none].mean()
        # Cohen's d: how many pooled standard deviations apart the two groups
        # are. Reported because with n in the hundreds a tiny, meaningless
        # difference in means can still look impressive as a p-value.
        sp = np.sqrt((dn[full].var(ddof=1) + dn[none].var(ddof=1)) / 2)
        d = (a - b) / sp if sp else np.nan
        print(f"fully shadowed (n={full.sum()}) mean DN {a:.1f}")
        print(f"not shadowed   (n={none.sum()}) mean DN {b:.1f}")
        print(f"difference {a-b:+.1f} DN,  Cohen's d = {d:+.2f}\n")

        if d < -0.8:
            print("  -> VERDICT (A): predicted shadows are genuinely dark.")
            print("     The shadow field is correctly placed, so r ~ 0 against")
            print("     shadow_local is a real finding: the DEM is blind to")
            print("     boulder-scale shadow. The architecture pivot stands.")
        elif d > -0.2:
            print("  -> VERDICT (B): predicted shadows are NOT dark.")
            print("     The shadow field is misplaced. Every DEM-vs-label")
            print("     correlation so far is uninterpretable, including the")
            print("     one the pivot rests on. Next step is to find out why:")
            print("       1. re-run raytrace_shadow with sun_az + 180 as a")
            print("          pure diagnostic. If predicted shadows go dark,")
            print("          the azimuth convention is inverted after all.")
            print("       2. if that does not fix it, the DEM and the OHRC")
            print("          control grid are not in the same frame, and the")
            print("          offset is larger than the +-600 m align_scan swept.")
        else:
            print("  -> AMBIGUOUS: a weak effect in the right direction.")
            print("     Consistent with a partly misplaced field, or with a")
            print("     DEM too coarse to place 10 m shadows. Treat the")
            print("     DEM-vs-label comparison as unproven either way.")
    else:
        print(f"too few patches at the extremes "
              f"(full={full.sum()}, none={none.sum()}) for a clean contrast")

    # The same question asked without binning, as a guard against the bin
    # edges doing the work.
    r = np.corrcoef(pred, dn)[0, 1]
    print(f"\nungrouped: pred_shadow vs dn_mean r = {r:+.3f} "
          f"(should be strongly NEGATIVE if the field is placed correctly)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
