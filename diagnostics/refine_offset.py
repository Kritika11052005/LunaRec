"""
Fine scan around a candidate offset, with a sign-agnostic metric.

Why a third script. The wide scan found the same feature in both clusters --
(2000, 4000) in C, (2400, 4000) in B, one grid step apart across two different
DEM strips and two different scenes 100 km apart. But it judged that feature by
two criteria that were both wrong:

  SIGN. It demanded slope~dn_std be positive. At 9.5 deg sun a steep uniform
  slope inside a 200 m patch is either lit or shadowed, and both compress the
  dynamic range, so steep ground can perfectly well show LESS texture than flat
  ground scattered with boulders. The sign is not predictable a priori, so no
  test should rest on it.

  SHARPNESS. It demanded an isolated spike. Slope fields are autocorrelated
  over kilometres, so a correct alignment MUST produce a broad maximum several
  cells across at a 400 m step. The criterion excluded exactly the shape a true
  positive has.

This replaces both with a question that has no sign and no shape assumption:
how much of the image's variation can the DEM explain at all? Regress the
detrended image channel on ALL detrended DEM channels at once and report R^2.
Registration raises it however the individual signs fall.

A far-field control region is scanned alongside the candidate, so the peak is
judged against offsets that are definitely wrong rather than against a guess.

Usage:
    python refine_offset.py <dem.tif> <patches.csv> [cx] [cy] [half] [step]
        cx, cy : centre of the fine scan (default 2200, 4000)
        half   : half-width of the scan box (default 1600 m)
        step   : grid step (default 200 m)
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio

# isolated interpreters drop the script's own directory from sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from find_offset import slope_raster, boxmean_nan, detrend, PATCH_M


def r2_multi(Y, Xcols):
    """R^2 of regressing Y on Xcols (both already detrended, intercept added).

    Sign-agnostic by construction: it asks how much of Y the DEM explains,
    not in which direction any single channel pushes.
    """
    A = np.column_stack([np.ones(len(Y))] + list(Xcols))
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan


def main(dem_path, patch_path, cx=2200.0, cy=4000.0, half=1600.0, step=200.0):
    cx, cy, half, step = float(cx), float(cy), float(half), float(step)
    df = pd.read_csv(patch_path)

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        nodata, tr = src.nodata, src.transform
        px_m = abs(tr.a)
        slope, valid = slope_raster(dem, nodata, px_m)
        win = max(3, int(round(PATCH_M / px_m)) | 1)
        slope_s = boxmean_nan(slope, valid, win)
        del slope
        # roughness: how much the surface departs from its own local mean, and
        # elevation itself -- three channels rather than one, so the regression
        # can use whatever the DEM actually carries.
        z = dem.astype(np.float32)
        z[~valid] = np.nan
        elev_s = boxmean_nan(z, valid, win)
        rough_s = boxmean_nan(np.abs(z - elev_s), valid, win)
        del z

        nrow, ncol = slope_s.shape
        xs, ys = df.patch_x.values, df.patch_y.values
        min_keep = int(0.90 * len(df))

        def score(dx, dy):
            r, c = rasterio.transform.rowcol(tr, xs + dx, ys + dy)
            r, c = np.asarray(r), np.asarray(c)
            ok = (r >= 0) & (r < nrow) & (c >= 0) & (c < ncol)
            if ok.sum() < min_keep:
                return None
            S = slope_s[r[ok], c[ok]].astype(float)
            E = elev_s[r[ok], c[ok]].astype(float)
            R = rough_s[r[ok], c[ok]].astype(float)
            T = df['dn_std'].values[ok].astype(float)
            M = df['dn_mean'].values[ok].astype(float)
            g = np.isfinite(S) & np.isfinite(E) & np.isfinite(R) & \
                np.isfinite(T) & np.isfinite(M)
            if g.sum() < min_keep:
                return None
            gx_, gy_ = xs[ok][g], ys[ok][g]
            Sd, Ed, Rd = (detrend(v[g], gx_, gy_) for v in (S, E, R))
            Td, Md = detrend(T[g], gx_, gy_), detrend(M[g], gx_, gy_)
            return (r2_multi(Td, [Sd, Ed, Rd]),
                    r2_multi(Md, [Sd, Ed, Rd]),
                    int(g.sum()))

        print(f"{patch_path}: {len(df)} patches")
        base = score(0.0, 0.0)
        if base:
            print(f"zero offset (current state): R2(dn_std) = {base[0]:.4f}, "
                  f"R2(dn_mean) = {base[1]:.4f}")

        # Far-field control: offsets that are certainly wrong, to establish
        # what R2 a mis-registered DEM produces on this data.
        # Shrink the control annulus until it actually fits on this DEM. A
        # small clip can push every far offset off the raster, and an empty
        # control is worse than a near one: it leaves the peak unjudged.
        # The control annulus MUST lie beyond the terrain's own correlation
        # length, or it is not a null. The fitted variograms give ranges of
        # 2.5-3.5 km, so an earlier 3.0-4.5 km annulus sat inside the
        # correlated zone and returned R2 up to 0.26 where ~0.002 is the true
        # null for 3 regressors on ~2000 points. One cluster then "passed" and
        # one "failed" on the same contaminated baseline.
        #
        # 12 km is ~4x the longest fitted range. The annulus is NOT shrunk if
        # it does not fit: a control inside the correlation length is worse
        # than no control, because it silently licenses a false positive.
        CTRL_MIN, CTRL_MAX = 12000.0, 25000.0
        rng = np.random.default_rng(0)
        ctrl = []
        for _ in range(200):
            a = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(CTRL_MIN, CTRL_MAX)
            s = score(rad * np.cos(a), rad * np.sin(a))
            if s:
                ctrl.append(s[0])
        ctrl = np.array(ctrl)
        print(f"control annulus {CTRL_MIN/1000:.0f}-{CTRL_MAX/1000:.0f} km "
              f"(>= 4x the fitted variogram range, so genuinely uncorrelated)")
        if len(ctrl) < 10:
            print(f"  only {len(ctrl)} such offsets keep enough patches on this")
            print(f"  DEM clip. The clip is too small to establish a valid null,")
            print(f"  so NO verdict is possible here -- a nearer annulus would sit")
            print(f"  inside the terrain correlation length and manufacture one.")
            print(f"  Re-clip the DEM with a wider margin, or scan along the")
            print(f"  strip's long axis where more room exists.\n")
        else:
            print(f"  {len(ctrl)} offsets: R2(dn_std) median {np.median(ctrl):.4f}, "
                  f"max {ctrl.max():.4f}\n")

        offs = np.arange(-half, half + 1, step)
        print(f"fine scan around ({cx:.0f}, {cy:.0f}) +-{half:.0f} m, "
              f"{step:.0f} m steps\n")
        print('%9s %9s %12s %12s %8s' % ('dx', 'dy', 'R2(dn_std)',
                                         'R2(dn_mean)', 'n'))
        rows = []
        for dy in offs:
            for dx in offs:
                s = score(cx + dx, cy + dy)
                if s:
                    rows.append((cx + dx, cy + dy, s[0], s[1], s[2]))
        if not rows:
            sys.exit("no offsets in the scan box kept enough patches")
        arr = np.array(rows)
        for i in np.argsort(-arr[:, 2])[:12]:
            print('%9.0f %9.0f %12.4f %12.4f %8d'
                  % (arr[i, 0], arr[i, 1], arr[i, 2], arr[i, 3], arr[i, 4]))

        best = arr[np.argmax(arr[:, 2])]
        # Compare against the control's MAXIMUM, not its p95. This scan takes
        # the best of many offsets, so its winner beats any percentile by
        # construction -- a selection effect, not evidence. On a fixture with
        # no real offset the p95 rule fired at R2 = 0.051; the max rule does
        # not. The absolute floor is the second gate: a DEM that explains under
        # ~10% of image texture is not registered to it in any useful sense.
        cmax = float(ctrl.max()) if len(ctrl) >= 10 else np.nan
        print(f"\nbest R2(dn_std) = {best[2]:.4f} at ({best[0]:.0f}, {best[1]:.0f}) m")
        if not np.isfinite(cmax):
            print("control        = unavailable (see above)")
            print("\n  -> NO VERDICT. The peak above may be real or may be what this")
            print("     terrain produces at any offset; without a valid null there")
            print("     is no way to tell, and reporting it either way would be")
            print("     guessing.")
            return
        print(f"control max     = {cmax:.4f}   ({best[2]/cmax:.1f}x)"
              if cmax > 0 else f"control max     = {cmax:.4f}")
        if best[2] > 2 * cmax and best[2] > 0.10:
            print("\n  -> The DEM explains substantially more of the image here than")
            print("     at offsets known to be wrong. Treat this as the registration")
            print("     offset ONLY if the other cluster's best lands within a few")
            print("     hundred metres of it. One product-level error appears in both;")
            print("     two unrelated numbers mean something else is going on.")
        else:
            print("\n  -> Not above the far-field control. The DEM explains no more of")
            print("     the image at this offset than at offsets that are certainly")
            print("     wrong, so no translation registers these datasets. The cause")
            print("     is rotation, a projection-parameter mismatch, or a control")
            print("     grid that is not in the DEM's CRS at all.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:7])
