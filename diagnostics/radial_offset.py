"""
The DEM sits radially outward from the OHRC grid. Which correction fixes it?

The two clusters each showed a strong coupling (7-9x a valid far-field control)
at an offset pointing along GRID NORTH at their own longitude -- 4.0 deg off in
cluster C, 0.6 deg in cluster B -- with magnitudes 4808 m and 4940 m. In a south
polar stereographic projection grid north is radially outward from the pole, so
this is one systematic error: the DEM places ground further from the pole than
the OHRC control grid does, by about 4.87 km (0.16 deg of latitude).

No single (dx, dy) can fix that, because the same radial displacement decomposes
differently at every longitude. That is why the translation scans failed and why
a +-600 m sweep never had a chance.

Two candidate forms, distinguishable because the clusters sit at different polar
radii (665 km vs 630 km):

  CONSTANT   r' = r + d           same metres everywhere
  SCALE      r' = r * s           metres proportional to r, so cluster C
                                  (larger radius) should show the larger shift

Observed magnitudes run the other way -- the smaller radius has the larger
shift -- which favours CONSTANT, but by 132 m, under one grid step of the scan
that produced them. This fits both forms properly, per cluster, at finer
resolution, and reports which is more consistent ACROSS clusters. That
cross-cluster consistency is the test: one physical cause must fit both with
one parameter.

Usage:
    python radial_offset.py <dem.tif> <patches.csv>
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from find_offset import slope_raster, boxmean_nan, detrend, PATCH_M
from refine_offset import r2_multi


def main(dem_path, patch_path):
    df = pd.read_csv(patch_path)
    xs, ys = df.patch_x.values.astype(float), df.patch_y.values.astype(float)
    rad = np.hypot(xs, ys)
    ux, uy = xs / rad, ys / rad          # unit vector radially outward = grid north
    cx, cy = xs.mean(), ys.mean()
    lon = np.degrees(np.arctan2(cx, cy)) % 360.0
    r_mean = float(np.hypot(cx, cy))
    print(f"{patch_path}: {len(df)} patches, lon {lon:.2f} deg, "
          f"mean polar radius {r_mean:.0f} m\n")

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        nodata, tr = src.nodata, src.transform
        px_m = abs(tr.a)
        slope, valid = slope_raster(dem, nodata, px_m)
        win = max(3, int(round(PATCH_M / px_m)) | 1)
        slope_s = boxmean_nan(slope, valid, win)
        del slope
        z = dem.astype(np.float32)
        z[~valid] = np.nan
        elev_s = boxmean_nan(z, valid, win)
        rough_s = boxmean_nan(np.abs(z - elev_s), valid, win)
        del z

        nrow, ncol = slope_s.shape
        min_keep = int(0.90 * len(df))

        def score_at(nx, ny):
            r, c = rasterio.transform.rowcol(tr, nx, ny)
            r, c = np.asarray(r), np.asarray(c)
            ok = (r >= 0) & (r < nrow) & (c >= 0) & (c < ncol)
            if ok.sum() < min_keep:
                return None
            S = slope_s[r[ok], c[ok]].astype(float)
            E = elev_s[r[ok], c[ok]].astype(float)
            R = rough_s[r[ok], c[ok]].astype(float)
            T = df['dn_std'].values[ok].astype(float)
            g = np.isfinite(S) & np.isfinite(E) & np.isfinite(R) & np.isfinite(T)
            if g.sum() < min_keep:
                return None
            gx_, gy_ = xs[ok][g], ys[ok][g]
            Sd, Ed, Rd = (detrend(v[g], gx_, gy_) for v in (S, E, R))
            Td = detrend(T[g], gx_, gy_)
            return r2_multi(Td, [Sd, Ed, Rd])

        print("CONSTANT radial shift:  r' = r + d")
        print('%10s %12s' % ('d (m)', 'R2(dn_std)'))
        best_d, best_rd = None, -1
        for d in np.arange(3000.0, 6501.0, 100.0):
            v = score_at(xs + d * ux, ys + d * uy)
            if v is None:
                continue
            if v > best_rd:
                best_d, best_rd = float(d), v
            if int(d) % 500 == 0:
                print('%10.0f %12.4f' % (d, v))

        print("\nSCALE factor:  r' = r * s")
        print('%10s %12s %12s' % ('s', 'implied d (m)', 'R2(dn_std)'))
        best_s, best_rs = None, -1
        for s in np.arange(1.0030, 1.01201, 0.00015):
            v = score_at(xs * s, ys * s)
            if v is None:
                continue
            if v > best_rs:
                best_s, best_rs = float(s), v
            if abs((s - 1.0030) % 0.0015) < 1e-9:
                print('%10.5f %12.0f %12.4f' % (s, (s - 1) * r_mean, v))

        base = score_at(xs, ys)
        print(f"\nuncorrected (current state): R2 = {base:.4f}" if base is not None
              else "\nuncorrected: unavailable")
        if best_d is None or best_s is None:
            print("\nNo corrected position kept 90% of the patches on this DEM clip.")
            print("The radial shift pushes the patches off the raster, so the clip")
            print("needs a wider margin -- re-run clip_dem.py with PAD raised to")
            print("about 8000 m and try again.")
            return
        print(f"best CONSTANT: d = {best_d:.0f} m        R2 = {best_rd:.4f}")
        print(f"best SCALE   : s = {best_s:.5f}  "
              f"(= {(best_s-1)*r_mean:.0f} m here)  R2 = {best_rs:.4f}")
        print("\nRun this on the OTHER cluster and compare. The correct form is the")
        print("one whose PARAMETER agrees across clusters -- d in metres for")
        print("CONSTANT, or s as a ratio for SCALE. Whichever agrees is the physical")
        print("cause; the other will need a different number per cluster, which is")
        print("not a cause but a curve fit.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
