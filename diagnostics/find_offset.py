"""
Where is the DEM relative to the OHRC grid?

Established, on all four scenes in both blocks: every DEM-derived quantity is
uncorrelated with every image-derived quantity. slope~dn_std, rough~dn_std,
slope~dn_mean, elev~dn_mean all sit at |r| < 0.14, while two OHRC-derived
columns correlate perfectly sensibly with each other (dn_std~shadow_local
+0.260). The image side is internally consistent; the DEM side is detached.

That is a registration failure, so this looks for the registration.

Why this is not align_scan.py again:
  - align_scan swept +-600 m. A real offset larger than that produces a flat
    field of noise, which is exactly what it reported.
  - align_scan scored with the RAY-TRACED shadow, which depends on the sun
    azimuth convention -- the thing we could not verify. If the azimuth is
    also wrong, that scan cannot find the offset even where it exists.

This uses DEM SLOPE against image TEXTURE (dn_std) instead. Rough, steep
ground scatters light and produces more brightness variance no matter which
direction the sun comes from, so the metric is azimuth-independent. Whatever
is wrong with the sun geometry cannot hide a true alignment from this test.

Reading the result:
  A clear peak well above the background  -> that offset IS the registration
      error. Re-run build_patches.py with it applied and the DEM features
      finally describe the same ground as the labels.
  No peak anywhere, flat near zero        -> not a translation. Next suspects
      are a rotation or a projection parameter mismatch (different central
      meridian, standard parallel, or body radius) between the DEM product
      and the OHRC control grid.

Usage:
    python find_offset.py <dem.tif> <patches.csv> [max_off_m] [step_m]
"""

import sys

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

PATCH_M = 200.0


def slope_raster(dem, nodata, px_m):
    """Slope in degrees, with nodata excluded rather than treated as terrain."""
    z = dem.astype(np.float32)
    valid = (dem != nodata) if nodata is not None else np.ones(dem.shape, bool)
    z[~valid] = np.nan
    gy, gx = np.gradient(z, px_m)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32), valid


def boxmean_nan(a, valid, size):
    """Mean over a box window, ignoring NaN.

    uniform_filter has no NaN handling: one NaN poisons every window touching
    it, which near a nodata edge silently blanks good data. Filtering the
    zero-filled values and the validity mask separately, then dividing, gives
    the mean of the valid cells in each window.
    """
    filled = np.where(valid, np.nan_to_num(a, nan=0.0), 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    num = uniform_filter(filled, size=size, mode='constant', cval=0.0)
    den = uniform_filter(w, size=size, mode='constant', cval=0.0)
    out = np.full(a.shape, np.nan, np.float32)
    good = den > 0.5
    out[good] = num[good] / den[good]
    return out


def detrend(v, xs, ys):
    """Remove a quadratic spatial trend.

    THIS IS THE WHOLE TEST, not a refinement. Both DEM slope and image texture
    carry broad gradients across a scene. Shifting one by a few km aligns or
    anti-aligns those gradients and manufactures |r| of 0.3-0.4 with no
    registration meaning whatever -- which is what the first version of this
    script reported, in opposite directions in the two clusters.

    Registration is a FINE-SCALE property: if two rasters are aligned, their
    local structure matches feature by feature. Removing the trend from both
    leaves only that, so a peak in the residual correlation means the terrain
    itself lines up rather than that two slopes happen to point the same way.
    """
    sx = (xs - xs.mean()) / max(xs.std(), 1.0)
    sy = (ys - ys.mean()) / max(ys.std(), 1.0)
    A = np.column_stack([np.ones_like(sx), sx, sy, sx * sx, sx * sy, sy * sy])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ coef


def main(dem_path, patch_path, max_off=8000.0, step=400.0):
    max_off, step = float(max_off), float(step)
    df = pd.read_csv(patch_path)
    for col in ('dn_std', 'dn_mean'):
        if col not in df:
            sys.exit(f"patches file has no {col}; rebuild with build_patches.py")

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        nodata, tr = src.nodata, src.transform
        px_m = abs(tr.a)
        print(f"DEM {src.width} x {src.height} at {px_m:.1f} m")
        print(f"patches {len(df)}  x {df.patch_x.min():.0f}..{df.patch_x.max():.0f}"
              f"  y {df.patch_y.min():.0f}..{df.patch_y.max():.0f}\n")

        slope, valid = slope_raster(dem, nodata, px_m)
        win = max(3, int(round(PATCH_M / px_m)) | 1)
        print(f"smoothing slope over {win} px (~{win*px_m:.0f} m) to match the patch scale")
        slope_s = boxmean_nan(slope, valid, win)
        del slope

        obs_t = df['dn_std'].values
        obs_m = df['dn_mean'].values
        xs, ys = df.patch_x.values, df.patch_y.values
        nrow, ncol = slope_s.shape

        offs = np.arange(-max_off, max_off + 1, step)
        print(f"scanning {len(offs)}x{len(offs)} offsets, +-{max_off:.0f} m "
              f"in {step:.0f} m steps\n")

        # Offsets that push most patches off the DEM must be discarded, not
        # ranked. The first version let an offset keeping 250 of 2010 patches
        # win outright: at that shift the survivors are a biased corner of the
        # scene, and a correlation over them says nothing about registration.
        min_keep = int(0.90 * len(df))
        print(f"requiring >= {min_keep} of {len(df)} patches on the DEM "
              f"(offsets that drop more are discarded)\n")

        results, dropped = [], 0
        for dy in offs:
            for dx in offs:
                r, c = rasterio.transform.rowcol(tr, xs + dx, ys + dy)
                r, c = np.asarray(r), np.asarray(c)
                ok = (r >= 0) & (r < nrow) & (c >= 0) & (c < ncol)
                if ok.sum() < min_keep:
                    dropped += 1
                    continue
                s = slope_s[r[ok], c[ok]]
                good = np.isfinite(s) & np.isfinite(obs_t[ok])
                if good.sum() < min_keep:
                    dropped += 1
                    continue
                gx_, gy_ = xs[ok][good], ys[ok][good]
                sd = detrend(s[good].astype(float), gx_, gy_)
                td = detrend(obs_t[ok][good].astype(float), gx_, gy_)
                md = detrend(obs_m[ok][good].astype(float), gx_, gy_)
                rt = np.corrcoef(sd, td)[0, 1]
                rm = np.corrcoef(sd, md)[0, 1]
                results.append((dx, dy, rt, rm, int(good.sum())))
        if dropped:
            print(f"({dropped} offsets discarded for falling off the DEM)\n")

        if not results:
            sys.exit("no offset kept enough patches inside the DEM -- check that "
                     "the patch coordinates and the DEM share a CRS")

        res = np.array([(a, b, c, d, e) for a, b, c, d, e in results])
        # Rank by |r| against texture: a strong NEGATIVE correlation is just as
        # informative as a positive one, and would itself say something is
        # inverted. What matters is whether ANY offset couples the two datasets.
        order = np.argsort(-np.abs(res[:, 2]))

        base = [r for r in results if r[0] == 0 and r[1] == 0]
        if base:
            print(f"at zero offset (current state): slope~dn_std r = {base[0][2]:+.3f}")
        bg = float(np.median(np.abs(res[:, 2])))
        print(f"background |r| across all offsets: median {bg:.3f}\n")

        print("strongest couplings found")
        print('%10s %10s %12s %12s %9s' % ('dx (m)', 'dy (m)', 'slope~dn_std',
                                           'slope~dn_mean', 'n'))
        for i in order[:10]:
            dx, dy, rt, rm, n = res[i]
            print('%10.0f %10.0f %12.3f %12.3f %9d' % (dx, dy, rt, rm, n))

        best = res[order[0]]
        peak, ratio = abs(best[2]), abs(best[2]) / bg if bg > 0 else np.inf
        p95 = float(np.percentile(np.abs(res[:, 2]), 95))
        sharp = peak / p95 if p95 > 0 else np.inf
        print(f"\npeak |r| = {peak:.3f} at ({best[0]:.0f}, {best[1]:.0f}) m, "
              f"{ratio:.1f}x the median, {sharp:.2f}x the 95th percentile")
        print(f"sign: {'POSITIVE' if best[2] > 0 else 'NEGATIVE'} "
              f"(rough terrain should give MORE image texture, so a real "
              f"alignment is positive)")

        # Three conditions, all required. The first version asked only for
        # magnitude and declared a find on a broad trend artefact that had the
        # wrong sign in one cluster and an incompatible offset in the other.
        ok_mag = peak > 0.30
        ok_sharp = sharp > 1.5       # isolated spike, not the top of a plateau
        ok_sign = best[2] > 0        # physically the right direction
        if ok_mag and ok_sharp and ok_sign:
            print("  -> REGISTRATION ERROR FOUND: strong, isolated, right sign.")
            print("     Confirm it reproduces in the OTHER cluster before acting:")
            print("     one product-level offset should appear in both.")
            print("     Then apply it in build_patches.py and rebuild.")
        elif peak > 0.20:
            why = []
            if not ok_sharp:
                why.append(f"not isolated (only {sharp:.2f}x the 95th pct -- a "
                           f"plateau, not a peak)")
            if not ok_sign:
                why.append("wrong sign (negative: rough terrain would mean LESS "
                           "texture)")
            if not ok_mag:
                why.append(f"too weak ({peak:.3f})")
            print("  -> NOT a registration find: " + "; ".join(why) + ".")
            print("     A residual correlation of this size after detrending is")
            print("     consistent with chance over ~1600 offsets tested.")
        else:
            print("  -> NO TRANSLATION EXPLAINS IT. The mismatch is not a simple shift.")
            print("     Next suspects, in order:")
            print("       1. rotation between the two grids")
            print("       2. a projection parameter mismatch -- central meridian,")
            print("          standard parallel, or body radius (1737.4 km vs 1737.1)")
            print("       3. the control grid is not in the DEM's CRS at all, and")
            print("          agreeing in range is a coincidence of the polar projection")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
