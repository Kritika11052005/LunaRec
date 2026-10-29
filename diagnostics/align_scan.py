"""
Is the DEM spatially aligned with the OHRC georeferencing?

The cast-shadow prediction produced the right AMOUNT of shadow (5.3% predicted
vs 5.3% observed) but in the wrong PLACES (r ~ 0). That pattern says
misalignment, not missing information: if the DEM were simply uninformative,
the predicted amount would not match either.

This computes the shadow mask once, then samples it at patch centres offset by
a grid of translations. If the DEM and OHRC agree, correlation peaks at (0,0).
If it peaks elsewhere, that offset IS the georeferencing error between two data
products -- which is itself a shared-dependency error of exactly the kind this
project is about.

Usage:
    python align_scan.py <dem.tif> <patches.csv> <sun_az> <sun_el> [max_off_m] [step_m]
"""

import math
import sys

import numpy as np
import pandas as pd
import rasterio

from raytrace_shadow import shadow_mask, COARSEN, MAX_SHADOW_M


def main(dem_path, patch_path, sun_az, sun_el, max_off=600.0, step=100.0):
    sun_az, sun_el = float(sun_az), float(sun_el)
    max_off, step = float(max_off), float(step)
    df = pd.read_csv(patch_path)
    obs = df['shadow_local'].values

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        nodata = src.nodata
        px_m = abs(src.transform.a)
        tr = src.transform

        cx, cy = df.patch_x.mean(), df.patch_y.mean()
        lon = math.degrees(math.atan2(cx, cy)) % 360.0
        az_grid = sun_az + lon

        nr = (dem.shape[0] // COARSEN) * COARSEN
        nc = (dem.shape[1] // COARSEN) * COARSEN
        blk = dem[:nr, :nc].astype(np.float32)
        blk[dem[:nr, :nc] == nodata] = np.nan
        blk = blk.reshape(nr // COARSEN, COARSEN, nc // COARSEN, COARSEN).mean(axis=(1, 3))
        cdem = np.where(np.isnan(blk), nodata, blk)
        cpx = px_m * COARSEN

        print(f"sun grid azimuth {az_grid % 360:.2f}, elevation {sun_el:.2f}")
        mask = shadow_mask(cdem, nodata, cpx, az_grid, sun_el)
        print(f"cast shadow {100*mask.mean():.1f}% of grid\n")

        half = 100.0
        offs = np.arange(-max_off, max_off + 1, step)
        best = (None, -9)
        print("correlation of DEM-predicted vs observed shadow_local")
        print("rows = dy (north +), cols = dx (east +), metres\n")
        hdr = "      " + "".join(f"{o:+7.0f}" for o in offs)
        print(hdr)
        for dy in offs[::-1]:
            row = f"{dy:+6.0f}"
            for dx in offs:
                pred = np.full(len(df), np.nan)
                for i, (x, y) in enumerate(zip(df.patch_x.values, df.patch_y.values)):
                    r0, c0 = rasterio.transform.rowcol(tr, x + dx - half, y + dy + half)
                    r1, c1 = rasterio.transform.rowcol(tr, x + dx + half, y + dy - half)
                    r0, r1 = max(0, min(r0, r1)) // COARSEN, min(nr, max(r0, r1)) // COARSEN
                    c0, c1 = max(0, min(c0, c1)) // COARSEN, min(nc, max(c0, c1)) // COARSEN
                    if r1 > r0 and c1 > c0:
                        pred[i] = mask[r0:r1, c0:c1].mean()
                ok = ~np.isnan(pred)
                r = np.corrcoef(pred[ok], obs[ok])[0, 1] if ok.sum() > 50 else np.nan
                row += f"{r:+7.3f}"
                if not np.isnan(r) and r > best[1]:
                    best = ((dx, dy), r)
            print(row)

        print(f"\nbest offset {best[0]} m, r = {best[1]:+.3f}")
        if best[0] == (0.0, 0.0):
            print("  -> aligned; the DEM genuinely does not predict these shadows")
        else:
            print("  -> peak away from zero suggests a georeferencing offset")
            print("     between the DEM and the OHRC control grid")


if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main(*sys.argv[1:7])
