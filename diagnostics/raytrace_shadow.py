"""
Does the DEM already explain the shadows the camera sees?

This is the test the whole premise rests on. If cast shadows predicted from the
10 m DEM account for what OHRC observes, then high-resolution imaging reveals
nothing the elevation model didn't already contain, and "verification" buys
nothing. Whatever is left over is the part only the camera can see.

The earlier cos_inc check only modelled LOCAL slope shading and explained ~0.6%.
It could not see shadows cast by distant topography -- crater rims throwing
shadows hundreds of metres at 9 deg sun. This marches a ray toward the sun from
every DEM cell and tests whether any terrain along the way rises above the solar
elevation, which is the mechanism that check was blind to.

GRID NORTH IS NOT TRUE NORTH: in south polar stereographic a point at longitude
lon has true north at grid azimuth lon, so az_grid = az_true + lon. At 32 E that
is a 32 deg rotation. (An earlier version of this file added a further 180 deg;
see the note at the az_grid line in main() for how that was caught.)

Usage:
    python raytrace_shadow.py <dem.tif> <patches.csv> <sun_az> <sun_el> [out.csv]
"""

import math
import sys

import numpy as np
import pandas as pd
import rasterio

MAX_SHADOW_M = 2500.0     # at 9 deg sun a 400 m rim casts ~2.5 km
COARSEN = 2               # work at 20 m: rim shadows are 100s of m, and a
                          # full-res march would be 83 Mpx x 250 steps

# NOTE: the march must step ONE cell at a time. Stepping 2 cells skipped
# alternate source rows, so a thin ridge was sampled by only half the cells
# downstream of it -- the shadow extent was right but came out stippled.


def shadow_mask(dem, nodata, px_m, az_grid_deg, sun_el_deg):
    """True where a cell is in cast shadow.

    Marches toward the sun in integer pixel offsets, tracking the largest
    elevation angle seen along the ray. A cell is shadowed once some terrain
    along that ray subtends a larger angle than the sun itself.
    """
    z = dem.astype(np.float32)
    valid = dem != nodata
    z[~valid] = np.nan

    az = math.radians(az_grid_deg)
    dx, dy = math.sin(az), math.cos(az)        # unit vector toward the sun
    tan_sun = math.tan(math.radians(sun_el_deg))

    nrow, ncol = z.shape
    max_angle = np.full(z.shape, -np.inf, np.float32)

    n_steps = int(MAX_SHADOW_M / px_m)
    for s in range(1, n_steps + 1):
        d = s
        dc = int(round(d * dx))
        dr = int(round(-d * dy))               # rows increase downward (-Y)
        if abs(dc) >= ncol or abs(dr) >= nrow:
            break

        shifted = np.full(z.shape, np.nan, np.float32)
        r0s, r1s = max(0, dr), min(nrow, nrow + dr)
        c0s, c1s = max(0, dc), min(ncol, ncol + dc)
        r0d, r1d = max(0, -dr), min(nrow, nrow - dr)
        c0d, c1d = max(0, -dc), min(ncol, ncol - dc)
        shifted[r0d:r1d, c0d:c1d] = z[r0s:r1s, c0s:c1s]

        dist = d * px_m
        with np.errstate(invalid='ignore'):
            ang = (shifted - z) / dist
        np.fmax(max_angle, np.nan_to_num(ang, nan=-np.inf), out=max_angle)

    return (max_angle > tan_sun) & valid


def main(dem_path, patch_path, sun_az, sun_el, out_path=None):
    sun_az, sun_el = float(sun_az), float(sun_el)
    df = pd.read_csv(patch_path)

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        nodata = src.nodata
        px_m = abs(src.transform.a)
        tr = src.transform

        # one representative longitude for the grid-north correction; the
        # patches span ~0.3 deg of longitude so a single value is fine here
        cx, cy = df.patch_x.mean(), df.patch_y.mean()
        lon = math.degrees(math.atan2(cx, cy)) % 360.0
        # In south polar stereographic, NORTH is radially OUTWARD from the
        # pole (moving north = latitude increasing = away from -90 = larger
        # radius). A point at longitude lon therefore has true north at grid
        # azimuth lon, so az_grid = az_true + lon. An earlier version added a
        # further 180 deg, which put every cast shadow on the wrong side of
        # every obstacle and produced r ~ 0 against the observed shadows.
        az_grid = sun_az + lon
        print(f"sun az {sun_az:.2f} true -> {az_grid % 360:.2f} grid "
              f"(lon {lon:.2f}), elevation {sun_el:.2f}")
        # coarsen before marching; nodata propagates so partial blocks are
        # excluded rather than averaged with -32768
        if COARSEN > 1:
            nr = (dem.shape[0] // COARSEN) * COARSEN
            nc = (dem.shape[1] // COARSEN) * COARSEN
            blk = dem[:nr, :nc].astype(np.float32)
            blk[dem[:nr, :nc] == nodata] = np.nan
            blk = blk.reshape(nr // COARSEN, COARSEN,
                              nc // COARSEN, COARSEN).mean(axis=(1, 3))
            cdem = np.where(np.isnan(blk), nodata, blk)
            cpx = px_m * COARSEN
        else:
            cdem, cpx = dem, px_m
        print(f"marching {int(MAX_SHADOW_M/cpx)} steps of {cpx:.0f} m, "
              f"up to {MAX_SHADOW_M:.0f} m, on a {cdem.shape[1]}x{cdem.shape[0]} grid")

        cmask = shadow_mask(cdem, nodata, cpx, az_grid, sun_el)
        mask = np.repeat(np.repeat(cmask, COARSEN, 0), COARSEN, 1) \
            if COARSEN > 1 else cmask
        # pad back to the original shape after the block trim
        full = np.zeros(dem.shape, bool)
        full[:mask.shape[0], :mask.shape[1]] = mask
        mask = full
        print(f"cast shadow covers {100*mask.mean():.1f}% of the DEM window\n")

        half = 100.0
        pred = np.full(len(df), np.nan)
        for i, (x, y) in enumerate(zip(df.patch_x.values, df.patch_y.values)):
            r0, c0 = rasterio.transform.rowcol(tr, x - half, y + half)
            r1, c1 = rasterio.transform.rowcol(tr, x + half, y - half)
            r0, r1 = max(0, min(r0, r1)), min(dem.shape[0], max(r0, r1))
            c0, c1 = max(0, min(c0, c1)), min(dem.shape[1], max(c0, c1))
            if r1 > r0 and c1 > c0:
                pred[i] = mask[r0:r1, c0:c1].mean()

    df['shadow_pred_dem'] = pred
    ok = ~np.isnan(pred)
    print(f"predicted for {ok.sum()} of {len(df)} patches")
    print(f"predicted shadow fraction: mean {pred[ok].mean():.4f} "
          f"p95 {np.percentile(pred[ok],95):.4f}")
    print()
    for col in ('shadow_local', 'shadow_frac', 'dn_mean'):
        if col in df:
            r = np.corrcoef(pred[ok], df[col].values[ok])[0, 1]
            print(f"  DEM-predicted vs observed {col:<13} r = {r:+.3f}  "
                  f"(r2 = {r*r:.3f})")

    if 'shadow_local' in df:
        y = df['shadow_local'].values[ok]
        x = pred[ok]
        coef = np.polyfit(x, y, 1)
        resid = y - np.polyval(coef, x)
        df.loc[ok, 'shadow_beyond_dem'] = resid
        print(f"\n  residual std {resid.std():.4f} vs observed std {y.std():.4f}"
              f"  ({100*resid.std()/y.std():.0f}% of signal unexplained by DEM)")

    out_path = out_path or patch_path.replace('.csv', '_rt.csv')
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main(*sys.argv[1:6])
