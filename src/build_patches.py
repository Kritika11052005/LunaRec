"""
Build the LunaRec patch table: DEM features + OHRC hazard statistics.

Both sources are aggregated PER PATCH rather than resampled to a common raster.
Downsampling 0.29 m OHRC to the 10 m DEM grid would destroy the boulder-scale
signal the labels depend on, so instead each 200 m patch gets:
  - terrain features from the DEM (20x20 px at 10 m)
  - hazard statistics from OHRC at native resolution (~690x690 px at 0.29 m)

The OHRC .img is a headerless uint8 array (Scan x Pixel, row-major), read via
memmap so only the blocks we need touch RAM.

Georeferencing comes from the _g_grd_ CSV: a complete 121 x 903 control grid
mapping (Pixel, Scan) -> (Longitude, Latitude), interpolated here.

NOTE ON LABELS: shadow fraction at low sun is a hazard PROXY, not a validated
boulder detector. At 19 deg sun a 1 m boulder casts a ~2.9 m shadow (~10 px),
so dark-pixel statistics do carry boulder signal -- but calling this a
"hazard label" in the paper requires either validation against a real boulder
catalogue or honest framing as a proxy. Do not skip that distinction.

Usage:
    python build_patches.py <grd.csv> <ohrc.img> <dem.tif> [out.csv]
"""

import os
import sys

import numpy as np
import rasterio
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import uniform_filter

R_MOON = 1737400.0
PATCH_M = 200.0          # patch size in metres
OHRC_SHAPE = None        # inferred from file size / 12000
OHRC_SAMPLES = 12000
SHADOW_PCTL = 15         # dark-pixel threshold percentile, scene-wide


def latlon_to_xy(lat, lon):
    colat = np.radians(90.0 + np.asarray(lat))
    r = 2.0 * R_MOON * np.tan(colat / 2.0)
    lonr = np.radians(np.asarray(lon))
    return r * np.sin(lonr), r * np.cos(lonr)


def load_grid(path):
    g = np.genfromtxt(path, delimiter=',', names=True)
    x, y = latlon_to_xy(g['Latitude'], g['Longitude'])
    pts = np.column_stack([x, y])
    ps = np.column_stack([g['Pixel'], g['Scan']])
    # inverse map: ground metres -> image coords
    inv = LinearNDInterpolator(pts, ps)

    # across-track ground sampling, derived per scene rather than assumed:
    # scenes differ (0.21 vs 0.29 m/px), and a fixed pixel window would mean a
    # different PHYSICAL window per scene, which would confound any comparison
    # of shadow_local between scenes.
    n_pix = len(np.unique(g['Pixel']))
    n_scan = len(np.unique(g['Scan']))
    X = x.reshape(n_scan, n_pix)
    Y = y.reshape(n_scan, n_pix)
    step = np.unique(g['Pixel'])[1] - np.unique(g['Pixel'])[0]
    px_m = float(np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1)).mean() / step)
    return inv, x, y, px_m


def terrain_features(dem, nodata, sun_az=None, sun_el=None, lon=None):
    """Slope, roughness, and predicted topographic shading from a DEM block.

    The shading prediction is what lets us separate boulder shadows from
    hillside shadows. The first two scenes had sun azimuths 272 deg apart, so
    self-shadowing fell on opposite faces and the raw shadow metrics did not
    correlate at all (+0.008) across the same ground. Predicting the
    topographic part means the residual is what topography does NOT explain.

    GRID NORTH IS NOT TRUE NORTH. In south polar stereographic, moving north
    means moving AWAY from the south pole, i.e. radially outward, so a point at
    longitude lon has true north along grid azimuth lon (not lon + 180, which
    an earlier version used and which reversed every shading computation).

    LIMITATION: this models local slope shading only, not shadows cast by
    distant topography (crater rims). At 9.6 deg sun those cast shadows are
    significant, and capturing them needs horizon ray-tracing over the DEM.
    """
    valid = dem != nodata
    # Must be FULLY valid. Accepting 90% and then computing over the whole
    # block let nodata (-32768) into the plane fit, which produced roughness
    # values in the thousands on patches near the DEM strip edge. A partially
    # covered 200 m patch is not worth rescuing -- drop it.
    if not valid.all() or dem.shape[0] < 3 or dem.shape[1] < 3:
        return None
    z = dem.astype(np.float32)
    gy, gx = np.gradient(z, 10.0, 10.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # roughness: std of elevation after removing a planar trend
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(z.size)])
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    resid = z.ravel() - A @ coef
    out = {
        'elev_mean': float(z.mean()),
        'slope_mean': float(slope.mean()),
        'slope_p95': float(np.percentile(slope, 95)),
        'slope_max': float(slope.max()),
        'roughness': float(resid.std()),
    }

    if sun_az is not None:
        # gradients in grid axes: rows run -Y (north-up raster), cols run +X
        zx = gx            # dz/dX
        zy = -gy           # dz/dY
        slope_r = np.arctan(np.hypot(zx, zy))
        # aspect = grid azimuth the slope faces, clockwise from +Y
        aspect = np.arctan2(-zx, -zy)
        az_grid = np.radians(sun_az + lon)
        el_r = np.radians(sun_el)
        cos_i = (np.sin(el_r) * np.cos(slope_r) +
                 np.cos(el_r) * np.sin(slope_r) * np.cos(az_grid - aspect))
        out['cos_inc'] = float(np.clip(cos_i, 0, None).mean())
        out['selfshadow_frac'] = float((cos_i <= 0).mean())
    return out


def hazard_stats(block, dark_thresh, px_m=0.289):
    """Shadow / texture statistics from a native-resolution OHRC block.

    The shadow metric is LOCAL: each pixel is compared against a smoothed
    background estimated over a window larger than a boulder shadow but much
    smaller than the scene's illumination drift. A single scene-wide threshold
    (dark_thresh) measures how bright a patch is, not how many shadows it
    holds -- it correlated -0.82 with patch mean DN on the first scene.

    At 19 deg sun a 1 m boulder casts ~2.9 m of shadow (~10 px at 0.29 m/px),
    so the background window is set to ~15 m: an order of magnitude above
    boulder shadows, well below the along-track brightness gradient.
    """
    if block.size == 0:
        return None
    b = block.astype(np.float32)

    win = max(3, int(round(15.0 / px_m)) | 1)     # ~15 m, odd
    if min(b.shape) > win:
        bg = uniform_filter(b, size=win, mode='nearest')
    else:
        bg = np.full_like(b, b.mean())

    rel = b / np.maximum(bg, 1.0)                 # brightness relative to local background
    shadow_local = rel < 0.65                     # clearly darker than surroundings

    gy, gx = np.gradient(b)
    grad = np.hypot(gx, gy)

    return {
        'dn_mean': float(b.mean()),
        'dn_std': float(b.std()),
        'shadow_frac': float((b <= dark_thresh).mean()),   # kept for comparison
        'shadow_local': float(shadow_local.mean()),        # the metric to use
        'rel_p05': float(np.percentile(rel, 5)),
        'grad_mean': float(grad.mean()),
        'n_px': int(b.size),
    }


def main(grd_path, img_path, dem_path, out_path='patches.csv',
         sun_az=None, sun_el=None):
    sun_az = float(sun_az) if sun_az is not None else None
    sun_el = float(sun_el) if sun_el is not None else None
    if sun_az is not None:
        print(f"sun geometry: azimuth {sun_az:.2f} deg, elevation {sun_el:.2f} deg")
    inv, gx, gy, px_m = load_grid(grd_path)
    print(f"control grid: x {gx.min():.0f}..{gx.max():.0f}  y {gy.min():.0f}..{gy.max():.0f}")
    print(f"sampling    : {px_m:.3f} m/px (from grid)  -> background window "
          f"{max(3, int(round(15.0/px_m)) | 1)} px = 15 m")

    n_scan = os.path.getsize(img_path) // OHRC_SAMPLES
    img = np.memmap(img_path, dtype=np.uint8, mode='r', shape=(n_scan, OHRC_SAMPLES))
    print(f"OHRC image : {n_scan} scans x {OHRC_SAMPLES} samples (memmap)")

    # scene-wide dark threshold from a strided sample (cheap, representative)
    sample = np.asarray(img[::200, ::20])
    dark_thresh = np.percentile(sample, SHADOW_PCTL)
    print(f"dark threshold (p{SHADOW_PCTL} of scene): DN {dark_thresh:.1f}")

    src = rasterio.open(dem_path)
    print(f"DEM        : {src.width} x {src.height}, nodata {src.nodata}\n")

    x0 = np.floor(gx.min() / PATCH_M) * PATCH_M
    y0 = np.floor(gy.min() / PATCH_M) * PATCH_M
    nx = int(np.ceil((gx.max() - x0) / PATCH_M))
    ny = int(np.ceil((gy.max() - y0) / PATCH_M))
    print(f"patch grid : {nx} x {ny} = {nx*ny} candidate patches at {PATCH_M:.0f} m\n")

    rows = []
    skipped = {'outside_image': 0, 'no_dem': 0}
    for j in range(ny):
        for i in range(nx):
            xmin = x0 + i * PATCH_M
            ymin = y0 + j * PATCH_M
            cx, cy = xmin + PATCH_M / 2, ymin + PATCH_M / 2

            corners = np.array([[xmin, ymin], [xmin + PATCH_M, ymin],
                                [xmin, ymin + PATCH_M],
                                [xmin + PATCH_M, ymin + PATCH_M]])
            ps = inv(corners)
            if np.isnan(ps).any():
                skipped['outside_image'] += 1
                continue
            p0, p1 = int(ps[:, 0].min()), int(np.ceil(ps[:, 0].max()))
            s0, s1 = int(ps[:, 1].min()), int(np.ceil(ps[:, 1].max()))
            p0, p1 = max(p0, 0), min(p1, OHRC_SAMPLES)
            s0, s1 = max(s0, 0), min(s1, n_scan)
            if p1 <= p0 or s1 <= s0:
                skipped['outside_image'] += 1
                continue

            win = rasterio.windows.from_bounds(
                xmin, ymin, xmin + PATCH_M, ymin + PATCH_M, transform=src.transform)
            dem_block = src.read(1, window=win, boundless=True,
                                 fill_value=src.nodata)
            lon = np.degrees(np.arctan2(cx, cy)) % 360.0
            tf = terrain_features(dem_block, src.nodata, sun_az, sun_el, lon)
            if tf is None:
                skipped['no_dem'] += 1
                continue

            hs = hazard_stats(np.asarray(img[s0:s1, p0:p1]), dark_thresh, px_m)
            if hs is None:
                skipped['outside_image'] += 1
                continue

            rows.append({'patch_x': cx, 'patch_y': cy,
                         'ohrc_p0': p0, 'ohrc_p1': p1,
                         'ohrc_s0': s0, 'ohrc_s1': s1,
                         **tf, **hs})
        if ny > 20 and j % max(1, ny // 10) == 0:
            print(f"  row {j}/{ny}  kept {len(rows)}")

    src.close()

    if not rows:
        sys.exit("no patches produced -- check that the DEM and scene overlap")

    keys = list(rows[0].keys())
    with open(out_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(f"{r[k]}" for k in keys) + '\n')

    print(f"\nkept {len(rows)} patches -> {out_path}")
    print(f"skipped: {skipped}")
    arr = {k: np.array([r[k] for r in rows], float) for k in
           ('slope_mean', 'slope_p95', 'roughness',
            'shadow_frac', 'shadow_local', 'dn_mean')}
    print()
    for k, v in arr.items():
        print(f"  {k:<13} mean {v.mean():8.3f}  p5 {np.percentile(v,5):8.3f}  p95 {np.percentile(v,95):8.3f}")

    if 'cos_inc' in rows[0]:
        ci = np.array([r['cos_inc'] for r in rows], float)
        sl = arr['shadow_local']
        r = np.corrcoef(ci, sl)[0, 1]
        print(f"\n  topographic shading explains the shadows?")
        print(f"    cos_inc vs shadow_local   r = {r:+.3f}  (r2 = {r*r:.3f})")
        # residual: the part of observed shadow that topography does NOT predict
        coef = np.polyfit(ci, sl, 1)
        resid = sl - np.polyval(coef, ci)
        for i, row in enumerate(rows):
            row['shadow_resid'] = float(resid[i])
        print(f"    residual std = {resid.std():.4f} (written as shadow_resid)")
        # rewrite with the residual column included
        keys = list(rows[0].keys())
        with open(out_path, 'w') as f:
            f.write(','.join(keys) + '\n')
            for row in rows:
                f.write(','.join(f"{row[k]}" for k in keys) + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:7])
