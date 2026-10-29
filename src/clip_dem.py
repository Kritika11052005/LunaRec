"""
Clip the TMC-2 DEM strip to the OHRC cluster footprints.

The delivered DEM is a full orbit-length strip (~15 GB, 68365 x 115812 at 10 m).
We need only the few hundred km2 under each OHRC cluster. Clipping gives ~250 MB
of useful raster instead, which is what goes to Kaggle -- derived data, not a
copy of the source product.

Also reports the valid-data fraction, which tells us whether the WFS footprint
polygon (86% for cluster B) or the bounding box (100%) is the honest number.

Usage:
    python clip_dem.py <path to ...d_dtm_d32.tif> [outdir]
"""

import os
import sys

import numpy as np
import rasterio
from rasterio.windows import from_bounds

# Cluster bounds in lunar polar stereographic metres (same CRS as the DEM),
# padded by 2 km so patch windows near the edge still have context.
PAD = 2000.0
CLUSTERS = {
    # name: (xmin, ymin, xmax, ymax)
    'clusterB_ch3site': (328131, 521277, 394709, 635242),
    'clusterC':         (231965, 609090, 270318, 733745),
}


def main(tif_path, outdir='.'):
    os.makedirs(outdir, exist_ok=True)

    with rasterio.open(tif_path) as src:
        print(f"source: {src.width} x {src.height}, {src.dtypes[0]}, nodata={src.nodata}")
        print(f"crs:    {src.crs.to_string()[:60]}...")
        b = src.bounds
        print(f"bounds: x {b.left:.0f}..{b.right:.0f}  y {b.bottom:.0f}..{b.top:.0f}\n")

        for name, (xmin, ymin, xmax, ymax) in CLUSTERS.items():
            win = from_bounds(xmin - PAD, ymin - PAD, xmax + PAD, ymax + PAD,
                              transform=src.transform)
            try:
                # a DEM strip covers one cluster, not both -- skip the other
                win = win.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height))
            except rasterio.errors.WindowError:
                print(f"{name}: outside this DEM strip, skipping\n")
                continue

            data = src.read(1, window=win)
            transform = src.window_transform(win)

            nodata = src.nodata
            valid = data != nodata if nodata is not None else np.ones_like(data, bool)
            frac = 100.0 * valid.sum() / valid.size

            out = os.path.join(outdir, f"dem_{name}.tif")
            profile = src.profile.copy()
            profile.update(height=data.shape[0], width=data.shape[1],
                           transform=transform, compress='deflate', tiled=True,
                           blockxsize=512, blockysize=512)
            with rasterio.open(out, 'w', **profile) as dst:
                dst.write(data, 1)

            mb = os.path.getsize(out) / 1e6
            if valid.any():
                vals = data[valid]
                rng = f"elev {vals.min()} .. {vals.max()}"
            else:
                rng = "no valid pixels"
            print(f"{name}")
            print(f"  window   {data.shape[1]} x {data.shape[0]} px")
            print(f"  valid    {frac:5.1f}% of window ({rng})")
            print(f"  written  {out}  ({mb:.1f} MB)\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '.')
