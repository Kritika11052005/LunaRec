"""
How much DEM actually sits under the OHRC footprints?

The clipped window is a rectangle; the DEM strip and the OHRC scenes are both
diagonal within it, so "percent of window with data" is not the number we need.
This rasterises the OHRC footprints and reports, per scene, the fraction of its
ground that has valid DEM beneath it.

That fraction is the real trainable area: OHRC gives the hazard label, the DEM
gives the features, and only their intersection is usable.

Usage:
    python check_coverage.py dem_clusterB_ch3site.tif clusterB_footprints.geojson
"""

import json
import sys

import numpy as np
import rasterio
from rasterio.features import geometry_mask


def main(dem_path, geojson_path):
    feats = json.load(open(geojson_path))['features']

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        nodata = src.nodata
        valid = dem != nodata if nodata is not None else np.ones_like(dem, bool)
        px_km2 = abs(src.transform.a * src.transform.e) / 1e6

        print(f"DEM {src.width} x {src.height} @ {abs(src.transform.a):.0f} m\n")
        print(f"{'date':<12} {'scene':<42} {'ground':>9} {'with DEM':>9} {'frac':>6}")

        total_area = total_cov = 0.0
        per_scene = []
        for f in feats:
            mask = geometry_mask([f['geometry']], out_shape=dem.shape,
                                 transform=src.transform, invert=True)
            n_in = int(mask.sum())
            n_cov = int((mask & valid).sum())
            area = n_in * px_km2
            cov = n_cov * px_km2
            total_area += area
            total_cov += cov
            per_scene.append(cov)
            frac = 100 * cov / area if area else 0.0
            print(f"{f['properties']['OBS_ST_TIME'][:10]:<12} "
                  f"{f['properties']['PRODUCT_ID']:<42} "
                  f"{area:7.1f}km2 {cov:7.1f}km2 {frac:5.0f}%")

        # union, not sum -- scenes overlap heavily
        union = np.zeros(dem.shape, bool)
        for f in feats:
            union |= geometry_mask([f['geometry']], out_shape=dem.shape,
                                   transform=src.transform, invert=True)
        u_area = int(union.sum()) * px_km2
        u_cov = int((union & valid).sum()) * px_km2

        print(f"\nunion of all scenes : {u_area:.1f} km2")
        print(f"  of which has DEM  : {u_cov:.1f} km2 ({100*u_cov/u_area:.0f}%)")
        print(f"\ntrainable area = {u_cov:.1f} km2")
        for size in (1000, 500, 200, 100):
            n = u_cov * 1e6 / size**2
            print(f"  at {size:>4} m patches -> {n:9,.0f} patches")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
