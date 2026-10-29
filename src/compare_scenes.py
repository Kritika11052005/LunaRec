"""
Do two OHRC scenes agree about the same ground?

The hazard label is an OBSERVATION, not a property of the terrain -- it is what
one camera saw under one sun. Before the label can carry an experiment, we need
to know how much of it is the ground and how much is the acquisition.

Cluster B gave the answer for matched illumination (r ~ 0.88) and for opposite
illumination (r ~ 0.01). This repeats that test in cluster C, which is the
held-out block, so it is a genuine replication rather than a second look at the
scenes the method was developed on.

INTEGRITY CONTROL, read this before believing any shadow number:
  Both scenes sample slope and roughness from the SAME DEM at the SAME patch
  centres. On correctly joined patches those columns must correlate ~1.0. They
  are not evidence about the label -- they are evidence that the join is real.
  If slope does not come back ~1.0, the merge is wrong and the shadow
  correlation below it is meaningless, however good it looks.

Usage:
    python compare_scenes.py <patches_A.csv> <patches_B.csv> [grid_m]
"""

import sys

import numpy as np
import pandas as pd


def main(path_a, path_b, grid_m=200.0):
    grid_m = float(grid_m)
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)

    for name, d in ((path_a, a), (path_b, b)):
        print(f"{name}: {len(d)} patches, "
              f"x {d.patch_x.min():.0f}..{d.patch_x.max():.0f}  "
              f"y {d.patch_y.min():.0f}..{d.patch_y.max():.0f}")

    # Snap to the patch grid before joining. Centres from two scenes land on the
    # same lattice by construction, but float formatting through CSV can differ
    # in the last place, and an exact-equality join would silently return zero
    # rows -- which looks identical to "the scenes do not overlap".
    #
    # FLOOR, not round. Patch centres sit at cell centres, so patch_x / grid_m
    # is a half-integer (232500 / 200 = 1162.5). numpy's round is round-half-to-
    # EVEN, so 1163.5 and 1164.5 both snap to 1164 -- two adjacent patch columns
    # collapse onto one key and the merge goes many-to-many. That produced 6870
    # "co-located" rows from inputs of 1875 and 1930, and dragged the integrity
    # control down to r = 0.83. floor maps each centre into its own cell
    # whatever the fractional part.
    for d in (a, b):
        d['gx'] = np.floor(d.patch_x / grid_m).astype(np.int64)
        d['gy'] = np.floor(d.patch_y / grid_m).astype(np.int64)

    for name, d in ((path_a, a), (path_b, b)):
        dup = d.duplicated(['gx', 'gy']).sum()
        if dup:
            print(f"\n{name}: {dup} duplicate grid keys -- patch centres are "
                  f"not on a {grid_m:.0f} m lattice; the join below would be "
                  f"many-to-many. Stopping.")
            return

    m = a.merge(b, on=['gx', 'gy'], suffixes=('_a', '_b'))
    print(f"\nco-located patches: {len(m)}")
    # A join can never yield more rows than the smaller input. If it does, the
    # keys are not unique and every number below it is a cross-product artefact.
    assert len(m) <= min(len(a), len(b)), (
        f"merge produced {len(m)} rows from {len(a)} and {len(b)} -- keys are "
        f"not unique, results would be meaningless")

    if len(m) < 50:
        print("\nToo little overlap for a correlation. These scenes image "
              "different ground, so the matched-illumination test cannot run "
              "on this pair. Comparing distributions instead:\n")
        for col in ('shadow_local', 'slope_mean', 'roughness', 'dn_mean'):
            if col in a and col in b:
                print(f"  {col:<13} A mean {a[col].mean():8.4f}  "
                      f"B mean {b[col].mean():8.4f}  "
                      f"A p95 {np.nanpercentile(a[col], 95):8.4f}  "
                      f"B p95 {np.nanpercentile(b[col], 95):8.4f}")
        print("\nSimilar distributions are consistent with the label behaving "
              "the same way in both scenes, but they are NOT reproducibility:\n"
              "two unrelated patches of the same terrain type would also match.")
        return

    print("\nintegrity control -- same DEM, same centres, must be ~1.0")
    for col in ('slope_mean', 'roughness'):
        if f'{col}_a' in m:
            ok = m[[f'{col}_a', f'{col}_b']].dropna()
            r = np.corrcoef(ok[f'{col}_a'], ok[f'{col}_b'])[0, 1]
            flag = '' if r > 0.99 else '   <-- JOIN IS WRONG, stop here'
            print(f"  {col:<13} r = {r:+.4f}  (n={len(ok)}){flag}")

    print("\nthe actual test -- independent observations of the same ground")
    for col in ('shadow_local', 'shadow_frac', 'dn_mean'):
        if f'{col}_a' in m:
            ok = m[[f'{col}_a', f'{col}_b']].dropna()
            if len(ok) < 50:
                continue
            r = np.corrcoef(ok[f'{col}_a'], ok[f'{col}_b'])[0, 1]
            print(f"  {col:<13} r = {r:+.4f}  (r2 = {r*r:.3f}, n={len(ok)})")

    ok = m[['shadow_local_a', 'shadow_local_b']].dropna()
    r = np.corrcoef(ok.shadow_local_a, ok.shadow_local_b)[0, 1]
    print(f"\n  shared variance {100*r*r:.0f}% -- the part of the label that is "
          f"the ground")
    print(f"  remaining {100*(1-r*r):.0f}% is acquisition, noise, or real "
          f"change between the two passes")
    print("\n  Note: this is an upper bound on what any model could predict "
          "from terrain alone.\n  A model scoring above it is fitting the "
          "acquisition, not the Moon.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
