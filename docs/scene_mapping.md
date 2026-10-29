# Scene provenance — LunaRec

Which patch file came from which delivered product. Written because the
short-named extraction folders (`sC1`, `sC2`, `demC`) were duplicates and were
deleted; without this table the mapping from a scene label to a PDS product ID
exists nowhere on disk.

This is the table an examiner or reviewer asks for when they want to know what
data a result rests on.

## Scenes

| label | cluster | PDS product | patch file | status |
|---|---|---|---|---|
| `0858` | B | `ch2_ohr_ncp_20230823T0858341949_d_img_n18` | `patches_0858_rt.csv` | current |
| `1647a` | B | `ch2_ohr_ncp_20230823T1647285085_d_img_n18` | `patches_1647a_rt.csv` | current |
| `1647b` | B | `ch2_ohr_ncp_20230823T1647285085_d_img_n18` (second product of the same bundle — CONFIRM from the XML label) | `patches_20230823_1647b.csv` | superseded |
| `C1` | C | `ch2_ohr_ncp_20200827T0030107497_d_img_d18` | `patches_C1.csv` | current |
| `C2` | C | `ch2_ohr_ncp_20200827T0226453039_d_img_d18` | `patches_C2.csv` | current |

The `_rt` suffix means the file carries extra columns written by
`diagnostics/raytrace_shadow.py`. Those columns are not used by the current
evaluation; the files are current because of their OHRC columns, not those.

## Elevation products

| product | role | status |
|---|---|---|
| `ch2_tmc_ndn_20230130T1900132182_d_dtm_d32` | DTM strip, cluster B | retained (15 GB) |
| `ch2_tmc_ndn_20231101T0125121377_d_dtm_d18` | DTM strip, cluster C | retained (4.6 GB) |
| `data/interim/dem_clusterB_ch3site.tif` | clip used by every diagnostic | **keep** |
| `data/interim/dem_clusterC.tif` | clip used by every diagnostic | **keep** |

DEM-derived attributes are NOT used by the current method. The products are
kept only as the evidence behind the registration finding below; every
diagnostic ran against the clips, not the strips, so the strips are redundant
for reproducing anything done so far.

## Sun geometry

Needed by `build_patches.py`. Taken from each product's XML label.

| scene | sun azimuth | sun elevation |
|---|---|---|
| `C2` | 61.141200 | 9.561470 |
| `C1` | _fill from label_ | _fill from label_ |
| `0858` | _fill from label_ | _fill from label_ |
| `1647a` | _fill from label_ | _fill from label_ |

C1 and C2 were acquired within roughly an hour of each other on adjacent
orbits: their sun angles agree to 0.3° in azimuth and 0.6° in elevation, and
their raw brightness correlates at r = 0.987. They are a matched-illumination
PAIR, not independent samples. The same holds for 0858 and 1647a
(`dn_mean` r = 0.987).

So the dataset carries **two independent illumination conditions across two
geographic blocks**, not four independent scenes. Any claim phrased as
"replicated across four scenes" overstates it.

## Fitted GP priors, per scene and attribute

Measured by `src/fit_variogram.py`, not assumed. An earlier revision hardcoded
1100 m and a nugget of 0.10, which were wrong by roughly 3× and 5× against
every scene below.

| scene | hazard (range / nugget) | illumination | texture |
|---|---|---|---|
| `0858` | 2535 m / 0.436 | 1117 m / 0.000 | 1850 m / 0.477 |
| `1647a` | 2750 m / 0.523 | 1221 m / 0.081 | 2523 m / 0.671 |
| `C1` | 3232 m / 0.451 | 2029 m / 0.122 | 2648 m / 0.267 |
| `C2` | 3525 m / 0.446 | 1984 m / 0.079 | 2506 m / 0.313 |

Variogram fit R² runs 0.75–0.99 across all twelve fits.

The hazard nugget of ~0.45 is a result in its own right, not a nuisance term:
read against the scene-to-scene repeatability of 0.88 (cluster B), it says
roughly half the hazard variance is real structure finer than the 200 m patch
grid rather than noise. Hazard cannot be extrapolated from neighbouring
patches, which is why high-resolution verification imaging is needed at all.

## Why DEM attributes are excluded

Every DEM-derived quantity is uncorrelated with every image-derived quantity on
all four scenes (|r| < 0.14), while two OHRC-derived columns correlate with each
other exactly as expected (`dn_std` ~ `shadow_local` = +0.26). The DEM describes
different ground than the labels.

The offset was measured: both clusters show strong coupling (7–9× a far-field
control at 12–25 km) at a displacement pointing along **grid north** at their own
longitude — 4.0° off in cluster C, 0.6° in cluster B — of 4808 m and 4940 m. In
a polar projection grid north is radially outward, and for a polar-orbiting
spacecraft that is the along-track direction. The magnitudes differ per scene,
so it is per-scene along-track georeferencing error, not one product-level
constant, and no single translation or scale factor corrects both.

All attributes are therefore drawn from the OHRC imagery itself, co-registered
by construction. The scripts behind this finding are in `diagnostics/`.
