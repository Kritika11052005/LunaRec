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

Measured by `src/fit_variogram.py`, not assumed.

| scene | hazard (range / nugget) | illumination | texture |
|---|---|---|---|
| `0858` | 1819 m / 0.235 | 1143 m / 0.000 | 1504 m / 0.345 |
| `1647a` | 1611 m / 0.288 | 1148 m / 0.000 | 1660 m / 0.527 |
| `C1` | 2394 m / 0.297 | 1896 m / 0.054 | 2496 m / 0.225 |
| `C2` | 2782 m / 0.332 | 1902 m / 0.039 | 2347 m / 0.266 |

Variogram fit R² runs 0.742–0.984.

**These supersede an earlier table** listing ranges of 2535–3525 m and hazard
nuggets of 0.436–0.523. Those came from a weighting defect in
`fit_variogram.fit()`: `sigma = 1/sqrt(counts)` weights each bin by the square
root of its pair count, and pair counts *grow* with lag (3,675 pairs at 241 m
against 11,621 at 2,035 m), so the fit was dominated by long lags, missed its
own shortest-lag point by ~0.19, and inflated the nugget to compensate. Cressie
weights, `N(h)/γ(h)²`, upweight short lags where γ is small. Shortest-lag fit
error improved from ~0.19 to 0.035–0.095.

### On the nugget, and what replaced it

An earlier version of this document read the ~0.45 hazard nugget as evidence
that half the label's variance is real structure finer than the 200 m grid, and
used it to argue that hazard cannot be extrapolated from neighbouring patches.
**That claim is withdrawn.** The nugget is variance at *zero* separation, the
patch grid is 200 m, and no estimator can observe below it — the quantity is
extrapolation, not measurement. Three estimators on the same data gave 0.44–0.52
(old weighting), 0.24–0.33 (Cressie) and ~0.00 (marginal likelihood).

The likelihood's zero is an artefact: `fit_gp.py` assumes Gaussianity, and the
disagreement tracks skewness exactly — every symmetric attribute shows the two
methods agreeing to within 0.08, every right-skewed one (hazard skew 0.87–1.71)
shows the likelihood collapsing the nugget to zero. The Cressie variogram is
distribution-free and is the defensible estimator here.

The argument the nugget was carrying is now made by direct measurement instead:

| | hazard |
|---|---|
| nearest neighbour, 200 m away, predicts a patch at | r = 0.686–0.768 |
| the same patch imaged again predicts it at | r = 0.884–0.955 |

In variance terms, imaging a patch explains **78–91%** of its hazard while its
neighbour explains **47–59%** — roughly 30 percentage points that only an
acquisition can buy. The gain is positive in all twelve scene-attribute
combinations. No model, no extrapolation, no estimator dependence.

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
