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

### Patch grid and scene extent

All four scenes are sampled on a **200 m grid, square to four decimal places**
(x and y spacing both 200.0 m, ratio 1.0000 on every scene). This was verified
rather than assumed: an unequal grid would make a neighbour in one direction
closer than in another, and a directional analysis would then report geometry
as terrain.

| scene | extent (km) | patches | max valid variogram lag |
|---|---|---|---|
| `0858` | 17.2 × 22.4 | 2010 | 8.6 km |
| `1647a` | 16.0 × 23.2 | 2008 | 8.0 km |
| `C1` | 11.2 × 25.2 | 1875 | 5.6 km |
| `C2` | 11.2 × 25.6 | 1930 | 5.6 km |

The last column matters for anything fitted from a variogram: a lag window must
not exceed roughly **half the smallest scene dimension**, beyond which there are
too few pairs, edge effects dominate, and the fit chases large-scale trend
instead of local structure. **Every fitted value in this document uses a 4 km
window**, valid for all four scenes. Refits at 6 km and 9 km were run and
discarded — 6 km exceeds cluster C's limit and 9 km exceeds all four, and their
isotropic ranges duly ran away to 4.7–7.9 km against 1.5–2.8 km at 4 km.

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

Measured by `src/fit_variogram.py` at a 4 km lag window, not assumed.

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

The evidence that this was a repair rather than a different arbitrary choice:
agreement with the independent marginal-likelihood estimator improved, nugget
difference median −0.186 → −0.085, while the range ratio barely moved
(0.73 → 0.77). A change that merely perturbed the fit would move both at
random.

### The identification range

The table above is the **operating point**, not a measurement. A second,
independent estimator — Gaussian marginal likelihood, `src/fit_gp.py` — gives
systematically different values on the same data:

| estimator | hazard range | hazard nugget |
|---|---|---|
| variogram, `1/√N(h)` weights | 2535 – 3525 m | 0.436 – 0.523 |
| variogram, Cressie weights | **1611 – 2782 m** | **0.235 – 0.332** |
| Gaussian marginal likelihood | 805 – 1872 m | 0.000 – 0.022 |

Against the operating point: range ratio 0.49–1.74× (median 0.77), nugget
difference median −0.085, maximum 0.354. Reporting any single fitted range as
a measured quantity would overstate what 200 m-spaced data support.

The sensitivity sweep tolerates range misspecification from 0.25× to 2.00× and
nugget values from 0.00 to 0.60, which covers the whole disagreement above — so
the result is stable across more than the parameters are uncertain by.

### Anisotropy: the fits are isotropic, the field is not

Directional variograms at 4 km give longest-to-shortest range ratios of
**1.70–4.57** across the twelve scene–attribute combinations, median 2.68.
The direction reproduces across independent acquisitions of the same ground —
cluster B illumination is longest at 45° and shortest at 135° in both scenes
(2.21 and 2.40), cluster C hazard is longest at 45° and shortest at 0° in both
(3.03 and 3.46), and cluster C's shortest axis is 0° in five of six fits.

Against the isotropic fits in the table above, the long axis is 1.54× the
fitted range at the median and 2.36× at the extreme; the short axis is 0.70×
and 0.33×. Two of twelve fall outside the swept band, both on 1647a and both
clamped lower bounds.

Setting each attribute's range to its own measured long axis costs 45 % of the
criterion's margin while leaving it significant (0.153, 17/20, t = 3.93 on C2);
the short axis costs 17 % (0.232, 18/20, t = 6.82). A geometric-anisotropy term
in the covariance is the correct fix and is recorded as future work.

Whether part of this is a pushbroom sampling artefact rather than terrain
**cannot be determined from these scenes**: cluster B's along-track direction
is 57.7° and cluster C's is 69.0°, only 11° apart, so "aligned with
along-track" and "aligned with roughly 60° for any reason" make identical
predictions. Scenes at widely separated longitudes would settle it.

### On the nugget, and what replaced it

An earlier version of this document read the ~0.45 hazard nugget as evidence
that half the label's variance is real structure finer than the 200 m grid, and
used it to argue that hazard cannot be extrapolated from neighbouring patches.
**That claim is withdrawn.** The nugget is variance at *zero* separation, the
patch grid is 200 m, and no estimator can observe below it — the quantity is
extrapolation, not measurement. Three estimators on the same data gave 0.44–0.52
(old weighting), 0.24–0.33 (Cressie) and ~0.00 (marginal likelihood).

The likelihood's zero is an artefact: `fit_gp.py` assumes Gaussianity, and the
disagreement tracks skewness exactly — skew against the nugget gap correlates at
**r = −0.888**, against only −0.306 for kurtosis. Every symmetric attribute
(|skew| < 0.3) shows the two methods agreeing to within 0.08; every right-skewed
one shows the likelihood collapsing the nugget to zero, and all six zeros are
skewed hazard or texture attributes. The Cressie variogram is distribution-free
and is the defensible estimator here.

The argument the nugget was carrying is now made by direct measurement instead:

| | hazard |
|---|---|
| nearest neighbour, 200 m away, predicts a patch at | r = 0.686–0.768 |
| the same patch imaged again predicts it at | r = 0.884–0.955 |

In variance terms, imaging a patch explains **78–91%** of its hazard while its
neighbour explains **47–59%** — roughly 30 percentage points that only an
acquisition can buy. The gain is positive in all twelve scene-attribute
combinations. No model, no extrapolation, no estimator dependence.

It also survives the anisotropy above. Measured per direction at matched
distance — 0° and 90° at 200 m, 45° and 135° at 283 m, since the grid places
diagonal neighbours further away and comparing across that would confound
direction with distance — imaging still beats the **best-direction** neighbour
in all twelve combinations, by +0.073 to +0.297, mean +0.142.

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

### The OHRC-to-OHRC join, by contrast, is sound

The same question asked of the two scenes within each cluster — the join the
label validation depends on — comes back clean. Regressing the per-patch
attribute difference on along-track and cross-track position gives a maximum
R² of **0.0215** across twelve scene–attribute–direction combinations, and the
gradient signs are *incoherent* between attributes (cluster B along-track
−,+,−; cluster C cross-track +,−,+). Three attributes cut from the same pixels
cannot disagree about which way the image moved, so this is not a shift. For
scale, the largest effect is 2.2 % of the difference variance against 91.2 %
explained by re-imaging the same patch.

## Open

- **Sun azimuth and elevation for `C1`, `0858` and `1647a`** — three of four
  rows above are still unfilled. From each product's XML label.
- **Confirm the `1647b` product ID** from the XML label; it is currently
  recorded as the second product of the `1647a` bundle, unverified. The file is
  superseded, so this blocks nothing.
- **DEM rasters are gitignored**, so the 4808 m / 4940 m offsets are cited here
  but not reproducible from the repository alone.