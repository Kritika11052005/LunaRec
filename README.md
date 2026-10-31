# LunaRec

**PRIVATE — do not make this repository public.**

India provides no grace period for patent novelty. Any public disclosure before
a provisional application is filed destroys novelty permanently, and making the
repository private again afterwards does not undo it. That applies to this
repository, to Kaggle notebooks, to posted slides, and to project showcases.

---

Verification-allocation for lunar landing-site selection, framed as
multi-stakeholder recommendation under acquisition.

A set of candidate surface patches must be narrowed to a shortlist. Several
stakeholders weigh the attributes differently and nobody knows the weights
exactly. The attributes themselves are unknown until a patch is imaged at high
resolution, and imaging is scarce. **Which patch should be imaged next?**

The contribution is the acquisition criterion: score each candidate by the
shortlist *disagreement* its measurement would resolve — the entropy of its
top-k membership pooled over both posterior draws and stakeholder weightings —
weighted by how much observing it also resolves correlated neighbours, divided
by what the measurement costs.

`src/criterion.py` states this independently of any model. It consumes posterior
samples and knows nothing about Gaussian processes, the Moon, or any sensor.

## Result

Four Chandrayaan-2 OHRC scenes across two geographic blocks at 60–70°S, 30
repetitions, priors fitted per scene, all policies sharing each repetition's
subsample so every test is paired.

| | 0858 | 1647a | C1 | C2 |
|---|---|---|---|---|
| Precision@K, criterion | 0.692 | 0.663 | 0.805 | 0.790 |
| Precision@K, best baseline | 0.567 | 0.517 | 0.610 | 0.613 |
| vs ablation, paired | +0.163 | +0.147 | +0.307 | +0.272 |
| | 27/30, p=1e-07 | 24/30, p=7e-06 | 29/30, p=7e-11 | 29/30, p=2e-12 |
| shortlist disagreement | +11.4 % | +13.2 % | +20.5 % | +18.0 % |
| acquisitions saved | 26 % | 29 % | 44 % | 37 % |

Sixteen comparisons — four baselines × four scenes — all positive, all
significant, all surviving Holm correction for multiple testing.

The ablation (`voi-value`) is identical to the criterion except that it is blind
to shortlist structure — same posterior variance, same spatial propagation, same
division by cost. The gap between them is what the shortlist framing buys.

**Technical effect.** To reach the best shortlist quality any baseline achieves,
the criterion needs **34 % fewer acquisitions** of the high-resolution
instrument (26–44 % across scenes). That is a measured reduction in operations
of a scarce resource rather than a change in a statistic.

**Where it applies.** The advantage holds above roughly half the best achievable
precision, and from about 75 acquisitions of 400 candidates. Below that, simpler
policies reach a low bar faster. This method addresses campaign-scale screening
of hundreds of candidate patches, not the selection of a handful of sites.

## Layout

```
data/raw/          delivered PDS products (gitignored, ~16 GB)
data/interim/      DEM clips, footprints, service responses
data/patches/      per-patch features and labels
src/               the pipeline
diagnostics/       the DEM registration investigation
results/           evaluation output
notebooks/         analysis notebooks
docs/              provenance, remaining tasks, the invention log
```

## Running it

```
python src/build_patches.py <control_grid.csv> <image.img> <dem.tif> \
                            <out.csv> <sun_az> <sun_el>
python src/fit_variogram.py    data/patches/current/*.csv
python src/evaluate.py         data/patches/current/patches_C2.csv 30 150
python src/sensitivity.py      data/patches/current/patches_C2.csv 20 150
python src/negative_control.py data/patches/current/patches_C2.csv 20 150 3
```

`evaluate.py` imports `fit_variogram.py`; keep them in the same directory. The
GP prior is fitted per scene at every run — nothing is hardcoded.

Each run writes structured output beside its input, and that is what the
notebooks read: `*_precision_curve.csv` (mean curve per policy), `*_per_rep.csv`
(one row per policy per repetition), `*_per_rep_curve.csv` (the full curve per
repetition) and `*_sensitivity.csv`. Every row records the priors in force, so
a result file can never be ambiguous about which model produced it.

## Known limitations

- **The criterion needs budget, and a high quality bar, before it helps.** It is
  reliably the best of the five policies only from about **75 acquisitions** of
  400 candidates, and the acquisition saving turns positive only above roughly
  half the best achievable precision — 12 % of it on 0858 but 75 % on C2. Below
  those thresholds `max-variance` and `coverage` reach a low bar faster. The
  mechanism: early on the posterior is close to the prior, every patch has
  inclusion probability near K/n, boundary uncertainty is flat across
  candidates, and the criterion degenerates toward its own ablation.

- **Shortlist length is the one real sensitivity.** The margin runs 0.440 at
  K = 5 down to 0.047 at K = 50 (14/20, t = 2.26). Every headline figure here is
  measured at K = 20.

- **Prior misspecification is tolerated from 0.25× to 2× the fitted range**, and
  across nugget values 0.00–0.60. At 4× the margin falls to 0.042 (11/20,
  t = 1.07) and is no longer significant. An earlier version of this file
  claimed the advantage "holds throughout" a 0.25×–4× sweep; that was wrong.

- **The hazard label is illumination-conditional.** `shadow_local` measures
  shadow fraction against a local background, and shadows are cast by the sun.
  Scenes imaged at a ~80° solar azimuth difference show the label decorrelating
  (r = 0.008) alongside raw brightness (r = −0.031); only scenes under matched
  illumination agree (r = 0.88–0.955). So the quantity is *shadow hazard under a
  given illumination*, not intrinsic boulder density, and a shortlist is valid
  only for the illumination it was measured under. A different landing window
  would require new imagery.

  This strengthens rather than weakens the case for acquisition: an
  illumination-conditional quantity cannot be supplied by a static elevation
  model, so the imaging cannot be replaced by computation.

- **The spatial model is partially identified.** Three estimators of the same
  parameters give hazard ranges spanning 805–3525 m: weighted-least-squares
  variogram under the superseded weighting (2535–3525 m), Cressie-weighted
  variogram (1611–2782 m, the operating point) and Gaussian marginal likelihood
  (805–1872 m). Reported against the operating point the disagreement is
  0.49×–1.74× on range and ≤ 0.354 on nugget, which sits inside the tolerated
  band above — so the result is stable across more than the parameters are
  uncertain by. A single fitted range should nevertheless not be reported as a
  measurement. The likelihood estimator's bias is itself characterised: it
  tracks attribute skew at r = −0.888, because a Gaussian likelihood cannot
  place mass in a long right tail and assigns it to signal instead.

- **The field is anisotropic and the GP is isotropic.** Directional ranges
  differ by 1.7–4.6×, reproducibly across independent acquisitions of the same
  ground, and the preferred direction depends on the lag — short-lag structure
  peaks at 90° in all twelve scene–attribute combinations while the kilometre-
  scale ranges peak elsewhere. Setting each attribute's range to its own
  measured long axis costs 45 % of the margin while leaving it significant
  (0.153, 17/20, t = 3.93). Whether part of the effect is a pushbroom sampling
  artefact cannot be determined from four scenes whose along-track directions
  span only 11°. A scene-difference test rules out an acquisition artefact in
  the registration itself (R² < 0.006 in both directions).

  Variogram windows are capped at 4 km: a lag window must not exceed about half
  the smallest scene dimension, and these scenes are 11.2–17.2 km across. Where
  a directional fit clamps at that width, the range exceeds what a scene this
  size can resolve and is reported as a lower bound.

- **DEM attributes are excluded.** TMC-2 DTM products are not registered to the
  OHRC grid: every DEM-derived quantity is uncorrelated with every image-derived
  quantity (|r| < 0.14) across all four scenes. The offset was measured at
  4.8–4.9 km along grid north, consistent with per-scene along-track
  georeferencing error. All attributes are drawn from OHRC itself. See
  `docs/scene_mapping.md` and `diagnostics/`.

- **Two illumination conditions, not four.** C1/C2 and 0858/1647a are
  matched-illumination pairs acquired roughly an hour apart. The dataset carries
  two independent conditions across two geographic blocks.

- **Stakeholder weights are modelled, not elicited.** No preference data from
  real mission planners exists. The margin is flat across a 25× span of weight
  spread (0.257–0.320, t = 6.6–11.1), so the result does not depend on the
  choice, but the weights remain scenarios.

## Where the numbers come from

`docs/remaining_tasks.md` lists what is measured, what is assumed and what is
still open. `notebooks/03_spatial_structure.ipynb` records the hypotheses tested
during validation — including the ones that failed and the code defects they
exposed.