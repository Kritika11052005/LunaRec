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
significant, all surviving Holm correction for multiple testing. These sixteen
are the confirmatory family and the correction covers exactly them; the
shortlist-length, prior-misspecification and anisotropy sweeps reported under
*Known limitations* are exploratory and are not inside it.

The ablation (`voi-value`) is identical to the criterion except that it is blind
to shortlist structure — same posterior variance, same spatial propagation, same
division by cost. The gap between them is what the shortlist framing buys.

**Technical effect.** To reach the best shortlist quality any baseline achieves,
the criterion needs **38 % ± 4.2 fewer acquisitions** of the high-resolution
instrument — a 95 % interval over 120 scene-repetitions, paired, each
repetition's target taken from the best baseline's own score on that same
repetition. The figure is a floor rather than a central estimate: 14 of the 120
repetitions never reach the target and are scored as zero saving rather than
dropped.

The per-scene row in the table above (26 %/29 %/44 %/37 %, mean 34 %) is the
same quantity computed on rep-averaged curves. The two differ because
threshold-crossing is a nonlinear function of a curve, so averaging the curves
before finding the crossing is not the same as averaging the crossings; the
per-repetition figure is the one carrying an interval and a paired design. Both
are given because quoting one alone invites the question of what the other
says. Either way it is a measured reduction in operations of a scarce
instrument rather than a change in a statistic.

**Where it applies.** The advantage holds above roughly half the best achievable
precision, and from about 75 acquisitions of 400 candidates — a threshold that
grows with the candidate set, roughly as its square root (see *Known
limitations*). Below it, simpler policies reach a low bar faster. This method
addresses campaign-scale screening of hundreds of candidate patches, not the
selection of a handful of sites.

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

Three further scripts produce the sensitivity results quoted below:
`src/scaling_n.py` (crossover against candidate-set size), `src/effect_vs_k.py`
(technical effect against shortlist length) and `src/targeted_aniso.py`
(per-attribute range perturbation). Their arguments differ from the pattern
above and are documented in each script's docstring.

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
  K = 5 down to 0.047 at K = 50 (14/20, t = 2.26; 20 repetitions). The
  acquisition saving falls the same way — 40 %, 39 %, 35 %, 19 %, 8 % at
  K = 5, 10, 20, 35, 50 — so the effect is real across the range a shortlist
  would plausibly take but shrinks fivefold across it. Every headline figure
  here is measured at K = 20.

- **Prior misspecification is tolerated from 0.25× to 3× the fitted range**, and
  across nugget values 0.00–0.60. The 3× point was measured rather than
  interpolated: margin 0.113 (t = 2.24, p = 0.037), still significant. At 4× it
  falls to 0.042 (11/20, t = 1.07) and is not. An earlier version of this file
  claimed the advantage "holds throughout" a 0.25×–4× sweep, which was wrong; a
  later one bracketed the edge at 2× without measuring between 2× and 4×.

- **The acquisition requirement grows with the candidate set.** The crossover —
  where the criterion becomes reliably best — scales as n^0.531, close to √n:
  75 acquisitions at n = 400, and at n = 1600 the fit from n = 400 and 800
  predicted 158 out-of-sample against 165 measured. The margin itself grows
  with n (0.317 → 0.388 → 0.429), so larger campaigns favour the method, but
  the budget before it pays off grows with them. Measured on C2; the other
  three scenes are assumed to behave similarly.

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
  differ by 1.7–4.6× at a 4 km lag window (median 2.68), reproducibly across
  independent acquisitions of the same ground: cluster B illumination is
  longest at 45° and shortest at 135° in both scenes, cluster C hazard longest
  at 45° in both.

  Setting each attribute's range to its own measured long axis retains
  86 %/143 %/81 %/55 % of the margin across the four scenes, mean **91 %** — so
  the cost is about 9 %. An earlier version of this file reported 45 %, which
  was the C2-only figure (0.153, 17/20, t = 3.93) generalised from one scene.

  The preferred direction is scale-dependent. At the 200 m grid spacing the
  strongest direction is 90° in all twelve scene–attribute combinations,
  measured against 0° at matched distance (median gap 0.121, max 0.211; the
  narrowest, 1647a texture, is 0.006 and is effectively a tie). The
  kilometre-scale fitted ranges peak elsewhere, at 45°. The two are not
  directly comparable, since the 45°/135° neighbours sit at 283 m against
  200 m on the axes, but a direction that dominates at the grid spacing while
  not being the longest-range direction is not something one spherical range
  per attribute can represent. Measured in `notebooks/03_spatial_structure.ipynb`
  §2's directional test, which uses fixed-distance neighbours and so is
  unaffected by the lag-window limit below.

  Whether part of the effect is a pushbroom sampling artefact cannot be
  determined from four scenes whose along-track directions span only 11°. A
  scene-difference test rules out an acquisition artefact in the registration
  itself (R² < 0.006 in both directions).

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