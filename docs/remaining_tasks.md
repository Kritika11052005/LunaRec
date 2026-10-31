# LunaRec — state of play and remaining work

Last updated 2026-10-31, 15:20 IST. Written so that a session starting cold
needs only this file and the repository.

---

## Status

| | state |
|---|---|
| notebook 01 — data audit | complete |
| notebook 02 — label validation | complete |
| notebook 03 — spatial structure | **complete** |
| notebook 04 — criterion evaluation | complete except one confidence interval |
| notebook 05 — sensitivity | not started; input does not exist yet |

`evaluate.py` has been run on all four scenes at 30 reps / budget 150 under
corrected (Cressie) priors. Those results are current and are what the README
and both conclusions quote.

---

## First commands in a new session

```powershell
cd "$env:USERPROFILE\OneDrive\Documents\Post Placement Projects\LunaRec"

Get-ChildItem results\evaluation\*.txt                   | Select Name,Length,LastWriteTime
Get-ChildItem data\patches\current\*_per_rep.csv         | Select Name,Length
Get-ChildItem data\patches\current\*_per_rep_curve.csv   | Select Name,Length
Get-ChildItem data\patches\current\*_sensitivity.csv     | Select Name,Length
```

Four `_per_rep.csv` (150 rows each) exist. The `_per_rep_curve.csv` and
`_sensitivity.csv` files appear only after the overnight queue below has run.

---

## Overnight queue

Add the per-rep-curve block to `evaluate.py` first (it is in the version
delivered 2026-10-31), set the machine not to sleep, then:

```powershell
$ErrorActionPreference="Continue"
foreach ($f in "patches_0858_rt","patches_1647a_rt","patches_C1","patches_C2") {
  python -u src\evaluate.py data\patches\current\$f.csv 30 150 > results\evaluation\$f.txt 2>&1
}
python -u src\sensitivity.py data\patches\current\patches_C2.csv 20 150 > results\evaluation\sensitivity_C2.txt 2>&1
```

About 6 hours: ~70 min for the four scenes, ~4.8 h for the 24-setting sweep.

What it buys: per-rep curves, so the 34 % technical effect gets a confidence
interval instead of being the one claim without one; and a sensitivity CSV, so
notebook 05 reads a file from the repository rather than numbers transcribed
from a chat. The sweep also adds a 3× range setting, which locates the
tolerance edge currently bracketed between 2× (significant) and 4× (not).

A useful side effect: the four scenes re-run with identical seeds and priors,
so the margins should come back bit-identical. If they don't, the pipeline is
not deterministic and that needs finding.

---

## Remaining tests

### Notebook 04

| # | test | cost | buys |
|---|---|---|---|
| 8 | negative control on shuffled data | ~50 min | proof the margin is not an artefact of the evaluation harness — the first thing a sceptic asks |
| 6 | per-rep curves → CI on the 34 % | overnight | an error bar on the patent's central number |
| 10 | technical effect vs K | ~40 min | K is the one real sensitivity (0.440 at K=5 → 0.047 at K=50) and the 34 % is measured at K=20 only |
| 7 | n-scaling (n = 800, 1600) | ~40 min | whether the crossover is a fixed count (75) or a fixed fraction (19 %). The claim is now campaign-scale, which *is* a claim about larger n |

Test 9 (is the 34 % an artefact of the target?) is **done**. It failed: the
saving is strongly target-dependent, negative below a scene-specific threshold,
and the conclusion was rewritten around that.

### New

**Cross-scene generalisation.** Fit priors on one scene, apply them unchanged
to another, and measure whether the criterion still beats the ablation. Right
now every evaluation fits priors on the scene it is evaluated on, so there is
no held-out test in the "train here, predict there" sense. A reviewer or
examiner will ask. `targeted_aniso.py` already overrides `E.PRIOR` and measures
the margin, so this is a small variant of it. ~40 min per pair; cluster B →
cluster C is the harder and more interesting direction, since the ranges differ
by about 1.5×.

This is also the clearest answer to "where is the training?" — the GP is
trained by marginal-likelihood maximisation offline and Bayesian conditioning
online, and a transfer result makes that visible.

### Notebook 05

Four cells, reading `*_sensitivity.csv`. Present the four axes, the tolerance
boundaries, and the negative control as a validation section.

Sweep results to present (from the completed run of 2026-10-30, to be replaced
by the CSV):

| axis | result |
|---|---|
| `W_SPREAD` 0.01–0.25 | flat 0.257–0.320 across a 25× span, t = 6.6–11.1 — no dependence |
| `K` 5–50 | 0.440 (K=5) → 0.047 (K=50, 14/20, t = 2.26) — **the one real sensitivity** |
| range × | 0.240 / 0.217 / 0.278 / 0.170 / **0.042 at 4× (11/20, t = 1.07)** |
| nugget 0.00–0.60 | 0.212 → 0.455, all positive and significant, increasing with nugget |

Tolerance: **0.25×–2.00×** on range, **0.00–0.60** on nugget. The nugget span
fully covers the disagreement between the three estimators.

---

## Documents

| file | state |
|---|---|
| `README.md` | **rewritten 2026-10-31** — result table, three corrected claims, new limitations |
| `docs/scene_mapping.md` | **needs rewriting** — see below |
| `src/fit_gp.py` docstring | **needs the Gaussian limitation** — numbers in notebook 03 §5 |
| `docs/remaining_tasks.md` | this file |

### `docs/scene_mapping.md`

- the prior table carries the superseded (`1/√N(h)`-weighted) values; replace
  with the Cressie column and annotate the likelihood column as the
  identification range
- the claim that a ~0.45 nugget means half the hazard variance is sub-grid
  structure is **withdrawn**; replace with the neighbour-versus-repeat
  measurement from notebook 03 §2
- three sun-geometry rows still read `_fill from label_` (`C1`, `0858`,
  `1647a`)
- the DEM section needs **no change** — those figures are correlations and a
  displacement scan, untouched by the weighting fix

### Claims corrected, do not reintroduce

- **"Robust to 4× range misspecification" is false.** Tolerance is 0.25×–2×.
- **"The label tracks the ground, not the acquisition" is backwards.** The
  hazard label is illumination-conditional; the recorded r = 0.008 across
  illumination conditions is a negative control for the method, not evidence of
  terrain-invariance.
- **"The criterion trails below 10–50 acquisitions"** — it is reliably best
  only from 75.
- **The Chandrayaan-3 framing** (20 candidate sites, 8 with stereo) motivated a
  regime in which the method does not win. The claim is campaign-scale
  screening.

---

## Outstanding, unrelated to the notebooks

- **Ask Kritika** for sun azimuth/elevation from the XML labels for `0858`,
  `1647a` and `C1`, and for `pip freeze` from the environment that produced the
  original recorded results.
- **`gate.py` is missing** — the 312-product inventory has no code behind it.
- **DEM rasters are gitignored**, so the 4.8–4.9 km registration offset is
  cited in the docs but not reproducible from the repository alone.

---

## Standing constraints

- **The repository stays PRIVATE until the provisional patent is filed.** India
  has no grace period; public disclosure destroys novelty permanently and making
  it private again does not undo it. Applies to the repository, Kaggle
  notebooks, posted slides and project showcases.
- Do not commit `data/raw` (~16 GB) or `.tif` files.
- **Edit pipeline files in place.** A parallel `fit_variogram2.py` once let the
  notebook and the pipeline diverge silently, and cost an evening.
- Variogram lag windows are capped at 4 km — half the smallest scene dimension.
  Wider windows are invalid for these scene sizes, not more informative.