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
repetitions, priors fitted per scene.

| | cluster C2 | cluster B 0858 |
|---|---|---|
| Precision@K, criterion | 0.770 | 0.740 |
| Precision@K, best baseline | 0.607 | 0.557 |
| vs ablation (paired) | +0.337, 14/15, p=9e-05 | +0.177, 14/15, p=6e-06 |
| shortlist disagreement | +18.2% | +13.9% |

The ablation (`voi-value`) is identical to the criterion except that it is blind
to shortlist structure — same spread, same spatial propagation. The gap between
them is what the shortlist framing buys.

Sensitivity was surveyed on three axes — stakeholder weight spread (0.01–0.25),
shortlist length (K = 5–50), and prior misspecification (0.25×–4× the fitted
range). The advantage holds throughout.

## Layout

```
data/raw/          delivered PDS products (gitignored, ~16 GB)
data/interim/      DEM clips, footprints, service responses
data/patches/      per-patch features and labels
src/               the pipeline
diagnostics/       the DEM registration investigation
results/           evaluation output
notebooks/         analysis notebooks
docs/              provenance and the invention log
```

## Running it

```
python src/build_patches.py <control_grid.csv> <image.img> <dem.tif> \
                            <out.csv> <sun_az> <sun_el>
python src/fit_variogram.py data/patches/current/*.csv
python src/evaluate.py data/patches/current/patches_C2.csv 30 150
python src/sensitivity.py data/patches/current/patches_C2.csv 20 150
```

`evaluate.py` imports `fit_variogram.py`; keep them in the same directory. The
GP prior is fitted per scene at every run — nothing is hardcoded.

## Known limitations

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
  real mission planners exists. The sensitivity sweep shows the result does not
  depend on the choice, but the weights remain scenarios.
- **The criterion needs budget before it helps.** With all three attributes
  unknown it trails simpler policies below roughly 10–50 acquisitions depending
  on the scene, then pulls away.
