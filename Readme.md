# Amazon–Cerrado Vegetation Resilience Synchrony

**IRIS National Fair 2026–27**
**Team:** Deepanshu Gupta (analysis/coding) · Ananya Dixit (research/writing)

## Research Question

Does the resilience state of one forest patch relate to the future resilience state of a distant patch, after controlling for the environmental conditions affecting both — and if so, at what distance, over what time lag, and under what conditions?

## Headline Finding

There is a statistically robust, negative association between a source patch's current resilience state and a target patch's future resilience state, specifically at **800–1,100 km separation**. The most rigorously tested estimate — a genuinely pairwise model with two-way clustered inference — gives **β ≈ −0.00723 (p = 0.0008)**. This survived dozens of alternative specifications (distance cutoffs, resilience metrics, temporal controls, inference methods) and an adversarial fake-neighbor permutation test targeted at the pairwise result. The physical transmission mechanism, however, was **not** identified: a surface-wind-alignment plausibility check gave internally inconsistent results, and a pre-registered moisture-availability proxy chain found no support for either of its two hypothesized links. This is reported as an open question for future work, not resolved by this project.

See [`docs/full_methodology.md`](docs/full_methodology.md) for the complete stage-by-stage narrative (why each step was taken, what was found, what came next) and [`docs/references.md`](docs/references.md) for the literature that motivated the methodology.

## Study Region & Data

- **Region:** Amazon–Cerrado transition zone, 4°S–18°S / 46°W–66°W (~2,500 km extent)
- **Patches:** 352 patches, each an aggregated ~100km × 100km block (4×4 VODCA pixels), 2003–2018 monthly
- **Data sources:** VODCA v2 (vegetation), CHIRPS (precipitation), ERA5 (temperature, dewpoint, wind u/v), ERA5-Land (solar radiation, root-zone soil moisture), TerraClimate (soil moisture, PDSI), NOAA ONI (ENSO), Hansen Global Forest Change (disturbance distance), MODIS (land-surface temperature, cloud fraction), SRTM (elevation → topographic wetness index)

## Resilience Metric

Vegetation Optical Depth (VOD) is deseasonalized into monthly anomalies, then summarized via **rolling AR(1)** (first-order autocorrelation, 24-month window, stepped monthly) — consistent with critical-slowing-down theory (Scheffer et al. 2009): increasing temporal persistence is a warning signal associated with reduced resilience, not a direct physical measurement of it. This time-varying metric (not a single static value) is the outcome variable for the project's core findings (Stages 41 onward). Sensitivity-checked against 36-month and 48-month windows (Stage 43).

## Methodology Summary

The project proceeds in four broad phases — full detail in `docs/full_methodology.md`:

1. **Establishing the basic spatial relationship** (Stages 1–13) — a real, robust neighbor-vegetation association (coef ≈ 0.087, p<0.0001) that, puzzlingly, did not decay with distance.
2. **Testing environmental explanations** (Stages 14–33) — systematically controlled for precipitation, temperature, soil moisture, drought (PDSI), ENSO, VPD, wind, solar radiation, root-zone soil moisture, canopy-air temperature difference, terrain wetness, and disturbance distance. Combined, these explained 41.4% of the original coefficient — VPD alone was the single largest driver (~25%).
3. **Testing predictive usefulness** (Stages 21–27, 34–38) — both large-sample linear models and a 4-architecture Graph Neural Network (with a real feature-normalization bug caught and fixed mid-investigation) found spatial information does **not** meaningfully improve vegetation prediction at any tested horizon.
4. **Reframing to resilience-to-resilience and stress-testing** (Stages 39–50) — the central methodological correction of the project: testing whether a neighbor's resilience *state* (not raw vegetation) predicts a target's own *future resilience*. This revealed a two-regime spatial pattern (near = fast/positive/symmetric, consistent with synchrony; far = slow/negative/asymmetric, more consistent with directional influence), the project's most consistent finding (target patch water availability and disturbance-distance modulate susceptibility at every lag tested), and an extensive, honestly-reported robustness and mechanism-search arc.

## Key Findings

| Finding | Result |
|---|---|
| Patches showing significant resilience-loss trend | 44.9% of 352 |
| Spatial clustering (Moran's I) | 0.727, p=0.001 |
| Original neighbor-vegetation coefficient | ≈0.087, p<0.0001 |
| Environmental drivers explained (12-factor model) | 41.4% (VPD alone ≈25%) |
| GNN vs. non-spatial baseline (prediction) | Baseline wins at every horizon, every architecture |
| Near-distance (75–650km) resilience association | Positive, fast (1-month peak), synchrony-like |
| Far-distance (800–1,100km) resilience association | Negative, slow (6-month peak), directional-looking |
| Susceptibility (every lag, 1/2/3/6mo) | Target's soil moisture, RZSM, TWI, disturbance-distance all significant |
| Final pairwise far-distance estimate | β ≈ −0.00723, p=0.0008 (two-way clustered) |
| Physical mechanism (wind alignment, moisture proxy) | Not established — mixed/negative, reported as future work |

## What This Project Does NOT Claim

- Does **not** claim one patch physically causes changes in another — all results are statistical associations after controlling for measured confounders, in an observational design.
- Does **not** claim to have identified the physical transmission mechanism — explicitly left as an open question.
- Does **not** hide results that weakened the central finding along the way (e.g., a two-way-clustering check and a fake-neighbor permutation test that were more equivocal before a more targeted pairwise re-test found a stronger result) — these are reported transparently as part of the analytical arc.

## Repository Structure

```
src/                    Numbered analysis scripts (01–50), run in order
gee/                    Google Earth Engine export scripts for raw data
data/processed/         Aggregated patch-level datasets and intermediate results
data/raw/                Downloaded raster/CSV inputs (not all committed — see gee/ to regenerate)
figures/                 All generated plots
docs/full_methodology.md         Complete stage-by-stage narrative and reasoning
docs/references.md               Literature that motivated the methodology
```

## Known Limitations

- A geographic bounding-box calculation bug affected Stages 14–26 (fixed for Stage 28 onward); a spot-check comparison suggests conclusions were not meaningfully distorted, but exact figures from that range were not formally reconciled.
- The GNN results (Stages 34–38) required a mid-project fix for unnormalized input features; final results are on the corrected pipeline, with the pre-fix run retained separately for comparison.
- The pairwise model (Stage 48) uses a reproducible random subsample of 3,000 (of 26,860 eligible) source-target pairs for computational tractability, keeping each pair's full time series intact.
- The physical transmission mechanism (e.g., atmospheric moisture transport) was not established with the data and methods available to this project; this would require dedicated multi-level atmospheric data and specialized moisture-tracking modeling beyond this project's scope.

## Reproducing the Analysis

Scripts in `src/` are numbered in the order they were run; each has a module docstring explaining its purpose, inputs, outputs, and key methodological notes. Most later stages (39+) reuse cached intermediate outputs (e.g., `patch_rolling_ar1.csv`) when present rather than recomputing from raw rasters.