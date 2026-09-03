# Amazon–Cerrado Vegetation Resilience Synchrony

**IRIS National Fair 2026–27**
**Team:** Deepanshu Gupta (analysis/coding) · Ananya Dixit (research/writing)

## Research Question

Does the resilience state of one forest patch relate to the future resilience state of a distant patch, after controlling for the environmental conditions affecting both — and if so, at what distance, over what time lag, and under what conditions?

## Headline Finding

There is a negative association between a source patch's current resilience state and a target patch's future resilience state at **800–1,100 km separation**, at a 3-month lag. How strong that claim is depends on how it is estimated, and the contrast between the two estimates is itself the finding:

- **Distance-band-averaged model** (each target's "exposure" = the mean resilience of all far-band patches): β ≈ −0.0185. The point estimate stays negative across roughly a dozen alternative specifications, **but it fails the two most demanding tests**: under two-way clustering by patch *and* month its p-value is **0.536**, and an adversarial fake-neighbor permutation test cannot distinguish it from random spatial pairings (permutation p ≈ 0.29).
- **Genuinely pairwise model** (every individual source→target pair kept separate instead of averaged into a band mean): **β ≈ −0.00723, p = 0.0008 under that same two-way clustering** — significant under all three clustering schemes tested, and similar across the three 100 km sub-bands of the 800–1,100 km range (−0.0081 / −0.0066 / −0.0070).

The pairwise estimate is the headline **because** it is the more granular and more conservative test and it holds where the band-averaged version does not — not because the result swept every check. The band-averaged failures are reported as part of the result, not a footnote.

The physical transmission mechanism was **not** identified: a surface-wind-alignment plausibility check gave internally inconsistent results, and a pre-registered moisture-availability proxy chain found no support for either of its two hypothesized links. This is reported as an open question for future work, not resolved by this project.

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
2. **Testing environmental explanations** (Stages 14–33) — systematically controlled for precipitation, temperature, soil moisture, drought (PDSI), ENSO, VPD, wind, solar radiation, root-zone soil moisture, canopy-air temperature difference, terrain wetness, and disturbance distance. Combined, these explained 41.4% of the original coefficient. VPD alone was the single largest driver, contributing ≈21.6 of those 41.4 percentage points (about half of everything explained); ENSO was second at ≈11 points; every other driver contributed ≤2 points, and TWI and disturbance distance contributed nothing.
3. **Testing predictive usefulness** (Stages 21–27, 34–38) — large-sample linear models found spatial information adds essentially nothing to prediction (best case ≈0.7% RMSE improvement). A 4-architecture Graph Neural Network comparison (after a real feature-normalization bug was caught and fixed) is more equivocal on the corrected pipeline: no architecture is reliably ahead — baseline, fixed-geographic GNN and attention GNN are within noise of each other at a 1-month horizon, the fixed-geographic GNN is ahead at 3 months, the baseline at 6 months. The defensible reading is that spatial structure yields **no consistent, exploitable prediction gain** — not that the non-spatial baseline dominates.
4. **Reframing to resilience-to-resilience and stress-testing** (Stages 39–50) — the central methodological correction of the project: testing whether a neighbor's resilience *state* (not raw vegetation) predicts a target's own *future resilience*. This revealed a two-regime spatial pattern (near = fast/positive/symmetric, consistent with synchrony; far = slow/negative/asymmetric, more consistent with directional influence), the project's most consistent finding (target patch soil moisture, terrain wetness and disturbance-distance modulate susceptibility at every lag tested; root-zone soil moisture at lags 1–3), and an extensive, honestly-reported robustness and mechanism-search arc.

## Key Findings

| Finding | Result |
|---|---|
| Patches showing significant resilience-loss trend | 44.9% of 352 |
| Spatial clustering (Moran's I) | 0.727, p=0.001 |
| Original neighbor-vegetation coefficient | ≈0.087, p<0.0001 |
| Environmental drivers explained (12-factor model) | 41.4% total; VPD alone ≈21.6 pts (~half of it), ENSO ≈11 pts |
| GNN vs. non-spatial baseline (corrected/normalized pipeline) | No consistent winner: baseline ≈ fixed-graph ≈ attention within noise at 1mo (baseline lowest RMSE in only 1 of 5 seeds); fixed-graph beats baseline by 7.5% at 3mo; baseline best at 6mo; adaptive/learned GNN worst at every horizon |
| Near-distance (75–650km) resilience association | Positive, fast (1-month peak), synchrony-like |
| Far-distance (800–1,100km) resilience association | Negative, slow (6-month peak), directional-looking |
| Susceptibility | Target's soil moisture, TWI, and disturbance-distance significant at all four lags (1/2/3/6mo); RZSM significant at lags 1–3 but not lag 6 (p=0.46) |
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
- The GNN results (Stages 34–38) required a mid-project fix for unnormalized input features. All three comparisons — single-horizon (Stage 35), multi-horizon (Stage 36), and 5-seed robustness (Stage 37) — have now been re-run on the corrected feature-standardized pipeline (the `*_normalized` scripts and CSVs). On that corrected pipeline the baseline's earlier clean sweep does **not** hold: spatial and non-spatial models fall within seed/horizon noise of each other, and the fixed-geographic GNN is ahead at the 3-month horizon. See `docs/full_methodology.md`, "Stage 36/37 (Corrected)".
- The methodology doc describes Stages 27, 39, and 40, but no script or output for those stages exists in this repository; their results are flagged inline in `docs/full_methodology.md` as unverified.
- The pairwise model (Stage 48) uses a reproducible random subsample of 3,000 (of 26,860 eligible) source-target pairs for computational tractability, keeping each pair's full time series intact.
- The physical transmission mechanism (e.g., atmospheric moisture transport) was not established with the data and methods available to this project; this would require dedicated multi-level atmospheric data and specialized moisture-tracking modeling beyond this project's scope.

## Reproducing the Analysis

Scripts in `src/` are numbered in the order they were run; each has a module docstring explaining its purpose, inputs, outputs, and key methodological notes. Most later stages (39+) reuse cached intermediate outputs (e.g., `patch_rolling_ar1.csv`) when present rather than recomputing from raw rasters.