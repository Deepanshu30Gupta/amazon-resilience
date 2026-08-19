# Amazon–Cerrado Vegetation Resilience: Spatial Synchrony vs. Shared Atmospheric Forcing

**IRIS National Fair project, 2026–27 cycle**

## Research Question

Can vegetation resilience loss in the Amazon–Cerrado transition zone be
explained by spatial mediation (one patch's resilience loss affecting
its neighbors) as distinct from shared atmospheric forcing (patches
merely responding to the same regional rainfall/climate)? And does any
such neighboring effect decay with distance, and if so, at what scale?

*(Original framing also asked whether this could be modeled with a
physics-constrained counterfactual graph neural network - see
"Findings" and "Status & Next Steps" below for how the project's
methodology evolved as the evidence came in.)*

## Study Area

The Cerrado–Amazon Transition (CAT) / "Arc of Deforestation" — the
world's largest tropical forest–savanna ecotone, spanning parts of
Mato Grosso, Pará, Rondônia, and neighboring Brazilian states.

- **v1 region** (initial pipeline test, archived): 8°S–14°S, 50°W–58°W
  (~1,000 km extent), 252 patches
- **v2 region** (current, expanded specifically to properly test
  distance decay): 4°S–18°S, 46°W–66°W (~2,500 km extent), 352 patches

## Data Sources

| Dataset | Variable | Resolution | Source |
|---|---|---|---|
| VODCA v2 | Vegetation Optical Depth (vegetation proxy) | ~0.25° | [TU Wien / GEE](https://doi.org/10.48436/t74ty-tcx62) |
| CHIRPS | Precipitation | ~0.05° | [UCSB Climate Hazards Center / GEE](https://www.chc.ucsb.edu/data/chirps) |

Both pulled via Google Earth Engine (`gee/vodca_chirps_export.js`),
clipped to the study region, aggregated to monthly values, covering
2003–2018 (192 months).

## Method Overview

1. Divide the study region into geography-based patches (aggregated
   pixel blocks — chosen over raw pixels or hand-drawn ecological
   boundaries as the realistic, defensible middle ground), each patch
   = one node in a spatial network.
2. Remove the seasonal (wet/dry) cycle from both vegetation and
   precipitation data, leaving anomalies.
3. Measure vegetation resilience per patch using lag-1 autocorrelation
   (AR1) of the vegetation anomaly — rising AR1 indicates resilience
   loss (Scheffer et al. 2009; Boulton et al. 2022).
4. Test whether a patch's neighbors' vegetation state predicts the
   patch's own future state, controlling for local and regional
   precipitation, to separate spatial association from shared forcing.
5. Repeat the neighbor-effect test across increasing distance bands to
   estimate the spatial scale of any influence.
6. Stress-test the finding with five robustness checks (placebo,
   time-lag direction test, alternative neighbor definitions,
   alternative patch sizes) before drawing conclusions or building any
   further model on top of it.

## Pipeline

Each stage is a standalone script in `src/`, run in order. Every
script reads/writes CSV or NumPy files in `data/processed/` so each
stage can be independently re-run and verified.

| # | Script | Purpose |
|---|---|---|
| 01 | `01_inspect_data.py` | Load and sanity-check the raw GeoTIFFs |
| 02 | `02_create_patches.py` | Build the patch grid from the VODCA pixel grid |
| 03 | `03_aggregate_data.py` | Aggregate CHIRPS into patches; build adjacency + combined time series |
| 04 | `04_deseasonalize.py` | Remove the seasonal cycle from VOD and precipitation |
| 05 | `05_calculate_ar1.py` | Compute rolling AR(1) resilience metric and test for a trend |
| 06 | `06_spatial_baseline.py` | Test the neighbor effect vs. local precipitation confounder |
| 07 | `07_distance_analysis.py` | Repeat the neighbor-effect test across distance bands |
| 08 | `08_regional_forcing.py` | Add a regional precipitation control to the distance analysis |
| 09 | `09_robustness_placebo.py` | Placebo test: does the effect also "work" backwards in time? |
| 10 | `10_robustness_timelags.py` | Compare forward vs. backward effect at 1/2/3-month lags |
| 11 | `11_robustness_direction_test.py` | Formal statistical test of forward-vs-backward difference |
| 12 | `12_robustness_neighbor_defs.py` | Test alternative neighbor definitions (diagonal, fixed-radius) |
| 13 | `13_robustness_patch_size.py` | Test an alternative patch size, self-contained |

Run from the repo root: `python src/01_inspect_data.py`, etc., in
numeric order — later stages depend on earlier stages' output files.

## Repository Structure

```
amazon-resilience/
├── README.md
├── requirements.txt
├── gee/                    # Google Earth Engine export script
├── src/                    # Pipeline scripts, numbered in run order
├── data/raw/                # Downloaded GeoTIFFs (not tracked in git - see below)
├── data/processed/          # Script outputs (CSVs, intermediate arrays)
└── figures/                 # Generated plots (current/v2 region)
    └── v1_region/            # Archived figures from the original smaller study region
```

Raw `.tif` files are not committed to git (too large) and are not kept
locally once superseded by a new region - they are fully regenerable
via `gee/vodca_chirps_export.js` (both the v1 and v2 bounding boxes are
documented in that script's comments).

## Setup

```bash
pip install -r requirements.txt
```

Run the GEE export script in the [Earth Engine Code Editor](https://code.earthengine.google.com),
download the resulting files from Google Drive into `data/raw/`, then
run the pipeline scripts in order.

## Findings

### 1. Resilience loss is real and measurable

Using lag-1 autocorrelation (AR1) of deseasonalized vegetation
anomalies as a resilience-loss indicator (Scheffer et al. 2009;
Boulton et al. 2022):

- **v1 region:** 33.3% of 252 patches (84) show a statistically
  significant increasing AR(1) trend (resilience loss), 2003–2018.
- **v2 region:** 44.9% of 352 patches (158) show the same, a larger
  share - likely reflecting the broader area sampled.
- Spatially, resilience-loss patches are not randomly scattered - they
  cluster together on the map, motivating the neighbor-effect
  investigation below.

### 2. A robust spatial association exists, and survives precipitation controls

A patch's neighboring patches' vegetation state significantly predicts
the patch's own future vegetation state, even after controlling for
local precipitation, regional (whole-area) precipitation, or both
together:

- **v1 region:** neighbor coefficient ≈ 0.077, p ≈ 0.006
- **v2 region:** neighbor coefficient ≈ 0.087, p < 0.001 (tighter
  estimate, larger sample: 67,232 observations)
- This result is **robust to**:
  - alternative neighbor definitions (edge-only, diagonal/8-connectivity,
    and fixed 150km-radius all give coef ≈ 0.087, p < 0.0001)
  - alternative patch size (a coarser 150-patch grid gives coef ≈
    0.088, p < 0.0001, nearly identical to the main 352-patch result)

### 3. The effect does not decay with distance

Tested across distance bands up to 800–1,100 km (v1 region) and up to
2,542 km (v2 region, more than double the extent), the neighbor
coefficient does not show a decay pattern - it fluctuates around a
roughly flat baseline (~0.03–0.09 depending on region/controls) at
every distance tested, including the farthest bands, both with and
without a regional precipitation control added.

### 4. Robustness testing reveals the association is primarily spatial SYNCHRONY, not confirmed directional influence

This is the project's most important methodological finding. Two
targeted robustness tests were run specifically to check whether the
neighbor effect reflects genuine forward-in-time causal influence
(neighbor's state today shaping the target's state tomorrow) or
symmetric synchrony (patches simply moving together, with no
meaningful time direction):

- **Placebo test:** using a neighbor's FUTURE state to "predict" the
  target's PAST state (a relationship that should not exist if the
  effect is genuinely forward-causal) returned a coefficient (0.079,
  p = 0.0001) nearly identical to the real forward-time result (0.087,
  p < 0.001).
- **Forward-vs-backward comparison at multiple lags (1, 2, 3 months):**
  forward coefficients were consistently slightly higher than backward
  at every lag tested, but a formal statistical test of that
  difference found it significant at only 1 of 3 lags (and that one
  right at the p = 0.05 threshold) - weak, borderline evidence at best.

**Conclusion:** the neighbor-target association is real, robust to
multiple confound controls and methodological choices, and replicates
across two independently-sized study regions - but the evidence
available does not support confidently framing it as one-directional
"spatial contagion." The more honest and better-supported description
is **spatial synchrony**: neighboring patches' vegetation dynamics are
strongly linked in a way that isn't explained by shared local or
regional precipitation, but the directionality of that link (does A
influence B, does B influence A, or do both simply move together due
to some other shared, unmeasured factor) remains an open question.

## Status & Next Steps

Given finding #4 above, the project's original framing (a
"physics-constrained counterfactual graph neural network" proving
directional spatial contagion) is not currently supported by the
evidence and will not be built as originally conceived - building a
sophisticated causal ML model on top of an unconfirmed directional
claim would be a real weakness under scrutiny.

Instead, the project is continuing to investigate the synchrony
finding on its own terms before any further modeling, including:
- Testing additional environmental drivers (temperature, soil
  moisture) to see whether they further explain the synchrony
- Spatial autocorrelation analysis (Moran's I / LISA) to formally
  characterize and map the clustering
- Classical spatial econometric models (spatial lag/error, per
  Elhorst 2010) as a more rigorous alternative/complement to the
  current linear panel regression
- A possible graph neural network extension - motivated specifically
  by the unexplained non-decay-with-distance result, to test whether a
  *learned* (rather than assumed-geographic) adjacency structure
  reveals that the true connectivity driving the synchrony is
  atmospheric/teleconnected rather than spatial - not framed as proof
  of causation, but as a descriptive/predictive-skill question

## References

Core papers underlying the methodology are listed in `docs/references.md`
(Scheffer et al. 2009; Boulton et al. 2022; Hirota et al. 2011; Elhorst
2010; Dakos et al. 2010; Zemp et al. 2017; DCRNN — Li et al. 2018;
Graph WaveNet — Wu et al. 2019; Raissi et al. 2019; CaST — 2023).

## Team

Deepanshu Gupta, Ananya Dixit — IRIS National Fair 2026–27