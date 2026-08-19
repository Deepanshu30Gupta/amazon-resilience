# Amazon–Cerrado Vegetation Resilience: Spatial Contagion vs. Shared Atmospheric Forcing

**IRIS National Fair project, 2026–27 cycle**

## Research Question

Can vegetation resilience loss in the Amazon–Cerrado transition zone be
explained by spatial mediation (one patch's resilience loss affecting
its neighbors) as distinct from shared atmospheric forcing (patches
merely responding to the same regional rainfall/climate)? And does any
such neighboring effect decay with distance, and if so, at what scale?

## Study Area

The Cerrado–Amazon Transition (CAT) / "Arc of Deforestation" — the
world's largest tropical forest–savanna ecotone, spanning parts of
Mato Grosso, Pará, Rondônia, and neighboring Brazilian states.

- **v1 region** (initial pipeline test, archived): 8°S–14°S, 50°W–58°W (~1,000 km extent)
- **v2 region** (current, expanded to properly test distance decay):
  4°S–18°S, 46°W–66°W (~2,500+ km extent)

## Data Sources

| Dataset | Variable | Resolution | Source |
|---|---|---|---|
| VODCA v2 | Vegetation Optical Depth (vegetation proxy) | ~0.25° | [TU Wien / GEE](https://doi.org/10.48436/t74ty-tcx62) |
| CHIRPS | Precipitation | ~0.05° | [UCSB Climate Hazards Center / GEE](https://www.chc.ucsb.edu/data/chirps) |

Both pulled via Google Earth Engine, clipped to the study region,
aggregated to monthly values (`gee/vodca_chirps_export.js`).

## Method Overview

1. Divide the study region into geography-based patches (aggregated
   pixel blocks), each patch = one node in a spatial network.
2. Remove the seasonal (wet/dry) cycle from both vegetation and
   precipitation data, leaving anomalies.
3. Measure vegetation resilience per patch using lag-1 autocorrelation
   (AR1) of the vegetation anomaly — rising AR1 indicates resilience
   loss (Scheffer et al. 2009; Boulton et al. 2022).
4. Test whether a patch's neighbors' vegetation state predicts the
   patch's own future state, controlling for local and regional
   precipitation, to separate spatial mediation from shared forcing.
5. Repeat the neighbor-effect test across increasing distance bands to
   estimate the spatial scale of any influence.

## Pipeline

Each stage is a standalone script in `src/`, run in order:

| Script | Purpose |
|---|---|
| `01_inspect_data.py` | Load and sanity-check the raw GeoTIFFs |
| `02_create_patches.py` | Build the patch grid from the VODCA pixel grid |
| `03_aggregate_data.py` | Aggregate CHIRPS into patches; build adjacency + combined time series |
| `04_deseasonalize.py` | Remove the seasonal cycle from VOD and precipitation |
| `05_calculate_ar1.py` | Compute rolling AR(1) resilience metric and test for a trend |
| `06_spatial_baseline.py` | Test the neighbor effect vs. local precipitation confounder |
| `07_distance_analysis.py` | Repeat the neighbor-effect test across distance bands |
| `08_regional_forcing.py` | Add a regional precipitation control to the distance analysis |

Run from the repo root: `python src/01_inspect_data.py`, etc.

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

Raw `.tif` files are not committed to git (too large) — see
`data/raw/README.md` for how to regenerate them via the GEE script.

## Setup

```bash
pip install -r requirements.txt
```

Run the GEE export script in the [Earth Engine Code Editor](https://code.earthengine.google.com),
download the resulting files from Google Drive into `data/raw/`, then
run the pipeline scripts in order.

## Key Findings

### v1 region (8°S–14°S, 50°W–58°W, ~1,000 km extent, 252 patches)
- 33.3% of patches show a statistically significant increasing AR(1) trend
  (resilience loss) between 2003–2018.
- Neighbor effect on future vegetation state: coef ≈ 0.077, p ≈ 0.006,
  surviving a local precipitation control.
- Effect does not decay with distance, tested up to ~1,100 km.
- See `figures/v1_region/` for the corresponding maps and plots.

### v2 region (4°S–18°S, 46°W–66°W, ~2,500 km extent, 352 patches)
- 44.9% of patches (158/352) show a statistically significant increasing
  AR(1) trend between 2003–2018 - a larger share than v1, likely
  reflecting the broader area sampled.
- Neighbor effect: coef ≈ 0.087, p < 0.001, essentially unchanged whether
  controlling for local precipitation, regional (whole-area) precipitation,
  or both together (local precipitation itself becomes non-significant
  once the regional control is added, suggesting most of the local
  precipitation signal was actually regional in origin).
- Effect still does not decay with distance, now tested up to 2,542 km -
  replicating the v1 finding on an independently larger region.
- See `figures/` (root) for these plots.

### Interpretation
The neighbor effect - vegetation state in nearby patches predicting a
target patch's future state - is a robust, repeated finding across two
independently-sized study regions, and survives both local and
regional-scale precipitation controls. It does not decay with distance
within either tested region, suggesting the true spatial scale of
influence (if it exists as a genuine local process) may exceed even our
larger ~2,500 km study area, or that an unmeasured factor beyond
precipitation contributes to the pattern. This is reported as an honest,
open finding rather than forced into a decay curve that the data does
not support.

**Note:** raw `.tif` files for the v1 region are no longer stored locally
(overwritten when the study area was expanded), but are fully
regenerable via `gee/vodca_chirps_export.js` using the v1 bounding box
noted in that script's comments.

## Robustness Testing (in progress)

Before proceeding to a causal/counterfactual model, the neighbor-effect
finding above is being stress-tested to rule out alternative
explanations:
- Placebo test (using future neighbor state to predict past target
  state - should show no effect if the methodology is sound)
- Different time delays (2-3 month lags, not just 1 month)
- Alternative neighbor definitions (diagonal adjacency, fixed-radius)
- Additional statistical controls (year/seasonal fixed effects)
- Sensitivity to patch size choice

Only once these hold up does the project proceed to the counterfactual
model and, time permitting, a graph neural network comparison.

## References

Core papers underlying the methodology are listed in `docs/references.md`
(Scheffer et al. 2009; Boulton et al. 2022; Hirota et al. 2011; Elhorst
2010; Dakos et al. 2010; Zemp et al. 2017; DCRNN — Li et al. 2018;
Graph WaveNet — Wu et al. 2019; Raissi et al. 2019; CaST — 2023).

## Team

Deepanshu Gupta and Ananya Dixit