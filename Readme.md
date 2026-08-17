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

- **v1 region** (initial pipeline test): 8°S–14°S, 50°W–58°W (~1,000 km extent)
- **v2 region** (current, expanded to properly test distance decay):
  4°S–18°S, 46°W–66°W (~2,000+ km extent)

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
└── figures/                 # Generated plots
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

## Key Findings (v1 region, 8°S–14°S / 50°W–58°W)

- 33.3% of 252 patches show a statistically significant increasing
  AR(1) trend (resilience loss) between 2003–2018.
- A patch's neighboring patches' vegetation state significantly
  predicts its own future state (coef ≈ 0.077, p ≈ 0.006), even after
  controlling for local precipitation — the effect is not just shared
  rainfall in disguise.
- This neighbor effect does not decay with distance within the v1
  region, even out to 800–1,100 km, and this holds after adding a
  regional (whole-area) precipitation control as well — motivating the
  region expansion for v2.

*(v2 region results to be added once the expanded-region pipeline run is complete.)*

## References

Core papers underlying the methodology are listed in `docs/references.md`
(Scheffer et al. 2009; Boulton et al. 2022; Hirota et al. 2011; Elhorst
2010; Dakos et al. 2010; Zemp et al. 2017; DCRNN — Li et al. 2018;
Graph WaveNet — Wu et al. 2019; Raissi et al. 2019; CaST — 2023).

## Team

[Your names], IRIS National Fair 2026–27