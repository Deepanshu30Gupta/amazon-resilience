# Amazon–Cerrado Vegetation Resilience: Spatial Synchrony vs. Shared Atmospheric Forcing

**IRIS National Fair project, 2026–27 cycle**  
**Deepanshu Gupta & Ananya Dixit**

---

## Research Question

Does the resilience state of one forest patch relate to the **future resilience state of a distant patch**, after accounting for environmental conditions affecting both patches?

More specifically:

- Does any cross-patch resilience association exist?
- At what distance does it appear?
- What time lag is associated with the strongest signal?
- Under what environmental conditions is the association stronger or weaker?
- Can the observed statistical relationship be distinguished from simple shared atmospheric forcing?

> **Important:** This is an observational study. The project does **not** claim that one patch physically causes another patch to change.

---

## Study Area

The study covers the **Amazon–Cerrado transition zone** in Brazil.

- **v1 region:** 8°S–14°S, 50°W–58°W — 252 patches
- **v2 region:** 4°S–18°S, 46°W–66°W — 352 patches
- v2 spans roughly **2,500 km**, allowing much stronger tests of long-distance relationships.
- Patch size is approximately **100 × 100 km** in the current v2 analysis.

The analysis uses monthly data from **January 2003 through December 2018 (192 months)**.

---

## Data Sources

| Dataset | Main variable | Role |
|---|---|---|
| **VODCA v2** | Vegetation Optical Depth (VOD) | Vegetation state / resilience indicator |
| **CHIRPS** | Precipitation | Local and regional rainfall controls |
| **ERA5 / ERA5-Land** | Temperature, dewpoint, wind, solar radiation, root-zone soil moisture | Atmospheric and land-surface controls |
| **TerraClimate** | Soil moisture, PDSI | Moisture / drought controls |
| **NOAA ONI** | ENSO / El Niño state | Basin-scale climate forcing |
| **Hansen Global Forest Change** | Disturbance information | Disturbance exposure |
| **MODIS** | Land-surface temperature, cloud fraction | Additional environmental controls |
| **SRTM DEM** | Elevation / terrain | Topographic Wetness Index (TWI) |

VODCA and CHIRPS were initially exported through Google Earth Engine using `gee/vodca_chirps_export.js`.

Raw GeoTIFFs are intentionally not committed to Git because of their size.

---

## Key Methodological Definitions

### Vegetation anomaly

Seasonality is removed separately for each patch by subtracting the typical value for the same calendar month.

This prevents the normal wet/dry seasonal cycle from being mistaken for a resilience signal.

### Resilience indicator

The project uses **lag-1 autocorrelation (AR(1)) of deseasonalized VOD anomalies** as an early-warning indicator associated with resilience loss.

A rising AR(1) means the vegetation state is becoming more persistent or "sticky," which is interpreted as a signal consistent with declining resilience.

**AR(1) is an indicator, not a direct measurement of ecological resilience or recovery speed.**

### Cross-patch resilience test

The central model asks whether:

> **Source patch A's resilience state at time t is statistically associated with target patch B's resilience state at a future time t + lag, after controlling for the target's current resilience and measured environmental conditions affecting both patches.**

A representative final pairwise model is:

\[
R_B(t+3)=\alpha+\beta R_A(t)+\gamma R_B(t)
+\theta'X_B(t)+\phi'X_A(t)+\rho\,ONI(t)+\epsilon
\]

where:

- \(R_A(t)\) = source patch resilience state
- \(R_B(t+3)\) = target patch resilience state three months later
- \(X_A, X_B\) = environmental controls for source and target
- \(ONI\) = basin-scale ENSO forcing
- \(\beta\) = the conditional statistical association of interest

This equation is **not a causal equation**. It describes a conditional observational association.

---

## Methodology Overview

The project evolved through several stages rather than assuming the original hypothesis was correct.

### Phase I — Establishing the spatial relationship

The first analysis tested whether neighboring **vegetation states** predicted future vegetation states.

This produced a strong positive association:

- v2 neighbor coefficient ≈ **0.087**
- p < **0.001**
- Robust to several alternative neighbor definitions
- Similar result under a coarser patch grid

However, the association initially showed little or no distance decay.

### Phase II — Testing environmental explanations

Environmental controls were added progressively, including:

- precipitation
- temperature
- soil moisture
- drought / PDSI
- ENSO / ONI
- VPD
- wind
- solar radiation
- root-zone soil moisture
- other land-surface variables

In the consolidated analysis, measured environmental factors reduced the original neighbor coefficient by about **41.4%**, with VPD being the strongest individual contributor.

This showed that shared environmental forcing explains a substantial part of the original vegetation association, but not all of it.

### Phase III — Testing predictive usefulness

Linear prediction models and multiple Graph Neural Network architectures were tested to ask whether explicit spatial information improves prediction.

The corrected GNN experiments used training-period standardization to avoid feature-scale domination and leakage.

**Result:** the non-spatial baseline generally performed best; spatial models did not provide a reliable predictive improvement across tested horizons.

This made the project less dependent on a black-box spatial ML claim.

### Phase IV — Reframing from vegetation to resilience

The key methodological correction was to stop treating raw vegetation state as the final object of interest.

Instead, the project constructed a rolling **resilience state** from vegetation anomalies using AR(1), then directly tested:

> **source resilience → future target resilience**

This is the analysis underlying the project's main result.

---

## Final Results

### 1. Spatial pattern of resilience loss

Using increasing AR(1) as the resilience-loss indicator:

- **44.9% of 352 patches (158 patches)** showed a statistically significant increasing AR(1) trend.
- Moran's I = **0.727**, p = **0.001**, indicating strong spatial clustering.

This means resilience-loss signals are not randomly distributed across the study area.

---

### 2. Near-distance relationship

For approximately **75–650 km** separation, the resilience-state analysis found a predominantly:

- **positive**
- **short-lag**
- approximately **1-month peak**

association.

The forward and backward tests were relatively similar in this regime.

The most defensible interpretation is that this pattern is **synchrony-like**: nearby patches tend to move together, rather than providing clear evidence that one patch drives another.

---

### 3. Far-distance relationship

At approximately **800–1,100 km** separation, the analysis found a **negative** association.

The distance-band analysis showed its strongest signal around a **6-month lag**, while a later targeted pairwise analysis at **3 months** independently recovered a significant negative association.

This distinction is important:

- **6-month result:** strongest peak in the distance-band analysis
- **3-month result:** final targeted pairwise specification used for the headline coefficient

The project therefore does **not** claim that 3 months is the unique causal delay.

---

## Headline Pairwise Result

A targeted pairwise re-analysis was performed for the **800–1,100 km** band.

Because constructing the full monthly panel for all eligible pairs exceeded available memory, the final computationally feasible analysis used:

- **3,000 sampled directed source-target pairs**
- **350 unique targets**
- **352 unique sources**
- **498,000 panel rows**
- the full available monthly history retained for each sampled pair
- two-way clustered inference by **target × source**

The estimated source-to-future-target association was:

\[
\boxed{\beta \approx -0.00723}
\]

with two-way clustered:

- 95% CI ≈ **[-0.01147, -0.00299]**
- **p = 0.0008**

Interpretation:

> Holding the measured controls and the target's current resilience state constant, higher source resilience at time \(t\) was associated with lower future target resilience three months later among the sampled 800–1,100 km pairs.

Again, this is a **statistical association, not proof of causation**.

The coefficient itself is unchanged by the clustering choice; clustering changes the estimated uncertainty around that coefficient.

---

## Robustness and Adversarial Testing

The project deliberately tested ways in which the headline interpretation could fail.

### Progressive control test

For the far-distance 800–1,100 km band at lag 3, the negative association remained negative after adding:

- local environmental controls
- regional controls
- latitude/longitude controls

The point estimate weakened as controls were added.

### Sensitivity to resilience construction

The negative lag-3 pattern remained negative when changing the resilience-window construction, including:

- 24-month AR(1)
- 36-month AR(1)
- 24-month rolling standard deviation

### Clustering sensitivity

A two-way clustered test on the earlier **band-averaged** design gave a much wider confidence interval and was not statistically significant.

This is an important caveat and is one reason the final claim is based on the more targeted **pairwise** specification rather than the earlier averaged-neighbor result.

### Random-shuffling test

A fake-neighbor permutation test produced:

- real statistic: about **−0.01846**
- permutation p ≈ **0.287**

Therefore, this test was **inconclusive** about whether the original band-averaged relationship was stronger than randomly reshuffled pairings.

The result is not presented as validation of the effect.

### Pairwise sub-bands

The final pairwise analysis remained negative within each of three sub-bands:

- **800–900 km:** β ≈ −0.00807
- **900–1000 km:** β ≈ −0.00663
- **1000–1100 km:** β ≈ −0.00698

---

## Susceptibility: Which Patches Are More Coupled?

A particularly consistent result was obtained from interactions between the cross-patch signal and target-side buffering variables.

Across tested lags, significant interactions were found for:

- target soil moisture
- root-zone soil moisture
- Topographic Wetness Index (TWI)
- distance to disturbance

The pattern suggests that more buffered / wetter / less disturbance-exposed patches may be **less strongly coupled** to the distant signal, while more vulnerable patches may show stronger coupling.

This is an association, not a demonstrated mechanism.

---

## Mechanism Search

The project then tested whether wind alignment and moisture availability could provide a physical explanation.

### Wind-direction test

A source-target wind-alignment interaction was statistically significant, but subgroup analysis gave contradictory behavior between wind-aligned and wind-opposed cases.

Therefore:

> **The wind test provides mixed evidence and does not establish a direct wind-transport mechanism.**

### Moisture-proxy chain

A conceptual proxy combining source resilience, wind alignment, and source root-zone soil moisture was used to test a possible source → atmospheric moisture → target pathway.

The required links did not show consistent statistical support, and the estimated direct association barely changed.

Therefore:

> **The physical transmission mechanism remains unresolved.**

A future mechanism study would require more direct atmospheric analysis, such as multi-level humidity/wind fields, evapotranspiration, moisture-recycling estimates, or back-trajectory / atmospheric-network methods.

---

## What the Project Does NOT Claim

This project does **not** claim that:

- one forest patch has been proven to physically cause another patch's resilience loss;
- the observed relationship is a demonstrated ecological contagion mechanism;
- wind transport has been established as the mechanism;
- the random-shuffling test validated the central relationship;
- AR(1) is a direct measurement of ecological resilience;
- all statistical uncertainty disappears simply because the final pairwise coefficient is significant.

The study is intentionally observational and reports both supporting and weakening evidence.

---

## Important Limitations

### Observational design

The data are observational. Unmeasured environmental or atmospheric processes could still contribute to the association.

### AR(1) as an indicator

AR(1) is an accepted early-warning indicator framework, but it is not identical to directly measuring ecological recovery dynamics.

### Pairwise sampling

The final pairwise analysis uses a **3,000-pair computational sample** rather than all eligible 26,860 directed pairs.

The 498,000 panel rows are repeated measurements of those pairs and should **not** be interpreted as 498,000 independent observations.

### Spatial scale

The project includes a patch-size sensitivity analysis, but the full possible multi-scale / MAUP space has not been exhausted.

### Mechanism

The statistical association is stronger than the current mechanistic evidence. The physical pathway remains an open question.

---

## Pipeline

The early pipeline is organized as numbered, independently runnable scripts in `src/`.

| # | Script | Purpose |
|---|---|---|
| 01 | `01_inspect_data.py` | Load and sanity-check raw GeoTIFFs |
| 02 | `02_create_patches.py` | Build the patch grid |
| 03 | `03_aggregate_data.py` | Aggregate CHIRPS and build adjacency/time series |
| 04 | `04_deseasonalize.py` | Remove the seasonal cycle |
| 05 | `05_calculate_ar1.py` | Compute rolling AR(1) and trend |
| 06 | `06_spatial_baseline.py` | Initial neighbor association |
| 07 | `07_distance_analysis.py` | Distance-band analysis |
| 08 | `08_regional_forcing.py` | Add regional precipitation control |
| 09 | `09_robustness_placebo.py` | Backward/future-state placebo |
| 10 | `10_robustness_timelags.py` | Forward vs backward lag tests |
| 11 | `11_robustness_direction_test.py` | Formal directionality comparison |
| 12 | `12_robustness_neighbor_defs.py` | Alternative neighbor definitions |
| 13 | `13_robustness_patch_size.py` | Alternative patch size |

Later stages extend the analysis into environmental controls, GNN prediction tests, rolling resilience-state models, robustness checks, pairwise inference, and mechanism tests. The complete stage-by-stage record is documented in:

`docs/full_methodology.md`

Run the numbered scripts from the repository root in dependency order.

---

## Repository Structure

```text
amazon-resilience/
├── README.md
├── requirements.txt
├── gee/
│   └── vodca_chirps_export.js
├── src/
│   ├── 01_*.py
│   ├── 02_*.py
│   ├── ...
│   └── later analysis stages
├── data/
│   ├── raw/          # downloaded GeoTIFFs; not tracked in Git
│   └── processed/    # generated CSV / NumPy outputs
├── figures/          # generated figures
└── docs/
    ├── full_methodology.md
    └── references.md
```

Large raw data files are excluded from Git and can be regenerated from the Earth Engine export workflow.

---

## Setup

```bash
pip install -r requirements.txt
```

Run the Earth Engine export script in the Google Earth Engine Code Editor, place the downloaded raw files in `data/raw/`, and run the analysis scripts from the repository root.

---

## References

Core methodological references are listed in:

`docs/references.md`

The project draws on work concerning early-warning indicators, vegetation resilience, spatial dependence, and spatiotemporal machine learning, including Scheffer et al. (2009), Boulton et al. (2022), Hirota et al. (2011), Elhorst (2010), Dakos et al. (2010), Zemp et al. (2017), Li et al. (2018), Wu et al. (2019), and related literature documented in the repository.

---

## Team

**Deepanshu Gupta & Ananya Dixit**  
IRIS National Fair 2026–27
