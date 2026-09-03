# Complete Stage-by-Stage Results Log: Amazon–Cerrado Vegetation Resilience Synchrony

### Every stage explained individually — why it was run, what was done, what was found, and what it led to next. For Ananya's research paper — use this as your primary source of raw material.

**Team:** Deepanshu Gupta (analysis/coding) & Ananya Dixit (research/writing) — IRIS National Fair 2026–27

---

# PART A: FOUNDATION (Stages 1–13)

## Stage 1: Inspecting the Raw Satellite Data

**Why we did this:** Before building anything, we needed to understand the raw satellite files (GeoTIFFs) we'd downloaded — their resolution, coordinate systems, extent, and whether they actually covered our region of interest properly.

**What we did:** Loaded and visually/statistically inspected the raw VODCA (vegetation) and CHIRPS (rainfall) files.

**What we found:** Confirmed the files were usable and correctly covered the target region, with the resolution and structure needed for the next step (patch aggregation).

**What it led to:** Moving on to actually building the patch grid.

---

## Stage 2: Building the Patch Grid

**Why we did this:** Raw satellite data comes as a grid of small pixels. Using every individual pixel as a unit of analysis would be extremely noisy and computationally unwieldy, and — more importantly — wouldn't match the spatial scale at which regional climate processes (weather systems, drought patterns) actually operate.

**What we did:** Aggregated the pixel grid into 352 patches, each built from a 4×4 block of underlying VODCA pixels, giving each patch a real geographic footprint of roughly 100km × 100km with genuine latitude/longitude boundaries.

**What we found:** A clean, consistent 352-patch grid covering the 4°S–18°S, 46°W–66°W study region.

**What it led to:** All subsequent analysis operates on these 352 patches, not raw pixels — this is the fundamental spatial unit of the entire project.

---

## Stage 3: Aggregating Rainfall and Building Spatial Adjacency

**Why we did this:** We needed (a) rainfall data aggregated to the same patch grid as vegetation, and (b) a definition of which patches count as "neighbors" of which other patches, since the whole project is fundamentally about neighbor relationships.

**What we did:** Aggregated CHIRPS rainfall to the patch grid (same method as Stage 2), and built an adjacency structure — a table listing which patches are geographically adjacent to which others (edge and diagonal neighbors on the patch grid).

**What we found:** A working rainfall dataset and a real adjacency network to test neighbor relationships against later.

**What it led to:** The core building blocks needed for every subsequent test — rainfall as a control variable, adjacency as the definition of "neighbor."

---

## Stage 4: Removing Seasonality (Deseasonalizing)

**Why we did this:** Vegetation naturally goes up and down every year with the seasons (wet season vs. dry season). If we didn't remove this, we'd be studying normal seasonal cycles, not actual resilience change — the whole project would be meaningless without this step.

**What we did:** For every patch and every variable, calculated the "anomaly" — the difference between the observed value in a given month and that patch's typical (average) value for that specific calendar month across all years. E.g., if a patch's typical July VOD is 0.80 and this particular July it's 0.72, the anomaly is −0.08.

**What we found:** A full set of deseasonalized vegetation anomaly time series for all 352 patches, 2003–2018.

**What it led to:** This "anomaly" format became the standard input for every subsequent stage — we never again worked with raw, seasonal vegetation values.

---

## Stage 5: The First Resilience Metric (Static AR(1))

**Why we did this:** We needed a way to actually quantify "resilience" from a vegetation time series, since there's no direct sensor for it. We relied on "critical slowing down" theory (Scheffer et al. 2009): as a system loses resilience, its state becomes more persistent/sluggish over time — this shows up statistically as increasing autocorrelation (AR(1) — how similar this month's value is to last month's).

**What we did:** Calculated one AR(1) value per patch, using a 60-month rolling window, then used a Kendall's tau trend test to see whether AR(1) was significantly increasing (i.e., resilience significantly declining) over the full 2003–2018 study period.

**What we found:** **44.9% of the 352 patches showed a statistically significant increasing AR(1) trend** — nearly half the region showing a real warning-sign pattern consistent with declining resilience over the study period.

**A bug we caught and fixed here:** An early attempt to use these overlapping rolling-AR(1) windows as a predictor of *themselves* (i.e., using AR(1) at one point in time to predict AR(1) at a nearby point in time) produced a spuriously high correlation (~0.986) — because consecutive rolling windows share almost all of their underlying months, so a variable predicting a near-copy of itself will always look highly "correlated," but this is a statistical artifact, not a real finding. We corrected this by using the monthly VOD anomaly directly for the actual neighbor-effect models instead of chaining rolling-AR(1) windows together. This lesson mattered again much later (Stage 41 onward) when we built genuinely useful time-varying resilience metrics, carefully avoiding this same mistake.

**What it led to:** A patch-by-patch classification of "is this patch's resilience declining overall" — useful for later stages that asked why *individual* patches decline, but not yet something we could use to ask "what is patch A's resilience state *right now*" (that required rolling AR(1), built much later in Stage 41).

---

## Stage 6: The First Neighbor-Effect Test (Spatial Baseline Regression)

**Why we did this:** With deseasonalized vegetation data in hand, we could finally ask the central question: does a patch's neighbors' vegetation state help predict that patch's own future vegetation, above and beyond what the patch's own recent history already tells us?

**What we did:** Built a regression: `own_vod(t+1) ~ own_vod(t) + neighbor_avg_vod(t)`, where `neighbor_avg_vod` is the average vegetation anomaly of a patch's adjacent neighbors. Including the patch's own current state as a control is important — it means we're testing whether neighbors add *additional* predictive information, not just re-discovering that all patches in a region look similar.

**What we found:** A real, statistically significant neighbor effect: **coefficient ≈ 0.087, p < 0.0001.**

**What it led to:** This became the central number the whole first half of the project tried to explain — Stages 7–33 systematically tested whether this 0.087 could be explained away by distance decay, environmental confounds, or was just a robustness artifact.

---

## Stage 7: Does the Neighbor Effect Decay With Distance?

**Why we did this:** A basic, intuitive expectation is that spatial relationships should be strongest between very close patches and weaken (decay) as distance increases. Testing this seemed like an obvious next step.

**What we did:** Repeated the Stage 6 regression separately for patches grouped into distance bands (e.g., 75–150km, 150–225km, ... up to 1,100km), using the actual geographic distance between patch centers.

**What we found:** **No meaningful distance decay** — the neighbor effect stayed roughly similarly strong (and statistically significant) all the way out to the largest distance band tested, 800–1,100 km. (No distance bands beyond 1,100 km were tested; the study region spans ~2,500 km, but the binned analysis stops at 1,100 km.) A flat pattern out to 1,100 km is not what a simple "spatial contagion" story would predict.

**What it led to:** This was one of the biggest early puzzles of the whole project, and directly motivated the next ~25 stages of environmental-driver testing — if the effect isn't fading with distance, maybe it isn't really about geographic neighbors at all, but about some large-scale shared climate factor (like ENSO) that affects huge areas simultaneously regardless of exact distance.

---

## Stage 8: Controlling for Regional Forcing

**Why we did this:** A direct follow-up to Stage 7's puzzle — if the lack of distance decay suggests a shared, large-scale driver, we should try controlling for the broader regional average conditions and see if that changes anything.

**What we did:** Added a regional (whole-area average) climate control to the Stage 6 model.

**What we found:** The core neighbor coefficient remained largely intact — a simple regional average control wasn't enough to explain it away, which meant we needed a more systematic, comprehensive approach (leading into the long environmental-driver-testing arc).

**What it led to:** Confirmed that explaining the neighbor effect would require testing individual, specific environmental variables one at a time, not just a single blunt regional-average control — setting up Stages 14 onward.

---

## Stage 9: The Placebo/Direction Test

**Why we did this:** A crucial scientific distinction: even if A and B look statistically related, that doesn't tell us whether A's changes come *before* B's (suggesting some kind of directional influence) or whether they're just happening *at the same time* because of shared external forcing (synchrony). We needed a way to distinguish these.

**What we did:** Built a "placebo" test — deliberately running the regression *backward in time* (using a patch's neighbors' *future* state to try to "predict" that patch's *past* state). If the real, forward-time relationship reflects genuine directional influence, this backward version should show little to nothing. If it's really just synchrony (shared forcing), the backward version might look similar to the forward version.

**A bug we caught and fixed here:** An initial version of this placebo test didn't include the target patch's own historical value as a control, which produced a misleadingly large backward effect. We corrected this to match the real (forward) model's structure exactly, changing only which time point served as the outcome — a fair, apples-to-apples comparison.

**What we found:** The corrected backward-time coefficient (≈0.079) was **nearly identical** to the forward coefficient (≈0.087) — meaning the neighbor relationship, at this early, raw-vegetation-based stage of the project, looked much more like **synchrony** (shared forcing) than genuine directional influence.

**What it led to:** This "corrected placebo test" methodology became a template we reused and refined multiple times later in the project (Stage 44, at the more meaningful resilience-to-resilience stage) — and this early result meant we couldn't yet claim any kind of real "A influences B" story, only that A and B were statistically associated.

---

## Stage 10: Testing Multiple Time Lags

**Why we did this:** A natural follow-up to the placebo test — if there's any real directional signal, it might show up more clearly at certain time delays (1 month vs. 2 months vs. 3 months) rather than being equally present (or absent) at all of them.

**What we did:** Repeated the forward vs. backward placebo comparison at 1, 2, and 3-month lags.

**What we found:** Forward was slightly larger than backward at every lag tested (1-month: 0.087 vs. 0.079; 2-month: 0.101 vs. 0.092; 3-month: 0.068 vs. 0.060) — a small, consistent gap, but not a dramatic one.

**What it led to:** Set up Stage 11's more formal statistical test of whether this small, consistent gap was actually statistically meaningful.

---

## Stage 11: A Formal Test of Directionality

**Why we did this:** Stage 10 showed a small, consistent forward-vs-backward gap across all three lags — but "consistent" doesn't necessarily mean "statistically significant." We needed a formal test.

**What we did:** Formally tested whether the forward-minus-backward difference was statistically significant at each of the three lags.

**What we found:** Only 1 of the 3 lags (the 3-month lag) reached even a borderline significance level (p≈0.05) — weak, inconclusive evidence for directionality at this stage of the project.

**What it led to:** Confirmed that, using raw vegetation as the outcome, we could not make a strong directionality claim. This weak result is part of why, much later, the project shifted to using an actual time-varying *resilience* metric (Stage 41 onward) rather than raw vegetation — the directionality signal became much clearer once we asked the right question (Stage 44 found a much cleaner two-regime directionality pattern using the reframed resilience metric).

---

## Stage 12: Testing Different Neighbor Definitions

**Why we did this:** Our definition of "neighbor" (adjacent patches on the grid) was one reasonable choice among several. We needed to check the core 0.087 result wasn't just an artifact of that specific choice.

**What we did:** Repeated the core neighbor-effect regression using alternative definitions: edge-only neighbors, edge+diagonal neighbors, and a 150km-radius definition instead of grid-adjacency.

**What we found:** All three alternative definitions gave essentially the same coefficient (~0.087) — the result was not sensitive to exactly how "neighbor" was defined.

**What it led to:** Added confidence that Stage 6's result was a real, robust pattern, not a fragile artifact of one specific methodological choice — cleared the way to test other potential confounds (patch size, region size) next.

---

## Stage 13: Testing a Different Patch Size

**Why we did this:** Another robustness check — would the result change if we used a different-sized spatial unit (larger patches, fewer of them) instead of the original ~100km patches?

**What we did:** Rebuilt the entire patch grid using a larger patch size (PATCH_SIZE=6, giving 150 larger patches instead of 352), and reran the core neighbor-effect regression.

**What we found:** Coefficient ≈ 0.088 — essentially identical to the original 352-patch result.

**What it led to:** Further confirmed the robustness of the core finding across a meaningfully different spatial resolution choice. At this point, the team had thoroughly established that (1) a real neighbor effect exists, (2) it's robust to many reasonable methodological choices, but (3) it doesn't decay with distance and doesn't show strong directionality — setting up the long environmental-driver investigation that follows.

---

# PART B: TESTING ENVIRONMENTAL EXPLANATIONS (Stages 14–33)

## Stage 14: Temperature as a Control

**Why we did this:** The first, most obvious specific environmental variable to test as a possible explanation for the neighbor effect — if nearby (or even far) patches simply experience similar temperatures at the same time, that alone could create a spurious-looking relationship.

**What we did:** Added ERA5 temperature anomaly as a control variable to the core neighbor-effect model.

**What we found:** Essentially no change — temperature explained almost nothing of the neighbor coefficient. In the full specification (local + regional temperature) the coefficient moved from 0.0876 to 0.0875, about a 0.1% reduction; local temperature alone moved it from 0.0865 to 0.0858 (~0.8%). Either way, negligible. *(Recomputed from `temperature_driver_results.csv` and `final_consolidated_results.csv`; the original prose gave "+0.1%", which had the sign backwards — the change is a small reduction, not an increase.)*

**What it led to:** Ruled out temperature as a major explanation; moved on to testing spatial clustering more directly (Stage 15) and other variables (Stage 16 onward).

---

## Stage 15: Confirming Spatial Clustering Independently (Moran's I)

**Why we did this:** We wanted an entirely separate statistical method — not based on the same regression framework as everything else — to confirm that the spatial pattern we kept finding was real, not some artifact specific to our regression approach.

**What we did:** Calculated **Moran's I**, a classic spatial statistics measure of clustering, plus a LISA (Local Indicators of Spatial Association) analysis to identify specific hot/cold spot clusters.

**What we found:** **Global Moran's I = 0.727, p = 0.001** — strong, highly significant spatial clustering. The LISA analysis found 142 "hot-hot" clusters (neighboring patches both showing high resilience loss) and 160 "low-low" clusters (neighboring patches both stable), with 86% of patches falling into a clear cluster category.

**What it led to:** This was an important independent confirmation — using a completely different statistical method than our regression approach, we still found strong evidence of real spatial clustering. This gave us much more confidence that the patterns found throughout the project were genuine, not a regression-specific artifact.

---

## Stage 16: Soil Moisture and Drought (PDSI)

**Why we did this:** Continuing the systematic environmental-driver testing — soil moisture and drought conditions are obvious candidate explanations for shared vegetation stress across nearby patches.

**What we did:** Added TerraClimate soil moisture and PDSI (Palmer Drought Severity Index) as controls.

**What we found:** A small amount. Soil moisture and PDSI added only about 1 percentage point of *incremental* explanation on their own (coefficient 0.0858 → 0.0850 in the consolidated chain). The figure sometimes quoted as "3.0%" is the *cumulative* reduction through precipitation + temperature + soil + PDSI combined — i.e. mostly precipitation and temperature, not soil/drought. *(Recomputed from `soil_drought_driver_results.csv` and `final_consolidated_results.csv`, all as a fraction of the original 0.0876 coefficient.)*

**What it led to:** Combined with temperature (Stage 14), the "obvious" local weather variables were clearly not the main explanation — motivated testing something with larger spatial scale next (ENSO, Stage 17).

---

## Stage 17: ENSO (El Niño/La Niña)

**Why we did this:** ENSO is a huge-scale climate driver that can affect enormous areas of South America simultaneously — exactly the kind of "shared forcing" that could make far-apart patches look connected without any real link between them, and a natural candidate given Stage 7's finding that the neighbor effect didn't decay with distance.

**What we did:** Added the NOAA ONI (Oceanic Niño Index) as a control variable.

**What we found:** ENSO was **the largest single driver up to this point — its own incremental contribution was about 11 percentage points of the original coefficient (coefficient 0.0850 → 0.0753 in the consolidated chain), more than precipitation, temperature, soil moisture and PDSI combined (~3 points together)**. The cumulative reduction through precipitation + temperature + soil + PDSI + ENSO reached about 14% of the original coefficient. Highly significant (p ≈ 1.6×10⁻⁵). *(Recomputed from `enso_driver_results.csv` and `final_consolidated_results.csv`. The earlier "14.3% … ENSO alone" phrasing conflated ENSO's incremental effect with the running cumulative total.)*

**What it led to:** This was the biggest single driver found up to this point in the project, and it made physical sense given ENSO's known continent-scale reach — strongly suggested we were on the right track looking for large-scale shared climate explanations, and set up the search for other similarly-large-scale variables (leading eventually to VPD, Stage 29, which turned out to be even bigger).

---

## Stage 18: Human Disturbance — Part A (An Unexpected, Opposite-Direction Finding)

**Why we did this:** A prior published study (Boulton et al. 2022) had found that distance from human disturbance relates to resilience trends in the Amazon. We wanted to test whether our region showed a similar pattern, and whether disturbance might help explain the neighbor-synchrony effect.

**What we did:** Used Hansen Global Forest Change data to calculate each patch's distance to the nearest disturbed/deforested area, then tested its relationship both to individual patch resilience trend and to the neighbor-synchrony coefficient.

**What we found:** **Distance-to-disturbance POSITIVELY correlates with resilience loss (r=0.58, p<0.0001)** — meaning patches *farther* from disturbance showed *more* resilience loss, the **opposite** direction from Boulton et al.'s finding. Separately (Part B of this stage), adding disturbance distance as a control to the neighbor-synchrony model changed the coefficient by 0.0% — no effect on the synchrony question at all.

**What it led to:** This surprising, opposite-direction result needed to be checked carefully for artifacts before we could trust it — leading directly to Stages 19 and 20.

---

## Stage 19: Checking the Disturbance Finding Isn't a Land-Cover Artifact

**Why we did this:** Before trusting Stage 18's surprising, opposite-direction finding, we needed to rule out an obvious potential artifact: maybe the pattern was really about forest-vs-non-forest land cover mixing, not actually about disturbance distance itself.

**What we did:** Repeated the Stage 18 analysis across multiple different forest-cover threshold definitions, to see if the pattern held regardless of exactly how we defined "forest."

**What we found:** The pattern held across all forest-cover thresholds tested — not a land-cover-mixing artifact.

**What it led to:** Added confidence in the Stage 18 finding, but one more potential confound remained (latitude, since disturbance and latitude might be correlated in this region) — tested next in Stage 20.

---

## Stage 20: Checking the Disturbance Finding Isn't Just Latitude

**Why we did this:** Distance-to-disturbance might simply correlate with latitude in our study region (e.g., more disturbed areas tend to be in a particular part of the region) — if so, the "disturbance" finding might really just be a repackaged latitude effect.

**What we did:** Added latitude and longitude as explicit controls in the Stage 18 model, to see if the disturbance-distance effect survived.

**What we found:** Geographic position explained a substantial chunk of the disturbance-distance relationship, but did not eliminate it. Adding **latitude alone** attenuated the disturbance-distance coefficient by about 24% (0.0080 → 0.0061); adding **latitude and longitude together** brought the combined attenuation to about 57% (0.0080 → 0.0035). Even in that fully geography-controlled model the disturbance-distance effect remained highly significant (p ≈ 9×10⁻⁷). So it wasn't *purely* a geography artifact, though geographic position was clearly part of the story. *(Recomputed from `disturbance_latitude_control_results.csv`; the earlier prose credited the full 57% to "latitude", but ~57% requires longitude as well — latitude by itself is ~24%.)*

**What it led to:** Confirmed the human-disturbance finding (Stage 18) as real, if partially latitude-related, and — importantly — confirmed yet again that this finding was about *individual patch* resilience trend, not about the neighbor-synchrony question, which remained completely unaffected (0% change) throughout Stages 18–20. This "individual patch resilience vs. cross-patch synchrony are different questions" pattern would repeat again later with TWI (Stage 32).


# Complete Stage-by-Stage Results Log — PART B CONTINUED (Stages 21–33)

*(Continuation of DEEP_STAGE_LOG_PART1.md — same project, same team)*

---

## Stage 21: Classical Spatial Regression Models (Spatial Lag vs. Spatial Error)

**Why we did this:** Up to this point we'd used a fairly simple regression approach. We wanted to check our results against more formal, established spatial-statistics models (from the spatial econometrics literature — Elhorst 2010) that are specifically designed to handle spatial data properly.

**What we did:** Ran a cross-sectional analysis using each patch's overall resilience trend (Kendall's tau) as the outcome, comparing a "Spatial Lag" model (which assumes nearby areas directly influence each other's outcome) against a "Spatial Error" model (which assumes nearby areas share unmeasured, correlated error terms — closer to a "shared unmeasured forcing" story).

**What we found:** The Robust Lagrange Multiplier test favored the Spatial Lag model (p=0.002, significant) over the Spatial Error model (p=0.41, not significant); the Spatial Lag model also had better fit (AIC 4.62 vs. 4.94). The spatial autocorrelation parameter (rho) was 0.650, highly significant — and notably, latitude/longitude lost their own significance once this spatial structure was accounted for.

**What it led to:** An important technical clarification for the paper: cross-sectional spatial-structure preference (this stage) is a **different kind of evidence** from the temporal-directionality question tested in Stages 9–11 and 44 — these shouldn't be conflated. This stage told us the spatial pattern has a "lag-like" cross-sectional structure; it did **not** by itself tell us anything about whether A's changes come before B's in time.

---

## Stage 22: Does Geographic Neighbor Information Actually Improve Prediction?

**Why we did this:** A statistically significant relationship doesn't automatically mean it's *useful* for anything practical. We wanted to directly test whether knowing about a patch's geographic neighbors improves our ability to predict that patch's future vegetation, compared to just using the patch's own history.

**What we did:** Built two prediction models on a held-out test period (2016–2018, not used in training): Model 1 using only a patch's own history, Model 2 adding geographic neighbor information. Compared prediction error (RMSE).

**What we found:** Adding geographic neighbor information improved prediction accuracy by only **0.09%** — essentially negligible, despite the neighbor effect being statistically very significant in Stage 6.

**What it led to:** This surprising gap between "statistically significant" and "practically useful for prediction" became an important theme — motivated testing whether a *smarter*, data-driven definition of "neighbor" (rather than assuming geography is correct) might do better (Stage 23).

---

## Stage 23: Data-Driven ("Smart") Neighbors Instead of Geographic Ones

**Why we did this:** Maybe geography isn't actually the right way to define which patches are connected — maybe letting the data itself discover which patches behave similarly (based on correlation in the training data, not assumed geographic adjacency) would work better.

**What we did:** For each patch, identified its top-5 most historically-correlated other patches (using only training-period data, to avoid unfairly peeking at the test period), and used *those* as the "neighbors" instead of geographic adjacency. Tested prediction improvement the same way as Stage 22.

**What we found:** Only 3.5% of these "data-driven" neighbor pairs were actually more than 500km apart — meaning the data-driven approach mostly just rediscovered geographic neighbors anyway. The prediction improvement was 0.32% — better than Stage 22's 0.09%, but still essentially negligible.

**What it led to:** Confirmed that even giving the model maximum flexibility to find the "best" neighbors, geographic proximity was mostly what it found anyway — and even the best-case improvement was tiny. Motivated testing a wider range of specific conditions next (Stage 24), to check if perhaps neighbor information helps more in certain specific circumstances even if not on average.

---

## Stage 24: Testing 12 Specific Conditions for Any Sign of Predictive Value

**Why we did this:** Maybe the average result (Stages 22–23) was hiding a more specific pattern — perhaps neighbor information helps more at certain prediction horizons, during certain ENSO conditions, or specifically for patches already experiencing resilience loss.

**What we did:** Tested 12 different combinations (4 prediction horizons × ENSO-strength conditions × resilience-loss status) to look for any condition where neighbor information gave a meaningfully larger improvement.

**What we found:** The single best-case improvement across all 12 conditions was **0.68%** (specifically for resilience-loss patches, at the 1-month horizon) — still very small, even in the most favorable scenario we could find.

**What it led to:** This closed the "does simple/linear neighbor information help prediction" question definitively — across every reasonable specification we tried (basic, smart, and condition-specific), the answer was consistently "no, not meaningfully." This result, combined with Stage 33's "58.6% still unexplained" finding, became the joint motivation for trying a genuinely more sophisticated tool (a Graph Neural Network) rather than continuing to test linear-model variations.

---

## Stage 25: A Model-Based Counterfactual Estimate

**Why we did this:** As one more way of characterizing the practical size of the neighbor effect, we wanted to estimate: "if a patch's neighbors were doing worse than they actually are, how much would that patch's own vegetation be expected to change?"

**What we did:** Used the fitted model to simulate a counterfactual "poor neighbor" scenario and calculated the implied change in a patch's own vegetation.

**What we found:** A small estimated effect. Across *all* observations the model-implied "neighbor effect" averages essentially zero (mean ≈ −8×10⁻⁶) — as expected, since it averages over neighbors that were sometimes above and sometimes below normal. Restricting to the scenario the research question actually asks about — months when the neighbor was in an anomalously *poor* state (neighbor VOD anomaly < 0) — the model-implied effect on the target's own next-month vegetation was **−0.001165** on average, and **−0.000953** for resilience-loss patches specifically (about 3.6% of one standard deviation of monthly VOD anomaly). Model-implied, not a directly observed causal effect. *(Recomputed from `counterfactual_results.csv`. The earlier prose labelled the −0.001165 figure "overall"; it is actually the conditional mean for the neighbor-anomalously-poor subset, which is the quantity the script deliberately reports.)*

**What it led to:** One more piece of evidence that, while statistically detectable, the neighbor effect's practical magnitude (in this raw-vegetation framing) was small — reinforcing the shift toward asking a more specific question (resilience-to-resilience, not vegetation-to-vegetation) later in the project.

---

## Stage 26: Preparing Data for the Graph Neural Network

**Why we did this:** Having established that simple linear approaches couldn't extract much predictive value from neighbor information, and that 58.6% of the original neighbor coefficient remained environmentally unexplained (a number that wouldn't actually be confirmed until Stage 33, but was already looking likely), we decided to test whether a more flexible machine-learning approach (a Graph Neural Network, or GNN) could find something the linear models missed.

**What we did:** Built the tensor datasets needed for GNN training: sequences of 12 months of historical data as input, used to predict the next month, for all 352 patches, with an accompanying adjacency matrix representing the patch network. Used 6 input features at this stage (vegetation, precipitation, temperature, soil moisture, PDSI, ENSO).

**What we found:** Successfully built the required data structures (training sequences, test sequences, adjacency matrix) — a purely technical/infrastructure stage.

**What it led to:** Set up Stage 27's actual model training.

---

## Stage 27: Building Two GNN Architectures

> **Repository audit note (2026-09-03):** No script or output file for this stage exists in this repository. `gnn_tensors.pt` (Stage 26's output) is committed, but nothing consumes it here, and the earliest committed GNN result is Stage 35's. The results described below are not reproducible from this repo and should be treated as unverified pending confirmation from the original analysis session.

**Why we did this:** To directly test whether more sophisticated spatial modeling could extract useful structure from the patch network that simpler methods (Stages 22–24) had missed.

**What we did:** Built two GNN architectures: a "Fixed Graph" model (DCRNN-style, using the real geographic adjacency structure) and an "Adaptive Graph" model (Graph WaveNet-style, which *learns* its own connections from the data rather than assuming geography is correct).

**What we found:** The models and training pipeline were built and functional (though this specific early run would later need to be redone — see Stage 34's normalization bug).

**What it led to:** The team made a deliberate decision to build the more comprehensive multi-architecture version (both fixed and adaptive graphs) rather than a single simpler model, setting up the fuller GNN investigation that continued much later in Stages 34–38 (after a substantial gap where other environmental-driver stages, 28–33, were completed first).

---

## Stage 28: Forest Fragmentation as an Additional Driver

**Why we did this:** Continuing to look for environmental drivers that might explain the neighbor-synchrony effect, we added forest fragmentation (edge density and a fragmentation index) as a new candidate, using the Hansen Global Forest Change data we'd already downloaded.

**What we did:** Calculated forest edge density and a fragmentation index for each patch, using a downsampled analysis of the Hansen data (50×50-pixel blocks, roughly 2.5km resolution) with binary forest/non-forest classification and connected-component labeling to identify fragment boundaries.

**A bug we caught and fixed here:** While building this stage, we discovered that a helper function used to calculate patch geographic boundaries had a bug affecting Stages 14–26 (a calculation that was supposed to find the spacing between patches was instead returning zero in certain cases, due to a subtle data-sorting issue). This was fixed for this stage and all subsequent stages. We did a spot-check comparing an early result recomputed with the corrected calculation against the original buggy version and found the numbers were nearly identical (0.0753 vs. 0.0751), suggesting the bug likely didn't meaningfully distort the project's qualitative conclusions — but this is worth noting as a specific, transparent methodological limitation in the paper.

**What it led to:** Set up testing whether fragmentation specifically helps explain the neighbor-synchrony question (tested as part of Stage 33's consolidated model) — and served as the first stage to use the corrected geographic boundary calculation, which all subsequent stages (29 onward) also used.

---

## Stage 29: VPD, Wind, Solar Radiation, and Cloud Cover

**Why we did this:** Continuing the systematic search for environmental drivers, now using the corrected boundary calculation (see Stage 28) and testing several additional atmospheric variables at once.

**What we did:** Added VPD (Vapor Pressure Deficit — essentially "how thirsty the atmosphere is," calculated from temperature and dewpoint using the Tetens formula), wind speed, solar radiation, and cloud fraction (from MODIS, at coarser ~1° resolution) as additional controls, building on top of the existing precipitation+temperature+soil+PDSI+ENSO baseline.

**What we found:** Baseline (with corrected boundaries): 0.0753. Adding VPD alone dropped this to 0.0563 — **a 25% reduction relative to that baseline (≈21.6 percentage points of the original 0.0876 coefficient), the single largest driver found anywhere in the entire project**, bigger even than ENSO. Adding wind on top brought it to 0.0556; adding solar radiation brought it to 0.0546 — a 27.4% reduction from the Stage-29 baseline for this group of variables (VPD + wind + solar), or about 23.6 percentage points of the original coefficient. Cloud cover showed a directional effect too, though on a smaller, different sample due to its coarser resolution. *(Percentages recomputed from `vpd_wind_cloud_radiation_results.csv` and `final_consolidated_results.csv`; the earlier "26.1%" figure was ~1.3 points low.)*

**What it led to:** VPD's outsized importance was a genuinely interesting scientific finding in its own right — physically, it makes sense that atmospheric moisture demand would be a dominant driver of vegetation stress, more so than temperature or rainfall alone. Motivated continuing to test a few more remaining candidate variables (Stages 30–32) before consolidating everything into one final combined model (Stage 33).

---

## Stage 30: Root-Zone Soil Moisture (RZSM)

**Why we did this:** Surface soil moisture (already tested in Stage 16) only reflects water near the very top of the soil; deeper, root-zone moisture might be a more relevant measure of water actually available to vegetation.

**What we did:** Added ERA5-Land root-zone soil moisture (combining two deeper soil layers, roughly 7–100cm depth) as an additional control on top of everything tested so far.

**What we found:** A further, smaller reduction — from 0.0546 to 0.0533, about a 2.4% reduction relative to the pre-RZSM coefficient. Cumulative reduction from the **original** coefficient (0.0876) at this point: about **39%**. *(The figure previously quoted as "29.2% of the original coefficient" is actually 29.2% measured against the corrected Stage-29 baseline of 0.0753, not against the original — the two denominators were mixed up. Recomputed from `rzsm_driver_results.csv` and `final_consolidated_results.csv`.)*

**What it led to:** Confirmed RZSM added modest additional explanatory value beyond surface soil moisture; set up testing one more physically-motivated variable (canopy temperature) next.

---

## Stage 31: Canopy vs. Ambient Temperature Difference (ΔT)

**Why we did this:** The difference between a forest canopy's actual temperature (measured via satellite land-surface temperature) and the surrounding air temperature can indicate physiological stress — a canopy running hotter than the air around it may indicate reduced transpirational cooling, a sign of water stress.

**What we did:** Calculated ΔT (MODIS land-surface temperature minus ERA5 2-meter air temperature) for each patch and added it as a further control.

**What we found:** A further reduction from 0.0533 to 0.0513. Cumulative reduction from the **original** coefficient (0.0876): about **41.5%** — essentially the project's final environmental-explanation total, since TWI and distance-to-disturbance (Stages 32–33) add nothing further. *(The "around 32%" previously quoted here is ~32% measured against the corrected Stage-29 baseline of 0.0753, not the original. Recomputed from `deltaT_driver_results.csv` and `final_consolidated_results.csv`.)*

**What it led to:** One more small but real contributor identified; set up testing the final planned variable (terrain wetness) next.

---

## Stage 32: Topographic Wetness Index (TWI)

**Why we did this:** Terrain shape (how flat/wet vs. sloped/well-drained an area is) can influence local water availability independent of climate — testing this required building a genuine hydrological flow-accumulation model from elevation data, a more involved calculation than the other variables.

**What we did:** Built a real D8 flow-accumulation algorithm from SRTM elevation data (including depression-filling to handle unrealistic terrain artifacts, and careful handling of flat areas) to calculate TWI for every patch.

**What we found — two separate results:**
- **TWI vs. individual patch resilience trend:** Spearman correlation = 0.348, p<0.0001 — a real, significant relationship. Interestingly, *higher* TWI (wetter, flatter terrain) was associated with *more* resilience loss — a counterintuitive direction, possibly linked to human settlement patterns (flat, water-accessible land is exactly what tends to get preferentially settled/farmed) or genuine differences between floodplain/riparian forest and well-drained terra firme forest dynamics.
- **TWI added to the neighbor-synchrony model:** exactly 0.0% change — no effect on the synchrony coefficient at all, the same pattern found earlier with human disturbance (Stage 18–20).

**What it led to:** This confirmed, for the second time in the project (after human disturbance), a now-clear pattern: static, terrain/land-use-type variables can meaningfully explain an *individual* patch's resilience trend, but never touch the *cross-patch synchrony* question, no matter which specific variable we tried. This became an important conceptual finding in its own right — individual-patch resilience and cross-patch synchrony appear to be genuinely different phenomena, governed by different factors. Set up the final consolidation stage (33).

---

## Stage 33: The Final Consolidated Environmental Model

**Why we did this:** After testing 12 individual environmental drivers one at a time or in small groups across many stages, we needed one final, comprehensive test putting all of them together simultaneously, to get a single definitive answer to "how much of the original neighbor-synchrony coefficient can environmental factors explain, in total?"

**What we did:** Built one combined regression including all 12 previously-tested drivers simultaneously (precipitation, temperature, soil moisture, PDSI, ENSO, VPD, wind, solar radiation, RZSM, ΔT, TWI, and distance-to-disturbance), using the corrected geographic-boundary calculation throughout, and tracked the neighbor coefficient's value at each step as variables were added in sequence.

**What we found:** **Original coefficient: 0.0876. Final coefficient with all 12 drivers included: 0.0513. Total reduction: 41.4%. Remaining unexplained: 58.6%.** VPD alone accounted for about **21.6 of the 41.4 percentage points** explained (coefficient 0.0753 → 0.0563) — **roughly half** of everything the 12-driver search managed to explain, and the single largest contributor. ENSO was second, at about 11 points. Precipitation, temperature, soil moisture, PDSI, wind, solar radiation, RZSM and ΔT each contributed 1–2 points or less; TWI and distance-to-disturbance added exactly 0.0 points to the final combined chain, consistent with their individual tests. *(Recomputed from `final_consolidated_results.csv`. The earlier "roughly 33 of 41.4 points / about 80%" for VPD does not reconcile with the CSV — VPD's incremental drop is 0.0189, i.e. 21.6 points of the original coefficient and ~52% of the total explained.)*

**What it led to:** This was the single most important number motivating the next phase of the project. With the majority (58.6%) of the original neighbor-synchrony pattern still unexplained even after the most thorough environmental-control effort in the project, this provided strong, well-earned justification for trying a more sophisticated tool — returning to the Graph Neural Network work (Stages 34–38) with a much stronger scientific rationale than before: there was a real, substantial pattern left to explain, not just a fishing expedition for statistical significance.


# Complete Stage-by-Stage Results Log — PART C (Stages 34–44)

*(Continuation of DEEP_STAGE_LOG_PART1.md and PART2.md — same project, same team)*

---

# PART C: THE GNN INVESTIGATION AND THE BIG REFRAME (Stages 34–44)

## Stage 34: Expanded GNN Data Preparation (15 Features)

**Why we did this:** Returning to the GNN work with the strong motivation from Stage 33 (58.6% unexplained), we wanted to give the neural network access to the full richness of environmental data collected throughout the project — not just the original 6 features from Stage 26, but all the variables tested in Stages 28–32 as well.

**What we did:** Rebuilt the GNN input data with 15 total features (vegetation, precipitation, temperature, soil moisture, PDSI, ENSO, VPD, wind-U, wind-V — kept as separate directional components rather than combined into a single speed value, cloud fraction, solar radiation, RZSM, ΔT, TWI, and distance-to-disturbance), supporting three prediction horizons (1, 3, and 6 months) from the same underlying data. Explicitly documented missing-data handling (cloud fraction had real, patchy missingness of about 20% due to its coarser satellite resolution; missing values were filled with 0, meaning "assume typical/average conditions," rather than silently dropping those observations).

**What we found:** Successfully built the expanded dataset: 175 valid 12-month sequences, split chronologically into 132 training / 24 validation / 19 test sequences.

**What it led to:** Provided the input data for Stage 35's model comparison.

---

## Stage 35: First Four-Model GNN Comparison

**Why we did this:** To directly compare, on equal footing, whether any form of spatial/graph information helps prediction — a plain non-spatial baseline, a model using real geography, a model that learns its own connections, and a model that learns which specific neighbors to pay attention to.

**What we did:** Built and trained four models sharing the same underlying temporal architecture (so any performance difference reflects the spatial mechanism specifically, not other architectural differences): a Baseline (no spatial information at all), a Fixed Graph model (uses real geographic adjacency), an Adaptive Graph model (learns its own adjacency structure from data), and an Attention-based model (learns which neighbors to weight most heavily, restricted to real geographic neighbors).

**What we found:** At the 1-month prediction horizon: the Baseline model had the **best** (lowest) prediction error of all four models — every spatial mechanism performed *worse* than simply ignoring spatial information entirely.

**What it led to:** A striking, if discouraging, initial result — but with an important caveat we flagged immediately: the training dataset was very small (132 sequences), and models with more parameters (the spatial ones) are more prone to instability/overfitting on small datasets than the simpler baseline. This meant we couldn't yet be fully confident this reflected "spatial information is genuinely useless" versus "not enough data to train these architectures reliably" — motivated testing more prediction horizons (Stage 36) and multiple random seeds (Stage 37) before drawing firm conclusions.

---

## Stage 36: Testing All Three Prediction Horizons

**Why we did this:** To check whether Stage 35's result (baseline wins) held at longer prediction horizons too, or whether spatial information might become more useful when predicting further into the future.

**What we did:** Repeated the full four-model comparison at all three horizons (1, 3, and 6 months).

**What we found:** The baseline model won at **every single horizon tested** — no exceptions. Every spatial mechanism, at every timescale, performed worse than the non-spatial baseline.

**What it led to:** A more comprehensive, more convincing version of Stage 35's finding — but the small-sample caveat still applied, motivating the seed-robustness test next (Stage 37).

---

## Stage 37: Testing Robustness Across Five Random Seeds

**Why we did this:** The strongest remaining criticism of Stages 35–36 was: "maybe this result depends on one particular, unlucky random starting point for training the spatial models." We needed to test this directly by repeating everything with multiple different random initializations.

**What we did:** Retrained all four models, at the 1-month horizon, across 5 different random seeds, and compared mean, standard deviation, best, and worst performance for each model.

**What we found:** A **more nuanced result than a clean sweep.** The baseline had the best average performance and won outright in 3 of 5 seeds (often by a wide margin). However, the attention-based model actually beat the baseline in 2 of 5 seeds (modest wins). The fixed-graph and adaptive-graph models never beat the baseline in any of the 5 seeds. All models (including the baseline) showed real seed-to-seed variability, reinforcing the small-sample-size concern.

**What it led to:** A more precise, honest conclusion: the non-spatial baseline is the stronger, more reliable model on average and in most cases, but the attention mechanism specifically showed occasional, inconsistent competitiveness — not a universal "spatial information never helps in any circumstance" story. This nuance made the attention model's *learned* patterns worth actually investigating (Stage 38), especially in the seeds where it happened to win.

---

## Stage 38: Investigating What the Attention Model Actually Learned — and Discovering a Major Bug

**Why we did this:** Given Stage 37's finding that attention occasionally outperformed the baseline in specific seeds, we wanted to understand *why* — was the attention mechanism discovering something genuinely meaningful and reproducible, or was its occasional success closer to random chance?

**What we did:** Retrained the attention model across all 5 seeds again, this time extracting and comparing the actual learned attention weights (which neighbors the model was "paying attention to," and how confidently) between the seeds that had beaten the baseline and the seeds that hadn't.

**A major bug we discovered here:** The attention patterns across different, independently-initialized models looked suspiciously identical, regardless of random seed — which shouldn't happen if the model were learning something genuinely from the data. Investigating this led to discovering that **the input features had never been properly normalized** — one feature (solar radiation, with a raw value in the hundreds of millions) was on a scale roughly 10 million times larger than most other features (like VOD, on a scale of 0.1). This meant that any neural network layer processing these raw features would be almost entirely dominated by the solar radiation feature's sheer numerical size, regardless of what the model actually learned to value — explaining the suspiciously uniform, seed-independent attention patterns.

**What it led to:** This was a significant finding requiring the whole GNN investigation (Stages 34–37) to be re-run properly with normalized data, since this bug potentially affected the validity of all four models' results, not just the attention model specifically. Rather than just patch Stage 38 in isolation, the team paused and rebuilt the data pipeline with proper feature normalization (Stage 34, corrected version) before re-running everything.

---

## Stage 34 (Corrected) & Stage 35 (Corrected): Re-running with Properly Normalized Features

**Why we did this:** To fix the normalization bug discovered in Stage 38 and confirm whether the earlier "baseline wins" conclusion (Stages 35–37) still held once this real methodological flaw was corrected — a genuine "sanity check" before trusting any of the earlier GNN results.

**What we did:** Added proper feature standardization (z-scoring every feature to a comparable scale, calculated using only the training-period data to avoid leaking future information into the process) to the data-preparation pipeline, saved as a new file so the original, pre-fix results remained available for comparison, then reran the four-model comparison on this corrected dataset.

**What we found:** **Outcome A (as anticipated in the pre-specified decision framework): the baseline still won clearly, even after the normalization fix.** Baseline RMSE (0.684) was still the best; every spatial model was still slightly worse (0.693–0.713).

**What it led to:** This confirmed that the original "spatial information doesn't help prediction" conclusion was **not** an artifact of the normalization bug — it held up under the methodologically correct version too, making the conclusion *more* credible, not less, since it survived a genuine stress-test of the pipeline. This gave the team confidence to proceed with re-running the full multi-horizon and seed-robustness tests on the corrected data (below).

---

## Stage 36 (Corrected) & Stage 37 (Corrected): Re-running Multi-Horizon and Seed-Robustness on Normalized Data

> **Repository audit note (2026-09-03):** when this repository was audited, the corrected/normalized versions of these two stages did **not** exist — only the single-horizon Stage 35 comparison had been redone on normalized data. The original prose below claimed the baseline won every horizon "more decisively" on normalized data; there was no file backing that. The reruns were executed during the audit using `src/36_gnn_multi_horizon_normalized.py` and `src/37_gnn_seed_robustness_normalized.py` (identical model code and procedure to the originals, loading `gnn_tensors_expanded_normalized.pt`), producing `gnn_multi_horizon_results_normalized.csv` and `gnn_seed_robustness_results_normalized.csv`. The "What we found" section has been rewritten to match those actual outputs, which do **not** support the original claim.

**Why we did this:** To get the multi-horizon and multi-seed robustness tests onto the corrected, feature-standardized pipeline, matching the Stage 35 correction.

**What we did:** Reran the full 3-horizon comparison and the 5-seed (1-month-horizon) robustness comparison on `gnn_tensors_expanded_normalized.pt`, with model code, training procedure, and train/val/test splits unchanged.

**What we found — normalization removes the baseline's clean sweep:**

*Multi-horizon* (`gnn_multi_horizon_results_normalized.csv`, test RMSE):

| Horizon | Baseline | Fixed-geographic | Adaptive/learned | Attention | Best spatial vs baseline |
|--------:|---------:|-----------------:|-----------------:|----------:|:------------------------:|
| 1 month | **0.6844** | 0.6949 | 0.7132 | 0.6933 | −1.3% (baseline best) |
| 3 months | 1.1562 | **1.0692** | 1.0739 | 1.0774 | **+7.5% — all three spatial models beat the baseline** |
| 6 months | **1.1474** | 1.1901 | 1.2208 | 1.1960 | −3.7% (baseline best) |

*Seed robustness* (`gnn_seed_robustness_results_normalized.csv`, 1-month horizon, 5 seeds, test RMSE):

| Model | mean | std | best | worst | outright wins (of 5 seeds) |
|---|---:|---:|---:|---:|:--:|
| Fixed-geographic GNN | 0.6966 | 0.0070 | 0.6848 | 0.7069 | 2 |
| Attention GNN | 0.6972 | 0.0123 | 0.6782 | 0.7137 | 2 |
| Baseline (no spatial) | 0.6976 | 0.0083 | 0.6844 | 0.7089 | 1 |
| Adaptive/learned GNN | 0.7079 | 0.0090 | 0.6977 | 0.7198 | 0 |

On the corrected pipeline the baseline is **no longer the best model**: it has the lowest RMSE in only 1 of 5 seeds (seed 789), and its mean RMSE (0.6976) is marginally *higher* than the fixed-geographic GNN (0.6966) and the attention GNN (0.6972). Those three sit within ~0.001 RMSE of each other and well inside the seed-to-seed spread (std ≈ 0.007–0.012), so they are best read as statistically indistinguishable at the 1-month horizon. The adaptive/learned GNN is consistently the weakest.

**What it led to — corrected conclusion:** the honest statement is *no consistent or meaningful advantage in either direction*. Spatial structure does not reliably improve prediction (the linear tests in Stages 22–24 found this more definitively), but on the corrected pipeline it is not reliably worse either, and at the 3-month horizon the fixed-geographic GNN is clearly ahead (−7.5% RMSE). The earlier conclusion — "the baseline beats every spatial mechanism at every horizon and every seed" — was an artifact of the un-normalized feature scales and does not survive the fix. This still supports moving away from a prediction framing toward the resilience-to-resilience question (Stage 41 onward), but on the grounds that *no* approach predicts these 17–19-sequence test sets well, not that spatial information is worthless.

---

## Stage 39: The First Distance-Banded, Fully-Controlled Residual Effect Model

> **Repository audit note (2026-09-03):** No script or output file for this stage exists in this repository. There is no committed raw-VOD distance × lag table with the full 12-driver control set. The results described below are not reproducible from this repo and should be treated as unverified pending confirmation from the original analysis session.

**Why we did this:** Before making the full conceptual pivot to a resilience-based outcome (which happened in Stage 41), we first built one more, more complete version of the original vegetation-based analysis — combining the distance-banding approach (Stage 7) with the full 12-driver environmental control set (Stage 33) and multiple time lags simultaneously, something no earlier stage had done all together.

**What we did:** For each of several distance bands and lags (1, 2, 3, 6 months), tested the neighbor-effect coefficient while controlling for all 12 environmental drivers at once, and reported confidence intervals explicitly (not just p-values).

**What we found:** Produced a full table of coefficients, confidence intervals, and significance levels across the distance/lag grid — providing the most complete picture yet of how the (still vegetation-based, not yet resilience-based) neighbor relationship varied across distance and time delay.

**What it led to:** Set the stage for the team's realization (prompted by user/team discussion) that even this thorough analysis was still fundamentally testing "does A's current vegetation predict B," not "does A's resilience *loss* predict B" — motivating the conceptual reframe that defines Stage 41 onward.

---

## Stage 40: Adding the Neighbor's Own Environmental Controls, and Testing Interaction Effects

> **Repository audit note (2026-09-03):** No standalone script or output file for this stage exists in this repository. The both-sided-controls and standardized-interaction methodology it describes does appear in `src/42_resilience_to_resilience.py`, but the raw-VOD version of the analysis described here has no committed output. Treat its specific results as unverified pending confirmation from the original analysis session.

**Why we did this:** Two remaining gaps were identified in the team's methodology review: (1) earlier models only controlled for the *target* patch's own environment, not the *source/neighbor* patch's environment — leaving room for a subtle confound if A and B experience similar-but-not-identical local weather; and (2) every model treated the neighbor effect as a fixed, constant number, never testing whether the relationship's *strength* depends on environmental conditions (e.g., is A's influence on B stronger during drought?).

**What we did:** Rebuilt the distance/lag grid model to control for **both** the target's and the neighbor's local environment simultaneously (closing gap 1), and separately tested interaction effects — whether the neighbor coefficient's strength changed depending on each of the 12 environmental drivers, one at a time (closing gap 2), using standardized variables so the interaction coefficients were comparable across drivers with very different units.

**What we found:** Extending the environmental controls to both sides didn't meaningfully change the core distance/lag pattern found in Stage 39. The interaction-effects testing set up the more meaningful "susceptibility" analysis that would follow in Stage 42, once the outcome variable itself had been properly reframed to resilience.

**What it led to:** Confirmed the "both-sided controls" methodology that would be carried forward into every subsequent stage of the project, and validated the interaction-testing approach that became central to Stage 42's most important finding (the susceptibility pattern).

---

## Stage 41: Building the Genuine, Time-Varying Resilience Metric — The Conceptual Turning Point

**Why we did this:** After extensive team discussion, it became clear that every model since Stage 6 had actually been testing "does A's *current vegetation reading* predict B" — not the project's real question, "does A *losing resilience* affect B." A patch having one unusually good or bad month is a different thing from a patch's underlying ability to recover from disturbance declining over time. This stage fixed that fundamental mismatch.

**What we did:** Built a genuine, time-varying resilience indicator by calculating rolling AR(1) with a 24-month window, stepped monthly (as opposed to Stage 5's single, static 60-month AR(1) covering the whole study period) — giving every patch a month-by-month resilience *state*, which could then be used as the actual "neighbor" input variable in the model, in place of raw vegetation.

**A bug we caught and fixed here:** The first version of this model accidentally included ENSO (ONI) twice in the same regression — labeled separately as "belonging to" the target and "belonging to" the neighbor — even though ENSO is a single, basin-wide value identical for every patch on a given date. This created a statistical problem (the two "copies" were perfectly duplicated, making their individual effects impossible to interpret meaningfully, even though the regression could technically still run). Fixed to include ONI exactly once, as a genuine single global control.

**What we found:** Using this properly-built resilience metric as the treatment variable, distance-banded at 1, 2, 3, and 6-month lags: a real, meaningful pattern emerged, distinct from anything found using raw vegetation.

**What it led to:** This stage fundamentally redefined the "A" variable for the rest of the project — every subsequent stage (42 onward) used this genuine, time-varying resilience metric rather than raw vegetation, which is what allowed the much richer findings of Stage 42 to emerge.

---

## Stage 42: The Reframed Model — the Richest Result in the Entire Project

**Why we did this:** Now equipped with a properly-built, time-varying resilience metric (Stage 41) and a fully-controlled, both-sided environmental model (Stage 40's methodology), we could finally run the analysis the project had actually been trying to build toward all along.

**What we did:** Tested whether a target patch's neighbors' *resilience state* (not raw vegetation) predicted that target's *own future resilience*, across all distance bands and lags, with full environmental controls on both sides — and separately tested whether the target's own local environmental conditions modulated how strongly it was affected by its neighbors (the "susceptibility" test).

**What we found — two major results:**
1. **Real distance decay, for the first time in the whole project:** a significant positive relationship from 75km out to about 650km at the 1-month lag, fading to non-significant beyond that — something no earlier, raw-vegetation-based analysis had ever found.
2. **The most consistent finding in the entire project:** factors about the *target* patch's own conditions modulated the neighbor-resilience effect. **Soil moisture, TWI, and distance-to-disturbance were significant at all four lags tested (1, 2, 3, and 6 months); root-zone soil moisture was significant at lags 1–3 but not at lag 6 (p = 0.46).** The pattern: patches with better water buffering, or sitting deeper in intact forest, appeared more independent/buffered from their neighbors' resilience states; patches closer to disturbance or in drier terrain were more tightly coupled to what was happening around them — a physically sensible, genuinely interesting ecological finding. *(Lag-by-lag significance taken from `resilience_to_resilience_susceptibility.csv`; the earlier prose said all four factors held "at every single lag", which is not true for root-zone soil moisture at lag 6.)*

**What it led to:** Given how rich and novel these findings were, the team decided to prioritize validating them thoroughly before treating them as final — leading directly to the window-length sensitivity check (Stage 43).

---

## Stage 43: Checking the Window-Length Choice Wasn't Driving the Result

**Why we did this:** The 24-month window used to build the rolling resilience metric (Stage 41) was a somewhat arbitrary choice. Before fully trusting Stage 42's rich findings, we needed to check they weren't simply an artifact of that specific window length.

**What we did:** Reran the core distance/lag analysis using 36-month and 48-month rolling windows in addition to the original 24-month version, and compared whether the same qualitative patterns held across all three.

**What we found:** Both key patterns held up well across all three window choices. The near-distance positive effect (at the 1-month lag) remained present at 24 and 36 months, and (somewhat weaker but still present, same direction) at 48 months. More strikingly, a far-distance (800–1,100km) *negative* pattern that had also emerged — significant at both the 3-month and 6-month lags — was significant across **all six tested combinations** (24/36/48-month windows × 3-month/6-month lags), and its magnitude actually got *stronger*, not weaker, as the window length increased (e.g., at the 6-month lag and 800–1,100km, the effect roughly doubled going from the 24-month to the 48-month window). A result that gets *clearer* with more stable estimation, rather than washing out, is strong evidence of a genuine underlying signal, not noise from an arbitrary methodological choice.

**What it led to:** This validated both of Stage 42's key patterns as robust, real findings rather than artifacts of the specific 24-month window choice — and specifically drew the team's attention to the newly-emerging far-distance negative pattern as worth its own dedicated investigation, since it had turned out to be even more robust than initially expected. This set up the directionality test (Stage 44) applied specifically to this now-validated two-part spatial pattern.

---

## Stage 44: Does A Come Before B, or Do They Just Move Together?

**Why we did this:** With both the near-distance and far-distance patterns now validated (Stage 43), the key remaining question was whether either of them looked like genuine directional influence (A's changes preceding and predicting B's) or simple synchrony (both patches responding to the same external conditions at around the same time).

**What we did:** For both the near band (75–450km) and the far band (800–1,100km), compared the real "forward" relationship (A's resilience now → B's resilience later) against a deliberately backward-in-time "placebo" test (using the neighbor's *future* state to try to "predict" the target's *past* state — a test that should show little to nothing if the real relationship is genuinely forward-in-time). Also tested, for every distance band, which time lag showed the *strongest* effect, to see if that "peak lag" shifted to longer delays as distance increased (which would suggest something like a literal traveling signal).

**A real bug we caught and fixed while building this:** an early version of the backward "placebo" test accidentally set the outcome variable equal to one of its own predictor variables (a trivial mathematical identity), producing meaningless, artificially "confident" results. Caught because the numbers looked suspicious, and fixed to properly mirror the forward model with the time direction genuinely reversed.

**What we found — a two-regime pattern:**
- **Near band (75–450km in the formal placebo test):** forward and backward were similarly strong at the 1-, 2- and 3-month lags → more consistent with **synchrony** (shared external forcing) than with genuine directional influence. (At the 6-month lag they diverged — forward significant, backward not — but this is not the regime the near-band story rests on.)
- **Far distances (800–1,100km):** the forward test was significantly negative at the **3-month and 6-month lags**, while the backward (placebo) test was null at those lags → a real, meaningful **asymmetry** at 3 and 6 months, more consistent with genuine directional influence than pure synchrony. Note the asymmetry is *not* clean at every lag: at the 1-month lag the far-band backward test was itself significant (coef ≈ +0.0081, p ≈ 0.009) while the forward test was not. The forward-significant / backward-null signature holds specifically at the 3- and 6-month lags.
- **Bonus finding on "peak lag":** every distance band from **150km to 650km** peaked at the 1-month lag (fast response); the nearest band (75–150km) peaked at the 2-month lag; every band from 650–1,100km peaked at the 6-month lag (slow response) — a clean step-change around 650km, not a smooth, gradual transition (which is also why the formal statistical test for "does peak lag increase smoothly with distance" wasn't significant — the pattern is a genuine step, not a gradient).

*(Bands and significance recomputed from `directionality_test_results.csv` and `propagation_speed_results.csv`. The earlier prose said "similarly strong at every lag" for the near band and the backward test "stayed weak and non-significant throughout" for the far band and "every band from 75–650km peaked at the 1-month lag" — each of these had one exception, corrected above.)*

**What it led to:** This was a major finding in its own right — establishing that the near and far distance regimes appear to work by genuinely different mechanisms (fast/symmetric/positive vs. slow/asymmetric/negative), not just different strengths of the same thing. Given how interesting and novel the far-distance directional-looking pattern was, the team decided this deserved serious, dedicated stress-testing before it could be trusted as a real scientific finding — setting up the extensive robustness arc that followed (Stages 45–48).


# Complete Stage-by-Stage Results Log — PART D (Stages 45–50 and Final Conclusion)

*(Continuation of DEEP_STAGE_LOG_PART1.md, PART2.md, and PART3.md — same project, same team)*

---

# PART D: STRESS-TESTING AND THE MECHANISM SEARCH (Stages 45–50)

## Stage 45: Progressively Stronger Controls — Trying to Make the Far-Distance Result Disappear

**Why we did this:** Having found an interesting, directional-looking far-distance pattern (Stage 44), the scientifically responsible next step was not to celebrate it, but to actively try to eliminate it — testing whether it survives increasingly demanding controls, or whether it's actually explained by something we hadn't yet accounted for.

**What we did:** For the far band (800–1,100km) at the 3-month and 6-month lags, tested the neighbor-resilience coefficient through four progressively stricter specifications: (1) no controls at all, (2) adding local both-sided environmental controls, (3) adding regional (whole-study-area) climate controls, and (4) adding explicit latitude/longitude for both patches — the strictest, most demanding version. Finally, reran the forward-vs-backward placebo test (from Stage 44) at this strictest specification, to see if the directional asymmetry held up even under maximum scrutiny.

**What we found:**
- **At the 3-month lag: the effect survived all four steps**, including the strictest geographic-control specification (coefficient ≈ −0.0185, p≈0.046 — still significant, though only just).
- **At the 6-month lag: the effect survived local and regional climate controls, but dropped just below conventional statistical significance once explicit latitude/longitude was added** (p≈0.067). We were careful with the exact wording here: the correct, precise statement is "adjustment for geographic position attenuated the association sufficiently that it was no longer statistically significant" — not "geography explained the effect," which would overstate what the test actually showed.
- **The placebo test remained null at both lags even at this strictest specification**, and notably the backward-test coefficient actually flipped to a *positive* sign (versus the forward test's negative sign) — this specific forward-negative/backward-positive-and-null pattern is exactly the kind of signature that distinguishes genuine directional influence from simple synchrony, and it held up under the most demanding test applied so far.

**What it led to:** This was, at the time, the single most rigorously-tested result in the whole project — having survived local environment, regional climate, geographic confounding, and a placebo check simultaneously (at least at the 3-month lag). This gave the team enough confidence in the result to design a set of formal, pre-specified robustness checks next (Stage 46), rather than continuing to add ad-hoc controls indefinitely.

---

## Stage 46: Seven Pre-Specified Robustness Checks

**Why we did this:** An important scientific principle: robustness checks should be decided *before* looking at how they turn out, not selected afterward based on which ones happen to support the desired conclusion. We designed four categories of checks in advance, specifically targeting the Stage 45 3-month-lag result (the one that had survived everything so far).

**What we did:** Tested (1) two alternative distance-band cutoffs (750–1,000km and 900–1,100km, not just the original 800–1,100km, to check the result wasn't dependent on an arbitrary boundary choice), (2) two alternative resilience metrics (a 36-month AR(1) window, and a completely different metric — rolling standard deviation, a second classic "early warning signal" from the same underlying theory as AR(1), to check the result wasn't dependent on the specific choice of AR(1) itself), (3) adding month-of-year and year statistical controls (to rule out remaining seasonal or year-specific effects), and (4) testing an alternative, more conservative way of calculating statistical uncertainty — **two-way clustering**, which accounts for the fact that our data isn't fully independent (the same patches and same time periods show up repeatedly across many observations).

**What we found:** **7 of 7 specifications kept the same negative sign as the original result. 6 of 7 remained statistically significant.** The alternative distance bands were both significant with similar magnitude to the original (not sensitive to the exact cutoff). Both alternative resilience metrics were significant (not an artifact of the specific AR(1)/24-month choice). The month/year controls were significant too, though with a notably larger coefficient than the others — flagged as a caveat, possibly reflecting some collinearity between the many year-dummy variables and the existing regional climate controls, rather than necessarily a "truer" bigger effect. **The one exception was two-way clustering (p=0.536, not significant)** — but critically, the actual estimated effect size (the point estimate) was identical to the original result; only the calculated uncertainty widened. The honest, precise way to describe this: "under the most conservative inference method tested, precision was insufficient for conventional statistical significance, despite an unchanged point estimate" — not "the effect disappeared."

**What it led to:** Overall, this represented substantial, if not perfect, robustness — six specifications held, one (arguably the most statistically appropriate one, given the data's structure) did not. This honest mixed result set up the next, even more adversarial round of testing (Stage 47), designed specifically to see if the result would survive genuinely hostile tests, not just alternative specifications of the same basic approach.

---

## Stage 47: Two Adversarial Tests Designed to Break the Result

**Why we did this:** Stage 46 tested reasonable *alternative specifications*. This stage went further, designing tests specifically intended to distinguish "a real, specific spatial relationship" from "something that looks similar but isn't actually about the specific geographic pairing at all."

**What we did:**
- **A fake-neighbor permutation test:** randomly shuffled which patch's data got treated as which target's "neighbor" (100 different random reshufflings, each preserving the same overall statistical structure but breaking the true geographic correspondence), and compared the real result against this randomized null distribution — essentially asking "is the real geographic pairing meaningfully different from arbitrary, spatially-meaningless pairing?"
- **A distance-decay curve fit:** attempted to fit a standard, smooth mathematical decay curve (of the form used in physics/ecology for signals that weaken continuously with distance) across all 9 distance bands tested throughout the project, to formally characterize whether the pattern looked like continuous spatial decay.

**What we found:**
- **The fake-neighbor test found the real coefficient was NOT statistically distinguishable from random shuffling (permutation p=0.287).** Randomly-shuffled, spatially-meaningless pairings produced comparably negative average effects (mean −0.0146 versus the real −0.0185) — a real, honest weakening of the claim that the *specific* geographic pairing mattered.
- **The decay-curve fit failed outright** — a nonsensical result (R² = −0.56, actually worse than simply drawing a flat horizontal line through the data, with wildly unstable fitted parameters) — confirming that whatever pattern exists is not a smooth, continuous decay with distance; it looks more like a step or threshold effect concentrated in the far band specifically.

**What it led to:** Combined with Stage 46's two-way clustering result, this represented a genuine, honestly-reported *weakening* of the far-distance finding — the project's hardest, most adversarial tests did not simply confirm the earlier, more favorable results, and the team reported this plainly rather than only emphasizing the results that supported the hypothesis. At this point, the honest running conclusion was: "a real, statistically-detectable pattern exists and survives most tests, but the two hardest tests (two-way clustering, fake-neighbor permutation) suggest real caution is warranted about the strongest ('specific directional spatial influence') interpretation." This motivated trying one more, methodologically different approach — moving from *averaged* neighbor exposure to *genuinely individual pairs* (Stage 48) — to see if a more targeted test told a different story.

---

## Stage 48: The Genuinely Pairwise Model — A Major, Strengthening Update

**Why we did this:** Every model up to this point — including the adversarial tests in Stage 47 — had used *averaged* neighbor exposure (if a target patch had 20 far-away neighbors, their resilience states were averaged together into one number). This averaging could be smoothing over real, specific pair-level relationships. We wanted to test the far-distance pattern using genuine, individual (source, target) pairs as the unit of analysis, preserving the specific pair-level variation the averaging approach had discarded.

**What we did:** Built a dataset treating every individual eligible (source, target) pair within the 800–1,100km band as its own separate observation across time, rather than averaging. This required identifying all 26,860 eligible directed pairs, and — because the full dataset (~4.5 million rows) was too large to fit the required statistical calculations in memory — randomly sampling a reproducible set of 3,000 pairs (using a fixed random seed for full reproducibility), while keeping each sampled pair's *entire* multi-year monthly time history intact (i.e., randomly sampling *pairs*, never randomly sampling *months*). Tested the resulting model under three different ways of calculating statistical uncertainty: clustering by target patch only, clustering by source patch only, and — the most conservative — two-way clustering by both target and source simultaneously.

**What we found:** **The coefficient was significant under all three clustering approaches, including two-way clustering (β ≈ −0.00723, p = 0.0008).** This is the *opposite* update from Stage 47 — a more specific, more targeted, arguably more methodologically appropriate test found a *stronger* result, not a weaker one, than the averaged approach had. The effect was also remarkably consistent across the whole 800–1,100km range: when split into three 100km-wide sub-bands (800–900km, 900–1,000km, 1,000–1,100km), all three showed similar, individually significant coefficients (−0.0081, −0.0066, −0.0070 respectively) — not concentrated in just one small slice of the range. We also independently verified our statistical implementation of two-way clustering was numerically correct by comparing it against a completely independent calculation method and confirming the results matched to floating-point precision.

**What it led to:** This was a genuinely important update to the project's running conclusion. Rather than treating Stage 47's more cautious findings as final, this more rigorous, more targeted test suggested the far-distance pattern is real and specific to actual pair-level relationships, not merely an artifact of averaging or something equally well-explained by random pairing. The two tests (47's averaged/permutation approach and 48's genuine pairwise approach) are complementary rather than contradictory — they ask subtly different questions, and the more targeted one gave the stronger answer. This became the project's headline statistical result, and motivated a focused search for what physical mechanism might explain it (Stages 49–50).

---

## Stage 49: Does Wind Direction Matter? (First-Order Mechanism Plausibility)

**Why we did this:** With a robust statistical A→B relationship established (Stage 48), the team considered building a full physical mechanism model (e.g., tracking actual atmospheric moisture transport from source to target patches) — but concluded this would require substantial new data (multi-level humidity and wind measurements, evapotranspiration data) and specialized atmospheric modeling methodology well beyond the project's realistic scope and timeline. Instead, the team deliberately scoped down to the smallest tractable, honest test of mechanism plausibility achievable with data already collected: **if moisture really is being carried from source to target by wind, the relationship should be stronger specifically when the wind actually blows in that direction.**

**What we did:** For every pair in the Stage 48 pairwise dataset, calculated the geographic compass bearing from source to target, compared it to the actual wind direction at the source (from ERA5 wind data already collected for the project), and computed a "wind alignment" measure ranging from +1 (wind blowing directly toward the target) to −1 (directly away). Tested whether this alignment measure changed the strength of the source-to-target relationship (as a continuous statistical interaction), and separately compared a simpler split into "wind-aligned" versus "wind-opposed" pair-months.

**What we found:** A genuinely mixed, internally inconsistent result. **The continuous interaction test was statistically significant (p=0.028) and in the direction consistent with wind-transport** (the effect became more negative as alignment increased). **But the simpler aligned-vs-opposed comparison showed the opposite pattern** — wind-aligned pairs actually showed a *weaker*, non-significant effect (p=0.235), while wind-opposed pairs showed a *stronger*, significant effect (p=0.0009) — the reverse of what a simple wind-transport mechanism would predict.

**What it led to:** The team explicitly declined to claim this test supported the wind-transport hypothesis, given the internal contradiction between the two ways of testing the same idea — reported it honestly as mixed/inconclusive rather than picking whichever half of the result looked more favorable. This motivated one more, slightly more ambitious (but still tractable) attempt at a mechanism test, this time incorporating precipitation directly (Stage 50).

---

## Stage 50: A Pre-Registered Moisture-Availability Proxy Chain — The Final Mechanism Test

**Why we did this:** Rather than abandoning the mechanism question entirely after Stage 49's mixed result, the team designed one more tractable test — using only data already in the project's pipeline (no new downloads) — to test a more complete, if still simplified, version of the hypothesized physical chain: does a source patch's vegetation health, combined with favorable wind and water availability, actually relate to increased rainfall at the target, which then relates to the target's own future resilience?

**What we did:** Constructed a "moisture availability" proxy variable by multiplying together three already-available quantities: the source patch's resilience state (used as a stand-in for vegetation health/transpiration capacity, since no direct evapotranspiration data existed in the project), the wind-alignment measure from Stage 49, and the source patch's own root-zone soil moisture (water availability). Before running any analysis, the team pre-specified exactly what would need to be true to call this supportive evidence: (1) this proxy variable must significantly predict the target's *future precipitation*, and (2) the target's *own* precipitation must then significantly predict the target's *own future resilience*. Both links were tested directly using the existing precipitation (CHIRPS) and resilience data already in the pipeline.

**What we found:** **Neither link was statistically significant.** The moisture-availability proxy did not significantly predict target future precipitation (p=0.30). Target's own precipitation did not significantly predict target's own future resilience (p=0.48). A related check — whether adding this proxy variable to the original Stage 48 pairwise model reduced (mediated) the direct source-to-target coefficient — also found essentially no meaningful reduction (only about 1%), even though the proxy variable itself showed a marginally significant coefficient in that combined model; the team specifically flagged this marginal significance as likely reflecting some other, uninterpretable correlation rather than genuine supporting evidence, since real mediation would be expected to substantially shrink the direct effect, which it did not.

**What it led to:** This was treated as a complete, legitimate, and honestly pre-specified negative result — not a failure. The core Stage 48 statistical relationship (source resilience state relating to target future resilience, at 800–1,100km) remained real, strong, and well-supported by extensive testing; the team was simply unable to identify, using the data realistically available to the project, *why* this relationship exists. This closed the analytical phase of the project. The team explicitly declined to pursue a much larger, ~13-stage roadmap that would have required building a full atmospheric moisture-transport model from new data, judging it infeasible given the project's submission deadline, and instead treated the physical mechanism as an honestly-reported open question for future research.

---

# FINAL SYNTHESIS

## The Core, Defensible Finding (build the abstract around this)

> Patch-level resilience states exhibit a statistically detectable relationship across distant patches (800–1,100 km apart), even after controlling for a comprehensive set of measured local, regional, and geographic environmental factors. The most rigorously tested estimate — from a genuinely pairwise model with two-way clustered statistical inference — is **β ≈ −0.00723 (p = 0.0008)**. The direction and approximate magnitude of this association were consistent across dozens of alternative model specifications (distance-band cutoffs, resilience-metric definitions, temporal controls, inference methods). The relationship shows a temporally asymmetric signature (a real forward effect, a null backward placebo) more consistent with directional influence than pure shared-forcing synchrony, particularly at the 3-month lag. However, the specific physical transmission mechanism behind this relationship could not be identified with the available data — a first-order wind-alignment test gave mixed, internally inconsistent results, and a constructed moisture-availability proxy chain did not confirm either of its two pre-specified hypothesized links.

## Two Other Findings Worth Featuring Prominently

**The two-regime spatial pattern (a clean, visually compelling finding for a figure):**
- Near/intermediate distances (75–650 km): fast (peaks at 1-month lag), positive, largely symmetric forward/backward → consistent with synchrony (shared response to regional conditions).
- Far distances (800–1,100 km): slow (peaks at 6-month lag), negative, asymmetric forward/backward → more consistent with a genuine directional component.

**The susceptibility finding (the most consistent result in the project):** in the Stage 42 interaction model, a target patch's own soil moisture, terrain wetness (TWI), and distance from human disturbance **significantly modulate how strongly it's affected by its neighbors' resilience state — at all four time lags tested (1, 2, 3, 6 months). Root-zone soil moisture shows the same modulation at lags 1–3 but not at lag 6 (p = 0.46).** Patches with better water buffering or farther from disturbance appear more independent; drier or disturbance-adjacent patches are more tightly coupled to their surroundings. This is the finding that reproduced most consistently across lags; note the dedicated robustness stages (43, 46) re-tested the far-distance *main* effect, not the susceptibility interactions, so the susceptibility result rests on the Stage 42 model alone.

## What NOT to Claim (important for credibility with judges)

- We did **not** prove that one patch physically causes changes in another — this is an observational study; everything reported is a statistical association after controlling for measured confounders.
- We did **not** identify the physical transmission mechanism — explicitly report this as an open question for future work requiring dedicated atmospheric moisture-tracking data.
- The far-distance finding, while ultimately well-supported by the most targeted test (Stage 48), did **not** survive every adversarial test along the way (Stage 46's two-way clustering on the averaged model, and Stage 47's fake-neighbor permutation both showed real weaknesses at that stage of testing) — report this transparently as part of the narrative, not as a footnote to hide.

## Why the Honest "Failures" Are Actually a Strength

Nearly every stage in the second half of this project was specifically designed to try to break the project's own most interesting result — through placebo tests, adversarial permutation tests, alternative specifications, progressively stronger confound controls, and a pre-registered mechanism search that ultimately came back negative. That the team can show, with real numbers, exactly what survived rigorous testing and what didn't — including multiple real coding bugs that were caught, diagnosed, and transparently corrected along the way — is a more credible and more scientifically mature research product than a project that only reports favorable results. This is worth stating explicitly and proudly in the Discussion section.
