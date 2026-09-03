# References

Literature and data sources that motivated the methodology. Referenced from
[`Readme.md`](../Readme.md) and [`full_methodology.md`](full_methodology.md).

> **Unverified item:** entry **10** below (the "CaST" causal spatiotemporal GNN
> reference) could **not** be confidently matched to a specific paper. Do not cite
> it as-is — check it against the original notes/reading list first, rather than
> guessing at the source.

## Core Academic References

1. **Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H., van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009).** Early-warning signals for critical transitions. *Nature, 461*(7260), 53–59. https://doi.org/10.1038/nature08227
   → Foundation for using AR(1) as a resilience/early-warning indicator.

2. **Boulton, C. A., Lenton, T. M., & Boers, N. (2022).** Pronounced loss of Amazon rainforest resilience since the early 2000s. *Nature Climate Change, 12*(3), 271–278. https://doi.org/10.1038/s41558-022-01287-8
   → Direct methodological precedent (AR(1) on VOD in the Amazon); our disturbance-distance finding was in the opposite direction to theirs.

3. **Hirota, M., Holmgren, M., Van Nes, E. H., & Scheffer, M. (2011).** Global resilience of tropical forest and savanna to critical transitions. *Science, 334*(6053), 232–235. https://doi.org/10.1126/science.1210657
   → Background on multi-stable-state theory for the Amazon–Cerrado transition zone.

4. **Elhorst, J. P. (2010).** Spatial Panel Data Models. In M. Fischer & A. Getis (Eds.), *Handbook of Applied Spatial Analysis* (pp. 377–407). Springer. https://doi.org/10.1007/978-3-642-03647-7_19
   → Basis for the spatial lag vs. spatial error model comparison (Stage 21). Note: this is a book chapter, not a standalone journal article.

5. **Dakos, V., van Nes, E. H., Donangelo, R., Fort, H., & Scheffer, M. (2010).** Spatial correlation as leading indicator of catastrophic shifts. *Theoretical Ecology, 3*, 163–174. https://doi.org/10.1007/s12080-009-0060-6
   → Background for spatial early-warning-signal theory (Moran's I usage, Stage 15).

6. **Zemp, D. C., Schleussner, C.-F., Barbosa, H. M. J., Hirota, M., Montade, V., Sampaio, G., Staal, A., Wang-Erlandsson, L., & Rammig, A. (2017).** Self-amplified Amazon forest loss due to vegetation–atmosphere feedbacks. *Nature Communications, 8*, 14681. https://doi.org/10.1038/ncomms14681
   → Correction from an earlier draft answer: this is *Nature Communications*, not GRL. Motivated the moisture-transport hypothesis (Stages 49–50); uses a complex-network approach linking forest patches by observation-based atmospheric water fluxes — exactly the kind of model this project explicitly scoped out as infeasible.

7. **Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018).** Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. *International Conference on Learning Representations (ICLR).*
   → Architectural basis for the "Fixed Graph" GNN model.

8. **Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019).** Graph WaveNet for Deep Spatial-Temporal Graph Modeling. *Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI-19)*, 1907–1913. https://doi.org/10.24963/ijcai.2019/264
   → Architectural basis for the "Adaptive Graph" (learned-adjacency) GNN model.

9. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics, 378*, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
   → Background reading only — **not implemented** in this project. Important not to cite this as if a physics-informed loss term was used.

10. **"CaST" (causal spatiotemporal GNN reference) — UNVERIFIED.** A specific, confidently-matching paper could not be identified. **Track down the exact source from the original notes/reading list before citing it**, rather than relying on memory.

## Data Source Citations

1. **Zotta, R.-M., Moesinger, L., van der Schalie, R., Vreugdenhil, M., Preimesberger, W., Frederikse, T., De Jeu, R., & Dorigo, W. (2024).** VODCA v2: multi-sensor, multi-frequency vegetation optical depth data for long-term canopy dynamics and biomass monitoring. *Earth System Science Data, 16*(10), 4573–4617. https://doi.org/10.5194/essd-16-4573-2024
   Note: this is the v2 paper, matching the VODCA v2 dataset used here. The original v1 paper is Moesinger et al. 2020, *ESSD 12*(1), 177–196, if the earlier version also needs citing.

2. **Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S., Husak, G., Rowland, J., Harrison, L., Hoell, A., & Michaelsen, J. (2015).** The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. *Scientific Data, 2*, 150066. https://doi.org/10.1038/sdata.2015.66

3. **Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., … Thépaut, J.-N. (2020).** The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society, 146*(730), 1999–2049. https://doi.org/10.1002/qj.3803

4. **Muñoz-Sabater, J., et al. (2021).** ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data, 13*(9), 4349–4383.
   Cite alongside Hersbach et al. 2020 for the ERA5-Land-specific variables: solar radiation, RZSM.

5. **Abatzoglou, J. T., Dobrowski, S. Z., Parks, S. A., & Hegewisch, K. C. (2018).** TerraClimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958–2015. *Scientific Data, 5*, 170191. https://doi.org/10.1038/sdata.2017.191

6. **Hansen, M. C., Potapov, P. V., Moore, R., Hancher, M., Turubanova, S. A., Tyukavina, A., Thau, D., Stehman, S. V., Goetz, S. J., Loveland, T. R., Kommareddy, A., Egorov, A., Chini, L., Justice, C. O., & Townshend, J. R. G. (2013).** High-resolution global maps of 21st-century forest cover change. *Science, 342*(6160), 850–853. https://doi.org/10.1126/science.1244693

7. **NOAA Climate Prediction Center. Oceanic Niño Index (ONI).** Cite as: NOAA CPC, ONI dataset, accessed via https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php (a dataset/product, not a journal article — cite per NOAA's data citation guidance).

8. **MODIS products** — cite the specific product documentation: MOD11A2 (Land Surface Temperature/Emissivity 8-Day) and MOD08_M3 (Atmosphere Monthly Global Product), both via NASA LP DAAC / LAADS DAAC product pages.

9. **NASA/USGS Shuttle Radar Topography Mission (SRTM).** Farr, T. G., et al. (2007). The Shuttle Radar Topography Mission. *Reviews of Geophysics, 45*(2), RG2004. https://doi.org/10.1029/2005RG000183
