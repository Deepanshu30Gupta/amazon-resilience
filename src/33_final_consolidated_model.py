"""
33_final_consolidated_model.py

Purpose: Consolidation step before returning to the GNN decision. Put
EVERY environmental/terrain driver tested so far into one final model
together, to answer: after accounting for everything measured, how
much of the original neighbor-synchrony coefficient is actually left
unexplained?

Combines:
  Monthly drivers: precipitation, temperature, soil moisture, PDSI,
    ENSO, VPD, wind speed, solar radiation, RZSM, deltaT
  Static (time-invariant) drivers: TWI, distance-to-disturbance, and
    forest fragmentation (if Stage 28 has been run - included
    automatically if the file exists, skipped with a note if not)

Uses the CORRECTED bounding-box calculation throughout.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/patch_forest_fragmentation.csv (optional)
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/era5_dewpoint_amazon_cerrado_monthly.tif
        data/raw/era5_wind_u_amazon_cerrado_monthly.tif
        data/raw/era5_wind_v_amazon_cerrado_monthly.tif
        data/raw/era5land_solar_radiation_amazon_cerrado_monthly.tif
        data/raw/era5land_rzsm_amazon_cerrado_monthly.tif
        data/raw/modis_lst_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
Output: data/processed/final_consolidated_results.csv
        printed full step-by-step chain and final remaining coefficient
"""

import rasterio
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os

OUT_DIR = "data/processed"
TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
DEWPOINT_PATH = "data/raw/era5_dewpoint_amazon_cerrado_monthly.tif"
WIND_U_PATH = "data/raw/era5_wind_u_amazon_cerrado_monthly.tif"
WIND_V_PATH = "data/raw/era5_wind_v_amazon_cerrado_monthly.tif"
SOLAR_PATH = "data/raw/era5land_solar_radiation_amazon_cerrado_monthly.tif"
RZSM_PATH = "data/raw/era5land_rzsm_amazon_cerrado_monthly.tif"
LST_PATH = "data/raw/modis_lst_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
ONI_PATH = "data/raw/oni_index.csv"
START_DATE = "2003-01-01"
PATCH_SIZE = 4

def reconstruct_vod_bounds(loc):
    n_patch_rows = loc["row"].max() + 1
    n_patch_cols = loc["col"].max() + 1
    lon_step_patch = loc[loc["row"] == 0].sort_values("col")["lon"].diff().dropna().median()
    lat_step_patch = -loc[loc["col"] == 0].sort_values("row")["lat"].diff().dropna().median()
    left = loc["lon"].min() - lon_step_patch / 2
    right = loc["lon"].max() + lon_step_patch / 2
    top = loc["lat"].max() + lat_step_patch / 2
    bottom = loc["lat"].min() - lat_step_patch / 2
    vh, vw = n_patch_rows * PATCH_SIZE, n_patch_cols * PATCH_SIZE
    return (left, right, top, bottom), vh, vw, n_patch_rows, n_patch_cols

def aggregate_raster_to_patches(path, vod_bounds, vh, vw, n_patch_rows, n_patch_cols):
    with rasterio.open(path) as src:
        data = src.read()
        bounds = src.bounds
    n_months, rh, rw = data.shape
    n = PATCH_SIZE
    lon_step_v = (vod_bounds[1] - vod_bounds[0]) / vw
    lat_step_v = (vod_bounds[2] - vod_bounds[3]) / vh
    lon_step_r = (bounds.right - bounds.left) / rw
    lat_step_r = (bounds.top - bounds.bottom) / rh
    r_lons = bounds.left + (np.arange(rw) + 0.5) * lon_step_r
    r_lats = bounds.top - (np.arange(rh) + 0.5) * lat_step_r
    patch_vals = np.full((n_months, n_patch_rows, n_patch_cols), np.nan)
    for pr in range(n_patch_rows):
        lat_top = vod_bounds[2] - (pr * n) * lat_step_v
        lat_bot = vod_bounds[2] - (pr * n + n) * lat_step_v
        row_mask = (r_lats <= lat_top) & (r_lats > lat_bot)
        for pc in range(n_patch_cols):
            lon_left = vod_bounds[0] + (pc * n) * lon_step_v
            lon_right = vod_bounds[0] + (pc * n + n) * lon_step_v
            col_mask = (r_lons >= lon_left) & (r_lons < lon_right)
            if row_mask.sum() == 0 or col_mask.sum() == 0:
                continue
            sub = data[:, row_mask, :][:, :, col_mask]
            patch_vals[:, pr, pc] = np.nanmean(sub, axis=(1, 2))
    return patch_vals, n_months

def to_long_anomaly(patch_vals, n_months, loc, colname):
    dates = pd.date_range(START_DATE, periods=n_months, freq="MS")
    records = []
    for _, r in loc.iterrows():
        pid, pr, pc = int(r.patch_id), int(r.row), int(r.col)
        for m in range(n_months):
            records.append((pid, dates[m], patch_vals[m, pr, pc]))
    df = pd.DataFrame(records, columns=["patch_id", "date", colname]).dropna()
    df["month"] = df["date"].dt.month
    df[colname + "_anomaly"] = df[colname] - df.groupby(["patch_id", "month"])[colname].transform("mean")
    return df[["patch_id", "date", colname + "_anomaly"]]

def saturation_vapor_pressure(temp_c):
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])
    twi_df = pd.read_csv(os.path.join(OUT_DIR, "patch_twi.csv"))
    dist_df = pd.read_csv(os.path.join(OUT_DIR, "patch_disturbance_distance.csv"))

    frag_path = os.path.join(OUT_DIR, "patch_forest_fragmentation.csv")
    have_frag = os.path.exists(frag_path)
    if have_frag:
        frag_df = pd.read_csv(frag_path)
        print("Forest fragmentation data found - including it.")
    else:
        print("NOTE: patch_forest_fragmentation.csv not found (Stage 28 not yet run) -")
        print("proceeding without it. Re-run this script after Stage 28 to include it.")

    print("\nAggregating all monthly environmental variables...")
    vod_bounds, vh, vw, n_patch_rows, n_patch_cols = reconstruct_vod_bounds(loc)

    temp_vals, n_m = aggregate_raster_to_patches(TEMP_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    dewpoint_vals, _ = aggregate_raster_to_patches(DEWPOINT_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    u_vals, _ = aggregate_raster_to_patches(WIND_U_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    v_vals, _ = aggregate_raster_to_patches(WIND_V_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    solar_vals, _ = aggregate_raster_to_patches(SOLAR_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    rzsm_vals, _ = aggregate_raster_to_patches(RZSM_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    lst_vals, _ = aggregate_raster_to_patches(LST_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    soil_vals, _ = aggregate_raster_to_patches(SOIL_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    pdsi_vals, _ = aggregate_raster_to_patches(PDSI_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)

    vpd_vals = saturation_vapor_pressure(temp_vals) - saturation_vapor_pressure(dewpoint_vals)
    wind_speed_vals = np.sqrt(u_vals**2 + v_vals**2)
    deltaT_vals = lst_vals - temp_vals

    temp_df = to_long_anomaly(temp_vals, n_m, loc, "temp")
    soil_df = to_long_anomaly(soil_vals, n_m, loc, "soil")
    pdsi_df = to_long_anomaly(pdsi_vals, n_m, loc, "pdsi")
    vpd_df = to_long_anomaly(vpd_vals, n_m, loc, "vpd")
    wind_df = to_long_anomaly(wind_speed_vals, n_m, loc, "wind")
    solar_df = to_long_anomaly(solar_vals, n_m, loc, "solar")
    rzsm_df = to_long_anomaly(rzsm_vals, n_m, loc, "rzsm")
    deltaT_df = to_long_anomaly(deltaT_vals, n_m, loc, "deltaT")

    merged = ts.merge(temp_df, on=["patch_id", "date"], how="inner") \
               .merge(soil_df, on=["patch_id", "date"], how="inner") \
               .merge(pdsi_df, on=["patch_id", "date"], how="inner") \
               .merge(oni, on="date", how="inner") \
               .merge(vpd_df, on=["patch_id", "date"], how="inner") \
               .merge(wind_df, on=["patch_id", "date"], how="inner") \
               .merge(solar_df, on=["patch_id", "date"], how="inner") \
               .merge(rzsm_df, on=["patch_id", "date"], how="inner") \
               .merge(deltaT_df, on=["patch_id", "date"], how="left") \
               .merge(twi_df, on="patch_id", how="left") \
               .merge(dist_df, on="patch_id", how="left")
    if have_frag:
        merged = merged.merge(frag_df[["patch_id", "forest_edge_density", "fragmentation_index"]],
                               on="patch_id", how="left")

    print(f"Merged dataset shape: {merged.shape}")

    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    vod_pivot = merged.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()

    monthly_cols = ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly",
                     "vpd_anomaly", "wind_anomaly", "solar_anomaly", "rzsm_anomaly", "deltaT_anomaly"]
    pivots = {c: merged.pivot(index="date", columns="patch_id", values=c).sort_index() for c in monthly_cols}
    oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()

    static_cols = ["twi", "dist_to_disturbance_km"]
    if have_frag:
        static_cols += ["forest_edge_density", "fragmentation_index"]
    static_map = {c: merged.drop_duplicates("patch_id").set_index("patch_id")[c] for c in static_cols}

    dates_list = vod_pivot.index.to_list()
    patches = vod_pivot.columns.to_list()

    neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patches:
        neighbors = [n for n in neighbor_map.get(pid, []) if n in vod_pivot.columns]
        neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    recs = []
    for pid in patches:
        own_t = vod_pivot[pid].values
        neigh_t = neighbor_vod[pid].values
        oni_t = oni_series.reindex(dates_list).values
        static_vals = {c: static_map[c].get(pid, np.nan) for c in static_cols}
        for i in range(len(dates_list) - 1):
            rec = {"patch_id": pid, "own_vod_t": own_t[i], "neighbor_vod_t": neigh_t[i],
                   "oni_t": oni_t[i], "own_vod_t1": own_t[i+1]}
            for c in monthly_cols:
                rec[c] = pivots[c][pid].values[i]
            for c in static_cols:
                rec[c] = static_vals[c]
            recs.append(rec)
    panel = pd.DataFrame(recs)
    print(f"Full panel shape (before dropna): {panel.shape}\n")

    # Build the chain step by step, matching the sequence already tested
    steps = [
        ("Original (no controls)", []),
        ("+ precipitation", ["precip_anomaly"]),
        ("+ temperature", ["precip_anomaly", "temp_anomaly"]),
        ("+ soil moisture", ["precip_anomaly", "temp_anomaly", "soil_anomaly"]),
        ("+ PDSI", ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly"]),
        ("+ ENSO", ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly", "oni_t"]),
        ("+ VPD", ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly", "oni_t", "vpd_anomaly"]),
        ("+ wind", ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly", "oni_t", "vpd_anomaly", "wind_anomaly"]),
        ("+ solar radiation", ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly", "oni_t", "vpd_anomaly", "wind_anomaly", "solar_anomaly"]),
        ("+ RZSM", ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly", "oni_t", "vpd_anomaly", "wind_anomaly", "solar_anomaly", "rzsm_anomaly"]),
        ("+ deltaT", ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly", "oni_t", "vpd_anomaly", "wind_anomaly", "solar_anomaly", "rzsm_anomaly", "deltaT_anomaly"]),
        ("+ TWI", ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly", "oni_t", "vpd_anomaly", "wind_anomaly", "solar_anomaly", "rzsm_anomaly", "deltaT_anomaly", "twi"]),
        ("+ distance to disturbance", ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly", "oni_t", "vpd_anomaly", "wind_anomaly", "solar_anomaly", "rzsm_anomaly", "deltaT_anomaly", "twi", "dist_to_disturbance_km"]),
    ]
    if have_frag:
        steps.append(("+ forest fragmentation", steps[-1][1] + ["forest_edge_density", "fragmentation_index"]))

    results = []
    print("===== FULL CONSOLIDATED CHAIN =====")
    for label, covariates in steps:
        cols_needed = ["own_vod_t1", "own_vod_t", "neighbor_vod_t"] + covariates
        sub_panel = panel.dropna(subset=cols_needed)
        formula = "own_vod_t1 ~ own_vod_t + neighbor_vod_t"
        if covariates:
            formula += " + " + " + ".join(covariates)
        m = smf.ols(formula, data=sub_panel).fit(cov_type="cluster", cov_kwds={"groups": sub_panel["patch_id"]})
        coef = m.params["neighbor_vod_t"]
        pval = m.pvalues["neighbor_vod_t"]
        print(f"{label:35s}: neighbor coef={coef:.4f}  p={pval:.4f}  n={len(sub_panel)}")
        results.append((label, coef, pval, len(sub_panel)))

    results_df = pd.DataFrame(results, columns=["model", "neighbor_coef", "neighbor_pval", "n_obs"])
    results_df.to_csv(os.path.join(OUT_DIR, "final_consolidated_results.csv"), index=False)

    first_coef = results_df.iloc[0]["neighbor_coef"]
    last_coef = results_df.iloc[-1]["neighbor_coef"]
    pct_explained = 100 * (first_coef - last_coef) / first_coef
    print(f"\n===== FINAL ANSWER =====")
    print(f"Original neighbor coefficient: {first_coef:.4f}")
    print(f"Remaining after ALL drivers tested: {last_coef:.4f}")
    print(f"Total explained: {pct_explained:.1f}%")
    print(f"Remaining UNEXPLAINED synchrony: {100-pct_explained:.1f}%")
    if pct_explained < 50:
        print("\n-> The MAJORITY of the neighbor-effect synchrony remains unexplained even")
        print("   after this extensive driver search. This is a strong, well-earned")
        print("   motivation for continuing to investigate a learned spatial structure")
        print("   (GNN) - there is a real, substantial pattern left to explain.")
    else:
        print("\n-> More than half of the original synchrony is now explained by measured")
        print("   environmental and terrain drivers combined.")

if __name__ == "__main__":
    main()