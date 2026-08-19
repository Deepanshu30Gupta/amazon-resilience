"""
21_spatial_lag_error_models.py

Purpose: Formalize the "spatial interaction vs shared spatial structure"
question using the classical spatial econometrics toolkit (Elhorst
2010), rather than the linear panel regression used so far.

Cross-sectional setup: one row per patch, outcome = resilience trend
(Kendall's tau from Stage 5), covariates = each patch's average local
climate conditions (temperature, precipitation, soil moisture, PDSI)
plus distance-to-disturbance and lat/lon.

Three models compared:
  1. OLS (no spatial structure at all - the naive baseline)
  2. Spatial Lag Model (SAR): tests whether NEIGHBORING patches'
     resilience trend directly predicts this patch's trend, after
     controlling for covariates - this is "spatial interaction."
  3. Spatial Error Model (SEM): tests whether the spatial similarity
     is better explained by an unobserved, spatially-structured factor
     affecting the error term - this is "shared spatial structure."

Lagrange Multiplier (LM) diagnostic tests on the OLS model indicate
which of the two spatial models (lag vs error) is the better-supported
specification for this data - the standard, recommended workflow in
spatial econometrics (per Elhorst 2010 and the broader literature).

Input:  data/processed/patch_locations.csv
        data/processed/patch_resilience_trend.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_disturbance_distance.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/processed/patch_timeseries.csv (for raw precipitation mean)
Output: data/processed/spatial_lag_error_results.csv
        printed OLS + LM diagnostics + Spatial Lag + Spatial Error results
"""

import rasterio
import numpy as np
import pandas as pd
import libpysal
import spreg
import os

OUT_DIR = "data/processed"
TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
PATCH_SIZE = 4

def reconstruct_vod_bounds(loc):
    n_patch_rows = loc["row"].max() + 1
    n_patch_cols = loc["col"].max() + 1
    lon_step_patch = loc.sort_values("col")["lon"].diff().dropna().median()
    lat_step_patch = -loc.sort_values("row")["lat"].diff().dropna().median()
    left = loc["lon"].min() - lon_step_patch / 2
    right = loc["lon"].max() + lon_step_patch / 2
    top = loc["lat"].max() + lat_step_patch / 2
    bottom = loc["lat"].min() - lat_step_patch / 2
    vh, vw = n_patch_rows * PATCH_SIZE, n_patch_cols * PATCH_SIZE
    return (left, right, top, bottom), vh, vw, n_patch_rows, n_patch_cols

def aggregate_raster_mean_to_patches(path, vod_bounds, vh, vw, n_patch_rows, n_patch_cols):
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
    patch_mean = np.full((n_patch_rows, n_patch_cols), np.nan)
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
            patch_mean[pr, pc] = np.nanmean(sub)
    return patch_mean

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    trend = pd.read_csv(os.path.join(OUT_DIR, "patch_resilience_trend.csv")).dropna(subset=["kendall_tau"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    dist_df = pd.read_csv(os.path.join(OUT_DIR, "patch_disturbance_distance.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries.csv"))

    print("Computing static (mean-over-time) covariates per patch...")
    vod_bounds, vh, vw, n_patch_rows, n_patch_cols = reconstruct_vod_bounds(loc)
    temp_mean = aggregate_raster_mean_to_patches(TEMP_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    soil_mean = aggregate_raster_mean_to_patches(SOIL_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    pdsi_mean = aggregate_raster_mean_to_patches(PDSI_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)

    static_records = []
    for _, r in loc.iterrows():
        pid, pr, pc = int(r.patch_id), int(r.row), int(r.col)
        static_records.append((pid, temp_mean[pr, pc], soil_mean[pr, pc], pdsi_mean[pr, pc]))
    static_df = pd.DataFrame(static_records, columns=["patch_id", "mean_temp", "mean_soil", "mean_pdsi"])

    precip_mean_df = ts.groupby("patch_id")["precip_mm"].mean().reset_index().rename(columns={"precip_mm": "mean_precip"})

    merged = trend.merge(loc[["patch_id", "lat", "lon", "row", "col"]], on="patch_id") \
                  .merge(static_df, on="patch_id") \
                  .merge(precip_mean_df, on="patch_id") \
                  .merge(dist_df, on="patch_id") \
                  .dropna().reset_index(drop=True)
    print(f"Final cross-sectional dataset: {len(merged)} patches\n")

    neighbor_dict = {}
    for pid in merged["patch_id"]:
        neighbor_dict[pid] = adj[adj["patch_id"] == pid]["neighbor_id"].tolist()
    valid_ids = set(merged["patch_id"])
    neighbor_dict = {pid: [n for n in neighbor_dict[pid] if n in valid_ids] for pid in neighbor_dict}
    w = libpysal.weights.W(neighbor_dict)
    w.transform = 'r'

    y = merged["kendall_tau"].values.reshape(-1, 1)
    x_cols = ["mean_precip", "mean_temp", "mean_soil", "mean_pdsi", "dist_to_disturbance_km", "lat", "lon"]
    x = merged[x_cols].values

    print("===== Model 1: OLS (no spatial structure) + LM diagnostics =====")
    ols = spreg.OLS(y, x, w=w, name_y="kendall_tau", name_x=x_cols, spat_diag=True, moran=True)
    print(ols.summary)

    print("\n===== Model 2: Spatial Lag Model (SAR) =====")
    lag_model = spreg.ML_Lag(y, x, w=w, name_y="kendall_tau", name_x=x_cols)
    print(lag_model.summary)

    print("\n===== Model 3: Spatial Error Model (SEM) =====")
    error_model = spreg.ML_Error(y, x, w=w, name_y="kendall_tau", name_x=x_cols)
    print(error_model.summary)

    results = pd.DataFrame([
        {"model": "OLS", "log_likelihood": np.nan,
         "aic": np.nan, "rho_or_lambda": np.nan, "pval": np.nan},
        {"model": "Spatial Lag (rho)", "log_likelihood": lag_model.logll,
         "aic": lag_model.aic, "rho_or_lambda": lag_model.rho, "pval": lag_model.z_stat[-1][1]},
        {"model": "Spatial Error (lambda)", "log_likelihood": error_model.logll,
         "aic": error_model.aic, "rho_or_lambda": error_model.lam, "pval": error_model.z_stat[-1][1]},
    ])
    results.to_csv(os.path.join(OUT_DIR, "spatial_lag_error_results.csv"), index=False)

    print("\n===== SUMMARY =====")
    print(results.to_string(index=False))
    print("\nLower AIC = better-fitting model. Compare Spatial Lag vs Spatial Error AIC")
    print("to see which specification (interaction vs shared unobserved structure)")
    print("the data better supports. Check the LM test p-values in the OLS output above")
    print("for the formal statistical guidance on which model is preferred.")

if __name__ == "__main__":
    main()