"""
34_gnn_data_prep_expanded.py

Purpose: Rebuild the GNN input tensors (originally from Stage 26) with
the full ~15-variable feature set, supporting multiple forecast
horizons, and with explicit, documented handling of missing data -
rather than silently dropping observations.

Monthly (time-varying) features, per patch per month:
  1. VOD anomaly (the target variable, also used as a lagged input)
  2. Precipitation anomaly
  3. Temperature anomaly
  4. Soil moisture anomaly (TerraClimate, surface)
  5. PDSI anomaly
  6. ENSO (ONI, same value across all patches in a given month)
  7. VPD anomaly
  8. Wind U anomaly (east-west component, kept SEPARATE from V per
     the team's request - directional info, not just speed)
  9. Wind V anomaly (north-south component)
  10. Cloud fraction anomaly (MODIS, ~1deg - coarser, more missing data)
  11. Solar radiation anomaly
  12. RZSM anomaly (root-zone soil moisture)
  13. deltaT anomaly (canopy minus ambient temperature)

Static (time-invariant) features, broadcast across all 12 months of
every sequence:
  14. TWI
  15. Distance to disturbance (km)
  16-17. Forest fragmentation (edge density, fragmentation index) -
     ONLY included if Stage 28 has been run; the script checks and
     reports which case applies.

MISSING DATA POLICY (explicit, not silent): cloud fraction and deltaT
(MODIS-derived) have real, patchy missing data due to cloud cover
during the underlying satellite retrieval. Rather than dropping any
patch-month with a missing value (which would shrink the dataset for
ALL features), missing anomaly values are imputed as 0 - i.e. "assume
typical/average conditions" for that specific patch-month, which is a
defensible choice specifically because these are ANOMALIES (0 = the
normal seasonal expectation) rather than raw values. The exact count
of imputed values per feature is printed and saved for transparency.

Forecast horizons: sequences support predicting 1, 3, and 6 months
ahead from the same 12-month input window - three separate target
tensors sharing the same input features.

Split: chronological, not random (this is a time series) -
  Train: 2003-2014
  Validation: 2015-2016
  Test: 2017-2018
(adjusted at the end of the script to fit the actual available months
after accounting for the 12-month lookback and up-to-6-month lookahead)

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
        data/raw/modis_cloud_fraction_amazon_cerrado_monthly.tif
        data/raw/era5land_solar_radiation_amazon_cerrado_monthly.tif
        data/raw/era5land_rzsm_amazon_cerrado_monthly.tif
        data/raw/modis_lst_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
Output: data/processed/gnn_tensors_expanded.pt
"""

import rasterio
import numpy as np
import pandas as pd
import torch
import os
import warnings

OUT_DIR = "data/processed"
TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
DEWPOINT_PATH = "data/raw/era5_dewpoint_amazon_cerrado_monthly.tif"
WIND_U_PATH = "data/raw/era5_wind_u_amazon_cerrado_monthly.tif"
WIND_V_PATH = "data/raw/era5_wind_v_amazon_cerrado_monthly.tif"
CLOUD_PATH = "data/raw/modis_cloud_fraction_amazon_cerrado_monthly.tif"
SOLAR_PATH = "data/raw/era5land_solar_radiation_amazon_cerrado_monthly.tif"
RZSM_PATH = "data/raw/era5land_rzsm_amazon_cerrado_monthly.tif"
LST_PATH = "data/raw/modis_lst_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
ONI_PATH = "data/raw/oni_index.csv"
START_DATE = "2003-01-01"
PATCH_SIZE = 4
SEQ_LEN = 12
HORIZONS = [1, 3, 6]
CLOUD_SCALE_FACTOR = 0.0001
VAL_START = "2015-01-01"
TEST_START = "2017-01-01"

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

def to_anomaly_grid(patch_vals):
    """(months, rows, cols) raw values -> same-shape anomaly (subtract per-cell calendar-month mean)."""
    n_months = patch_vals.shape[0]
    anomaly = np.full_like(patch_vals, np.nan)
    with warnings.catch_warnings():  # off-patch grid cells can be all-NaN; harmless, suppressed for clean output
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for month_of_year in range(12):
            idx = np.arange(month_of_year, n_months, 12)
            month_mean = np.nanmean(patch_vals[idx], axis=0)
            anomaly[idx] = patch_vals[idx] - month_mean
    return anomaly

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
    frag_df = pd.read_csv(frag_path) if have_frag else None

    print("===== STAGE 34 DATA PREPARATION =====\n")
    patches = sorted(loc["patch_id"].unique().tolist())
    idx_map = {pid: i for i, pid in enumerate(patches)}
    n_nodes = len(patches)
    print(f"Patches: {n_nodes}")

    vod_bounds, vh, vw, n_patch_rows, n_patch_cols = reconstruct_vod_bounds(loc)

    print("\nAggregating monthly rasters...")
    temp_raw, n_months = aggregate_raster_to_patches(TEMP_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    dewpoint_raw, _ = aggregate_raster_to_patches(DEWPOINT_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    u_raw, _ = aggregate_raster_to_patches(WIND_U_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    v_raw, _ = aggregate_raster_to_patches(WIND_V_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    cloud_raw, _ = aggregate_raster_to_patches(CLOUD_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    cloud_raw = cloud_raw * CLOUD_SCALE_FACTOR
    solar_raw, _ = aggregate_raster_to_patches(SOLAR_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    rzsm_raw, _ = aggregate_raster_to_patches(RZSM_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    lst_raw, _ = aggregate_raster_to_patches(LST_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    soil_raw, _ = aggregate_raster_to_patches(SOIL_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    pdsi_raw, _ = aggregate_raster_to_patches(PDSI_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    print(f"Months: {n_months}")

    vpd_raw = saturation_vapor_pressure(temp_raw) - saturation_vapor_pressure(dewpoint_raw)
    deltaT_raw = lst_raw - temp_raw

    # Convert each to anomaly (grid form, rows/cols, not yet flattened to patch_id)
    grids = {
        "precip": None,  # comes from ts (already anomaly, patch_id/date long format) - handled separately
        "temp": to_anomaly_grid(temp_raw),
        "soil": to_anomaly_grid(soil_raw),
        "pdsi": to_anomaly_grid(pdsi_raw),
        "vpd": to_anomaly_grid(vpd_raw),
        "wind_u": to_anomaly_grid(u_raw),
        "wind_v": to_anomaly_grid(v_raw),
        "cloud": to_anomaly_grid(cloud_raw),
        "solar": to_anomaly_grid(solar_raw),
        "rzsm": to_anomaly_grid(rzsm_raw),
        "deltaT": to_anomaly_grid(deltaT_raw),
    }

    # Build a (months, n_nodes) array for each grid-based feature, in patch_id order
    def grid_to_node_series(grid):
        out = np.full((n_months, n_nodes), np.nan)
        for _, r in loc.iterrows():
            pid, pr, pc = int(r.patch_id), int(r.row), int(r.col)
            out[:, idx_map[pid]] = grid[:, pr, pc]
        return out

    feature_series = {name: grid_to_node_series(g) for name, g in grids.items() if g is not None}

    # VOD and precipitation come from the existing long-format anomaly file
    dates_sorted = pd.date_range(START_DATE, periods=n_months, freq="MS")
    vod_pivot = ts.pivot(index="date", columns="patch_id", values="vod_anomaly").reindex(dates_sorted)
    precip_pivot = ts.pivot(index="date", columns="patch_id", values="precip_anomaly").reindex(dates_sorted)
    feature_series["vod"] = vod_pivot[patches].values
    feature_series["precip"] = precip_pivot[patches].values

    # ENSO: same value for every patch each month
    oni_series = oni.set_index("date")["oni_value"].reindex(dates_sorted).values
    feature_series["enso"] = np.tile(oni_series.reshape(-1, 1), (1, n_nodes))

    # ---- Report missing values BEFORE imputing (transparency) ----
    print("\n===== Missing values per feature (before imputation) =====")
    missing_report = {}
    for name, arr in feature_series.items():
        n_missing = np.isnan(arr).sum()
        pct = 100 * n_missing / arr.size
        missing_report[name] = (n_missing, pct)
        print(f"  {name:10s}: {n_missing:7d} / {arr.size} ({pct:.2f}%)")

    # Impute missing anomalies as 0 ("assume typical/average conditions")
    for name in feature_series:
        feature_series[name] = np.nan_to_num(feature_series[name], nan=0.0)

    # ---- Static features (broadcast across all months) ----
    twi_vec = np.array([twi_df.set_index("patch_id")["twi"].get(pid, np.nan) for pid in patches])
    dist_vec = np.array([dist_df.set_index("patch_id")["dist_to_disturbance_km"].get(pid, np.nan) for pid in patches])
    print(f"\nStatic feature missing: TWI={np.isnan(twi_vec).sum()}, "
          f"distance-to-disturbance={np.isnan(dist_vec).sum()}")
    twi_vec = np.nan_to_num(twi_vec, nan=np.nanmean(twi_vec))
    dist_vec = np.nan_to_num(dist_vec, nan=np.nanmean(dist_vec))

    static_features = {"twi": twi_vec, "dist_disturbance": dist_vec}
    if have_frag:
        edge_vec = np.array([frag_df.set_index("patch_id")["forest_edge_density"].get(pid, np.nan) for pid in patches])
        frag_vec = np.array([frag_df.set_index("patch_id")["fragmentation_index"].get(pid, np.nan) for pid in patches])
        edge_vec = np.nan_to_num(edge_vec, nan=np.nanmean(edge_vec))
        frag_vec = np.nan_to_num(frag_vec, nan=np.nanmean(frag_vec))
        static_features["edge_density"] = edge_vec
        static_features["fragmentation"] = frag_vec
        print("Forest fragmentation: INCLUDED (Stage 28 output found)")
    else:
        print("Forest fragmentation: NOT included (Stage 28 not yet run)")

    # ---- Assemble full feature tensor: (months, n_nodes, n_features) ----
    monthly_feature_names = ["vod", "precip", "temp", "soil", "pdsi", "enso", "vpd",
                              "wind_u", "wind_v", "cloud", "solar", "rzsm", "deltaT"]
    static_feature_names = list(static_features.keys())
    all_feature_names = monthly_feature_names + static_feature_names
    n_features = len(all_feature_names)
    print(f"\nTotal features: {n_features} -> {all_feature_names}")

    full_tensor = np.zeros((n_months, n_nodes, n_features), dtype=np.float32)
    for i, name in enumerate(monthly_feature_names):
        full_tensor[:, :, i] = feature_series[name]
    for i, name in enumerate(static_feature_names):
        full_tensor[:, :, len(monthly_feature_names) + i] = static_features[name][np.newaxis, :]  # broadcast

    print("\nFeature ranges (post-imputation):")
    for i, name in enumerate(all_feature_names):
        print(f"  {name:18s}: {full_tensor[:,:,i].min():.4f} to {full_tensor[:,:,i].max():.4f}")

    # ---- Adjacency matrix ----
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _, row in adj.iterrows():
        if row["patch_id"] in idx_map and row["neighbor_id"] in idx_map:
            A[idx_map[row["patch_id"]], idx_map[row["neighbor_id"]]] = 1.0
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums

    # ---- Build sequences for each horizon ----
    max_horizon = max(HORIZONS)
    n_sequences = n_months - SEQ_LEN - max_horizon + 1
    print(f"\nTotal valid 12-month sequences (supporting up to {max_horizon}-month horizon): {n_sequences}")

    X_list = []
    y_by_horizon = {h: [] for h in HORIZONS}
    seq_target_dates = []
    for i in range(n_sequences):
        X_list.append(full_tensor[i:i+SEQ_LEN])
        for h in HORIZONS:
            y_by_horizon[h].append(full_tensor[i+SEQ_LEN+h-1, :, 0])  # VOD is feature index 0
        seq_target_dates.append(dates_sorted[i + SEQ_LEN])  # 1-month-ahead date used as the sequence's reference date

    X = np.stack(X_list)
    seq_target_dates = pd.to_datetime(seq_target_dates)

    train_mask = seq_target_dates < pd.Timestamp(VAL_START)
    val_mask = (seq_target_dates >= pd.Timestamp(VAL_START)) & (seq_target_dates < pd.Timestamp(TEST_START))
    test_mask = seq_target_dates >= pd.Timestamp(TEST_START)

    print(f"\nSplit -> Train: {train_mask.sum()} | Validation: {val_mask.sum()} | Test: {test_mask.sum()}")

    tensors = {
        "X_train": torch.tensor(X[train_mask], dtype=torch.float32),
        "X_val": torch.tensor(X[val_mask], dtype=torch.float32),
        "X_test": torch.tensor(X[test_mask], dtype=torch.float32),
        "adjacency": torch.tensor(A_norm, dtype=torch.float32),
        "patch_ids": patches,
        "n_nodes": n_nodes,
        "n_features": n_features,
        "seq_len": SEQ_LEN,
        "feature_names": all_feature_names,
        "horizons": HORIZONS,
        "missing_report": missing_report,
    }
    for h in HORIZONS:
        y_h = np.stack(y_by_horizon[h])
        tensors[f"y_train_{h}"] = torch.tensor(y_h[train_mask], dtype=torch.float32)
        tensors[f"y_val_{h}"] = torch.tensor(y_h[val_mask], dtype=torch.float32)
        tensors[f"y_test_{h}"] = torch.tensor(y_h[test_mask], dtype=torch.float32)

    torch.save(tensors, os.path.join(OUT_DIR, "gnn_tensors_expanded.pt"))

    print("\n===== SAVED =====")
    print(f"X_train: {tensors['X_train'].shape}")
    print(f"X_val:   {tensors['X_val'].shape}")
    print(f"X_test:  {tensors['X_test'].shape}")
    for h in HORIZONS:
        print(f"y_train_{h}: {tensors[f'y_train_{h}'].shape}  "
              f"y_val_{h}: {tensors[f'y_val_{h}'].shape}  "
              f"y_test_{h}: {tensors[f'y_test_{h}'].shape}")
    print(f"adjacency: {tensors['adjacency'].shape}")
    print(f"\nSaved to {OUT_DIR}/gnn_tensors_expanded.pt")

if __name__ == "__main__":
    main()