"""
26_gnn_data_prep.py

Purpose: Build the tensors needed for the spatiotemporal GNN models
(Stage 27) - a proper sequence-based, graph-structured dataset,
distinct from the flat panel format used by the linear regression
models (Stages 06-25).

Produces, for every valid 12-month window:
  - input sequence: (12 months, 352 patches, 6 features) where
    features = [own VOD anomaly, precipitation, temperature,
    soil moisture, PDSI, ENSO]
  - target: (352 patches,) - next month's VOD anomaly for every patch

Same train (2003-2015) / test (2016-2018) split as Stages 22-25, so
GNN results are directly comparable to the linear baselines.

Also saves the row-standardized adjacency matrix as a tensor (used by
the fixed-graph DCRNN-style model in Stage 27; the adaptive/Graph
WaveNet-style model ignores this and learns its own).

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
Output: data/processed/gnn_tensors.pt (a dict of all tensors, via torch.save)
"""

import rasterio
import numpy as np
import pandas as pd
import torch
import os

OUT_DIR = "data/processed"
TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
ONI_PATH = "data/raw/oni_index.csv"
START_DATE = "2003-01-01"
PATCH_SIZE = 4
TEST_START = "2016-01-01"
SEQ_LEN = 12

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

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])

    print("Aggregating temperature, soil moisture, and PDSI...")
    vod_bounds, vh, vw, n_patch_rows, n_patch_cols = reconstruct_vod_bounds(loc)
    temp_vals, n_m1 = aggregate_raster_to_patches(TEMP_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    soil_vals, n_m2 = aggregate_raster_to_patches(SOIL_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    pdsi_vals, n_m3 = aggregate_raster_to_patches(PDSI_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    temp_df = to_long_anomaly(temp_vals, n_m1, loc, "temp")
    soil_df = to_long_anomaly(soil_vals, n_m2, loc, "soil")
    pdsi_df = to_long_anomaly(pdsi_vals, n_m3, loc, "pdsi")

    merged = ts.merge(temp_df, on=["patch_id", "date"], how="inner") \
               .merge(soil_df, on=["patch_id", "date"], how="inner") \
               .merge(pdsi_df, on=["patch_id", "date"], how="inner") \
               .merge(oni, on="date", how="inner")

    patches = sorted(loc["patch_id"].unique().tolist())
    idx_map = {pid: i for i, pid in enumerate(patches)}
    n_nodes = len(patches)

    dates_sorted = sorted(merged["date"].unique())
    n_months = len(dates_sorted)
    date_idx = {d: i for i, d in enumerate(dates_sorted)}

    features = np.full((n_months, n_nodes, 6), np.nan, dtype=np.float32)
    for _, row in merged.iterrows():
        t = date_idx[row["date"]]
        p = idx_map[row["patch_id"]]
        features[t, p, 0] = row["vod_anomaly"]
        features[t, p, 1] = row["precip_anomaly"]
        features[t, p, 2] = row["temp_anomaly"]
        features[t, p, 3] = row["soil_anomaly"]
        features[t, p, 4] = row["pdsi_anomaly"]
        features[t, p, 5] = row["oni_value"]

    print(f"Feature tensor shape: {features.shape} (months, nodes, features)")
    n_missing = np.isnan(features).sum()
    print(f"Missing values in feature tensor: {n_missing} "
          f"({100*n_missing/features.size:.2f}%) - will be filled with 0")
    features = np.nan_to_num(features, nan=0.0)

    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _, row in adj.iterrows():
        if row["patch_id"] in idx_map and row["neighbor_id"] in idx_map:
            A[idx_map[row["patch_id"]], idx_map[row["neighbor_id"]]] = 1.0
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums

    X_list, y_list, target_dates = [], [], []
    for i in range(n_months - SEQ_LEN):
        X_list.append(features[i:i+SEQ_LEN])
        y_list.append(features[i+SEQ_LEN, :, 0])
        target_dates.append(dates_sorted[i+SEQ_LEN])

    X = np.stack(X_list)
    y = np.stack(y_list)
    target_dates = pd.to_datetime(target_dates)

    train_mask = target_dates < pd.Timestamp(TEST_START)
    test_mask = ~train_mask

    print(f"\nTotal sequences: {len(X)} | Train: {train_mask.sum()} | Test: {test_mask.sum()}")

    tensors = {
        "X_train": torch.tensor(X[train_mask], dtype=torch.float32),
        "y_train": torch.tensor(y[train_mask], dtype=torch.float32),
        "X_test": torch.tensor(X[test_mask], dtype=torch.float32),
        "y_test": torch.tensor(y[test_mask], dtype=torch.float32),
        "adjacency": torch.tensor(A_norm, dtype=torch.float32),
        "patch_ids": patches,
        "n_nodes": n_nodes,
        "n_features": 6,
        "seq_len": SEQ_LEN,
    }
    torch.save(tensors, os.path.join(OUT_DIR, "gnn_tensors.pt"))
    print(f"\nSaved tensors to {OUT_DIR}/gnn_tensors.pt")
    print(f"X_train shape: {tensors['X_train'].shape}")
    print(f"y_train shape: {tensors['y_train'].shape}")
    print(f"X_test shape: {tensors['X_test'].shape}")
    print(f"y_test shape: {tensors['y_test'].shape}")
    print(f"Adjacency shape: {tensors['adjacency'].shape}")

if __name__ == "__main__":
    main()