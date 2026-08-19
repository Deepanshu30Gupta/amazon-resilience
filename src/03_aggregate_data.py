"""
03_aggregate_data.py

Purpose: 
1. Aggregate the (finer-resolution) CHIRPS precipitation pixels into the
   same 252-patch boundaries defined in Stage 2 (which used VODCA's grid).
2. Combine VOD + precipitation into one long-format time series table
   (one row per patch per month).
3. Build the first-order adjacency structure (which patches share a
   boundary) - this is our baseline spatial weights matrix.

Input:  data/raw/chirps_amazon_cerrado_monthly.tif
        data/processed/patch_vod.npy
        data/processed/patch_locations.csv
Output: data/processed/patch_timeseries.csv (patch_id, date, vod, precip_mm)
        data/processed/patch_adjacency.csv (patch_id, neighbor_id)
"""

import rasterio
import numpy as np
import pandas as pd
import os

VODCA_PATH = "data/raw/vodca_amazon_cerrado_monthly.tif"
CHIRPS_PATH = "data/raw/chirps_amazon_cerrado_monthly.tif"
OUT_DIR = "data/processed"
PATCH_SIZE = 4  # MUST match Stage 2's PATCH_SIZE exactly, or the CHIRPS aggregation
                # will misalign with the VODCA patch grid
START_DATE = "2003-01-01"

def aggregate_chirps_to_patches():
    with rasterio.open(CHIRPS_PATH) as src:
        chirps_data = src.read()
        chirps_bounds = src.bounds
    n_months, ch, cw = chirps_data.shape

    with rasterio.open(VODCA_PATH) as src:
        vod_bounds = src.bounds
        vh, vw = src.height, src.width

    n = PATCH_SIZE
    n_patch_rows = vh // n
    n_patch_cols = vw // n
    lon_step_v = (vod_bounds.right - vod_bounds.left) / vw
    lat_step_v = (vod_bounds.top - vod_bounds.bottom) / vh

    lon_step_c = (chirps_bounds.right - chirps_bounds.left) / cw
    lat_step_c = (chirps_bounds.top - chirps_bounds.bottom) / ch
    chirps_lons = chirps_bounds.left + (np.arange(cw) + 0.5) * lon_step_c
    chirps_lats = chirps_bounds.top - (np.arange(ch) + 0.5) * lat_step_c

    patch_precip = np.full((n_months, n_patch_rows, n_patch_cols), np.nan)
    for pr in range(n_patch_rows):
        lat_top = vod_bounds.top - (pr * n) * lat_step_v
        lat_bot = vod_bounds.top - (pr * n + n) * lat_step_v
        row_mask = (chirps_lats <= lat_top) & (chirps_lats > lat_bot)
        for pc in range(n_patch_cols):
            lon_left = vod_bounds.left + (pc * n) * lon_step_v
            lon_right = vod_bounds.left + (pc * n + n) * lon_step_v
            col_mask = (chirps_lons >= lon_left) & (chirps_lons < lon_right)
            if row_mask.sum() == 0 or col_mask.sum() == 0:
                continue
            sub = chirps_data[:, row_mask, :][:, :, col_mask]
            patch_precip[:, pr, pc] = np.nanmean(sub, axis=(1, 2))

    n_missing = np.isnan(patch_precip).any(axis=0).sum()
    print(f"CHIRPS aggregated to patch grid. Patches with any missing months: "
          f"{n_missing} / {n_patch_rows * n_patch_cols}")
    return patch_precip

def build_timeseries(patch_vod, patch_precip, patch_meta):
    n_months = patch_vod.shape[0]
    dates = pd.date_range(START_DATE, periods=n_months, freq="MS")
    records = []
    for _, row in patch_meta.iterrows():
        pid, pr, pc = int(row.patch_id), int(row.row), int(row.col)
        vod_series = patch_vod[:, pr, pc]
        precip_series = patch_precip[:, pr, pc]
        for m in range(n_months):
            records.append((pid, dates[m], vod_series[m], precip_series[m]))
    return pd.DataFrame(records, columns=["patch_id", "date", "vod", "precip_mm"])

def build_adjacency(patch_meta):
    lookup = {(int(r.row), int(r.col)): int(r.patch_id) for _, r in patch_meta.iterrows()}
    edges = []
    for _, r in patch_meta.iterrows():
        pr, pc, pid = int(r.row), int(r.col), int(r.patch_id)
        for nr, nc in [(pr-1, pc), (pr+1, pc), (pr, pc-1), (pr, pc+1)]:
            if (nr, nc) in lookup:
                edges.append((pid, lookup[(nr, nc)]))
    return pd.DataFrame(edges, columns=["patch_id", "neighbor_id"])

def main():
    patch_meta = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    patch_vod = np.load(os.path.join(OUT_DIR, "patch_vod.npy"))

    patch_precip = aggregate_chirps_to_patches()
    ts_df = build_timeseries(patch_vod, patch_precip, patch_meta)
    ts_df.to_csv(os.path.join(OUT_DIR, "patch_timeseries.csv"), index=False)
    print(f"\nSaved patch_timeseries.csv - shape {ts_df.shape}")
    print(ts_df.head())

    adj_df = build_adjacency(patch_meta)
    adj_df.to_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"), index=False)
    print(f"\nSaved patch_adjacency.csv - {len(adj_df)} edges, "
          f"avg {adj_df.groupby('patch_id').size().mean():.2f} neighbors/patch")

if __name__ == "__main__":
    main()