"""
14_environmental_driver_temperature.py

Purpose: Investigate whether temperature explains some of the spatial
synchrony found in Stage 6/8, on top of precipitation. This follows the
same logic as adding the regional precipitation control in Stage 8:
add one more environmental variable and see whether the neighbor
coefficient shrinks.

We build up the model step by step so we can see exactly how much each
addition changes the neighbor coefficient:
  Model 1: own_vod(t+1) ~ own_vod(t) + neighbor_vod(t)                         [no controls]
  Model 2: + local precipitation
  Model 3: + local precipitation + local temperature
  Model 4: + local precipitation + regional precipitation + local temperature

Input:  data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_locations.csv
        data/processed/patch_adjacency.csv
Output: data/processed/temperature_driver_results.csv
        printed step-by-step comparison
"""

import rasterio
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os

TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
OUT_DIR = "data/processed"
START_DATE = "2003-01-01"

def aggregate_temp_to_patches(patch_locations, vod_bounds, vh, vw, patch_size):
    with rasterio.open(TEMP_PATH) as src:
        temp_data = src.read()
        temp_bounds = src.bounds
    n_months, th, tw = temp_data.shape

    n = patch_size
    n_patch_rows = vh // n
    n_patch_cols = vw // n
    lon_step_v = (vod_bounds[1] - vod_bounds[0]) / vw   # (left, right, top, bottom) tuple passed in
    lat_step_v = (vod_bounds[2] - vod_bounds[3]) / vh

    lon_step_t = (temp_bounds.right - temp_bounds.left) / tw
    lat_step_t = (temp_bounds.top - temp_bounds.bottom) / th
    temp_lons = temp_bounds.left + (np.arange(tw) + 0.5) * lon_step_t
    temp_lats = temp_bounds.top - (np.arange(th) + 0.5) * lat_step_t

    patch_temp = np.full((n_months, n_patch_rows, n_patch_cols), np.nan)
    for pr in range(n_patch_rows):
        lat_top = vod_bounds[2] - (pr * n) * lat_step_v
        lat_bot = vod_bounds[2] - (pr * n + n) * lat_step_v
        row_mask = (temp_lats <= lat_top) & (temp_lats > lat_bot)
        for pc in range(n_patch_cols):
            lon_left = vod_bounds[0] + (pc * n) * lon_step_v
            lon_right = vod_bounds[0] + (pc * n + n) * lon_step_v
            col_mask = (temp_lons >= lon_left) & (temp_lons < lon_right)
            if row_mask.sum() == 0 or col_mask.sum() == 0:
                continue
            sub = temp_data[:, row_mask, :][:, :, col_mask]
            patch_temp[:, pr, pc] = np.nanmean(sub, axis=(1, 2))
    return patch_temp, n_months

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))

    # We need VODCA's original bounds/dimensions to align the patch grid -
    # re-derive from patch_locations' row/col/lon/lat spacing
    patch_size = 4  # must match the main pipeline's PATCH_SIZE
    n_patch_rows = loc["row"].max() + 1
    n_patch_cols = loc["col"].max() + 1
    # Approximate VODCA bounds from patch centers (good enough for aggregation)
    lon_step_patch = loc.sort_values("col")["lon"].diff().dropna().median()
    lat_step_patch = -loc.sort_values("row")["lat"].diff().dropna().median()
    left = loc["lon"].min() - lon_step_patch / 2
    right = loc["lon"].max() + lon_step_patch / 2
    top = loc["lat"].max() + lat_step_patch / 2
    bottom = loc["lat"].min() - lat_step_patch / 2
    vod_bounds = (left, right, top, bottom)
    vh, vw = n_patch_rows * patch_size, n_patch_cols * patch_size

    print("Aggregating ERA5 temperature into the existing patch grid...")
    patch_temp, n_months = aggregate_temp_to_patches(loc, vod_bounds, vh, vw, patch_size)

    n_missing = np.isnan(patch_temp).any(axis=0).sum()
    print(f"Patches with any missing temperature months: {n_missing} / {n_patch_rows * n_patch_cols}")

    # Build temperature time series + deseasonalize
    dates = pd.date_range(START_DATE, periods=n_months, freq="MS")
    records = []
    for _, r in loc.iterrows():
        pid, pr, pc = int(r.patch_id), int(r.row), int(r.col)
        for m in range(n_months):
            records.append((pid, dates[m], patch_temp[m, pr, pc]))
    temp_ts = pd.DataFrame(records, columns=["patch_id", "date", "temp_c"]).dropna()
    temp_ts["month"] = temp_ts["date"].dt.month
    temp_ts["temp_anomaly"] = temp_ts["temp_c"] - temp_ts.groupby(["patch_id", "month"])["temp_c"].transform("mean")

    # Merge with existing VOD/precip anomaly data
    merged = ts.merge(temp_ts[["patch_id", "date", "temp_anomaly"]], on=["patch_id", "date"], how="inner")
    print(f"\nMerged dataset shape: {merged.shape} (should be close to original {ts.shape})")

    # Regional temperature control too, same pattern as Stage 8
    regional_temp = merged.groupby("date")["temp_anomaly"].mean().rename("regional_temp_anom")
    merged = merged.merge(regional_temp, on="date")

    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    vod_pivot = merged.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = merged.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    temp_pivot = merged.pivot(index="date", columns="patch_id", values="temp_anomaly").sort_index()
    regional_temp_series = merged.drop_duplicates("date").set_index("date")["regional_temp_anom"].sort_index()
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
        precip_t = precip_pivot[pid].values
        temp_t = temp_pivot[pid].values
        reg_temp_t = regional_temp_series.reindex(dates_list).values
        for i in range(len(dates_list) - 1):
            recs.append((pid, own_t[i], neigh_t[i], precip_t[i], temp_t[i], reg_temp_t[i], own_t[i+1]))
    panel = pd.DataFrame(recs, columns=[
        "patch_id", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "temp_anom_t",
        "regional_temp_anom_t", "own_vod_t1"
    ]).dropna()

    print(f"\nFinal regression panel shape: {panel.shape}\n")

    results = []
    specs = [
        ("No controls", "own_vod_t1 ~ own_vod_t + neighbor_vod_t"),
        ("+ local precipitation", "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t"),
        ("+ local precip + local temp", "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + temp_anom_t"),
        ("+ local precip + local temp + regional temp",
         "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + temp_anom_t + regional_temp_anom_t"),
    ]
    for label, formula in specs:
        m = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
        coef = m.params["neighbor_vod_t"]
        pval = m.pvalues["neighbor_vod_t"]
        print(f"{label:45s}: neighbor coef={coef:.4f}  p={pval:.4f}")
        results.append((label, coef, pval))

    results_df = pd.DataFrame(results, columns=["model", "neighbor_coef", "neighbor_pval"])
    results_df.to_csv(os.path.join(OUT_DIR, "temperature_driver_results.csv"), index=False)

    first_coef = results_df.iloc[0]["neighbor_coef"]
    last_coef = results_df.iloc[-1]["neighbor_coef"]
    pct_change = 100 * (first_coef - last_coef) / first_coef
    print(f"\n===== SUMMARY =====")
    print(f"Neighbor coefficient: {first_coef:.4f} (no controls) -> {last_coef:.4f} (full controls)")
    print(f"Change: {pct_change:.1f}%")
    if abs(pct_change) < 15:
        print("Temperature (and precipitation) explain relatively little of the synchrony -")
        print("the neighbor effect remains largely unexplained by these environmental drivers.")
    else:
        print("Temperature explains a meaningful share of the apparent synchrony.")

if __name__ == "__main__":
    main()