"""
13_robustness_patch_size.py

Purpose: Robustness Test 1 (final one). So far all analysis has used
PATCH_SIZE=4 (16x22=352 patches). This tests whether the core finding
(own_vod(t+1) ~ own_vod(t) + neighbor_vod(t) + precip(t), neighbor
coefficient) holds up at a different patch size - here, PATCH_SIZE=6
(coarser patches, fewer of them).

This script is self-contained: it re-derives patches, aggregates
CHIRPS, deseasonalizes, and runs the baseline regression, all at the
alternative patch size, WITHOUT overwriting any of your existing
verified data/processed files from the main pipeline (uses a separate
output filename).

If the neighbor coefficient stays similarly sized and significant at
this alternative patch size too, that's evidence the main finding
isn't an artifact of the specific PATCH_SIZE=4 choice.

Input:  data/raw/vodca_amazon_cerrado_monthly.tif
        data/raw/chirps_amazon_cerrado_monthly.tif
Output: data/processed/patch_size_sensitivity_results.csv
        printed comparison to the main PATCH_SIZE=4 result
"""

import rasterio
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os

VODCA_PATH = "data/raw/vodca_amazon_cerrado_monthly.tif"
CHIRPS_PATH = "data/raw/chirps_amazon_cerrado_monthly.tif"
OUT_DIR = "data/processed"
ALT_PATCH_SIZE = 6  # alternative to the main pipeline's PATCH_SIZE=4
START_DATE = "2003-01-01"

def main():
    # ---- Rebuild patches at the alternative size ----
    with rasterio.open(VODCA_PATH) as src:
        vod_data = src.read()
        vod_bounds = src.bounds
    n_months, h, w = vod_data.shape
    n = ALT_PATCH_SIZE

    h_trim, w_trim = (h // n) * n, (w // n) * n
    vod_trim = vod_data[:, :h_trim, :w_trim]
    reshaped = vod_trim.reshape(n_months, h_trim // n, n, w_trim // n, n)
    patch_vod = np.nanmean(reshaped, axis=(2, 4))
    n_patch_rows, n_patch_cols = patch_vod.shape[1], patch_vod.shape[2]
    print(f"Alternative patch grid (size={n}): {n_patch_rows} x {n_patch_cols} = "
          f"{n_patch_rows * n_patch_cols} patches (main pipeline used 352 patches at size 4)")

    lon_step_v = (vod_bounds.right - vod_bounds.left) / w
    lat_step_v = (vod_bounds.top - vod_bounds.bottom) / h
    rows, cols, lons, lats, pids = [], [], [], [], []
    pid = 0
    for pr in range(n_patch_rows):
        for pc in range(n_patch_cols):
            lon_center = vod_bounds.left + (pc*n + pc*n + n) / 2 * lon_step_v
            lat_center = vod_bounds.top - (pr*n + pr*n + n) / 2 * lat_step_v
            pids.append(pid); rows.append(pr); cols.append(pc)
            lons.append(lon_center); lats.append(lat_center)
            pid += 1
    loc = pd.DataFrame({"patch_id": pids, "row": rows, "col": cols, "lon": lons, "lat": lats})

    # ---- Aggregate CHIRPS into the same alternative patches ----
    with rasterio.open(CHIRPS_PATH) as src:
        chirps_data = src.read()
        chirps_bounds = src.bounds
    n_m, ch, cw = chirps_data.shape
    lon_step_c = (chirps_bounds.right - chirps_bounds.left) / cw
    lat_step_c = (chirps_bounds.top - chirps_bounds.bottom) / ch
    chirps_lons = chirps_bounds.left + (np.arange(cw) + 0.5) * lon_step_c
    chirps_lats = chirps_bounds.top - (np.arange(ch) + 0.5) * lat_step_c

    patch_precip = np.full((n_m, n_patch_rows, n_patch_cols), np.nan)
    for pr in range(n_patch_rows):
        lat_top = vod_bounds.top - (pr*n) * lat_step_v
        lat_bot = vod_bounds.top - (pr*n + n) * lat_step_v
        row_mask = (chirps_lats <= lat_top) & (chirps_lats > lat_bot)
        for pc in range(n_patch_cols):
            lon_left = vod_bounds.left + (pc*n) * lon_step_v
            lon_right = vod_bounds.left + (pc*n + n) * lon_step_v
            col_mask = (chirps_lons >= lon_left) & (chirps_lons < lon_right)
            if row_mask.sum() == 0 or col_mask.sum() == 0:
                continue
            sub = chirps_data[:, row_mask, :][:, :, col_mask]
            patch_precip[:, pr, pc] = np.nanmean(sub, axis=(1, 2))

    # ---- Build time series + deseasonalize ----
    dates = pd.date_range(START_DATE, periods=n_months, freq="MS")
    records = []
    for _, r in loc.iterrows():
        pid, pr, pc = int(r.patch_id), int(r.row), int(r.col)
        for m in range(n_months):
            records.append((pid, dates[m], patch_vod[m, pr, pc], patch_precip[m, pr, pc]))
    ts = pd.DataFrame(records, columns=["patch_id", "date", "vod", "precip_mm"]).dropna()
    ts["month"] = ts["date"].dt.month
    ts["vod_anomaly"] = ts["vod"] - ts.groupby(["patch_id", "month"])["vod"].transform("mean")
    ts["precip_anomaly"] = ts["precip_mm"] - ts.groupby(["patch_id", "month"])["precip_mm"].transform("mean")

    # ---- Build adjacency (first-order) ----
    lookup = {(int(r.row), int(r.col)): int(r.patch_id) for _, r in loc.iterrows()}
    edges = []
    for _, r in loc.iterrows():
        pr, pc, pid = int(r.row), int(r.col), int(r.patch_id)
        for nr, nc in [(pr-1,pc),(pr+1,pc),(pr,pc-1),(pr,pc+1)]:
            if (nr, nc) in lookup:
                edges.append((pid, lookup[(nr, nc)]))
    adj = pd.DataFrame(edges, columns=["patch_id", "neighbor_id"])
    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()

    # ---- Run the baseline regression ----
    vod_pivot = ts.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = ts.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    dates_list = vod_pivot.index.to_list()
    patches = vod_pivot.columns.to_list()

    neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patches:
        neighbors = [x for x in neighbor_map.get(pid, []) if x in vod_pivot.columns]
        neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    recs = []
    for pid in patches:
        own_t = vod_pivot[pid].values
        neigh_t = neighbor_vod[pid].values
        precip_t = precip_pivot[pid].values
        for i in range(len(dates_list) - 1):
            recs.append((pid, own_t[i], neigh_t[i], precip_t[i], own_t[i+1]))
    panel = pd.DataFrame(recs, columns=[
        "patch_id", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "own_vod_t1"
    ]).dropna()

    m = smf.ols("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
    coef, pval = m.params["neighbor_vod_t"], m.pvalues["neighbor_vod_t"]

    print(f"\nPatch size {n} result: coef={coef:.4f}  p={pval:.4f}  n={len(panel)}")
    print("Main pipeline (patch size 4) result: coef=0.0865  p<0.0001")

    result_df = pd.DataFrame([{
        "patch_size": n, "n_patches": n_patch_rows * n_patch_cols,
        "coef": coef, "pval": pval, "n_obs": len(panel)
    }])
    result_df.to_csv(os.path.join(OUT_DIR, "patch_size_sensitivity_results.csv"), index=False)

    if pval < 0.05:
        print("\nResult holds up at this alternative patch size - supports the main")
        print("finding not being an artifact of the specific PATCH_SIZE=4 choice.")
    else:
        print("\nResult does NOT hold up at this alternative patch size - worth")
        print("discussing as a limitation/sensitivity in the write-up.")

if __name__ == "__main__":
    main()