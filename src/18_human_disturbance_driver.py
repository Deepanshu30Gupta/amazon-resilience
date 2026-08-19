"""
18_human_disturbance_driver.py

Purpose: Test the human disturbance / deforestation driver, inspired
by Boulton et al. 2022, in two parts:

PART A (Boulton-style replication): does a patch's distance to the
nearest significantly-disturbed area predict its OWN resilience trend
(Kendall's tau from Stage 5)? Boulton et al. found this effect fades
out around 200-250km - testing whether the same decay pattern shows up
here would be a nice methodological contrast to your main finding
(which does NOT decay with distance).

PART B (synchrony driver test): add distance-to-disturbance as another
control in the same step-by-step neighbor-effect chain used for
temperature/soil/PDSI/ENSO (Stages 14/16/17) - does it explain any of
the remaining synchrony?

Disturbance definition: a ~2.5km grid cell (50x50 block of the Hansen
data's native ~30m/aggregated ~100m pixels) is flagged "disturbed" if
more than 50% of its pixels show forest loss during 2001-2018 OR were
already below 10% tree cover in 2000. This threshold was chosen after
checking value counts - looser thresholds (10-30%) flagged the
majority of this heavily-impacted region as "disturbed," giving no
meaningful contrast for a distance analysis.

Input:  data/raw/hansen_forest_change_amazon_cerrado.tif
        data/processed/patch_locations.csv
        data/processed/patch_resilience_trend.csv
        data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
Output: data/processed/patch_disturbance_distance.csv
        data/processed/disturbance_resilience_results.csv (Part A)
        data/processed/disturbance_driver_results.csv (Part B)
        figures/disturbance_distance_vs_resilience.png
"""

import rasterio
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

HANSEN_PATH = "data/raw/hansen_forest_change_amazon_cerrado.tif"
TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
ONI_PATH = "data/raw/oni_index.csv"
OUT_DIR = "data/processed"
FIG_DIR = "figures"
START_DATE = "2003-01-01"
PATCH_SIZE = 4
BLOCK = 50            # ~2.5km coarse grid for disturbance mask
DISTURBANCE_THRESHOLD = 0.5

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_distance_to_disturbance(loc):
    with rasterio.open(HANSEN_PATH) as src:
        treecover = src.read(1)
        lossyear = src.read(2)
        bounds = src.bounds
    h, w = treecover.shape
    h_trim, w_trim = (h // BLOCK) * BLOCK, (w // BLOCK) * BLOCK
    tc_blocks = treecover[:h_trim, :w_trim].reshape(h_trim//BLOCK, BLOCK, w_trim//BLOCK, BLOCK)
    ly_blocks = lossyear[:h_trim, :w_trim].reshape(h_trim//BLOCK, BLOCK, w_trim//BLOCK, BLOCK)

    loss_frac = ((ly_blocks >= 1) & (ly_blocks <= 18)).mean(axis=(1, 3))
    cleared_frac = (tc_blocks < 10).mean(axis=(1, 3))
    disturbed_mask = (loss_frac > DISTURBANCE_THRESHOLD) | (cleared_frac > DISTURBANCE_THRESHOLD)
    print(f"Disturbed cells: {disturbed_mask.sum()} / {disturbed_mask.size} "
          f"({100*disturbed_mask.mean():.1f}%)")

    lon_step = (bounds.right - bounds.left) / w * BLOCK
    lat_step = (bounds.top - bounds.bottom) / h * BLOCK
    rows_idx, cols_idx = np.where(disturbed_mask)
    d_lons = bounds.left + (cols_idx + 0.5) * lon_step
    d_lats = bounds.top - (rows_idx + 0.5) * lat_step

    tree = cKDTree(np.column_stack([d_lons, d_lats]))
    patch_coords = np.column_stack([loc["lon"].values, loc["lat"].values])
    _, idx = tree.query(patch_coords, k=1)
    dist_km = haversine(loc["lat"].values, loc["lon"].values, d_lats[idx], d_lons[idx])
    return dist_km

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))

    print("Computing distance to nearest disturbance for each patch...")
    loc["dist_to_disturbance_km"] = compute_distance_to_disturbance(loc)
    loc[["patch_id", "dist_to_disturbance_km"]].to_csv(
        os.path.join(OUT_DIR, "patch_disturbance_distance.csv"), index=False)
    print(loc["dist_to_disturbance_km"].describe())

    # ============================================================
    # PART A: Boulton-style - does distance-to-disturbance predict
    # a patch's OWN resilience trend (kendall_tau)?
    # ============================================================
    trend = pd.read_csv(os.path.join(OUT_DIR, "patch_resilience_trend.csv")).dropna(subset=["kendall_tau"])
    merged_a = trend.merge(loc[["patch_id", "dist_to_disturbance_km"]], on="patch_id")

    corr, corr_p = stats.spearmanr(merged_a["dist_to_disturbance_km"], merged_a["kendall_tau"])
    print(f"\n===== PART A: distance-to-disturbance vs resilience trend =====")
    print(f"Spearman correlation: {corr:.4f}, p={corr_p:.4f}")

    bin_edges = [0, 10, 25, 50, 100, 250]
    band_results = []
    for i in range(len(bin_edges)-1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        sub = merged_a[(merged_a["dist_to_disturbance_km"] > lo) & (merged_a["dist_to_disturbance_km"] <= hi)]
        if len(sub) < 5:
            continue
        mean_tau = sub["kendall_tau"].mean()
        band_results.append((lo, hi, mean_tau, len(sub)))
        print(f"Distance {lo}-{hi}km: mean resilience trend (tau)={mean_tau:.4f}  n={len(sub)}")

    band_df = pd.DataFrame(band_results, columns=["dist_lo", "dist_hi", "mean_tau", "n_patches"])
    band_df.to_csv(os.path.join(OUT_DIR, "disturbance_resilience_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(merged_a["dist_to_disturbance_km"], merged_a["kendall_tau"], alpha=0.4, s=20)
    mids = [(row.dist_lo + row.dist_hi)/2 for _, row in band_df.iterrows()]
    ax.plot(mids, band_df["mean_tau"], 'ro-', label="Binned mean")
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Distance to nearest disturbance (km)")
    ax.set_ylabel("Resilience trend (Kendall's tau)")
    ax.set_title("Human disturbance distance vs. resilience trend\n(higher tau = more resilience loss)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "disturbance_distance_vs_resilience.png"), dpi=130, bbox_inches="tight")
    print("Saved figures/disturbance_distance_vs_resilience.png")

    # ============================================================
    # PART B: does distance-to-disturbance explain any of the
    # remaining synchrony, added to the full driver chain?
    # ============================================================
    print("\n===== PART B: adding distance-to-disturbance to the synchrony driver chain =====")
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])

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
               .merge(oni, on="date", how="inner") \
               .merge(loc[["patch_id", "dist_to_disturbance_km"]], on="patch_id", how="inner")

    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    vod_pivot = merged.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = merged.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    temp_pivot = merged.pivot(index="date", columns="patch_id", values="temp_anomaly").sort_index()
    soil_pivot = merged.pivot(index="date", columns="patch_id", values="soil_anomaly").sort_index()
    pdsi_pivot = merged.pivot(index="date", columns="patch_id", values="pdsi_anomaly").sort_index()
    oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
    dist_map = merged.drop_duplicates("patch_id").set_index("patch_id")["dist_to_disturbance_km"]
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
        soil_t = soil_pivot[pid].values
        pdsi_t = pdsi_pivot[pid].values
        oni_t = oni_series.reindex(dates_list).values
        dist_val = dist_map.get(pid, np.nan)
        for i in range(len(dates_list) - 1):
            recs.append((pid, own_t[i], neigh_t[i], precip_t[i], temp_t[i], soil_t[i],
                         pdsi_t[i], oni_t[i], dist_val, own_t[i+1]))
    panel = pd.DataFrame(recs, columns=[
        "patch_id", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "temp_anom_t",
        "soil_anom_t", "pdsi_anom_t", "oni_t", "dist_disturbance", "own_vod_t1"
    ]).dropna()

    specs = [
        ("Prior best (all drivers thru ENSO)",
         "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + temp_anom_t + soil_anom_t + pdsi_anom_t + oni_t"),
        ("+ distance to disturbance",
         "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + temp_anom_t + soil_anom_t + pdsi_anom_t + oni_t + dist_disturbance"),
    ]
    results = []
    for label, formula in specs:
        m = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
        coef = m.params["neighbor_vod_t"]
        pval = m.pvalues["neighbor_vod_t"]
        print(f"{label:38s}: neighbor coef={coef:.4f}  p={pval:.4f}")
        results.append((label, coef, pval))

    results_df = pd.DataFrame(results, columns=["model", "neighbor_coef", "neighbor_pval"])
    results_df.to_csv(os.path.join(OUT_DIR, "disturbance_driver_results.csv"), index=False)

    first_coef, last_coef = results_df.iloc[0]["neighbor_coef"], results_df.iloc[-1]["neighbor_coef"]
    pct_change = 100 * (first_coef - last_coef) / first_coef
    print(f"\nAdditional change from disturbance distance: {pct_change:.1f}%")

if __name__ == "__main__":
    main()