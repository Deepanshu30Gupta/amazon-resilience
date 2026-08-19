"""
19_forest_mask_sensitivity.py

Purpose: Stage 18's Part A found a surprising result (resilience loss
INCREASES with distance from disturbance, opposite to Boulton et al.).
The leading hypothesis is that patches near disturbance are averaging
over a mix of intact forest AND already-cleared agricultural/pasture
land, which has fundamentally different vegetation dynamics than the
forest-resilience-loss signal being measured.

This script tests that hypothesis directly: compute each patch's mean
tree cover (from Hansen treecover2000), then re-run the Stage 18 Part A
analysis (distance-to-disturbance vs. resilience trend) restricted to
only high-forest-cover patches. If the relationship changes
substantially once non-forest patches are excluded, that supports the
"land-cover mixing" explanation. If it stays the same, this is a
genuinely more robust and unexpected ecological result.

Input:  data/raw/hansen_forest_change_amazon_cerrado.tif
        data/processed/patch_locations.csv
        data/processed/patch_resilience_trend.csv
        data/processed/patch_disturbance_distance.csv
Output: data/processed/forest_mask_sensitivity_results.csv
        figures/forest_mask_sensitivity_comparison.png
        printed comparison: all patches vs. forest-only patches
"""

import rasterio
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os

HANSEN_PATH = "data/raw/hansen_forest_change_amazon_cerrado.tif"
OUT_DIR = "data/processed"
FIG_DIR = "figures"
PATCH_SIZE = 4
FOREST_THRESHOLDS = [50, 70, 80]

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

def compute_patch_forest_cover(loc):
    with rasterio.open(HANSEN_PATH) as src:
        treecover = src.read(1).astype(np.float32)
        bounds = src.bounds
    rh, rw = treecover.shape
    vod_bounds, vh, vw, n_patch_rows, n_patch_cols = reconstruct_vod_bounds(loc)

    n = PATCH_SIZE
    lon_step_v = (vod_bounds[1] - vod_bounds[0]) / vw
    lat_step_v = (vod_bounds[2] - vod_bounds[3]) / vh
    lon_step_r = (bounds.right - bounds.left) / rw
    lat_step_r = (bounds.top - bounds.bottom) / rh
    r_lons = bounds.left + (np.arange(rw) + 0.5) * lon_step_r
    r_lats = bounds.top - (np.arange(rh) + 0.5) * lat_step_r

    patch_forest_pct = np.full((n_patch_rows, n_patch_cols), np.nan)
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
            sub = treecover[row_mask, :][:, col_mask]
            patch_forest_pct[pr, pc] = np.nanmean(sub)

    records = []
    for _, r in loc.iterrows():
        pid, pr, pc = int(r.patch_id), int(r.row), int(r.col)
        records.append((pid, patch_forest_pct[pr, pc]))
    return pd.DataFrame(records, columns=["patch_id", "mean_treecover_pct"])

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    trend = pd.read_csv(os.path.join(OUT_DIR, "patch_resilience_trend.csv")).dropna(subset=["kendall_tau"])
    dist_df = pd.read_csv(os.path.join(OUT_DIR, "patch_disturbance_distance.csv"))

    print("Computing patch-level mean tree cover...")
    forest_df = compute_patch_forest_cover(loc)
    print(forest_df["mean_treecover_pct"].describe())

    merged = trend.merge(dist_df, on="patch_id").merge(forest_df, on="patch_id")

    print("\n===== Comparison: all patches vs. forest-cover-filtered patches =====")
    results = []

    corr_all, p_all = stats.spearmanr(merged["dist_to_disturbance_km"], merged["kendall_tau"])
    print(f"ALL patches (n={len(merged)}): Spearman r={corr_all:.4f}, p={p_all:.4f}")
    results.append(("All patches", len(merged), corr_all, p_all))

    for thresh in FOREST_THRESHOLDS:
        sub = merged[merged["mean_treecover_pct"] >= thresh]
        if len(sub) < 10:
            print(f"Threshold {thresh}%: too few patches (n={len(sub)}) to test reliably")
            continue
        corr, p = stats.spearmanr(sub["dist_to_disturbance_km"], sub["kendall_tau"])
        print(f"Forest cover >= {thresh}% (n={len(sub)}): Spearman r={corr:.4f}, p={p:.4f}")
        results.append((f"Forest >= {thresh}%", len(sub), corr, p))

    results_df = pd.DataFrame(results, columns=["subset", "n_patches", "spearman_r", "p_value"])
    results_df.to_csv(os.path.join(OUT_DIR, "forest_mask_sensitivity_results.csv"), index=False)

    fig, axes = plt.subplots(1, len(FOREST_THRESHOLDS) + 1, figsize=(5*(len(FOREST_THRESHOLDS)+1), 4.5), sharey=True)
    axes[0].scatter(merged["dist_to_disturbance_km"], merged["kendall_tau"], alpha=0.4, s=20, color='gray')
    axes[0].set_title(f"All patches (n={len(merged)})\nr={corr_all:.3f}, p={p_all:.4f}")
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.7)
    axes[0].set_xlabel("Distance to disturbance (km)")
    axes[0].set_ylabel("Resilience trend (tau)")

    for i, thresh in enumerate(FOREST_THRESHOLDS, start=1):
        sub = merged[merged["mean_treecover_pct"] >= thresh]
        if len(sub) < 10:
            continue
        corr, p = stats.spearmanr(sub["dist_to_disturbance_km"], sub["kendall_tau"])
        axes[i].scatter(sub["dist_to_disturbance_km"], sub["kendall_tau"], alpha=0.5, s=20, color='darkgreen')
        axes[i].set_title(f"Forest >= {thresh}% (n={len(sub)})\nr={corr:.3f}, p={p:.4f}")
        axes[i].axhline(0, color='black', linestyle='--', linewidth=0.7)
        axes[i].set_xlabel("Distance to disturbance (km)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "forest_mask_sensitivity_comparison.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/forest_mask_sensitivity_comparison.png")

    print("\n===== INTERPRETATION GUIDE =====")
    print("If the correlation weakens substantially or flips sign as the forest")
    print("threshold increases -> supports the land-cover-mixing explanation for")
    print("Stage 18's surprising result.")
    print("If the positive correlation persists even in high-forest-cover patches ->")
    print("this is a more robust, genuinely unexpected finding worth deeper discussion.")

if __name__ == "__main__":
    main()