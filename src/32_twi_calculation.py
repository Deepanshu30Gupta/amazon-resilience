"""
32_twi_calculation.py

Purpose: Compute a genuine Topographic Wetness Index (TWI) from the
SRTM DEM, using a real D8 flow-accumulation algorithm - per the team's
explicit choice to do this properly rather than approximate it.

TWI = ln( (flow_accumulation_area) / tan(slope) )
Higher TWI = flatter, more "downhill-converging" terrain that tends to
stay wetter (valley bottoms, floodplains). Lower TWI = steep, well-
drained terrain (ridges, hillslopes).

METHOD (standard hydrology algorithm, implemented from scratch since
GEE has no built-in flow accumulation):
1. Depression filling (Priority-Flood algorithm, Barnes et al. 2014):
   raises any local sink/pit up to its lowest possible outlet
   elevation, so every cell has a monotonic downhill path to the
   region's edge. Without this, flow gets trapped in the many small
   spurious pits present in any real-world DEM.
2. Flat-area tiebreak: filling creates flat plateaus where flow
   direction can't be resolved. Fixed by adding a tiny fraction of the
   ORIGINAL (unfilled) elevation as a tiebreaker - a standard practical
   technique that restores a resolvable direction everywhere while
   preserving the filled DEM's drainage structure.
3. D8 flow direction: each cell flows to whichever of its 8 neighbors
   has the steepest downhill gradient.
4. Flow accumulation: processing cells from highest to lowest
   elevation, each cell accumulates its own area plus everything that
   has already flowed into it.
5. Slope: the steepest downhill gradient found in step 3, with a small
   minimum floor to avoid division by near-zero in the TWI formula.

IMPORTANT LIMITATION, stated honestly: flow accumulation is computed
only within this clipped regional DEM. Rivers whose true watersheds
extend beyond the region boundary show artificially low accumulated
area near the edges, since their real upstream contribution from
outside the box isn't visible. This is a standard, expected limitation
of any regionally-clipped flow analysis, not a bug - interior areas
away from the boundary are the most reliable.

Input:  data/raw/srtm_dem_amazon_cerrado.tif
        data/processed/patch_locations.csv
        data/processed/patch_resilience_trend.csv
        data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
Output: data/processed/patch_twi.csv
        data/processed/twi_resilience_results.csv (Part A)
        data/processed/twi_driver_results.csv (Part B)
        figures/twi_map.png
"""

import rasterio
import numpy as np
import pandas as pd
import heapq
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

DEM_PATH = "data/raw/srtm_dem_amazon_cerrado.tif"
OUT_DIR = "data/processed"
FIG_DIR = "figures"
PATCH_SIZE = 4
TIEBREAK_EPSILON = 1e-5
MIN_SLOPE_RADIANS = 0.001

NEIGHBORS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def fill_depressions(dem):
    h, w = dem.shape
    filled = dem.copy()
    visited = np.zeros((h, w), dtype=bool)
    pq = []
    for r in range(h):
        heapq.heappush(pq, (dem[r, 0], r, 0)); visited[r, 0] = True
        heapq.heappush(pq, (dem[r, w-1], r, w-1)); visited[r, w-1] = True
    for c in range(w):
        if not visited[0, c]:
            heapq.heappush(pq, (dem[0, c], 0, c)); visited[0, c] = True
        if not visited[h-1, c]:
            heapq.heappush(pq, (dem[h-1, c], h-1, c)); visited[h-1, c] = True
    while pq:
        elev, r, c = heapq.heappop(pq)
        for dr, dc in NEIGHBORS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                visited[nr, nc] = True
                filled[nr, nc] = max(filled[nr, nc], elev)
                heapq.heappush(pq, (filled[nr, nc], nr, nc))
    return filled

def compute_flow_and_slope(routing_dem, cell_size_x, cell_size_y):
    h, w = routing_dem.shape
    dist_factors = [np.sqrt(cell_size_x**2+cell_size_y**2), cell_size_y, np.sqrt(cell_size_x**2+cell_size_y**2),
                     cell_size_x, cell_size_x,
                     np.sqrt(cell_size_x**2+cell_size_y**2), cell_size_y, np.sqrt(cell_size_x**2+cell_size_y**2)]
    padded = np.pad(routing_dem, 1, mode='edge')
    flow_dir = np.full((h, w), -1, dtype=np.int8)
    max_slope_gradient = np.full((h, w), 0.0)
    for idx, (dr, dc) in enumerate(NEIGHBORS):
        neighbor_vals = padded[1+dr:1+dr+h, 1+dc:1+dc+w]
        slope = (routing_dem - neighbor_vals) / dist_factors[idx]
        better = slope > max_slope_gradient
        flow_dir[better] = idx
        max_slope_gradient[better] = slope[better]
    return flow_dir, max_slope_gradient

def compute_flow_accumulation(routing_dem, flow_dir):
    h, w = routing_dem.shape
    flow_accum = np.ones((h, w), dtype=np.float64)
    flat_idx_sorted = np.argsort(-routing_dem.ravel())
    rows_sorted, cols_sorted = np.unravel_index(flat_idx_sorted, (h, w))
    flow_dir_flat = flow_dir.ravel()
    for i in range(len(flat_idx_sorted)):
        idx = flat_idx_sorted[i]
        d = flow_dir_flat[idx]
        if d == -1:
            continue
        r, c = rows_sorted[i], cols_sorted[i]
        dr, dc = NEIGHBORS[d]
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            flow_accum[nr, nc] += flow_accum[r, c]
    return flow_accum

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))

    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float64)
        bounds = src.bounds
        res = src.res
    h, w = dem.shape

    lat_center = (bounds.top + bounds.bottom) / 2
    cell_size_y = res[1] * 111320
    cell_size_x = res[0] * 111320 * np.cos(np.radians(lat_center))
    cell_area = cell_size_x * cell_size_y

    print("Filling depressions (this takes ~20-30 seconds)...")
    filled = fill_depressions(dem)

    print("Computing flow direction and accumulation...")
    routing_dem = filled + TIEBREAK_EPSILON * dem
    flow_dir, max_slope_gradient = compute_flow_and_slope(routing_dem, cell_size_x, cell_size_y)
    flow_accum = compute_flow_accumulation(routing_dem, flow_dir)

    print(f"Flow accumulation range: {flow_accum.min():.1f} - {flow_accum.max():.1f} cells")
    print(f"(NOTE: max values near the region's edges are artificially low - see docstring)")

    slope_radians = np.arctan(max_slope_gradient)
    slope_radians = np.maximum(slope_radians, MIN_SLOPE_RADIANS)
    specific_catchment_area = flow_accum * cell_area / cell_size_x
    twi = np.log(specific_catchment_area / np.tan(slope_radians))

    print(f"TWI range: {twi.min():.2f} - {twi.max():.2f}")

    n_patch_rows = loc["row"].max() + 1
    n_patch_cols = loc["col"].max() + 1
    lon_step_patch = loc[loc["row"] == 0].sort_values("col")["lon"].diff().dropna().median()
    lat_step_patch = -loc[loc["col"] == 0].sort_values("row")["lat"].diff().dropna().median()
    patch_left = loc["lon"].min() - lon_step_patch / 2
    patch_top = loc["lat"].max() + lat_step_patch / 2

    lon_step_r = (bounds.right - bounds.left) / w
    lat_step_r = (bounds.top - bounds.bottom) / h

    results = []
    for _, r in loc.iterrows():
        pid, pr, pc = int(r.patch_id), int(r.row), int(r.col)
        patch_lon_left = patch_left + pc * lon_step_patch
        patch_lon_right = patch_lon_left + lon_step_patch
        patch_lat_top = patch_top - pr * lat_step_patch
        patch_lat_bottom = patch_lat_top - lat_step_patch

        col_start = int((patch_lon_left - bounds.left) / lon_step_r)
        col_end = int((patch_lon_right - bounds.left) / lon_step_r)
        row_start = int((bounds.top - patch_lat_top) / lat_step_r)
        row_end = int((bounds.top - patch_lat_bottom) / lat_step_r)
        col_start, col_end = max(0, col_start), min(w, col_end)
        row_start, row_end = max(0, row_start), min(h, row_end)

        if col_end <= col_start or row_end <= row_start:
            results.append((pid, np.nan))
            continue
        patch_twi = twi[row_start:row_end, col_start:col_end]
        results.append((pid, np.nanmean(patch_twi)))

    twi_df = pd.DataFrame(results, columns=["patch_id", "twi"])
    twi_df.to_csv(os.path.join(OUT_DIR, "patch_twi.csv"), index=False)
    print("\n===== Patch-level TWI =====")
    print(twi_df["twi"].describe())

    merged_map = twi_df.merge(loc[["patch_id", "lat", "lon"]], on="patch_id")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(merged_map["lon"], merged_map["lat"], c=merged_map["twi"], cmap="Blues", s=80, marker='s')
    plt.colorbar(sc, ax=ax, label="TWI (higher = wetter/flatter)")
    ax.set_title("Topographic Wetness Index per patch")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "twi_map.png"), dpi=130, bbox_inches="tight")
    print("Saved figures/twi_map.png")

    trend = pd.read_csv(os.path.join(OUT_DIR, "patch_resilience_trend.csv")).dropna(subset=["kendall_tau"])
    merged_a = trend.merge(twi_df, on="patch_id").dropna()
    corr, corr_p = stats.spearmanr(merged_a["twi"], merged_a["kendall_tau"])
    print(f"\n===== PART A: TWI vs resilience trend =====")
    print(f"Spearman correlation: {corr:.4f}, p={corr_p:.4f}")
    pd.DataFrame([{"spearman_r": corr, "p_value": corr_p, "n": len(merged_a)}]).to_csv(
        os.path.join(OUT_DIR, "twi_resilience_results.csv"), index=False)

    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    merged_b = ts.merge(twi_df, on="patch_id", how="left")

    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    vod_pivot = merged_b.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = merged_b.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    twi_map_dict = merged_b.drop_duplicates("patch_id").set_index("patch_id")["twi"]
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
        twi_val = twi_map_dict.get(pid, np.nan)
        for i in range(len(dates_list) - 1):
            recs.append((pid, own_t[i], neigh_t[i], precip_t[i], twi_val, own_t[i+1]))
    panel = pd.DataFrame(recs, columns=["patch_id", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "twi", "own_vod_t1"]).dropna()

    m1 = smf.ols("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
    m2 = smf.ols("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + twi", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})

    print(f"\n===== PART B: TWI added to synchrony driver chain =====")
    print(f"Without TWI: neighbor coef={m1.params['neighbor_vod_t']:.4f}  p={m1.pvalues['neighbor_vod_t']:.4f}")
    print(f"With TWI:    neighbor coef={m2.params['neighbor_vod_t']:.4f}  p={m2.pvalues['neighbor_vod_t']:.4f}")

    pd.DataFrame([
        {"model": "without TWI", "neighbor_coef": m1.params['neighbor_vod_t'], "neighbor_pval": m1.pvalues['neighbor_vod_t']},
        {"model": "with TWI", "neighbor_coef": m2.params['neighbor_vod_t'], "neighbor_pval": m2.pvalues['neighbor_vod_t']},
    ]).to_csv(os.path.join(OUT_DIR, "twi_driver_results.csv"), index=False)

if __name__ == "__main__":
    main()