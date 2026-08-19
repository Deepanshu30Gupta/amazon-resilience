"""
12_robustness_neighbor_defs.py

Purpose: Robustness Test 2. So far "neighbor" has meant first-order
adjacency only (patches sharing a grid edge). This tests whether the
core finding (Stage 6/8: own_vod(t+1) ~ own_vod(t) + neighbor_vod(t) +
precip(t)) holds up under two alternative neighbor definitions:

1. Edge + diagonal adjacency (8-connectivity instead of 4-connectivity)
2. Fixed-radius neighbors (all patches within 150km, regardless of
   grid position)

If the neighbor coefficient stays similarly sized and significant
under all three definitions (original edge-only, diagonal, radius),
that's good evidence the finding isn't an artifact of one particular
adjacency choice.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_locations.csv
Output: data/processed/neighbor_def_results.csv
        printed comparison across the three definitions
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

OUT_DIR = "data/processed"
RADIUS_KM = 150

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_diagonal_adjacency(loc):
    lookup = {(int(r.row), int(r.col)): int(r.patch_id) for _, r in loc.iterrows()}
    edges = []
    for _, r in loc.iterrows():
        pr, pc, pid = int(r.row), int(r.col), int(r.patch_id)
        # 8-connectivity: edge neighbors + 4 diagonals
        offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr, dc in offsets:
            if (pr+dr, pc+dc) in lookup:
                edges.append((pid, lookup[(pr+dr, pc+dc)]))
    return pd.DataFrame(edges, columns=["patch_id", "neighbor_id"])

def build_radius_adjacency(loc, radius_km):
    n = len(loc)
    lats, lons, pids = loc["lat"].values, loc["lon"].values, loc["patch_id"].values
    edges = []
    for i in range(n):
        d = haversine(lats[i], lons[i], lats, lons)
        within = np.where((d > 0) & (d <= radius_km))[0]
        for j in within:
            edges.append((pids[i], pids[j]))
    return pd.DataFrame(edges, columns=["patch_id", "neighbor_id"])

def test_neighbor_def(name, adj_df, vod_pivot, precip_pivot, dates, patches):
    neighbor_map = adj_df.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patches:
        neighbors = [n for n in neighbor_map.get(pid, []) if n in vod_pivot.columns]
        neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    records = []
    for pid in patches:
        own_t = vod_pivot[pid].values
        neigh_t = neighbor_vod[pid].values
        precip_t = precip_pivot[pid].values
        for i in range(len(dates) - 1):
            records.append((pid, own_t[i], neigh_t[i], precip_t[i], own_t[i+1]))
    panel = pd.DataFrame(records, columns=[
        "patch_id", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "own_vod_t1"
    ]).dropna()

    m = smf.ols("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
    coef = m.params["neighbor_vod_t"]
    pval = m.pvalues["neighbor_vod_t"]
    avg_neighbors = adj_df.groupby("patch_id").size().mean()
    print(f"{name:30s}: coef={coef:.4f}  p={pval:.4f}  avg_neighbors={avg_neighbors:.2f}  n={len(panel)}")
    return name, coef, pval, avg_neighbors, len(panel)

def main():
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    original_adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))

    vod_pivot = ts.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = ts.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    dates = vod_pivot.index.to_list()
    patches = vod_pivot.columns.to_list()

    diagonal_adj = build_diagonal_adjacency(loc)
    radius_adj = build_radius_adjacency(loc, RADIUS_KM)

    print(f"Testing three neighbor definitions (original result: coef~0.087, p<0.001):\n")
    results = []
    results.append(test_neighbor_def("Edge-only (original)", original_adj, vod_pivot, precip_pivot, dates, patches))
    results.append(test_neighbor_def("Edge + diagonal (8-conn)", diagonal_adj, vod_pivot, precip_pivot, dates, patches))
    results.append(test_neighbor_def(f"Fixed radius ({RADIUS_KM}km)", radius_adj, vod_pivot, precip_pivot, dates, patches))

    results_df = pd.DataFrame(results, columns=["definition", "coef", "pval", "avg_neighbors", "n_obs"])
    results_df.to_csv(os.path.join(OUT_DIR, "neighbor_def_results.csv"), index=False)

    print("\n===== SUMMARY =====")
    if (results_df["pval"] < 0.05).all():
        print("The neighbor effect remains significant under all three neighbor")
        print("definitions - supports the finding not being an artifact of the")
        print("specific (edge-only) adjacency choice used in the main analysis.")
    else:
        print("The neighbor effect is NOT significant under at least one alternative")
        print("definition - this needs further discussion in the write-up.")

if __name__ == "__main__":
    main()