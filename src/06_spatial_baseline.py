"""
06_spatial_baseline.py

Purpose: Test the core question - does a patch's neighbors' current
state predict the patch's OWN future state, even after controlling for
the patch's own local precipitation? This is the direct test of
"spatially mediated resilience loss" vs "shared atmospheric forcing."

We use the monthly deseasonalized VOD anomaly (not the AR1 series from
Stage 5) as the outcome, because AR1 values from overlapping rolling
windows are not valid for this kind of lagged-predictor regression (see
note in 05_calculate_ar1.py).

Model: own_vod(t+1) ~ own_vod(t) + neighbor_mean_vod(t) [+ precip_anom(t)]
Neighbors = first-order adjacency (patches sharing a grid boundary),
from Stage 3's patch_adjacency.csv.
Standard errors are clustered by patch, since each patch contributes
many repeated observations over time.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
Output: data/processed/regression_panel_monthly.csv
        printed regression summaries (with and without precip control)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

OUT_DIR = "data/processed"

def main():
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()

    vod_pivot = ts.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = ts.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    dates = vod_pivot.index.to_list()
    patches = vod_pivot.columns.to_list()

    # Neighbor mean VOD anomaly at each time step
    neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patches:
        neighbors = [n for n in neighbor_map.get(pid, []) if n in vod_pivot.columns]
        neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    # Build the panel: target = own VOD anomaly at t+1
    records = []
    for pid in patches:
        own_t = vod_pivot[pid].values
        neigh_t = neighbor_vod[pid].values
        precip_t = precip_pivot[pid].values
        for i in range(len(dates) - 1):
            records.append((pid, dates[i], own_t[i], neigh_t[i], precip_t[i], own_t[i + 1]))

    panel = pd.DataFrame(records, columns=[
        "patch_id", "date", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "own_vod_t1"
    ]).dropna()
    panel.to_csv(os.path.join(OUT_DIR, "regression_panel_monthly.csv"), index=False)
    print("Panel shape:", panel.shape, "| unique patches:", panel["patch_id"].nunique())

    m1 = smf.ols("own_vod_t1 ~ own_vod_t + neighbor_vod_t", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
    m2 = smf.ols("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})

    print("\n===== WITHOUT precipitation control =====")
    print(m1.summary().tables[1])
    print("\n===== WITH precipitation control =====")
    print(m2.summary().tables[1])
    print("\nKey check: does the neighbor_vod_t coefficient survive, and stay roughly")
    print("the same size, after adding the precipitation control? If yes, that's")
    print("evidence the neighbor effect isn't just shared rainfall in disguise.")

if __name__ == "__main__":
    main()