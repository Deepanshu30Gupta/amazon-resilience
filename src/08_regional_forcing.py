"""
08_regional_forcing.py

Purpose: Resolve the open question from Stage 7 - the neighbor effect
did not decay with distance, staying significant even at 800-1100km.
That's suspicious: real short-range spatial contagion should fade with
distance, so a flat pattern suggests our precipitation control (each
patch's own local rainfall) isn't capturing broader regional-scale
atmospheric forcing that could be shared across the WHOLE study area at
once (e.g. a large-scale drought system affecting everything together).

Fix: add a REGIONAL precipitation control - the average precipitation
anomaly across all 252 patches at each time step - as an additional
confounder, alongside each patch's own local precipitation. Then re-run
both the baseline (Stage 6) and distance-band (Stage 7) regressions.

Two possible outcomes, both are legitimate findings:
1. If the neighbor effect NOW decays with distance once we add this
   control -> confirms shared regional forcing was masking the true,
   shorter-range spatial contagion effect. Report the new decay curve
   as the real answer.
2. If the neighbor effect STILL doesn't decay even with this control ->
   genuine evidence of long-range spatial coherence in this system,
   worth reporting and discussing as a real (if unexpected) finding.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_locations.csv
Output: data/processed/distance_decay_results_regional_controlled.csv
        figures/distance_decay_plot_regional_controlled.png
        printed comparison to Stage 6/7 results
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

OUT_DIR = "data/processed"
FIG_DIR = "figures"
BIN_EDGES = [0, 75, 150, 225, 300, 375, 450, 550, 650, 800, 1100]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def decay_func(d, A, psi):
    return A * np.exp(-d / psi)

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])

    # ---- Build the regional (whole-area average) precipitation control ----
    regional_precip = ts.groupby("date")["precip_anomaly"].mean().rename("regional_precip_anom")
    ts = ts.merge(regional_precip, on="date")
    print("Regional precip control built. Example values:")
    print(regional_precip.head())

    # Distance matrix (same as Stage 7)
    n = len(loc)
    lats, lons, pids = loc["lat"].values, loc["lon"].values, loc["patch_id"].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine(lats[i], lons[i], lats, lons)
    dist_df = pd.DataFrame(dist_matrix, index=pids, columns=pids)

    vod_pivot = ts.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = ts.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    regional_series = ts.drop_duplicates("date").set_index("date")["regional_precip_anom"].sort_index()
    dates = vod_pivot.index.to_list()
    patches = vod_pivot.columns.to_list()

    # ---- Re-run baseline (first-order neighbors) WITH regional control ----
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patches:
        neighbors = [x for x in neighbor_map.get(pid, []) if x in vod_pivot.columns]
        neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    records = []
    for pid in patches:
        own_t, neigh_t, precip_t = vod_pivot[pid].values, neighbor_vod[pid].values, precip_pivot[pid].values
        reg_t = regional_series.reindex(dates).values
        for i in range(len(dates) - 1):
            records.append((own_t[i], neigh_t[i], precip_t[i], reg_t[i], own_t[i+1], pid))
    panel = pd.DataFrame(records, columns=[
        "own_vod_t", "neighbor_vod_t", "precip_anom_t", "regional_precip_anom_t", "own_vod_t1", "patch_id"
    ]).dropna()

    m = smf.ols("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + regional_precip_anom_t",
                data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
    print("\n===== Baseline (first-order neighbors) WITH regional control =====")
    print(m.summary().tables[1])
    print("\nCompare neighbor_vod_t coefficient here to Stage 6's 0.077 - did it shrink?")

    # ---- Re-run distance-band analysis WITH regional control ----
    results = []
    for b in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        neighbor_band_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
        for pid in patches:
            d = dist_df.loc[pid]
            in_band = [p for p in d[(d > lo) & (d <= hi)].index.tolist() if p in vod_pivot.columns]
            neighbor_band_vod[pid] = vod_pivot[in_band].mean(axis=1) if in_band else np.nan

        recs = []
        for pid in patches:
            own_t, neigh_t, precip_t = vod_pivot[pid].values, neighbor_band_vod[pid].values, precip_pivot[pid].values
            reg_t = regional_series.reindex(dates).values
            for i in range(len(dates) - 1):
                recs.append((own_t[i], neigh_t[i], precip_t[i], reg_t[i], own_t[i+1], pid))
        p = pd.DataFrame(recs, columns=[
            "own_vod_t", "neighbor_vod_t", "precip_anom_t", "regional_precip_anom_t", "own_vod_t1", "patch_id"
        ]).dropna()
        if len(p) < 100:
            continue
        mb = smf.ols("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + regional_precip_anom_t",
                     data=p).fit(cov_type="cluster", cov_kwds={"groups": p["patch_id"]})
        coef, pval = mb.params["neighbor_vod_t"], mb.pvalues["neighbor_vod_t"]
        ci_low, ci_high = mb.conf_int().loc["neighbor_vod_t"]
        mid = (lo + hi) / 2
        results.append((lo, hi, mid, coef, pval, ci_low, ci_high, len(p)))
        print(f"Distance {lo}-{hi}km: coef={coef:.4f}  p={pval:.4f}  n={len(p)}")

    decay_df = pd.DataFrame(results, columns=[
        "dist_lo", "dist_hi", "dist_mid", "coef", "pval", "ci_low", "ci_high", "n_obs"
    ])
    decay_df.to_csv(os.path.join(OUT_DIR, "distance_decay_results_regional_controlled.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(decay_df["dist_mid"], decay_df["coef"],
                yerr=[decay_df["coef"]-decay_df["ci_low"], decay_df["ci_high"]-decay_df["coef"]],
                fmt='o-', capsize=4, color='darkgreen')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Distance between patches (km)")
    ax.set_ylabel("Neighbor effect coefficient")
    ax.set_title("Neighbor effect vs. distance (WITH regional precipitation control)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "distance_decay_plot_regional_controlled.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/distance_decay_plot_regional_controlled.png")

    try:
        popt, _ = curve_fit(decay_func, decay_df["dist_mid"], decay_df["coef"], p0=[0.05, 300], maxfev=5000)
        print(f"\nFitted exponential decay: A={popt[0]:.4f}, correlation length psi={popt[1]:.1f} km")
    except Exception as e:
        print("Curve fit failed:", e)

if __name__ == "__main__":
    main()