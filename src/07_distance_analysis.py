"""
07_distance_analysis.py

Purpose: Extend the Stage 6 neighbor-effect test from immediate
(first-order) neighbors to increasingly distant patches, to answer the
second half of the research question: at what distance does the
conditional neighbor effect become statistically indistinguishable from
zero?

For each distance band, we redefine "neighbor" as any patch whose
center falls in that distance range, then re-run the same regression as
Stage 6 (own_vod(t+1) ~ own_vod(t) + band_neighbor_vod(t) + precip(t)).

KNOWN RESULT / OPEN QUESTION (see Stage 8): when this was first run, the
neighbor effect did NOT decay toward zero even out to 800-1100km - it
stayed roughly constant (~0.03-0.04) and significant at every distance
tested. That's unexpected if this were a genuine short-range spatial
contagion effect, and suggests the patch-level precipitation control
isn't fully capturing broader regional-scale atmospheric forcing that
could be shared across the whole study area. Stage 8 tests this directly
by adding a REGIONAL (whole-area average) precipitation control.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_locations.csv
Output: data/processed/distance_decay_results.csv
        figures/distance_decay_plot.png
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

    # Pairwise distance matrix
    n = len(loc)
    lats, lons, pids = loc["lat"].values, loc["lon"].values, loc["patch_id"].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine(lats[i], lons[i], lats, lons)
    dist_df = pd.DataFrame(dist_matrix, index=pids, columns=pids)
    print("Max distance in region (km):", round(dist_matrix.max(), 1))

    vod_pivot = ts.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = ts.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    dates = vod_pivot.index.to_list()
    patches = vod_pivot.columns.to_list()

    results = []
    for b in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        neighbor_band_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
        for pid in patches:
            d = dist_df.loc[pid]
            in_band = [p for p in d[(d > lo) & (d <= hi)].index.tolist() if p in vod_pivot.columns]
            neighbor_band_vod[pid] = vod_pivot[in_band].mean(axis=1) if in_band else np.nan

        records = []
        for pid in patches:
            own_t = vod_pivot[pid].values
            neigh_t = neighbor_band_vod[pid].values
            precip_t = precip_pivot[pid].values
            for i in range(len(dates) - 1):
                records.append((own_t[i], neigh_t[i], precip_t[i], own_t[i+1], pid))
        panel = pd.DataFrame(records, columns=[
            "own_vod_t", "neighbor_vod_t", "precip_anom_t", "own_vod_t1", "patch_id"
        ]).dropna()

        if len(panel) < 100:
            continue
        m = smf.ols("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t", data=panel).fit(
            cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
        coef, pval = m.params["neighbor_vod_t"], m.pvalues["neighbor_vod_t"]
        ci_low, ci_high = m.conf_int().loc["neighbor_vod_t"]
        mid = (lo + hi) / 2
        results.append((lo, hi, mid, coef, pval, ci_low, ci_high, len(panel)))
        print(f"Distance {lo}-{hi}km: coef={coef:.4f}  p={pval:.4f}  n={len(panel)}")

    decay_df = pd.DataFrame(results, columns=[
        "dist_lo", "dist_hi", "dist_mid", "coef", "pval", "ci_low", "ci_high", "n_obs"
    ])
    decay_df.to_csv(os.path.join(OUT_DIR, "distance_decay_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(decay_df["dist_mid"], decay_df["coef"],
                yerr=[decay_df["coef"]-decay_df["ci_low"], decay_df["ci_high"]-decay_df["coef"]],
                fmt='o-', capsize=4, color='darkred')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Distance between patches (km)")
    ax.set_ylabel("Neighbor effect coefficient")
    ax.set_title("Neighbor effect vs. distance (patch-level precipitation control only)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "distance_decay_plot.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/distance_decay_plot.png")

    try:
        popt, _ = curve_fit(decay_func, decay_df["dist_mid"], decay_df["coef"], p0=[0.05, 300], maxfev=5000)
        print(f"\nFitted exponential decay: A={popt[0]:.4f}, correlation length psi={popt[1]:.1f} km")
        print("(A very large psi relative to the region size means the effect is NOT")
        print("meaningfully decaying within this data - see Stage 8.)")
    except Exception as e:
        print("Curve fit failed:", e)

if __name__ == "__main__":
    main()
    