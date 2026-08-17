"""
05_calculate_ar1.py

Purpose: Compute lag-1 autocorrelation (AR1) of the deseasonalized VOD
anomaly for each patch, over a rolling window. Rising AR1 over time is
our resilience-loss indicator (per Scheffer et al. 2009 / Boulton et al.
2022 methodology): as a system loses resilience, it recovers more slowly
from small disturbances, which shows up as the current state becoming
more strongly correlated with the previous state.

We then test whether each patch's AR1 shows a significant increasing
trend over 2003-2018, using Kendall's tau (robust to non-normal data,
standard in this literature).

IMPORTANT NOTE for later stages: this AR1 series uses OVERLAPPING
60-month windows stepped by 1 month. That's fine for measuring a TREND
(Kendall's tau across the whole sequence), but do NOT use consecutive
AR1 values as a lagged predictor of each other in a regression (e.g.
"does AR1 at time t predict AR1 at t+1") - consecutive windows share 59
of 60 months of data, which creates a spurious near-perfect correlation
that has nothing to do with real dynamics. Stage 6/7 use the monthly
VOD anomaly directly for that reason, not this AR1 series.

Input:  data/processed/patch_timeseries_anomaly.csv
Output: data/processed/patch_ar1_timeseries.csv (patch_id, window_end_date, ar1)
        data/processed/patch_resilience_trend.csv (patch_id, kendall_tau, p_value)
        figures/resilience_trend_map.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

OUT_DIR = "data/processed"
FIG_DIR = "figures"
WINDOW = 60   # months (5 years)
STEP = 1      # months

def lag1_autocorr(x):
    x = np.asarray(x)
    if len(x) < 3:
        return np.nan
    return np.corrcoef(x[:-1], x[1:])[0, 1]

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])

    results = []
    for pid in ts["patch_id"].unique():
        sub = ts[ts["patch_id"] == pid].sort_values("date").reset_index(drop=True)
        anomaly = sub["vod_anomaly"].values
        dates = sub["date"].values
        n = len(anomaly)
        for start in range(0, n - WINDOW + 1, STEP):
            window = anomaly[start:start + WINDOW]
            ar1 = lag1_autocorr(window)
            results.append((pid, dates[start + WINDOW - 1], ar1))

    ar1_df = pd.DataFrame(results, columns=["patch_id", "window_end_date", "ar1"])
    ar1_df.to_csv(os.path.join(OUT_DIR, "patch_ar1_timeseries.csv"), index=False)
    print("AR1 time series shape:", ar1_df.shape)

    # Trend test per patch
    trend_results = []
    for pid in ts["patch_id"].unique():
        sub = ar1_df[ar1_df["patch_id"] == pid].sort_values("window_end_date")
        vals = sub["ar1"].values
        x = np.arange(len(vals))
        if len(vals) < 3 or np.isnan(vals).any():
            tau, pval = np.nan, np.nan
        else:
            tau, pval = stats.kendalltau(x, vals)
        trend_results.append((pid, tau, pval))

    trend_df = pd.DataFrame(trend_results, columns=["patch_id", "kendall_tau", "p_value"])
    trend_df.to_csv(os.path.join(OUT_DIR, "patch_resilience_trend.csv"), index=False)

    n_increasing = (trend_df["kendall_tau"] > 0).sum()
    n_sig = ((trend_df["kendall_tau"] > 0) & (trend_df["p_value"] < 0.05)).sum()
    print(f"\nPatches with increasing AR1: {n_increasing}/{len(trend_df)} "
          f"({100*n_increasing/len(trend_df):.1f}%)")
    print(f"Patches with SIGNIFICANT increasing AR1 (p<0.05): {n_sig}/{len(trend_df)} "
          f"({100*n_sig/len(trend_df):.1f}%)")

    # Map the result spatially
    loc_df = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    merged = trend_df.merge(loc_df, on="patch_id")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(merged["lon"], merged["lat"], c=merged["kendall_tau"],
                     cmap="RdBu_r", vmin=-0.5, vmax=0.5, s=110, marker='s')
    plt.colorbar(sc, ax=ax, label="Kendall's tau (AR1 trend)")
    ax.set_title("Resilience trend per patch\n(red = losing resilience, blue = gaining)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "resilience_trend_map.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/resilience_trend_map.png")

if __name__ == "__main__":
    main()