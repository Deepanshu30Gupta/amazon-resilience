"""
10_robustness_timelags.py

Purpose: Robustness Test 3 (time delays), directly following up on the
Stage 9 placebo result. Compare the neighbor effect at 1, 2, and 3
month FORWARD lags (real, causally plausible direction) against the
same lags BACKWARD (placebo direction, should not work if the effect
is genuinely directional).

Interpretation:
- If forward stays clearly stronger/more significant than backward at
  every lag -> some support for genuine directional influence, even
  though the 1-month placebo was concerning.
- If forward and backward stay roughly equal at every lag -> supports
  the Stage 9 conclusion that this is spatial synchrony, not directional
  causal influence. In that case, the counterfactual/GNN framing of the
  original research question needs to be reconsidered or reframed
  around "synchrony conditional on shared drivers" rather than
  "spatial contagion."

Model at each lag k (k = 1, 2, 3 months):
  FORWARD:  own_vod(t+k) ~ own_vod(t) + neighbor_vod(t) + precip_anom(t)
  BACKWARD: own_vod(t-k) ~ own_vod(t) + neighbor_vod(t) + precip_anom(t)

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
Output: data/processed/timelag_results.csv
        figures/timelag_forward_vs_backward.png
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

OUT_DIR = "data/processed"
FIG_DIR = "figures"
LAGS = [1, 2, 3]

def build_panel(vod_pivot, neighbor_vod, precip_pivot, dates, patches, lag, direction):
    """direction = 'forward' or 'backward'"""
    records = []
    for pid in patches:
        own = vod_pivot[pid].values
        neigh = neighbor_vod[pid].values
        precip = precip_pivot[pid].values
        n = len(dates)
        if direction == "forward":
            # predictors at t, outcome at t+lag
            for i in range(n - lag):
                records.append((pid, own[i], neigh[i], precip[i], own[i + lag]))
        else:
            # predictors at t, outcome at t-lag (placebo)
            for i in range(lag, n):
                records.append((pid, own[i], neigh[i], precip[i], own[i - lag]))
    return pd.DataFrame(records, columns=[
        "patch_id", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "outcome"
    ]).dropna()

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()

    vod_pivot = ts.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = ts.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    dates = vod_pivot.index.to_list()
    patches = vod_pivot.columns.to_list()

    neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patches:
        neighbors = [n for n in neighbor_map.get(pid, []) if n in vod_pivot.columns]
        neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    results = []
    for lag in LAGS:
        for direction in ["forward", "backward"]:
            panel = build_panel(vod_pivot, neighbor_vod, precip_pivot, dates, patches, lag, direction)
            m = smf.ols("outcome ~ own_vod_t + neighbor_vod_t + precip_anom_t", data=panel).fit(
                cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
            coef = m.params["neighbor_vod_t"]
            pval = m.pvalues["neighbor_vod_t"]
            ci_low, ci_high = m.conf_int().loc["neighbor_vod_t"]
            results.append((lag, direction, coef, pval, ci_low, ci_high, len(panel)))
            print(f"Lag {lag} months, {direction:9s}: coef={coef:.4f}  p={pval:.4f}  n={len(panel)}")

    results_df = pd.DataFrame(results, columns=[
        "lag_months", "direction", "coef", "pval", "ci_low", "ci_high", "n_obs"
    ])
    results_df.to_csv(os.path.join(OUT_DIR, "timelag_results.csv"), index=False)

    # Plot forward vs backward side by side
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for direction, color, marker in [("forward", "darkred", "o"), ("backward", "steelblue", "s")]:
        sub = results_df[results_df["direction"] == direction]
        ax.errorbar(sub["lag_months"], sub["coef"],
                     yerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                     fmt=f'{marker}-', capsize=4, color=color, label=direction, markersize=8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Neighbor effect coefficient")
    ax.set_title("Forward (real) vs. Backward (placebo) neighbor effect by lag")
    ax.legend()
    ax.set_xticks(LAGS)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "timelag_forward_vs_backward.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/timelag_forward_vs_backward.png")

    print("\n===== INTERPRETATION GUIDE =====")
    print("If forward coefficients stay clearly ABOVE backward at every lag,")
    print("and/or forward stays significant while backward weakens or loses")
    print("significance -> supports some genuine directional influence.")
    print("If forward and backward tracks stay close together at every lag ->")
    print("supports the Stage 9 conclusion: spatial synchrony, not directional")
    print("causal influence. Reconsider the counterfactual/GNN framing in that case.")

if __name__ == "__main__":
    main()