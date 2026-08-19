"""
11_robustness_direction_test.py

Purpose: Stage 10 showed forward coefficients consistently slightly
higher than backward at every lag (1, 2, 3 months), but comparing two
separate p-values by eye is a weak way to judge whether that gap is
real. This script tests it directly and formally: pool the forward and
backward panels together, add a "direction" indicator and its
interaction with the neighbor effect, and test whether that
interaction term is itself statistically significant.

If the interaction term IS significant -> the forward and backward
effects are genuinely, statistically different from each other,
supporting some real directional component on top of the dominant
synchrony.
If the interaction term is NOT significant -> we cannot statistically
distinguish forward from backward; the small gaps seen in Stage 10 are
consistent with noise, and synchrony (not directional influence) should
be treated as the primary finding going forward.

Model (pooled): outcome ~ own_vod_t + neighbor_vod_t + precip_anom_t
                + is_forward + neighbor_vod_t:is_forward
The coefficient on neighbor_vod_t:is_forward is the key test.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
Output: data/processed/direction_test_results.csv
        printed regression summary + interpretation
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

OUT_DIR = "data/processed"
LAGS = [1, 2, 3]

def build_panel(vod_pivot, neighbor_vod, precip_pivot, dates, patches, lag, direction):
    records = []
    for pid in patches:
        own = vod_pivot[pid].values
        neigh = neighbor_vod[pid].values
        precip = precip_pivot[pid].values
        n = len(dates)
        if direction == "forward":
            for i in range(n - lag):
                records.append((pid, own[i], neigh[i], precip[i], own[i + lag], 1))
        else:
            for i in range(lag, n):
                records.append((pid, own[i], neigh[i], precip[i], own[i - lag], 0))
    return pd.DataFrame(records, columns=[
        "patch_id", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "outcome", "is_forward"
    ]).dropna()

def main():
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

    all_results = []
    for lag in LAGS:
        fwd = build_panel(vod_pivot, neighbor_vod, precip_pivot, dates, patches, lag, "forward")
        bwd = build_panel(vod_pivot, neighbor_vod, precip_pivot, dates, patches, lag, "backward")
        pooled = pd.concat([fwd, bwd], ignore_index=True)

        m = smf.ols(
            "outcome ~ own_vod_t + neighbor_vod_t + precip_anom_t + is_forward "
            "+ neighbor_vod_t:is_forward",
            data=pooled
        ).fit(cov_type="cluster", cov_kwds={"groups": pooled["patch_id"]})

        interaction_coef = m.params["neighbor_vod_t:is_forward"]
        interaction_pval = m.pvalues["neighbor_vod_t:is_forward"]
        print(f"\n--- Lag {lag} months ---")
        print(f"Forward-vs-backward difference (interaction term): "
              f"coef={interaction_coef:.4f}  p={interaction_pval:.4f}")
        if interaction_pval < 0.05:
            print("-> Forward and backward ARE significantly different at this lag.")
        else:
            print("-> Cannot statistically distinguish forward from backward at this lag.")

        all_results.append((lag, interaction_coef, interaction_pval))

    results_df = pd.DataFrame(all_results, columns=["lag_months", "interaction_coef", "interaction_pval"])
    results_df.to_csv(os.path.join(OUT_DIR, "direction_test_results.csv"), index=False)

    n_sig = (results_df["interaction_pval"] < 0.05).sum()
    print(f"\n===== SUMMARY: {n_sig} / {len(LAGS)} lags show a significant forward-vs-backward difference =====")
    if n_sig == 0:
        print("No lag shows a statistically significant difference between forward and")
        print("backward. This supports treating SYNCHRONY (not directional causal")
        print("influence) as the primary, defensible finding. Recommend reframing the")
        print("counterfactual/GNN component around 'conditional synchrony' rather than")
        print("one-directional spatial contagion, or treating directional causality as")
        print("an open question for future work rather than a settled result.")
    else:
        print("At least one lag shows a statistically significant forward-vs-backward")
        print("difference, providing some support for a genuine directional component")
        print("on top of the dominant synchrony pattern.")

if __name__ == "__main__":
    main()