"""
09_robustness_placebo.py

Purpose: Robustness Test 4 (placebo test). Test whether the SAME
predictors used in the real model (Stage 6/8) - a patch's own current
state, its neighbors' current state, and precipitation - can also
"predict" something they should have no real causal claim on: the
target patch's PAST value (before the predictors were even measured).

This mirrors the real model's exact structure (same own-lag control,
same predictors) and only changes which time point is treated as the
outcome - from the FUTURE (t+1, the real test) to the PAST (t-1, the
placebo). If neighbor_vod_t's relationship with the target is a real,
forward-time influence, it should show a WEAKER and less certain
relationship with the target's past than with the target's future.

NOTE: an earlier draft of this placebo dropped the own-lag control
entirely, which produced a misleadingly huge "effect" - that was a
flaw in the placebo's design (missing the same control the real model
uses), not a problem with the real result. This version fixes that by
keeping the model structure identical to Stage 6, only changing the outcome.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
Output: data/processed/placebo_panel_monthly.csv
        printed regression summary + comparison to the real (forward) result
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

    neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patches:
        neighbors = [n for n in neighbor_map.get(pid, []) if n in vod_pivot.columns]
        neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    # Same structure as the real model (own_t, neighbor_t, precip_t as
    # predictors) but the outcome is own_vod at t-1 (the PAST) instead
    # of t+1 (the future, as in Stage 6/8).
    records = []
    for pid in patches:
        own_t = vod_pivot[pid].values
        neigh_t = neighbor_vod[pid].values
        precip_t = precip_pivot[pid].values
        for i in range(1, len(dates)):
            records.append((pid, own_t[i], neigh_t[i], precip_t[i], own_t[i-1]))

    panel = pd.DataFrame(records, columns=[
        "patch_id", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "own_vod_past"
    ]).dropna()
    panel.to_csv(os.path.join(OUT_DIR, "placebo_panel_monthly.csv"), index=False)
    print("Placebo panel shape:", panel.shape)

    m = smf.ols("own_vod_past ~ own_vod_t + neighbor_vod_t + precip_anom_t", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
    print("\n===== PLACEBO TEST: predicting target's PAST value =====")
    print(m.summary().tables[1])

    coef = m.params["neighbor_vod_t"]
    pval = m.pvalues["neighbor_vod_t"]
    print(f"\nPlacebo neighbor coefficient: {coef:.4f}, p-value: {pval:.4f}")
    print("Compare to the REAL (forward-time, Stage 6/8) result: coef ~0.087, p<0.001")
    print("\nA good placebo result shows a SMALLER and/or less significant coefficient")
    print("here than in the real forward-time test - some residual correlation is")
    print("expected since neighboring patches share conditions regardless of time")
    print("direction, but it should clearly be weaker than the real effect.")
    if pval < 0.05 and coef >= 0.07:
        print("\nWARNING: placebo coefficient is comparable in size AND significant -")
        print("this would need investigation before trusting the real result.")
    else:
        print("\nPlacebo result is weaker/less certain than the real result - supports")
        print("the real forward-time finding reflecting genuine dynamics.")

if __name__ == "__main__":
    main()