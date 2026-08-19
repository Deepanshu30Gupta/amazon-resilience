"""
25_counterfactual_analysis.py

Purpose: Answer the project's original counterfactual question in a
concrete, quantified way: "What would a patch's vegetation trajectory
have looked like if its neighboring patch had NOT experienced its
observed (often anomalously poor) state?"

IMPORTANT FRAMING: because the underlying model is a linear regression,
this is a MODEL-BASED counterfactual, computed under the model's
assumptions (linearity, the specific confounders controlled for, no
unmeasured confounding) - NOT a causal proof. Given Stages 9-11 found
only weak/borderline evidence of genuine directional influence (as
opposed to synchrony), this analysis should be read as "what the
fitted model implies," not "what would actually have happened in
reality." This caveat is stated explicitly in the output.

Method: fit the full model (own history + precip + temp + soil + PDSI
+ ENSO + neighbor) on the complete dataset. For every observation,
compute two predictions:
  - Observed: using the neighbor's actual VOD anomaly at time t
  - Counterfactual: using neighbor VOD anomaly = 0 (i.e. "neighbor was
    at its normal/expected state, not anomalously good or bad")
The difference between these two predictions is the model-implied
"neighbor effect" for that specific observation.

We report this overall, and specifically for the scientifically
relevant scenario: months where the neighbor was in an anomalously
POOR state (neighbor_vod_t < 0) - the direct counterfactual the
original research question asked about.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_resilience_trend.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
        data/processed/patch_locations.csv
Output: data/processed/counterfactual_results.csv
        figures/counterfactual_effect_distribution.png
        printed summary
"""

import rasterio
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

OUT_DIR = "data/processed"
FIG_DIR = "figures"
TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
ONI_PATH = "data/raw/oni_index.csv"
START_DATE = "2003-01-01"
PATCH_SIZE = 4

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

def aggregate_raster_to_patches(path, vod_bounds, vh, vw, n_patch_rows, n_patch_cols):
    with rasterio.open(path) as src:
        data = src.read()
        bounds = src.bounds
    n_months, rh, rw = data.shape
    n = PATCH_SIZE
    lon_step_v = (vod_bounds[1] - vod_bounds[0]) / vw
    lat_step_v = (vod_bounds[2] - vod_bounds[3]) / vh
    lon_step_r = (bounds.right - bounds.left) / rw
    lat_step_r = (bounds.top - bounds.bottom) / rh
    r_lons = bounds.left + (np.arange(rw) + 0.5) * lon_step_r
    r_lats = bounds.top - (np.arange(rh) + 0.5) * lat_step_r
    patch_vals = np.full((n_months, n_patch_rows, n_patch_cols), np.nan)
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
            sub = data[:, row_mask, :][:, :, col_mask]
            patch_vals[:, pr, pc] = np.nanmean(sub, axis=(1, 2))
    return patch_vals, n_months

def to_long_anomaly(patch_vals, n_months, loc, colname):
    dates = pd.date_range(START_DATE, periods=n_months, freq="MS")
    records = []
    for _, r in loc.iterrows():
        pid, pr, pc = int(r.patch_id), int(r.row), int(r.col)
        for m in range(n_months):
            records.append((pid, dates[m], patch_vals[m, pr, pc]))
    df = pd.DataFrame(records, columns=["patch_id", "date", colname]).dropna()
    df["month"] = df["date"].dt.month
    df[colname + "_anomaly"] = df[colname] - df.groupby(["patch_id", "month"])[colname].transform("mean")
    return df[["patch_id", "date", colname + "_anomaly"]]

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    trend = pd.read_csv(os.path.join(OUT_DIR, "patch_resilience_trend.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])

    print("Aggregating temperature, soil moisture, and PDSI...")
    vod_bounds, vh, vw, n_patch_rows, n_patch_cols = reconstruct_vod_bounds(loc)
    temp_vals, n_m1 = aggregate_raster_to_patches(TEMP_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    soil_vals, n_m2 = aggregate_raster_to_patches(SOIL_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    pdsi_vals, n_m3 = aggregate_raster_to_patches(PDSI_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    temp_df = to_long_anomaly(temp_vals, n_m1, loc, "temp")
    soil_df = to_long_anomaly(soil_vals, n_m2, loc, "soil")
    pdsi_df = to_long_anomaly(pdsi_vals, n_m3, loc, "pdsi")

    merged = ts.merge(temp_df, on=["patch_id", "date"], how="inner") \
               .merge(soil_df, on=["patch_id", "date"], how="inner") \
               .merge(pdsi_df, on=["patch_id", "date"], how="inner") \
               .merge(oni, on="date", how="inner")

    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    vod_pivot = merged.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = merged.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    temp_pivot = merged.pivot(index="date", columns="patch_id", values="temp_anomaly").sort_index()
    soil_pivot = merged.pivot(index="date", columns="patch_id", values="soil_anomaly").sort_index()
    pdsi_pivot = merged.pivot(index="date", columns="patch_id", values="pdsi_anomaly").sort_index()
    oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
    dates_list = vod_pivot.index.to_list()
    patch_cols = vod_pivot.columns.to_list()

    neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patch_cols:
        neighbors = [n for n in neighbor_map.get(pid, []) if n in vod_pivot.columns]
        neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    recs = []
    for pid in patch_cols:
        own_t = vod_pivot[pid].values
        neigh_t = neighbor_vod[pid].values
        precip_t = precip_pivot[pid].values
        temp_t = temp_pivot[pid].values
        soil_t = soil_pivot[pid].values
        pdsi_t = pdsi_pivot[pid].values
        oni_t = oni_series.reindex(dates_list).values
        for i in range(len(dates_list) - 1):
            recs.append((pid, dates_list[i+1], own_t[i], neigh_t[i], precip_t[i], temp_t[i],
                         soil_t[i], pdsi_t[i], oni_t[i], own_t[i+1]))
    panel = pd.DataFrame(recs, columns=[
        "patch_id", "target_date", "own_vod_t", "neighbor_vod_t", "precip_anom_t",
        "temp_anom_t", "soil_anom_t", "pdsi_anom_t", "oni_t", "own_vod_t1"
    ]).dropna()

    formula = "own_vod_t1 ~ own_vod_t + precip_anom_t + temp_anom_t + soil_anom_t + pdsi_anom_t + oni_t + neighbor_vod_t"
    model = smf.ols(formula, data=panel).fit()
    print("Model fitted. Neighbor coefficient:", model.params["neighbor_vod_t"],
          "p-value:", model.pvalues["neighbor_vod_t"])

    observed_pred = model.predict(panel)
    counterfactual_panel = panel.copy()
    counterfactual_panel["neighbor_vod_t"] = 0.0
    counterfactual_pred = model.predict(counterfactual_panel)

    panel["counterfactual_effect"] = observed_pred - counterfactual_pred

    print("\n===== Counterfactual effect: observed vs. 'neighbor at normal state' =====")
    print(panel["counterfactual_effect"].describe())

    poor_neighbor = panel[panel["neighbor_vod_t"] < 0]
    print(f"\n===== Scenario: neighbor in anomalously POOR state (n={len(poor_neighbor)}) =====")
    print(f"Mean counterfactual effect: {poor_neighbor['counterfactual_effect'].mean():.6f}")
    print(f"(Negative value = target patch's predicted vegetation was WORSE than it would")
    print(f" have been if the neighbor had been at its normal state)")

    loss_patches = set(trend[(trend["kendall_tau"] > 0) & (trend["p_value"] < 0.05)]["patch_id"])
    loss_poor_neighbor = poor_neighbor[poor_neighbor["patch_id"].isin(loss_patches)]
    print(f"\n===== Same scenario, resilience-LOSS patches only (n={len(loss_poor_neighbor)}) =====")
    print(f"Mean counterfactual effect: {loss_poor_neighbor['counterfactual_effect'].mean():.6f}")

    panel.to_csv(os.path.join(OUT_DIR, "counterfactual_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(panel["counterfactual_effect"], bins=60, color='steelblue', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label="No effect")
    ax.set_xlabel("Model-implied counterfactual effect\n(observed prediction - 'neighbor at normal state' prediction)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of model-based counterfactual effects\n(all patches, all months)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "counterfactual_effect_distribution.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/counterfactual_effect_distribution.png")

    print("\n===== IMPORTANT CAVEAT =====")
    print("This is a MODEL-BASED counterfactual, computed under the fitted linear")
    print("regression's assumptions - it shows what the model implies would change,")
    print("not a proven causal effect. Given Stages 9-11 found only weak/borderline")
    print("evidence for genuine directional influence (vs. symmetric synchrony), this")
    print("counterfactual should be reported as a description of the fitted model's")
    print("behavior, not as evidence that neighboring patches causally affect each other.")

if __name__ == "__main__":
    main()