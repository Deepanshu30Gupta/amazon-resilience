"""
22_prediction_comparison.py

Purpose: Move from "is there a statistically significant relationship"
to "does spatial information actually improve prediction." This is the
bridge between the statistical analysis so far and the eventual GNN
comparison - if a simple model with neighbor information doesn't
predict noticeably better than one without it, that's an important
result in itself (and argues against investing in a more complex GNN).

Train/test split: by TIME, not by patch - train on 2003-2015, test on
2016-2018 (last 36 months held out). This respects the temporal
structure of the data (a real forecasting scenario) and avoids the
data leakage that a random train/test split would cause in a panel.

Two models compared, both predicting next-month VOD anomaly:
  Model A (NO SPATIAL INFO): own history + precipitation + temperature
    + soil moisture + PDSI + ENSO
  Model B (WITH SPATIAL INFO): Model A + neighbor's mean VOD anomaly

Metrics: RMSE and MAE on the held-out test period. Lower = better.
Also reports the % improvement from adding spatial information.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
        data/processed/patch_locations.csv
Output: data/processed/prediction_comparison_results.csv
        printed comparison
"""

import rasterio
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os

OUT_DIR = "data/processed"
TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
ONI_PATH = "data/raw/oni_index.csv"
START_DATE = "2003-01-01"
PATCH_SIZE = 4
TEST_START = "2016-01-01"

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

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
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
        temp_t = temp_pivot[pid].values
        soil_t = soil_pivot[pid].values
        pdsi_t = pdsi_pivot[pid].values
        oni_t = oni_series.reindex(dates_list).values
        for i in range(len(dates_list) - 1):
            recs.append((pid, dates_list[i+1], own_t[i], neigh_t[i], precip_t[i], temp_t[i],
                         soil_t[i], pdsi_t[i], oni_t[i], own_t[i+1]))
    panel = pd.DataFrame(recs, columns=[
        "patch_id", "target_date", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "temp_anom_t",
        "soil_anom_t", "pdsi_anom_t", "oni_t", "own_vod_t1"
    ]).dropna()

    train = panel[panel["target_date"] < TEST_START]
    test = panel[panel["target_date"] >= TEST_START]
    print(f"\nTrain: {len(train)} obs (through {TEST_START}), Test: {len(test)} obs (from {TEST_START})")

    formula_a = "own_vod_t1 ~ own_vod_t + precip_anom_t + temp_anom_t + soil_anom_t + pdsi_anom_t + oni_t"
    formula_b = formula_a + " + neighbor_vod_t"

    model_a = smf.ols(formula_a, data=train).fit()
    model_b = smf.ols(formula_b, data=train).fit()

    pred_a = model_a.predict(test)
    pred_b = model_b.predict(test)
    actual = test["own_vod_t1"].values

    rmse_a, mae_a = rmse(actual, pred_a), mae(actual, pred_a)
    rmse_b, mae_b = rmse(actual, pred_b), mae(actual, pred_b)

    print("\n===== OUT-OF-SAMPLE PREDICTION RESULTS (test period only) =====")
    print(f"Model A (NO spatial info):   RMSE={rmse_a:.5f}  MAE={mae_a:.5f}")
    print(f"Model B (WITH spatial info): RMSE={rmse_b:.5f}  MAE={mae_b:.5f}")

    rmse_improvement = 100 * (rmse_a - rmse_b) / rmse_a
    mae_improvement = 100 * (mae_a - mae_b) / mae_a
    print(f"\nRMSE improvement from adding spatial info: {rmse_improvement:.2f}%")
    print(f"MAE improvement from adding spatial info:  {mae_improvement:.2f}%")

    results = pd.DataFrame([
        {"model": "A (no spatial)", "rmse": rmse_a, "mae": mae_a},
        {"model": "B (with spatial)", "rmse": rmse_b, "mae": mae_b},
    ])
    results.to_csv(os.path.join(OUT_DIR, "prediction_comparison_results.csv"), index=False)

    print("\n===== INTERPRETATION GUIDE =====")
    if rmse_improvement > 5:
        print("Spatial information provides a MEANINGFUL prediction improvement (>5% RMSE)")
        print("-> justifies investigating whether a more sophisticated spatial model (GNN)")
        print("   could improve on this further.")
    elif rmse_improvement > 0:
        print("Spatial information provides a SMALL prediction improvement (<5% RMSE).")
        print("-> a GNN might still be worth trying, but expectations should be modest -")
        print("   the geographic-neighbor model is already capturing most of the useful signal.")
    else:
        print("Spatial information does NOT improve prediction on held-out data, despite")
        print("being statistically significant in-sample. This is an important, genuine")
        print("finding: statistical significance and predictive usefulness are not the same")
        print("thing. This would argue AGAINST investing further effort in a GNN, since a")
        print("simple model already captures the useful signal just as well.")

if __name__ == "__main__":
    main()