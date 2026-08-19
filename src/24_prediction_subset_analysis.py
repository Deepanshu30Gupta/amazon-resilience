"""
24_prediction_subset_analysis.py

Purpose: Stage 22/23 showed geographic and data-driven neighbor info
barely improve 1-month-ahead prediction on average. This checks
whether that average is hiding a real effect in specific conditions,
before making a final call on whether a GNN is justified:

  A) Different forecast horizons (1, 2, 3, 6 months ahead)
  B) Strong vs. weak ENSO months (|ONI| >= 1.0 vs < 1.0)
  C) Resilience-loss patches vs. others

Uses the SAME train (2003-2015) / test (2016-2018) split as Stage
22/23.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/data_driven_adjacency.csv
        data/processed/patch_resilience_trend.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
        data/processed/patch_locations.csv
Output: data/processed/prediction_subset_results.csv
        data/processed/prediction_subset_enso_results.csv
        data/processed/prediction_subset_resilience_results.csv
        printed results for all three sub-analyses
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
HORIZONS = [1, 2, 3, 6]

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

def main():
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

    geo_neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    vod_pivot = merged.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = merged.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    temp_pivot = merged.pivot(index="date", columns="patch_id", values="temp_anomaly").sort_index()
    soil_pivot = merged.pivot(index="date", columns="patch_id", values="soil_anomaly").sort_index()
    pdsi_pivot = merged.pivot(index="date", columns="patch_id", values="pdsi_anomaly").sort_index()
    oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
    dates_list = vod_pivot.index.to_list()
    patch_cols = vod_pivot.columns.to_list()

    geo_neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patch_cols:
        neighbors = [n for n in geo_neighbor_map.get(pid, []) if n in vod_pivot.columns]
        geo_neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    base_formula = "own_vod_t1 ~ own_vod_t + precip_anom_t + temp_anom_t + soil_anom_t + pdsi_anom_t + oni_t"
    spatial_formula = base_formula + " + neighbor_vod_t"

    def build_panel_for_horizon(h):
        recs = []
        for pid in patch_cols:
            own_t = vod_pivot[pid].values
            neigh_t = geo_neighbor_vod[pid].values
            precip_t = precip_pivot[pid].values
            temp_t = temp_pivot[pid].values
            soil_t = soil_pivot[pid].values
            pdsi_t = pdsi_pivot[pid].values
            oni_t = oni_series.reindex(dates_list).values
            for i in range(len(dates_list) - h):
                recs.append((pid, dates_list[i+h], own_t[i], neigh_t[i], precip_t[i], temp_t[i],
                             soil_t[i], pdsi_t[i], oni_t[i], own_t[i+h]))
        return pd.DataFrame(recs, columns=[
            "patch_id", "target_date", "own_vod_t", "neighbor_vod_t", "precip_anom_t",
            "temp_anom_t", "soil_anom_t", "pdsi_anom_t", "oni_t", "own_vod_t1"
        ]).dropna()

    print("\n===== PART A: Spatial info improvement by forecast horizon =====")
    horizon_results = []
    panel_1mo = None
    for h in HORIZONS:
        panel_h = build_panel_for_horizon(h)
        train_h = panel_h[panel_h["target_date"] < TEST_START]
        test_h = panel_h[panel_h["target_date"] >= TEST_START]

        m_base = smf.ols(base_formula, data=train_h).fit()
        m_spatial = smf.ols(spatial_formula, data=train_h).fit()
        actual = test_h["own_vod_t1"].values
        rmse_base = rmse(actual, m_base.predict(test_h))
        rmse_spatial = rmse(actual, m_spatial.predict(test_h))
        improvement = 100 * (rmse_base - rmse_spatial) / rmse_base
        print(f"Horizon {h} month(s): RMSE no-spatial={rmse_base:.5f}  "
              f"RMSE with-spatial={rmse_spatial:.5f}  improvement={improvement:.2f}%")
        horizon_results.append((h, rmse_base, rmse_spatial, improvement))

        if h == 1:
            panel_1mo = panel_h

    horizon_df = pd.DataFrame(horizon_results, columns=["horizon_months", "rmse_no_spatial", "rmse_spatial", "pct_improvement"])

    print("\n===== PART B: Spatial info improvement, strong vs weak ENSO =====")
    train_1 = panel_1mo[panel_1mo["target_date"] < TEST_START]
    test_1 = panel_1mo[panel_1mo["target_date"] >= TEST_START]
    m_base_1 = smf.ols(base_formula, data=train_1).fit()
    m_spatial_1 = smf.ols(spatial_formula, data=train_1).fit()

    enso_results = []
    for label, mask in [("Strong ENSO (|ONI|>=1.0)", test_1["oni_t"].abs() >= 1.0),
                         ("Weak/neutral ENSO (|ONI|<1.0)", test_1["oni_t"].abs() < 1.0)]:
        sub = test_1[mask]
        if len(sub) < 20:
            continue
        actual = sub["own_vod_t1"].values
        rb = rmse(actual, m_base_1.predict(sub))
        rs = rmse(actual, m_spatial_1.predict(sub))
        imp = 100 * (rb - rs) / rb
        print(f"{label:32s}: n={len(sub):5d}  RMSE no-spatial={rb:.5f}  "
              f"RMSE with-spatial={rs:.5f}  improvement={imp:.2f}%")
        enso_results.append((label, len(sub), rb, rs, imp))

    print("\n===== PART C: Spatial info improvement, resilience-loss patches vs others =====")
    loss_patches = set(trend[(trend["kendall_tau"] > 0) & (trend["p_value"] < 0.05)]["patch_id"])
    test_1 = test_1.copy()
    test_1["is_loss_patch"] = test_1["patch_id"].isin(loss_patches)

    resilience_results = []
    for label, mask in [("Resilience-LOSS patches", test_1["is_loss_patch"]),
                         ("Other patches", ~test_1["is_loss_patch"])]:
        sub = test_1[mask]
        if len(sub) < 20:
            continue
        actual = sub["own_vod_t1"].values
        rb = rmse(actual, m_base_1.predict(sub))
        rs = rmse(actual, m_spatial_1.predict(sub))
        imp = 100 * (rb - rs) / rb
        print(f"{label:25s}: n={len(sub):5d}  RMSE no-spatial={rb:.5f}  "
              f"RMSE with-spatial={rs:.5f}  improvement={imp:.2f}%")
        resilience_results.append((label, len(sub), rb, rs, imp))

    horizon_df.to_csv(os.path.join(OUT_DIR, "prediction_subset_results.csv"), index=False)
    pd.DataFrame(enso_results, columns=["subset", "n", "rmse_no_spatial", "rmse_spatial", "pct_improvement"]) \
        .to_csv(os.path.join(OUT_DIR, "prediction_subset_enso_results.csv"), index=False)
    pd.DataFrame(resilience_results, columns=["subset", "n", "rmse_no_spatial", "rmse_spatial", "pct_improvement"]) \
        .to_csv(os.path.join(OUT_DIR, "prediction_subset_resilience_results.csv"), index=False)

    print("\n===== OVERALL SUMMARY =====")
    max_improvement = max(
        horizon_df["pct_improvement"].max(),
        max([r[4] for r in enso_results], default=0),
        max([r[4] for r in resilience_results], default=0)
    )
    print(f"Largest improvement found across all subsets tested: {max_improvement:.2f}%")
    if max_improvement > 5:
        print("-> A meaningful improvement DOES exist in at least one specific condition.")
        print("   This is a more targeted, defensible justification for trying a GNN than")
        print("   the flat average - worth understanding which condition specifically.")
    else:
        print("-> No subset tested shows a meaningful improvement. The negligible average")
        print("   result from Stage 22/23 is NOT hiding a real effect in these specific")
        print("   conditions. This is fairly strong evidence against investing in a GNN")
        print("   for prediction purposes.")

if __name__ == "__main__":
    main()