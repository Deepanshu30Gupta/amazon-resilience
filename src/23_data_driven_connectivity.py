"""
23_data_driven_connectivity.py

Purpose: Stage 22 showed geographic neighbors barely improve
prediction (0.09% RMSE). This asks a different question: instead of
assuming "neighbor" means "geographically close," can we find which
patches actually behave similarly over time, regardless of distance -
and does THAT connectivity improve prediction where geography didn't?

IMPORTANT LEAKAGE SAFEGUARD: patch similarity is computed using ONLY
the training period (2003-2015) VOD anomaly data - never the held-out
test period (2016-2018).

Part A: build the data-driven connectivity network (top-K most
correlated patches per patch, K=5 to roughly match geographic
adjacency's average neighbor count), and check how geographically
close/far these data-driven neighbors actually are.

Part B: compare three models on the same held-out 2016-2018 test
period as Stage 22:
  Model 1: no spatial info
  Model 2: geographic neighbors
  Model 3: data-driven neighbors (NEW)

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
Output: data/processed/data_driven_adjacency.csv
        data/processed/data_driven_connectivity_results.csv
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
TOP_K = 5

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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])

    vod_pivot_all = ts.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    vod_train_only = vod_pivot_all[vod_pivot_all.index < TEST_START]
    print(f"Computing patch-to-patch similarity using {len(vod_train_only)} training months only...")

    corr_matrix = vod_train_only.corr()
    patches = corr_matrix.columns.to_list()

    data_driven_edges = []
    for pid in patches:
        sims = corr_matrix[pid].drop(pid).sort_values(ascending=False)
        top_neighbors = sims.head(TOP_K).index.tolist()
        for n in top_neighbors:
            data_driven_edges.append((pid, n))
    dd_adj = pd.DataFrame(data_driven_edges, columns=["patch_id", "neighbor_id"])
    dd_adj.to_csv(os.path.join(OUT_DIR, "data_driven_adjacency.csv"), index=False)

    loc_lookup = loc.set_index("patch_id")[["lat", "lon"]]
    dd_adj_geo = dd_adj.merge(loc_lookup, left_on="patch_id", right_index=True) \
                       .merge(loc_lookup, left_on="neighbor_id", right_index=True, suffixes=("_p", "_n"))
    dd_adj_geo["dist_km"] = haversine(dd_adj_geo["lat_p"], dd_adj_geo["lon_p"],
                                        dd_adj_geo["lat_n"], dd_adj_geo["lon_n"])

    print(f"\n===== PART A: How geographically distant are data-driven neighbors? =====")
    print(dd_adj_geo["dist_km"].describe())
    frac_far = (dd_adj_geo["dist_km"] > 500).mean()
    print(f"\nFraction of data-driven neighbor pairs that are >500km apart: {100*frac_far:.1f}%")
    if frac_far > 0.3:
        print("-> A substantial share of 'learned' connections are NOT geographically close -")
        print("   consistent with the connectivity reflecting something other than simple")
        print("   physical proximity (e.g. shared atmospheric patterns).")
    else:
        print("-> Most 'learned' connections ARE geographically close - the data-driven")
        print("   network largely reproduces geographic adjacency rather than finding")
        print("   something new.")

    print("\nAggregating temperature, soil moisture, and PDSI...")
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
    dd_neighbor_map = dd_adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()

    vod_pivot = merged.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    precip_pivot = merged.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    temp_pivot = merged.pivot(index="date", columns="patch_id", values="temp_anomaly").sort_index()
    soil_pivot = merged.pivot(index="date", columns="patch_id", values="soil_anomaly").sort_index()
    pdsi_pivot = merged.pivot(index="date", columns="patch_id", values="pdsi_anomaly").sort_index()
    oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
    dates_list = vod_pivot.index.to_list()
    patch_cols = vod_pivot.columns.to_list()

    def build_neighbor_avg(neighbor_map):
        nv = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
        for pid in patch_cols:
            neighbors = [n for n in neighbor_map.get(pid, []) if n in vod_pivot.columns]
            nv[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan
        return nv

    geo_neighbor_vod = build_neighbor_avg(geo_neighbor_map)
    dd_neighbor_vod = build_neighbor_avg(dd_neighbor_map)

    recs = []
    for pid in patch_cols:
        own_t = vod_pivot[pid].values
        geo_t = geo_neighbor_vod[pid].values
        dd_t = dd_neighbor_vod[pid].values
        precip_t = precip_pivot[pid].values
        temp_t = temp_pivot[pid].values
        soil_t = soil_pivot[pid].values
        pdsi_t = pdsi_pivot[pid].values
        oni_t = oni_series.reindex(dates_list).values
        for i in range(len(dates_list) - 1):
            recs.append((pid, dates_list[i+1], own_t[i], geo_t[i], dd_t[i], precip_t[i],
                         temp_t[i], soil_t[i], pdsi_t[i], oni_t[i], own_t[i+1]))
    panel = pd.DataFrame(recs, columns=[
        "patch_id", "target_date", "own_vod_t", "geo_neighbor_vod_t", "dd_neighbor_vod_t",
        "precip_anom_t", "temp_anom_t", "soil_anom_t", "pdsi_anom_t", "oni_t", "own_vod_t1"
    ]).dropna()

    train = panel[panel["target_date"] < TEST_START]
    test = panel[panel["target_date"] >= TEST_START]
    print(f"\nTrain: {len(train)} obs, Test: {len(test)} obs")

    base_formula = "own_vod_t1 ~ own_vod_t + precip_anom_t + temp_anom_t + soil_anom_t + pdsi_anom_t + oni_t"
    formulas = {
        "Model 1 (no spatial)": base_formula,
        "Model 2 (geographic neighbors)": base_formula + " + geo_neighbor_vod_t",
        "Model 3 (data-driven neighbors)": base_formula + " + dd_neighbor_vod_t",
    }

    results = []
    print("\n===== PART B: 3-way out-of-sample prediction comparison =====")
    for label, formula in formulas.items():
        m = smf.ols(formula, data=train).fit()
        pred = m.predict(test)
        actual = test["own_vod_t1"].values
        r, a = rmse(actual, pred), mae(actual, pred)
        print(f"{label:35s}: RMSE={r:.5f}  MAE={a:.5f}")
        results.append((label, r, a))

    results_df = pd.DataFrame(results, columns=["model", "rmse", "mae"])
    results_df.to_csv(os.path.join(OUT_DIR, "data_driven_connectivity_results.csv"), index=False)

    rmse_1 = results_df.iloc[0]["rmse"]
    rmse_2 = results_df.iloc[1]["rmse"]
    rmse_3 = results_df.iloc[2]["rmse"]
    print(f"\n===== SUMMARY =====")
    print(f"Geographic neighbor improvement over no-spatial:  {100*(rmse_1-rmse_2)/rmse_1:.2f}%")
    print(f"Data-driven neighbor improvement over no-spatial: {100*(rmse_1-rmse_3)/rmse_1:.2f}%")
    if rmse_3 < rmse_2 and (rmse_1 - rmse_3) / rmse_1 > 0.02:
        print("\n-> Data-driven connectivity outperforms geographic adjacency meaningfully.")
        print("   Geographic distance misses important relationships - THIS JUSTIFIES")
        print("   trying a GNN with a learned adjacency structure.")
    elif rmse_3 < rmse_2:
        print("\n-> Data-driven connectivity slightly outperforms geographic adjacency,")
        print("   but the improvement is modest - a GNN might help a little, tempered")
        print("   expectations warranted.")
    else:
        print("\n-> Data-driven connectivity does NOT outperform geographic adjacency.")
        print("   Neither notion of 'neighbor' meaningfully improves prediction - the")
        print("   synchrony appears to be real but not predictively useful by either")
        print("   definition. This argues against investing further in a GNN for")
        print("   prediction purposes.")

if __name__ == "__main__":
    main()