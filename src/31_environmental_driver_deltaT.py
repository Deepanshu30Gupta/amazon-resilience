"""
31_environmental_driver_deltaT.py

Purpose: Test Canopy vs. Ambient Temperature (deltaT) as an additional
driver, continuing the chain from Stage 30 (baseline + VPD + wind +
solar + RZSM, which ended at neighbor coefficient 0.0533).

deltaT = MODIS Land Surface Temperature (canopy/surface) - ERA5 2m air
temperature (ambient). A positive deltaT means the canopy surface runs
warmer than the surrounding air - a known indicator of plant water
stress, since healthy transpiration cools the canopy via evaporative
cooling; when stomata close under stress, that cooling effect weakens
and the canopy surface heats up relative to ambient air.

Uses the CORRECTED bounding-box calculation.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/era5_dewpoint_amazon_cerrado_monthly.tif
        data/raw/era5_wind_u_amazon_cerrado_monthly.tif
        data/raw/era5_wind_v_amazon_cerrado_monthly.tif
        data/raw/era5land_solar_radiation_amazon_cerrado_monthly.tif
        data/raw/era5land_rzsm_amazon_cerrado_monthly.tif
        data/raw/modis_lst_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
Output: data/processed/deltaT_driver_results.csv
        printed step-by-step comparison
"""

import rasterio
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os

OUT_DIR = "data/processed"
TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
DEWPOINT_PATH = "data/raw/era5_dewpoint_amazon_cerrado_monthly.tif"
WIND_U_PATH = "data/raw/era5_wind_u_amazon_cerrado_monthly.tif"
WIND_V_PATH = "data/raw/era5_wind_v_amazon_cerrado_monthly.tif"
SOLAR_PATH = "data/raw/era5land_solar_radiation_amazon_cerrado_monthly.tif"
RZSM_PATH = "data/raw/era5land_rzsm_amazon_cerrado_monthly.tif"
LST_PATH = "data/raw/modis_lst_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
ONI_PATH = "data/raw/oni_index.csv"
START_DATE = "2003-01-01"
PATCH_SIZE = 4

def reconstruct_vod_bounds(loc):
    n_patch_rows = loc["row"].max() + 1
    n_patch_cols = loc["col"].max() + 1
    lon_step_patch = loc[loc["row"] == 0].sort_values("col")["lon"].diff().dropna().median()
    lat_step_patch = -loc[loc["col"] == 0].sort_values("row")["lat"].diff().dropna().median()
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

def saturation_vapor_pressure(temp_c):
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])

    print("Aggregating all environmental variables...")
    vod_bounds, vh, vw, n_patch_rows, n_patch_cols = reconstruct_vod_bounds(loc)

    temp_vals, n_m = aggregate_raster_to_patches(TEMP_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    dewpoint_vals, _ = aggregate_raster_to_patches(DEWPOINT_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    u_vals, _ = aggregate_raster_to_patches(WIND_U_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    v_vals, _ = aggregate_raster_to_patches(WIND_V_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    solar_vals, _ = aggregate_raster_to_patches(SOLAR_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    rzsm_vals, _ = aggregate_raster_to_patches(RZSM_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    lst_vals, _ = aggregate_raster_to_patches(LST_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    soil_vals, _ = aggregate_raster_to_patches(SOIL_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    pdsi_vals, _ = aggregate_raster_to_patches(PDSI_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)

    vpd_vals = saturation_vapor_pressure(temp_vals) - saturation_vapor_pressure(dewpoint_vals)
    wind_speed_vals = np.sqrt(u_vals**2 + v_vals**2)
    deltaT_vals = lst_vals - temp_vals

    print("deltaT range:", np.nanmin(deltaT_vals), "-", np.nanmax(deltaT_vals), "C")
    n_missing_patches = np.isnan(deltaT_vals).any(axis=0).sum()
    print(f"Patches with any missing deltaT months: {n_missing_patches} / {n_patch_rows*n_patch_cols}")

    temp_df = to_long_anomaly(temp_vals, n_m, loc, "temp")
    soil_df = to_long_anomaly(soil_vals, n_m, loc, "soil")
    pdsi_df = to_long_anomaly(pdsi_vals, n_m, loc, "pdsi")
    vpd_df = to_long_anomaly(vpd_vals, n_m, loc, "vpd")
    wind_df = to_long_anomaly(wind_speed_vals, n_m, loc, "wind")
    solar_df = to_long_anomaly(solar_vals, n_m, loc, "solar")
    rzsm_df = to_long_anomaly(rzsm_vals, n_m, loc, "rzsm")
    deltaT_df = to_long_anomaly(deltaT_vals, n_m, loc, "deltaT")

    merged = ts.merge(temp_df, on=["patch_id", "date"], how="inner") \
               .merge(soil_df, on=["patch_id", "date"], how="inner") \
               .merge(pdsi_df, on=["patch_id", "date"], how="inner") \
               .merge(oni, on="date", how="inner") \
               .merge(vpd_df, on=["patch_id", "date"], how="inner") \
               .merge(wind_df, on=["patch_id", "date"], how="inner") \
               .merge(solar_df, on=["patch_id", "date"], how="inner") \
               .merge(rzsm_df, on=["patch_id", "date"], how="inner") \
               .merge(deltaT_df, on=["patch_id", "date"], how="left")
    print(f"Merged dataset shape: {merged.shape}")

    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    vod_pivot = merged.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    pivots = {}
    for col in ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly",
                "vpd_anomaly", "wind_anomaly", "solar_anomaly", "rzsm_anomaly", "deltaT_anomaly"]:
        pivots[col] = merged.pivot(index="date", columns="patch_id", values=col).sort_index()
    oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
    dates_list = vod_pivot.index.to_list()
    patches = vod_pivot.columns.to_list()

    neighbor_vod = pd.DataFrame(index=vod_pivot.index, columns=vod_pivot.columns, dtype=float)
    for pid in patches:
        neighbors = [n for n in neighbor_map.get(pid, []) if n in vod_pivot.columns]
        neighbor_vod[pid] = vod_pivot[neighbors].mean(axis=1) if neighbors else np.nan

    recs = []
    for pid in patches:
        row = {"own_vod_t": vod_pivot[pid].values, "neighbor_vod_t": neighbor_vod[pid].values}
        for col, piv in pivots.items():
            row[col] = piv[pid].values
        row["oni_t"] = oni_series.reindex(dates_list).values
        for i in range(len(dates_list) - 1):
            rec = {"patch_id": pid, "own_vod_t1": row["own_vod_t"][i+1]}
            for key in ["own_vod_t", "neighbor_vod_t"] + list(pivots.keys()) + ["oni_t"]:
                rec[key] = row[key][i]
            recs.append(rec)
    panel = pd.DataFrame(recs)
    print(f"Full panel shape (before any dropna): {panel.shape}\n")

    base_formula = ("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anomaly + temp_anomaly + "
                     "soil_anomaly + pdsi_anomaly + oni_t + vpd_anomaly + wind_anomaly + "
                     "solar_anomaly + rzsm_anomaly")

    vars_no_deltaT = [c for c in panel.columns if c != "deltaT_anomaly"]
    main_panel = panel.dropna(subset=vars_no_deltaT)
    m_main = smf.ols(base_formula, data=main_panel).fit(cov_type="cluster", cov_kwds={"groups": main_panel["patch_id"]})
    coef_main = m_main.params["neighbor_vod_t"]
    pval_main = m_main.pvalues["neighbor_vod_t"]
    print(f"{'Baseline (thru RZSM, full sample)':42s}: neighbor coef={coef_main:.4f}  p={pval_main:.4f}  n={len(main_panel)}")

    deltaT_panel = panel.dropna()
    deltaT_formula = base_formula + " + deltaT_anomaly"
    m_deltaT = smf.ols(deltaT_formula, data=deltaT_panel).fit(cov_type="cluster", cov_kwds={"groups": deltaT_panel["patch_id"]})
    coef_deltaT = m_deltaT.params["neighbor_vod_t"]
    pval_deltaT = m_deltaT.pvalues["neighbor_vod_t"]
    print(f"{'+ deltaT (separate, smaller sample)':42s}: neighbor coef={coef_deltaT:.4f}  p={pval_deltaT:.4f}  n={len(deltaT_panel)}")

    results_df = pd.DataFrame([
        {"model": "Baseline (thru RZSM)", "neighbor_coef": coef_main, "neighbor_pval": pval_main, "n": len(main_panel)},
        {"model": "+ deltaT", "neighbor_coef": coef_deltaT, "neighbor_pval": pval_deltaT, "n": len(deltaT_panel)},
    ])
    results_df.to_csv(os.path.join(OUT_DIR, "deltaT_driver_results.csv"), index=False)

    print(f"\n===== SUMMARY =====")
    print("Note: the two models above use different (though overlapping) samples due to")
    print("deltaT's cloud-driven missing data, so compare them as a directional signal")
    print("rather than an exact before/after difference.")

if __name__ == "__main__":
    main()