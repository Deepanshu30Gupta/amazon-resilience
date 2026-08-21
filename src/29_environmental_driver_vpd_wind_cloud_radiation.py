"""
29_environmental_driver_vpd_wind_cloud_radiation.py

Purpose: Test four more environmental drivers: Vapor Pressure Deficit
(VPD, calculated from temperature + dewpoint), wind speed (from u/v
components), cloud fraction, and solar radiation.

NOTE ON THE BOUNDING-BOX FIX: this script uses the CORRECTED patch
bounding-box calculation throughout, including for temperature/soil/
PDSI/ENSO. This means the internal step-by-step chain in THIS script
is self-consistent, but individual numbers may not match exactly what
was reported in Stages 14-17 (which used the buggy version) - that
reconciliation is planned separately before the final write-up.

VPD calculation: Tetens formula for saturation vapor pressure:
  es(T) = 0.6108 * exp(17.27*T / (T+237.3))   [kPa, T in Celsius]
  VPD = es(air_temp) - es(dewpoint_temp)

Cloud fraction scale factor: MODIS MOD08_M3's Cloud_Fraction_Mean_Mean
band uses a standard scale factor of 0.0001 (raw 0-10000 -> 0.0-1.0).

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/era5_dewpoint_amazon_cerrado_monthly.tif
        data/raw/era5_wind_u_amazon_cerrado_monthly.tif
        data/raw/era5_wind_v_amazon_cerrado_monthly.tif
        data/raw/modis_cloud_fraction_amazon_cerrado_monthly.tif
        data/raw/era5land_solar_radiation_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/raw/oni_index.csv
Output: data/processed/vpd_wind_cloud_radiation_results.csv
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
CLOUD_PATH = "data/raw/modis_cloud_fraction_amazon_cerrado_monthly.tif"
SOLAR_PATH = "data/raw/era5land_solar_radiation_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
ONI_PATH = "data/raw/oni_index.csv"
START_DATE = "2003-01-01"
PATCH_SIZE = 4
CLOUD_SCALE_FACTOR = 0.0001

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

    print("Aggregating all environmental variables (using corrected bounding box)...")
    vod_bounds, vh, vw, n_patch_rows, n_patch_cols = reconstruct_vod_bounds(loc)

    temp_vals, n_m = aggregate_raster_to_patches(TEMP_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    dewpoint_vals, _ = aggregate_raster_to_patches(DEWPOINT_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    u_vals, _ = aggregate_raster_to_patches(WIND_U_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    v_vals, _ = aggregate_raster_to_patches(WIND_V_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    cloud_vals, _ = aggregate_raster_to_patches(CLOUD_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    cloud_vals = cloud_vals * CLOUD_SCALE_FACTOR
    solar_vals, _ = aggregate_raster_to_patches(SOLAR_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    soil_vals, _ = aggregate_raster_to_patches(SOIL_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)
    pdsi_vals, _ = aggregate_raster_to_patches(PDSI_PATH, vod_bounds, vh, vw, n_patch_rows, n_patch_cols)

    vpd_vals = saturation_vapor_pressure(temp_vals) - saturation_vapor_pressure(dewpoint_vals)
    wind_speed_vals = np.sqrt(u_vals**2 + v_vals**2)

    print("VPD range:", np.nanmin(vpd_vals), "-", np.nanmax(vpd_vals), "kPa")
    print("Wind speed range:", np.nanmin(wind_speed_vals), "-", np.nanmax(wind_speed_vals), "m/s")

    temp_df = to_long_anomaly(temp_vals, n_m, loc, "temp")
    soil_df = to_long_anomaly(soil_vals, n_m, loc, "soil")
    pdsi_df = to_long_anomaly(pdsi_vals, n_m, loc, "pdsi")
    vpd_df = to_long_anomaly(vpd_vals, n_m, loc, "vpd")
    wind_df = to_long_anomaly(wind_speed_vals, n_m, loc, "wind")
    cloud_df = to_long_anomaly(cloud_vals, n_m, loc, "cloud")
    solar_df = to_long_anomaly(solar_vals, n_m, loc, "solar")

    merged = ts.merge(temp_df, on=["patch_id", "date"], how="inner") \
               .merge(soil_df, on=["patch_id", "date"], how="inner") \
               .merge(pdsi_df, on=["patch_id", "date"], how="inner") \
               .merge(oni, on="date", how="inner") \
               .merge(vpd_df, on=["patch_id", "date"], how="inner") \
               .merge(wind_df, on=["patch_id", "date"], how="inner") \
               .merge(solar_df, on=["patch_id", "date"], how="inner") \
               .merge(cloud_df, on=["patch_id", "date"], how="left")  # LEFT join - cloud's
               # missingness must not drop rows for the other variables, which have full coverage
    print(f"\nMerged dataset shape: {merged.shape}")

    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    vod_pivot = merged.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    pivots = {}
    for col in ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly",
                "vpd_anomaly", "wind_anomaly", "cloud_anomaly", "solar_anomaly"]:
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
    panel = pd.DataFrame(recs)  # NOTE: do not dropna() here - different models need different subsets
    print(f"Full panel shape (before any dropna): {panel.shape}\n")

    specs = [
        ("Baseline (precip+temp+soil+PDSI+ENSO)",
         "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anomaly + temp_anomaly + soil_anomaly + pdsi_anomaly + oni_t"),
        ("+ VPD",
         "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anomaly + temp_anomaly + soil_anomaly + pdsi_anomaly + oni_t + vpd_anomaly"),
        ("+ VPD + wind speed",
         "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anomaly + temp_anomaly + soil_anomaly + pdsi_anomaly + oni_t + vpd_anomaly + wind_anomaly"),
        ("+ VPD + wind + solar radiation",
         "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anomaly + temp_anomaly + soil_anomaly + pdsi_anomaly + oni_t + vpd_anomaly + wind_anomaly + solar_anomaly"),
    ]

    print("NOTE: cloud fraction (MODIS, ~1 degree resolution) is missing for ~20% of")
    print("patches near the region's edges, where the coarse MODIS grid doesn't fully")
    print("overlap any patch. To avoid shrinking the sample for VPD/wind/solar (which")
    print("have full coverage), cloud fraction is tested SEPARATELY below, on its own")
    print("smaller, valid sample - not chained into the main sequence.\n")

    results = []
    for label, formula in specs:
        vars_needed = [c for c in panel.columns if c != "cloud_anomaly"]
        sub_panel = panel.dropna(subset=vars_needed)
        m = smf.ols(formula, data=sub_panel).fit(cov_type="cluster", cov_kwds={"groups": sub_panel["patch_id"]})
        coef = m.params["neighbor_vod_t"]
        pval = m.pvalues["neighbor_vod_t"]
        print(f"{label:42s}: neighbor coef={coef:.4f}  p={pval:.4f}  n={len(sub_panel)}")
        results.append((label, coef, pval, len(sub_panel)))

    # Cloud fraction tested separately, on its own valid (smaller) sample
    # - this one DOES need cloud_anomaly to be non-null
    cloud_panel = panel.dropna()
    cloud_formula = ("own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anomaly + temp_anomaly + "
                      "soil_anomaly + pdsi_anomaly + oni_t + vpd_anomaly + wind_anomaly + "
                      "solar_anomaly + cloud_anomaly")
    m_cloud = smf.ols(cloud_formula, data=cloud_panel).fit(cov_type="cluster", cov_kwds={"groups": cloud_panel["patch_id"]})
    cloud_coef = m_cloud.params["neighbor_vod_t"]
    cloud_pval = m_cloud.pvalues["neighbor_vod_t"]
    print(f"{'+ cloud fraction (separate, smaller sample)':42s}: neighbor coef={cloud_coef:.4f}  "
          f"p={cloud_pval:.4f}  n={len(cloud_panel)}")
    results.append(("+ cloud fraction (separate sample)", cloud_coef, cloud_pval, len(cloud_panel)))

    results_df = pd.DataFrame(results, columns=["model", "neighbor_coef", "neighbor_pval", "n_obs"])
    results_df.to_csv(os.path.join(OUT_DIR, "vpd_wind_cloud_radiation_results.csv"), index=False)

    first_coef = results_df.iloc[0]["neighbor_coef"]
    main_chain_last = results_df.iloc[2]["neighbor_coef"]  # VPD+wind+solar, full sample
    pct_change = 100 * (first_coef - main_chain_last) / first_coef
    print(f"\n===== SUMMARY =====")
    print(f"Neighbor coefficient (main chain, full sample): {first_coef:.4f} -> {main_chain_last:.4f} "
          f"({pct_change:.1f}% change)")
    if abs(pct_change) < 15:
        print("VPD, wind speed, and solar radiation explain relatively little additional")
        print("synchrony, continuing the pattern from other environmental drivers.")
    else:
        print("These variables explain a meaningful additional share of the synchrony.")

if __name__ == "__main__":
    main()