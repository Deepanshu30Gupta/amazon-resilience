"""
17_environmental_driver_enso.py

Purpose: Test whether ENSO (via the Oceanic Nino Index, ONI) explains
some of the spatial synchrony that precipitation, temperature, soil
moisture, and drought (PDSI) could not (Stages 6/8/14/16). ENSO is
different in kind from those drivers - it's a basin/global-scale
climate oscillation, not a locally-measured variable, so it could
plausibly explain why the neighbor effect does not decay with distance
(Stage 7): if patches are moving together because they're all
responding to the same large-scale teleconnection, that would look
exactly like "no distance decay," without requiring any local
patch-to-patch influence.

Unlike the other environmental drivers, ONI is a SINGLE value per
month (not spatial) - the same ONI value applies to every patch in a
given month, since ENSO affects the whole basin (and beyond)
simultaneously.

Model progression (same step-by-step logic as Stage 14/16):
  Model 1: no controls
  Model 2: + precipitation
  Model 3: + precipitation + temperature
  Model 4: + precipitation + temperature + soil moisture + PDSI
  Model 5: + precipitation + temperature + soil moisture + PDSI + ONI

CAUTION: even if ONI substantially reduces the neighbor coefficient,
that supports an ASSOCIATION between the synchrony and the ENSO
signal, not proof that ENSO CAUSES the synchrony - report accordingly.

Input:  data/raw/oni_index.csv
        data/raw/era5_temperature_amazon_cerrado_monthly.tif
        data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif
        data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif
        data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_locations.csv
        data/processed/patch_adjacency.csv
Output: data/processed/enso_driver_results.csv
        printed step-by-step comparison
"""

import rasterio
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import os

ONI_PATH = "data/raw/oni_index.csv"
TEMP_PATH = "data/raw/era5_temperature_amazon_cerrado_monthly.tif"
SOIL_PATH = "data/raw/terraclimate_soil_moisture_amazon_cerrado_monthly.tif"
PDSI_PATH = "data/raw/terraclimate_pdsi_amazon_cerrado_monthly.tif"
OUT_DIR = "data/processed"
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
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])

    vod_bounds, vh, vw, n_patch_rows, n_patch_cols = reconstruct_vod_bounds(loc)

    print("Aggregating temperature, soil moisture, and PDSI into the patch grid...")
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
    print(f"Merged dataset shape: {merged.shape}")

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
            recs.append((pid, own_t[i], neigh_t[i], precip_t[i], temp_t[i], soil_t[i],
                         pdsi_t[i], oni_t[i], own_t[i+1]))
    panel = pd.DataFrame(recs, columns=[
        "patch_id", "own_vod_t", "neighbor_vod_t", "precip_anom_t", "temp_anom_t",
        "soil_anom_t", "pdsi_anom_t", "oni_t", "own_vod_t1"
    ]).dropna()
    print(f"Final regression panel shape: {panel.shape}\n")

    specs = [
        ("No controls", "own_vod_t1 ~ own_vod_t + neighbor_vod_t"),
        ("+ precipitation", "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t"),
        ("+ precip + temp", "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + temp_anom_t"),
        ("+ precip + temp + soil + PDSI",
         "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + temp_anom_t + soil_anom_t + pdsi_anom_t"),
        ("+ precip + temp + soil + PDSI + ONI (ENSO)",
         "own_vod_t1 ~ own_vod_t + neighbor_vod_t + precip_anom_t + temp_anom_t + soil_anom_t + pdsi_anom_t + oni_t"),
    ]
    results = []
    for label, formula in specs:
        m = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
        coef = m.params["neighbor_vod_t"]
        pval = m.pvalues["neighbor_vod_t"]
        print(f"{label:45s}: neighbor coef={coef:.4f}  p={pval:.4f}")
        results.append((label, coef, pval))
        if "ONI" in label:
            oni_coef = m.params["oni_t"]
            oni_pval = m.pvalues["oni_t"]
            print(f"{'  (ONI itself)':45s}: coef={oni_coef:.4f}  p={oni_pval:.4f}")

    results_df = pd.DataFrame(results, columns=["model", "neighbor_coef", "neighbor_pval"])
    results_df.to_csv(os.path.join(OUT_DIR, "enso_driver_results.csv"), index=False)

    first_coef, last_coef = results_df.iloc[0]["neighbor_coef"], results_df.iloc[-1]["neighbor_coef"]
    pct_change = 100 * (first_coef - last_coef) / first_coef
    print(f"\n===== SUMMARY =====")
    print(f"Neighbor coefficient: {first_coef:.4f} -> {last_coef:.4f} ({pct_change:.1f}% change)")
    if abs(pct_change) < 15:
        print("ENSO explains relatively little of the synchrony on top of the other drivers -")
        print("same pattern as everything else tested. The neighbor effect remains largely")
        print("unexplained by environmental drivers, including large-scale teleconnections.")
    else:
        print("ENSO explains a meaningful share of the apparent synchrony - the long-distance")
        print("coherence may be substantially associated with this large-scale climate")
        print("oscillation, though this is an association, not proof of causation.")

if __name__ == "__main__":
    main()