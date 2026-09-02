"""
46_robustness_lag3.py

Purpose: A TARGETED, pre-specified robustness test for Stage 45's
headline result (far-band 800-1100km, lag=3, negative association
surviving local+regional climate and geographic controls, with a null
backward placebo). NOT a fishing expedition for significance - each
check below is chosen independently of the result, run once, and
reported honestly regardless of outcome.

Four checks, focused specifically on the lag=3 result at the Stage 45
Step 4 (most stringent) control specification:

1. ALTERNATIVE DISTANCE BANDS: does the result depend on the exact
   800-1100km cutoff? Tests 750-1000km and 900-1100km as nearby
   alternatives to the original 800-1100km band.

2. ALTERNATIVE RESILIENCE SPECIFICATION: the outcome (rolling AR(1))
   is itself an estimated quantity, not directly observed. Tests two
   reasonable alternatives:
   (a) 36-month rolling AR(1) window (already computed and validated
       in Stage 43, reused here - not the original 24-month window)
   (b) rolling STANDARD DEVIATION (a different, equally standard
       critical-slowing-down early-warning-signal from Scheffer et al.
       2009 - variance-based, not autocorrelation-based)

3. STRONGER TEMPORAL CONTROLS: adds month-of-year fixed effects
   (seasonal dummies) and year fixed effects, to rule out the result
   being driven by recurring seasonal patterns or broad year-level
   shocks not fully captured by the existing controls.

4. ALTERNATIVE INFERENCE: the original clustering was by patch_id
   only. Tests two-way clustering (patch_id AND date simultaneously),
   which is more conservative and appropriate given the data is
   structured both spatially (patches) and temporally (repeated
   months) - a single-dimension cluster may understate the true
   uncertainty.

Each check is reported as: does the lag=3 far-band coefficient remain
negative and statistically significant (or at least directionally and
qualitatively similar) under this alternative? A clean pass/fail
table, not a search for the best-looking specification.

WORDING DISCIPLINE: consistent survival across these checks would
support the finding being robust to reasonable specification choices -
it does not upgrade the finding to a proven causal or physical effect.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/patch_rolling_ar1.csv (24-month, reused)
        data/processed/patch_rolling_ar1_36mo.csv (from Stage 43, reused if present)
        data/raw/[all environmental raster files]
        data/raw/oni_index.csv
Output: data/processed/robustness_lag3_results.csv
        figures/robustness_lag3_summary.png
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
LOCAL_CONTROL_COLS = ["precip_anomaly", "temp_anomaly", "soil_anomaly", "pdsi_anomaly",
                       "vpd_anomaly", "wind_anomaly", "solar_anomaly", "rzsm_anomaly",
                       "deltaT_anomaly", "twi", "dist_to_disturbance_km"]
GLOBAL_CONTROL_COLS = ["oni_value"]  # ENSO is a single basin-wide value, not patch-specific -
                                       # must be included ONCE, not duplicated per target/neighbor
                                       # (a real bug found in Stages 40-41's first draft: since
                                       # oni_value is identical for every patch on a given date,
                                       # target_oni_value and neighbor_oni_value were exact
                                       # duplicates - perfect collinearity)
ROLLING_WINDOW = 24   # months - shorter than Stage 5's 60, for more time resolution
LAGS = [1, 2, 3, 6]
DIST_BIN_EDGES = [0, 75, 150, 225, 300, 375, 450, 550, 650, 800, 1100]

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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def lag1_autocorr(x):
    x = np.asarray(x)
    if len(x) < 3:
        return np.nan
    return np.corrcoef(x[:-1], x[1:])[0, 1]

LAG = 3  # focused specifically on the result that survived everything in Stage 45
ORIGINAL_BAND = (800, 1100)
ALT_BANDS = [(750, 1000), (900, 1100)]

def compute_rolling_metric(ts, window, metric="ar1"):
    records = []
    for pid in ts["patch_id"].unique():
        sub = ts[ts["patch_id"] == pid].sort_values("date").reset_index(drop=True)
        anomaly = sub["vod_anomaly"].values
        dates = sub["date"].values
        n_t = len(anomaly)
        for start in range(0, n_t - window + 1):
            w = anomaly[start:start + window]
            val = lag1_autocorr(w) if metric == "ar1" else np.std(w)
            records.append((pid, dates[start + window - 1], val))
    return pd.DataFrame(records, columns=["patch_id", "date", "resilience_metric"])


def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])
    twi_df = pd.read_csv(os.path.join(OUT_DIR, "patch_twi.csv"))
    dist_disturbance_df = pd.read_csv(os.path.join(OUT_DIR, "patch_disturbance_distance.csv"))

    print("Aggregating all environmental variables (shared across all checks)...")
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

    temp_df = to_long_anomaly(temp_vals, n_m, loc, "temp")
    soil_df = to_long_anomaly(soil_vals, n_m, loc, "soil")
    pdsi_df = to_long_anomaly(pdsi_vals, n_m, loc, "pdsi")
    vpd_df = to_long_anomaly(vpd_vals, n_m, loc, "vpd")
    wind_df = to_long_anomaly(wind_speed_vals, n_m, loc, "wind")
    solar_df = to_long_anomaly(solar_vals, n_m, loc, "solar")
    rzsm_df = to_long_anomaly(rzsm_vals, n_m, loc, "rzsm")
    deltaT_df = to_long_anomaly(deltaT_vals, n_m, loc, "deltaT")
    env_dfs = [temp_df, soil_df, pdsi_df, vpd_df, wind_df, solar_df, rzsm_df, deltaT_df]

    n = len(loc)
    lats, lons, pids = loc["lat"].values, loc["lon"].values, loc["patch_id"].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine(lats[i], lons[i], lats, lons)
    dist_df = pd.DataFrame(dist_matrix, index=pids, columns=pids)
    latlon_df = loc.set_index("patch_id")[["lat", "lon"]]

    def build_dataset(rolling_df):
        merged = ts.merge(temp_df, on=["patch_id", "date"], how="inner") \
                   .merge(soil_df, on=["patch_id", "date"], how="inner") \
                   .merge(pdsi_df, on=["patch_id", "date"], how="inner") \
                   .merge(oni, on="date", how="inner") \
                   .merge(vpd_df, on=["patch_id", "date"], how="inner") \
                   .merge(wind_df, on=["patch_id", "date"], how="inner") \
                   .merge(solar_df, on=["patch_id", "date"], how="inner") \
                   .merge(rzsm_df, on=["patch_id", "date"], how="inner") \
                   .merge(deltaT_df, on=["patch_id", "date"], how="left") \
                   .merge(twi_df, on="patch_id", how="left") \
                   .merge(dist_disturbance_df, on="patch_id", how="left") \
                   .merge(rolling_df, on=["patch_id", "date"], how="inner")
        merged = merged.dropna(subset=LOCAL_CONTROL_COLS + GLOBAL_CONTROL_COLS + ["resilience_metric"])
        for c in ["precip_anomaly", "temp_anomaly", "vpd_anomaly", "pdsi_anomaly"]:
            merged[f"regional_{c}"] = merged.groupby("date")[c].transform("mean")
        return merged

    def fit_far_band_lag3(merged, band, extra_terms="", cluster_two_way=False):
        resilience_pivot = merged.pivot(index="date", columns="patch_id", values="resilience_metric").sort_index()
        local_pivots = {c: merged.pivot(index="date", columns="patch_id", values=c).sort_index() for c in LOCAL_CONTROL_COLS}
        oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
        regional_series = {c: merged.drop_duplicates("date").set_index("date")[f"regional_{c}"].sort_index()
                            for c in ["precip_anomaly", "temp_anomaly", "vpd_anomaly", "pdsi_anomaly"]}
        dates_list = resilience_pivot.index.to_list()
        patches = resilience_pivot.columns.to_list()

        lo, hi = band
        band_map = {}
        for pid in patches:
            d = dist_df.loc[pid]
            band_map[pid] = [p for p in d[(d > lo) & (d <= hi)].index.tolist() if p in resilience_pivot.columns]

        def build_neighbor_avg(pivot):
            out = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
            for pid in patches:
                neighbors = [p for p in band_map.get(pid, []) if p in pivot.columns]
                out[pid] = pivot[neighbors].mean(axis=1) if neighbors else np.nan
            return out

        neighbor_resilience = build_neighbor_avg(resilience_pivot)
        neighbor_controls = {c: build_neighbor_avg(local_pivots[c]) for c in LOCAL_CONTROL_COLS}
        neighbor_latlon = {}
        for coord in ["lat", "lon"]:
            out = pd.Series(index=patches, dtype=float)
            for pid in patches:
                neighbors = band_map.get(pid, [])
                out[pid] = latlon_df.loc[neighbors, coord].mean() if neighbors else np.nan
            neighbor_latlon[coord] = out

        recs = []
        for pid in patches:
            own_res = resilience_pivot[pid].values
            neigh_res = neighbor_resilience[pid].values
            oni_vals = oni_series.reindex(dates_list).values
            regional_vals = {c: regional_series[c].reindex(dates_list).values for c in regional_series}
            for i in range(len(dates_list) - LAG):
                rec = {"patch_id": pid, "date": dates_list[i], "own_resilience_t": own_res[i],
                       "neighbor_resilience_state": neigh_res[i], "own_resilience_future": own_res[i + LAG],
                       "oni_value": oni_vals[i], "target_lat": latlon_df.loc[pid, "lat"],
                       "target_lon": latlon_df.loc[pid, "lon"], "neighbor_lat": neighbor_latlon["lat"][pid],
                       "neighbor_lon": neighbor_latlon["lon"][pid], "month": dates_list[i].month,
                       "year": dates_list[i].year}
                for c in LOCAL_CONTROL_COLS:
                    rec[f"target_{c}"] = local_pivots[c][pid].values[i]
                    rec[f"neighbor_{c}"] = neighbor_controls[c][pid].values[i]
                for c in regional_vals:
                    rec[f"regional_{c}"] = regional_vals[c][i]
                recs.append(rec)
        panel = pd.DataFrame(recs).dropna()

        local_terms = " + ".join([f"target_{c}" for c in LOCAL_CONTROL_COLS] +
                                  [f"neighbor_{c}" for c in LOCAL_CONTROL_COLS] + GLOBAL_CONTROL_COLS)
        regional_terms = "regional_precip_anomaly + regional_temp_anomaly + regional_vpd_anomaly + regional_pdsi_anomaly"
        latlon_terms = "target_lat + target_lon + neighbor_lat + neighbor_lon"
        rhs = f"own_resilience_t + neighbor_resilience_state + {local_terms} + {regional_terms} + {latlon_terms}"
        if extra_terms:
            rhs += f" + {extra_terms}"
        formula = f"own_resilience_future ~ {rhs}"

        if cluster_two_way:
            groups_array = panel[["patch_id", "date"]].apply(lambda col: pd.factorize(col)[0])
            m = smf.ols(formula, data=panel).fit(cov_type="cluster",
                cov_kwds={"groups": groups_array})
        else:
            m = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
        return m.params["neighbor_resilience_state"], m.pvalues["neighbor_resilience_state"], \
               m.conf_int().loc["neighbor_resilience_state"], len(panel)

    print("Computing 24-month rolling AR(1) (baseline, reused if cached)...")
    ar1_24_path = os.path.join(OUT_DIR, "patch_rolling_ar1.csv")
    if os.path.exists(ar1_24_path):
        ar1_24 = pd.read_csv(ar1_24_path, parse_dates=["date"]).rename(columns={"resilience_ar1": "resilience_metric"})
    else:
        ar1_24 = compute_rolling_metric(ts, 24, "ar1")
    dataset_24 = build_dataset(ar1_24)

    results = []

    # ---- BASELINE (Stage 45 Step 4 equivalent) ----
    print("\n===== BASELINE (Stage 45 result, for reference) =====")
    coef, pval, ci, n_obs = fit_far_band_lag3(dataset_24, ORIGINAL_BAND)
    print(f"  800-1100km, lag=3: coef={coef:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}] p={pval:.4f} n={n_obs}")
    results.append(("Baseline (800-1100km, 24mo AR1, patch-clustered SE)", coef, pval, ci[0], ci[1], n_obs))

    # ---- CHECK 1: alternative distance bands ----
    print("\n===== CHECK 1: Alternative distance bands =====")
    for band in ALT_BANDS:
        coef, pval, ci, n_obs = fit_far_band_lag3(dataset_24, band)
        label = f"{band[0]}-{band[1]}km"
        sig = "*" if pval < 0.05 else " "
        print(f"  {label:15s}: coef={coef:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}] p={pval:.4f}{sig} n={n_obs}")
        results.append((f"Alt distance band ({label})", coef, pval, ci[0], ci[1], n_obs))

    # ---- CHECK 2: alternative resilience specification ----
    print("\n===== CHECK 2: Alternative resilience specification =====")
    ar1_36_path = os.path.join(OUT_DIR, "patch_rolling_ar1_36mo.csv")
    if os.path.exists(ar1_36_path):
        print("  Reusing cached 36-month rolling AR(1) from Stage 43...")
        ar1_36 = pd.read_csv(ar1_36_path, parse_dates=["date"]).rename(columns={"resilience_ar1": "resilience_metric"})
    else:
        print("  Computing 36-month rolling AR(1) (Stage 43 cache not found)...")
        ar1_36 = compute_rolling_metric(ts, 36, "ar1")
    dataset_36 = build_dataset(ar1_36)
    coef, pval, ci, n_obs = fit_far_band_lag3(dataset_36, ORIGINAL_BAND)
    sig = "*" if pval < 0.05 else " "
    print(f"  36-month rolling AR(1): coef={coef:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}] p={pval:.4f}{sig} n={n_obs}")
    results.append(("Alt resilience metric (36mo AR1)", coef, pval, ci[0], ci[1], n_obs))

    print("  Computing 24-month rolling STANDARD DEVIATION (variance-based EWS)...")
    std_24 = compute_rolling_metric(ts, 24, "std")
    dataset_std = build_dataset(std_24)
    coef, pval, ci, n_obs = fit_far_band_lag3(dataset_std, ORIGINAL_BAND)
    sig = "*" if pval < 0.05 else " "
    print(f"  24-month rolling STD:   coef={coef:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}] p={pval:.4f}{sig} n={n_obs}")
    results.append(("Alt resilience metric (24mo rolling std)", coef, pval, ci[0], ci[1], n_obs))

    # ---- CHECK 3: stronger temporal controls ----
    print("\n===== CHECK 3: Month + year fixed effects =====")
    coef, pval, ci, n_obs = fit_far_band_lag3(dataset_24, ORIGINAL_BAND, extra_terms="C(month) + C(year)")
    sig = "*" if pval < 0.05 else " "
    print(f"  + month/year FE: coef={coef:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}] p={pval:.4f}{sig} n={n_obs}")
    results.append(("+ Month/year fixed effects", coef, pval, ci[0], ci[1], n_obs))

    # ---- CHECK 4: alternative inference (two-way clustering) ----
    print("\n===== CHECK 4: Two-way clustering (patch AND date) =====")
    coef, pval, ci, n_obs = fit_far_band_lag3(dataset_24, ORIGINAL_BAND, cluster_two_way=True)
    sig = "*" if pval < 0.05 else " "
    print(f"  Two-way clustered SE: coef={coef:+.5f} [{ci[0]:+.5f},{ci[1]:+.5f}] p={pval:.4f}{sig} n={n_obs}")
    results.append(("Two-way clustered SE (patch + date)", coef, pval, ci[0], ci[1], n_obs))

    results_df = pd.DataFrame(results, columns=["specification", "coef", "pval", "ci_low", "ci_high", "n_obs"])
    results_df.to_csv(os.path.join(OUT_DIR, "robustness_lag3_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(results_df))
    for i, row in results_df.iterrows():
        color = 'darkred' if row["pval"] < 0.05 else 'gray'
        ax.errorbar([row["coef"]], [i], xerr=[[row["coef"]-row["ci_low"]], [row["ci_high"]-row["coef"]]],
                    fmt='o', capsize=4, color=color)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df["specification"], fontsize=8)
    ax.set_xlabel("Far-band (800-1100km) lag=3 coefficient")
    ax.set_title("Robustness of the lag=3 far-distance result across alternative specifications\n(red = still significant, gray = not)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "robustness_lag3_summary.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/robustness_lag3_summary.png")

    n_sig_same_sign = ((results_df["pval"] < 0.05) & (results_df["coef"] < 0)).sum()
    n_same_sign = (results_df["coef"] < 0).sum()
    print(f"\n===== OVERALL SUMMARY =====")
    print(f"Same sign (negative) as baseline: {n_same_sign} / {len(results_df)} specifications")
    print(f"Still significant AND same sign: {n_sig_same_sign} / {len(results_df)} specifications")
    if n_same_sign == len(results_df) and n_sig_same_sign >= len(results_df) - 1:
        print("\n-> ROBUST: the lag=3 far-distance result is qualitatively consistent (same sign,")
        print("   mostly still significant) across all pre-specified alternative specifications.")
        print("   This is meaningfully stronger evidence than any single specification alone.")
    elif n_same_sign >= len(results_df) - 1:
        print("\n-> MOSTLY ROBUST: the result keeps the same sign under nearly all alternatives,")
        print("   though statistical significance is not uniform - report honestly as a")
        print("   directionally consistent but not universally significant finding.")
    else:
        print("\n-> NOT FULLY ROBUST: the result's sign or significance varies meaningfully across")
        print("   specifications - this should temper confidence in the Stage 45 finding and be")
        print("   reported as such.")

if __name__ == "__main__":
    main()