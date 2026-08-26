"""
43_window_sensitivity.py

Purpose: The recommended sensitivity check for Stage 42's main result -
repeat the distance x lag analysis with 36-month and 48-month rolling
AR(1) windows (in addition to the original 24-month window), to test
whether the two headline patterns hold up or were artifacts of the
specific window length:
  1. The clean distance-decay pattern at lag=1 (significant 75-650km,
     fading beyond)
  2. The reversal to significant NEGATIVE effect at 800-1100km for
     lags 3 and 6

A longer window gives more stable AR(1) estimates (more lag-pairs per
estimate) but less temporal resolution and even MORE window overlap
between consecutive months (a 48-month window overlaps 47/48 months
at lag=1, versus 23/24 for the original 24-month window) - so this is
a genuine trade-off, not a strictly "better" choice; the point is
checking whether the SUBSTANTIVE FINDING (decay pattern, long-distance
reversal) is consistent across window choices, which would make it
much more trustworthy than any single window length alone.

Only re-runs the MAIN distance x lag grid (not the full susceptibility
interaction grid) to keep runtime manageable - the susceptibility
finding is about interaction with environmental conditions, a
different question from the window-overlap concern this check
specifically addresses.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/resilience_to_resilience_distance_lag.csv (Stage 42's
          24-month result, included in the final comparison)
        data/raw/[all environmental raster files]
        data/raw/oni_index.csv
Output: data/processed/window_sensitivity_36month.csv
        data/processed/window_sensitivity_48month.csv
        data/processed/window_sensitivity_comparison_all.csv
        figures/window_sensitivity_comparison.png
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

WINDOWS_TO_TEST = [36, 48]  # 24-month already done in Stage 42

def compute_rolling_ar1(ts, window):
    records = []
    for pid in ts["patch_id"].unique():
        sub = ts[ts["patch_id"] == pid].sort_values("date").reset_index(drop=True)
        anomaly = sub["vod_anomaly"].values
        dates = sub["date"].values
        n_t = len(anomaly)
        for start in range(0, n_t - window + 1):
            w = anomaly[start:start + window]
            ar1 = lag1_autocorr(w)
            records.append((pid, dates[start + window - 1], ar1))
    return pd.DataFrame(records, columns=["patch_id", "date", "resilience_ar1"])


def run_main_grid_for_window(window, loc, ts, adj, oni, twi_df, dist_disturbance_df,
                              temp_vals, dewpoint_vals, u_vals, v_vals, solar_vals, rzsm_vals,
                              lst_vals, soil_vals, pdsi_vals, n_m, vpd_vals, wind_speed_vals, deltaT_vals):
    print(f"\n{'='*60}\nROLLING WINDOW: {window} months\n{'='*60}")
    rolling_ar1_path = os.path.join(OUT_DIR, f"patch_rolling_ar1_{window}mo.csv")
    if os.path.exists(rolling_ar1_path):
        print(f"Reusing cached {window}-month rolling AR(1)...")
        rolling_ar1_df = pd.read_csv(rolling_ar1_path, parse_dates=["date"])
    else:
        print(f"Computing {window}-month rolling AR(1) per patch...")
        rolling_ar1_df = compute_rolling_ar1(ts, window)
        rolling_ar1_df.to_csv(rolling_ar1_path, index=False)
    print(f"Rolling AR1: {rolling_ar1_df.shape[0]} patch-months (first available: {rolling_ar1_df['date'].min()})")

    for lag in LAGS:
        overlap_months = max(0, window - lag)
        overlap_pct = 100 * overlap_months / window
        print(f"  Lag {lag}mo: {overlap_months}/{window} months overlap ({overlap_pct:.0f}%)")

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
               .merge(deltaT_df, on=["patch_id", "date"], how="left") \
               .merge(twi_df, on="patch_id", how="left") \
               .merge(dist_disturbance_df, on="patch_id", how="left") \
               .merge(rolling_ar1_df, on=["patch_id", "date"], how="inner")

    merged = merged.dropna(subset=LOCAL_CONTROL_COLS + GLOBAL_CONTROL_COLS + ["resilience_ar1"])
    print(f"Merged dataset shape: {merged.shape}")

    n = len(loc)
    lats, lons, pids = loc["lat"].values, loc["lon"].values, loc["patch_id"].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine(lats[i], lons[i], lats, lons)
    dist_df = pd.DataFrame(dist_matrix, index=pids, columns=pids)

    resilience_pivot = merged.pivot(index="date", columns="patch_id", values="resilience_ar1").sort_index()
    local_pivots = {c: merged.pivot(index="date", columns="patch_id", values=c).sort_index() for c in LOCAL_CONTROL_COLS}
    oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
    dates_list = resilience_pivot.index.to_list()
    patches = resilience_pivot.columns.to_list()

    def build_neighbor_avg(pivot, neighbor_map_or_band):
        out = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
        for pid in patches:
            neighbors = [p for p in neighbor_map_or_band.get(pid, []) if p in pivot.columns]
            out[pid] = pivot[neighbors].mean(axis=1) if neighbors else np.nan
        return out

    results = []
    for lag in LAGS:
        for b in range(len(DIST_BIN_EDGES) - 1):
            lo, hi = DIST_BIN_EDGES[b], DIST_BIN_EDGES[b+1]
            band_map = {}
            for pid in patches:
                d = dist_df.loc[pid]
                band_map[pid] = [p for p in d[(d > lo) & (d <= hi)].index.tolist() if p in resilience_pivot.columns]

            neighbor_resilience = build_neighbor_avg(resilience_pivot, band_map)
            neighbor_controls = {c: build_neighbor_avg(local_pivots[c], band_map) for c in LOCAL_CONTROL_COLS}

            recs = []
            for pid in patches:
                own_res_t = resilience_pivot[pid].values
                a_resilience_t = neighbor_resilience[pid].values
                oni_t = oni_series.reindex(dates_list).values
                for i in range(len(dates_list) - lag):
                    rec = {"patch_id": pid, "own_resilience_t": own_res_t[i], "neighbor_resilience_state": a_resilience_t[i],
                           "own_resilience_future": own_res_t[i + lag], "oni_value": oni_t[i]}
                    for c in LOCAL_CONTROL_COLS:
                        rec[f"target_{c}"] = local_pivots[c][pid].values[i]
                        rec[f"neighbor_{c}"] = neighbor_controls[c][pid].values[i]
                    recs.append(rec)
            panel = pd.DataFrame(recs).dropna()
            if len(panel) < 100:
                continue

            control_terms = " + ".join([f"target_{c}" for c in LOCAL_CONTROL_COLS] +
                                        [f"neighbor_{c}" for c in LOCAL_CONTROL_COLS] +
                                        GLOBAL_CONTROL_COLS)
            formula = f"own_resilience_future ~ own_resilience_t + neighbor_resilience_state + {control_terms}"
            m = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
            coef = m.params["neighbor_resilience_state"]
            pval = m.pvalues["neighbor_resilience_state"]
            ci_low, ci_high = m.conf_int().loc["neighbor_resilience_state"]
            mid = (lo + hi) / 2
            results.append((window, lag, lo, hi, mid, coef, pval, ci_low, ci_high, len(panel)))
            sig = "*" if pval < 0.05 else " "
            print(f"  Lag{lag} {lo:4d}-{hi:4d}km: coef={coef:+.5f} p={pval:.4f}{sig} n={len(panel)}")

    return pd.DataFrame(results, columns=[
        "window_months", "lag_months", "dist_lo", "dist_hi", "dist_mid", "coef", "pval", "ci_low", "ci_high", "n_obs"
    ])


def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])
    twi_df = pd.read_csv(os.path.join(OUT_DIR, "patch_twi.csv"))
    dist_disturbance_df = pd.read_csv(os.path.join(OUT_DIR, "patch_disturbance_distance.csv"))

    print("Aggregating all environmental variables (once, reused across all window sizes)...")
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

    all_results = [run_main_grid_for_window(
        w, loc, ts, adj, oni, twi_df, dist_disturbance_df,
        temp_vals, dewpoint_vals, u_vals, v_vals, solar_vals, rzsm_vals,
        lst_vals, soil_vals, pdsi_vals, n_m, vpd_vals, wind_speed_vals, deltaT_vals
    ) for w in WINDOWS_TO_TEST]

    for w, df in zip(WINDOWS_TO_TEST, all_results):
        df.to_csv(os.path.join(OUT_DIR, f"window_sensitivity_{w}month.csv"), index=False)

    # Combine with the original 24-month result from Stage 42, if available
    stage42_path = os.path.join(OUT_DIR, "resilience_to_resilience_distance_lag.csv")
    combined_frames = []
    if os.path.exists(stage42_path):
        s42 = pd.read_csv(stage42_path)
        s42["window_months"] = 24
        combined_frames.append(s42[["window_months", "lag_months", "dist_lo", "dist_hi", "dist_mid",
                                     "coef", "pval", "ci_low", "ci_high", "n_obs"]])
        print("\nIncluded Stage 42's original 24-month result in the comparison.")
    else:
        print("\nNOTE: Stage 42's 24-month result file not found - comparison will only show 36/48-month.")
    combined_frames.extend(all_results)
    combined_df = pd.concat(combined_frames, ignore_index=True)
    combined_df.to_csv(os.path.join(OUT_DIR, "window_sensitivity_comparison_all.csv"), index=False)

    # ---- Plot: one panel per lag, one line per window size ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    window_colors = {24: 'darkred', 36: 'darkorange', 48: 'darkblue'}
    for idx, lag in enumerate(LAGS):
        ax = axes[idx // 2, idx % 2]
        for w in sorted(combined_df["window_months"].unique()):
            sub = combined_df[(combined_df["window_months"] == w) & (combined_df["lag_months"] == lag)].sort_values("dist_mid")
            if len(sub) == 0:
                continue
            ax.errorbar(sub["dist_mid"], sub["coef"],
                         yerr=[sub["coef"]-sub["ci_low"], sub["ci_high"]-sub["coef"]],
                         fmt='o-', capsize=3, color=window_colors.get(w, 'gray'), label=f"{w}-month window", alpha=0.8)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f"Lag = {lag} month(s)")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Neighbor resilience effect")
        ax.legend(fontsize=8)
    plt.suptitle("Window sensitivity: does the distance-decay pattern hold across 24/36/48-month windows?")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "window_sensitivity_comparison.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/window_sensitivity_comparison.png")

    print("\n===== SUMMARY: does the lag=1 distance-decay pattern hold across window sizes? =====")
    for w in sorted(combined_df["window_months"].unique()):
        sub = combined_df[(combined_df["window_months"] == w) & (combined_df["lag_months"] == 1)].sort_values("dist_mid")
        if len(sub) == 0:
            continue
        n_sig_near = (sub[sub["dist_mid"] < 400]["pval"] < 0.05).sum()
        n_sig_far = (sub[sub["dist_mid"] >= 800]["pval"] < 0.05).sum()
        print(f"  {w}-month window: {n_sig_near} significant near bands (<400km), "
              f"{n_sig_far} significant far bands (>=800km)")

    print("\n===== SUMMARY: does the long-lag, long-distance reversal hold across window sizes? =====")
    for w in sorted(combined_df["window_months"].unique()):
        for lag in [3, 6]:
            sub = combined_df[(combined_df["window_months"] == w) & (combined_df["lag_months"] == lag) &
                               (combined_df["dist_mid"] >= 800)]
            if len(sub) == 0:
                continue
            for _, row in sub.iterrows():
                sig = "SIGNIFICANT" if row["pval"] < 0.05 else "not significant"
                print(f"  {w}-month window, lag={lag}, {row['dist_lo']:.0f}-{row['dist_hi']:.0f}km: "
                      f"coef={row['coef']:+.5f} ({sig})")

    print("\n===== INTERPRETATION GUIDE =====")
    print("If the near-distance significance and far-distance non-significance (or reversal)")
    print("pattern is CONSISTENT across 24/36/48-month windows, that's strong evidence the")
    print("Stage 42 finding is real, not a window-length artifact. If the pattern changes")
    print("substantially between window sizes, treat the Stage 42 result with more caution -")
    print("it may be sensitive to the specific smoothing choice rather than a robust signal.")

if __name__ == "__main__":
    main()