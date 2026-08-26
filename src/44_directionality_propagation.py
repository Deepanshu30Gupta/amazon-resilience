"""
44_directionality_propagation.py

Purpose: Stage 42/43 established a robust, window-validated pattern:
positive resilience linkage at short distances/short lags, and a
negative reversal at long distances (800-1100km) that strengthens at
longer lags. This tests whether that pattern looks like genuine
DIRECTIONAL PROPAGATION (A's resilience change precedes and predicts
B's) or SYMMETRIC SYNCHRONY (A and B move together with no clear time
direction, most likely from shared forcing) - the same fundamental
distinction established in Stages 9-11, now applied to the new
resilience-to-resilience framework and specifically tested for both
the near-distance and far-distance patterns.

PART A: Placebo/direction test (same logic as Stages 9-11). For the
near band (75-450km, where Stage 42/43 found a robust POSITIVE effect)
and the far band (800-1100km, where a robust NEGATIVE effect was
found), compares:
  FORWARD:  neighbor_resilience(t)   -> own_resilience(t+lag)  [the
            real, established Stage 42 test]
  BACKWARD: neighbor_resilience(t+lag) -> own_resilience(t)    [placebo -
            using the neighbor's FUTURE state to "predict" B's PAST
            state; this relationship should NOT exist if the effect is
            genuinely forward-directional]
If forward and backward are comparably strong, that indicates symmetric
synchrony, not directional propagation - consistent with how this
same test was interpreted throughout the project.

PART B: Propagation-speed test. For each distance band, finds which
lag (1, 2, 3, 6) shows the strongest/most significant effect. If the
effect emerges at a systematically LONGER lag as distance increases,
that is suggestive of a signal actually traveling across space (a
"propagation speed"). If the peak lag doesn't shift with distance,
that argues against literal geographic propagation.

WORDING DISCIPLINE: even strong directional asymmetry (forward >>
backward) would support propagation being MORE LIKELY than pure
synchrony - it would not by itself prove a physical/causal mechanism.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/patch_rolling_ar1.csv (reused if present)
        data/raw/[all environmental raster files]
        data/raw/oni_index.csv
Output: data/processed/directionality_test_results.csv (Part A)
        data/processed/propagation_speed_results.csv (Part B)
        figures/directionality_forward_vs_backward.png
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

NEAR_BAND = (75, 450)   # where Stage 42/43 found robust POSITIVE effect
FAR_BAND = (800, 1100)  # where Stage 42/43 found robust NEGATIVE effect

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])
    twi_df = pd.read_csv(os.path.join(OUT_DIR, "patch_twi.csv"))
    dist_disturbance_df = pd.read_csv(os.path.join(OUT_DIR, "patch_disturbance_distance.csv"))

    rolling_ar1_path = os.path.join(OUT_DIR, "patch_rolling_ar1.csv")
    if os.path.exists(rolling_ar1_path):
        print(f"Reusing existing rolling AR(1) ({rolling_ar1_path})...")
        rolling_ar1_df = pd.read_csv(rolling_ar1_path, parse_dates=["date"])
    else:
        print(f"Computing rolling {ROLLING_WINDOW}-month AR(1) per patch...")
        rolling_records = []
        for pid in ts["patch_id"].unique():
            sub = ts[ts["patch_id"] == pid].sort_values("date").reset_index(drop=True)
            anomaly = sub["vod_anomaly"].values
            dates = sub["date"].values
            n_t = len(anomaly)
            for start in range(0, n_t - ROLLING_WINDOW + 1):
                window = anomaly[start:start + ROLLING_WINDOW]
                ar1 = lag1_autocorr(window)
                rolling_records.append((pid, dates[start + ROLLING_WINDOW - 1], ar1))
        rolling_ar1_df = pd.DataFrame(rolling_records, columns=["patch_id", "date", "resilience_ar1"])
        rolling_ar1_df.to_csv(rolling_ar1_path, index=False)
    print(f"Rolling AR1: {rolling_ar1_df.shape[0]} patch-months")

    print("\nAggregating all environmental variables...")
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

    def build_neighbor_avg(pivot, band_map):
        out = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
        for pid in patches:
            neighbors = [p for p in band_map.get(pid, []) if p in pivot.columns]
            out[pid] = pivot[neighbors].mean(axis=1) if neighbors else np.nan
        return out

    def band_map_for(lo, hi):
        band_map = {}
        for pid in patches:
            d = dist_df.loc[pid]
            band_map[pid] = [p for p in d[(d > lo) & (d <= hi)].index.tolist() if p in resilience_pivot.columns]
        return band_map

    def fit_directional_model(band_map, lag, direction):
        """direction='forward': neighbor(t) -> own(t+lag) [real]
           direction='backward': neighbor(t+lag) -> own(t) [placebo]"""
        neighbor_resilience = build_neighbor_avg(resilience_pivot, band_map)
        neighbor_controls = {c: build_neighbor_avg(local_pivots[c], band_map) for c in LOCAL_CONTROL_COLS}
        recs = []
        for pid in patches:
            own_res = resilience_pivot[pid].values
            neigh_res = neighbor_resilience[pid].values
            oni_vals = oni_series.reindex(dates_list).values
            for i in range(len(dates_list) - lag):
                if direction == "forward":
                    own_val, neigh_val, own_future = own_res[i], neigh_res[i], own_res[i + lag]
                    ctrl_idx = i
                else:  # backward placebo: mirror the forward model exactly, but with the
                    # whole time arrow reversed - own(t+lag) and neighbor(t+lag) are the
                    # "starting point" (matching forward's same-time own(t)+neighbor(t)
                    # pairing), and own(t) is the "outcome" playing the later role
                    own_val, neigh_val, own_future = own_res[i + lag], neigh_res[i + lag], own_res[i]
                    ctrl_idx = i + lag
                oni_val = oni_vals[ctrl_idx]
                rec = {"patch_id": pid, "own_resilience_t": own_val, "neighbor_resilience_state": neigh_val,
                       "own_resilience_future": own_future, "oni_value": oni_val}
                for c in LOCAL_CONTROL_COLS:
                    rec[f"target_{c}"] = local_pivots[c][pid].values[ctrl_idx]
                    rec[f"neighbor_{c}"] = neighbor_controls[c][pid].values[ctrl_idx]
                recs.append(rec)
        panel = pd.DataFrame(recs).dropna()
        control_terms = " + ".join([f"target_{c}" for c in LOCAL_CONTROL_COLS] +
                                    [f"neighbor_{c}" for c in LOCAL_CONTROL_COLS] +
                                    GLOBAL_CONTROL_COLS)
        formula = f"own_resilience_future ~ own_resilience_t + neighbor_resilience_state + {control_terms}"
        m = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["patch_id"]})
        return m.params["neighbor_resilience_state"], m.pvalues["neighbor_resilience_state"], \
               m.conf_int().loc["neighbor_resilience_state"], len(panel)

    # ================================================================
    # PART A: Forward vs Backward (placebo) directionality test
    # ================================================================
    print(f"\n===== PART A: Directionality test (forward vs backward/placebo) =====")
    direction_results = []
    for band_name, (lo, hi) in [("NEAR (75-450km, positive effect)", NEAR_BAND),
                                  ("FAR (800-1100km, negative effect)", FAR_BAND)]:
        print(f"\n--- {band_name} ---")
        band_map = band_map_for(lo, hi)
        for lag in LAGS:
            f_coef, f_pval, f_ci, f_n = fit_directional_model(band_map, lag, "forward")
            b_coef, b_pval, b_ci, b_n = fit_directional_model(band_map, lag, "backward")
            print(f"  Lag {lag}mo: FORWARD coef={f_coef:+.5f} p={f_pval:.4f} (n={f_n})  |  "
                  f"BACKWARD(placebo) coef={b_coef:+.5f} p={b_pval:.4f} (n={b_n})")
            direction_results.append((band_name, lo, hi, lag, "forward", f_coef, f_pval, f_ci[0], f_ci[1], f_n))
            direction_results.append((band_name, lo, hi, lag, "backward", b_coef, b_pval, b_ci[0], b_ci[1], b_n))

    direction_df = pd.DataFrame(direction_results, columns=[
        "band", "dist_lo", "dist_hi", "lag_months", "direction", "coef", "pval", "ci_low", "ci_high", "n_obs"
    ])
    direction_df.to_csv(os.path.join(OUT_DIR, "directionality_test_results.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for idx, (band_name, (lo, hi)) in enumerate([("NEAR (75-450km)", NEAR_BAND), ("FAR (800-1100km)", FAR_BAND)]):
        ax = axes[idx]
        for direction, color in [("forward", "darkred"), ("backward", "gray")]:
            sub = direction_df[(direction_df["band"].str.startswith(band_name.split(" (")[0])) &
                                (direction_df["direction"] == direction)].sort_values("lag_months")
            ax.errorbar(sub["lag_months"], sub["coef"],
                         yerr=[sub["coef"]-sub["ci_low"], sub["ci_high"]-sub["coef"]],
                         fmt='o-', capsize=3, color=color, label=direction, alpha=0.8)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(band_name)
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Neighbor resilience effect")
        ax.legend()
    plt.suptitle("Forward (real) vs Backward (placebo) - similar strength suggests synchrony, not propagation")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "directionality_forward_vs_backward.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/directionality_forward_vs_backward.png")

    # ================================================================
    # PART B: Propagation-speed test - does the peak effect shift to
    # longer lags as distance increases?
    # ================================================================
    print(f"\n===== PART B: Propagation-speed test (does peak lag shift with distance?) =====")
    speed_results = []
    for b in range(len(DIST_BIN_EDGES) - 1):
        lo, hi = DIST_BIN_EDGES[b], DIST_BIN_EDGES[b+1]
        band_map = band_map_for(lo, hi)
        lag_coefs = []
        for lag in LAGS:
            coef, pval, ci, n_obs = fit_directional_model(band_map, lag, "forward")
            lag_coefs.append((lag, coef, pval))
        # peak = largest ABSOLUTE coefficient among lags that are significant; if none
        # significant, report the lag with the largest absolute coefficient anyway (marked)
        sig_lags = [(l, c, p) for l, c, p in lag_coefs if p < 0.05]
        if sig_lags:
            peak_lag, peak_coef, peak_pval = max(sig_lags, key=lambda x: abs(x[1]))
            note = ""
        else:
            peak_lag, peak_coef, peak_pval = max(lag_coefs, key=lambda x: abs(x[1]))
            note = " (none significant - largest |coef| shown)"
        mid = (lo + hi) / 2
        print(f"  {lo:4d}-{hi:4d}km: peak lag={peak_lag}mo, coef={peak_coef:+.5f}, p={peak_pval:.4f}{note}")
        speed_results.append((lo, hi, mid, peak_lag, peak_coef, peak_pval))

    speed_df = pd.DataFrame(speed_results, columns=["dist_lo", "dist_hi", "dist_mid", "peak_lag", "peak_coef", "peak_pval"])
    speed_df.to_csv(os.path.join(OUT_DIR, "propagation_speed_results.csv"), index=False)

    corr, corr_p = np.nan, np.nan
    from scipy import stats as sstats
    if speed_df["peak_lag"].nunique() > 1 and speed_df["dist_mid"].nunique() > 1:
        corr, corr_p = sstats.spearmanr(speed_df["dist_mid"], speed_df["peak_lag"])
        print(f"\nCorrelation between distance and peak lag: Spearman r={corr:.3f}, p={corr_p:.4f}")
    else:
        print(f"\nCorrelation between distance and peak lag: not computable (peak lag was "
              f"constant across the distance bands tested - likely too few bands in this run)")
    if not np.isnan(corr_p) and corr_p < 0.05 and corr > 0:
        print("-> Peak lag INCREASES with distance - suggestive of an actual propagating signal")
        print("   (a 'speed' of spread), not just simultaneous synchrony.")
    else:
        print("-> No significant relationship between distance and peak lag - does NOT support")
        print("   literal geographic propagation at a consistent speed; more consistent with")
        print("   simultaneous synchrony or shared large-scale forcing.")

    print("\n===== OVERALL INTERPRETATION GUIDE =====")
    print("If forward and backward coefficients are SIMILAR in Part A, that indicates SYNCHRONY")
    print("(shared forcing / simultaneous co-movement), not directional propagation - the same")
    print("conclusion reached for the original VOD-based analysis in Stages 9-11. If forward is")
    print("clearly and consistently stronger than backward, that supports genuine directional")
    print("influence being more likely (though still not proof of a physical causal mechanism).")

if __name__ == "__main__":
    main()