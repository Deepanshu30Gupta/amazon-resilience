"""
42_resilience_to_resilience.py

Purpose: The final, correctly-framed version of the core research
question, confirmed with the team: "When patch A loses resilience,
does that loss propagate to reduce patch B's OWN resilience" - not
"does A's state predict B's next month's vegetation reading" (Stage
41's outcome) and not "does A's current condition predict B" (Stages
6-40's outcome). This is a genuinely different outcome variable:

  Stage 41: neighbor_resilience_state(t) -> B's raw VOD(t+lag)
  Stage 42: neighbor_resilience_state(t) -> B's OWN resilience(t+lag)

Both B's own resilience(t) [control, "B's baseline before the window"]
and B's own resilience(t+lag) [outcome] are built from the SAME rolling
24-month AR(1) metric computed in Stage 41 (patch_rolling_ar1.csv,
reused if present) - this is intentional, standard autoregressive
panel design (Y_t+lag ~ Y_t + X_t + controls), directly analogous to
how every VOD-based model since Stage 6 controlled for B's own past
VOD to predict B's own future VOD.

IMPORTANT METHODOLOGICAL CAVEAT, stated explicitly and QUANTIFIED in
the printed output: because B's resilience(t) and B's resilience(t+lag)
are both 24-month rolling windows, they OVERLAP substantially for any
lag shorter than 24 months - e.g. at lag=1, the two windows share 23
of 24 months (96% overlap); even at lag=6, they still share 18 of 24
months (75% overlap). This means the four lags (1, 2, 3, 6) are NOT
independent tests of each other the way Stage 39-41's raw-VOD-based
lags were - they are highly correlated views of mostly-the-same
underlying window data. This script prints the exact overlap
percentage for every lag tested; results should be read as related,
overlapping evidence, not four separate confirmations. This is a
milder, different-in-kind issue than the original Stage 6 bug (a
variable predicting ITSELF via mechanically overlapping windows); here,
B's future resilience is a genuinely later time period than B's
current resilience, just one that shares much of the same underlying
monthly data due to the smoothing window - a known, standard trade-off
in rolling-window panel designs, not a computational error.

Answers, directly:
  1. Does A's resilience loss affect B's resilience at all?
  2. Does the effect weaken with distance? (distance-banded)
  3. How quickly does it appear? (lags 1, 2, 3, 6 - with overlap caveat)
  4. Does B's own environment make it more/less susceptible?
     (interaction/susceptibility test, standardized)

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/patch_rolling_ar1.csv (from Stage 41, reused if present)
        data/raw/[all environmental raster files]
        data/raw/oni_index.csv
Output: data/processed/resilience_to_resilience_distance_lag.csv (main result)
        data/processed/resilience_to_resilience_susceptibility.csv (interaction result)
        figures/resilience_to_resilience_by_distance.png
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

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))
    oni = pd.read_csv(ONI_PATH, parse_dates=["date"])
    twi_df = pd.read_csv(os.path.join(OUT_DIR, "patch_twi.csv"))
    dist_disturbance_df = pd.read_csv(os.path.join(OUT_DIR, "patch_disturbance_distance.csv"))

    # ================================================================
    # STEP 1: rolling 24-month AR(1) per patch - the genuine time-
    # varying resilience-loss metric
    # ================================================================
    rolling_ar1_path = os.path.join(OUT_DIR, "patch_rolling_ar1.csv")
    if os.path.exists(rolling_ar1_path):
        print(f"Reusing existing rolling AR(1) from Stage 41 ({rolling_ar1_path})...")
        rolling_ar1_df = pd.read_csv(rolling_ar1_path, parse_dates=["date"])
    else:
        print(f"Computing rolling {ROLLING_WINDOW}-month AR(1) per patch (Stage 41 output not found)...")
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
    print(f"Rolling AR1: {rolling_ar1_df.shape[0]} patch-months "
          f"(first available date: {rolling_ar1_df['date'].min()})")

    # ---- Overlap quantification (the key methodological caveat) ----
    print(f"\n===== WINDOW OVERLAP BY LAG (read this before interpreting results) =====")
    for lag in LAGS:
        overlap_months = max(0, ROLLING_WINDOW - lag)
        overlap_pct = 100 * overlap_months / ROLLING_WINDOW
        print(f"  Lag {lag} month(s): B's resilience(t) and resilience(t+{lag}) windows share "
              f"{overlap_months}/{ROLLING_WINDOW} months ({overlap_pct:.0f}% overlap)")
    print("These lags are NOT independent tests of each other - treat them as related,")
    print("overlapping views of mostly the same underlying data, not separate confirmations.\n")

    # ================================================================
    # Aggregate environmental variables (same as Stage 40)
    # ================================================================
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
               .merge(rolling_ar1_df, on=["patch_id", "date"], how="inner")  # INNER - only
               # months where rolling AR1 is available (after the first 24-month warmup)

    merged = merged.dropna(subset=LOCAL_CONTROL_COLS + GLOBAL_CONTROL_COLS + ["resilience_ar1"])
    print(f"Merged dataset shape (after 24-month warmup): {merged.shape}")

    n = len(loc)
    lats, lons, pids = loc["lat"].values, loc["lon"].values, loc["patch_id"].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine(lats[i], lons[i], lats, lons)
    dist_df = pd.DataFrame(dist_matrix, index=pids, columns=pids)

    vod_pivot = merged.pivot(index="date", columns="patch_id", values="vod_anomaly").sort_index()
    resilience_pivot = merged.pivot(index="date", columns="patch_id", values="resilience_ar1").sort_index()
    local_pivots = {c: merged.pivot(index="date", columns="patch_id", values=c).sort_index() for c in LOCAL_CONTROL_COLS}
    # ONI: single global series, NOT patch-pivoted - included once, never duplicated
    oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
    dates_list = vod_pivot.index.to_list()
    patches = vod_pivot.columns.to_list()

    def build_neighbor_avg(pivot, neighbor_map_or_band):
        out = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
        for pid in patches:
            neighbors = [p for p in neighbor_map_or_band.get(pid, []) if p in pivot.columns]
            out[pid] = pivot[neighbors].mean(axis=1) if neighbors else np.nan
        return out

    def zscore(series):
        """Standardize a numeric column - makes interaction coefficients comparable
        across drivers with very different raw units/scales."""
        return (series - series.mean()) / series.std()

    # ================================================================
    # STEP 2: main model - A's (or, for banded distances, the average of
    # patches at that distance's) resilience STATE -> B's future VOD,
    # by distance band and lag, both-side environmental controls, ONI
    # included once as a global control
    # ================================================================
    print(f"\n===== MAIN RESULT: neighbor resilience exposure -> B's future resilience =====")
    print("(by distance band and lag, controlling for BOTH patches' local environments + ONI once)\n")
    print("NOTE: for distance bands with multiple patches, this is 'neighbor resilience EXPOSURE'")
    print("(the average resilience state of patches at that distance), not a single-pair A->B link -")
    print("consistent with how neighbor effects have been defined since Stage 6.\n")
    main_results = []
    for lag in LAGS:
        print(f"--- Lag {lag} month(s) ---")
        for b in range(len(DIST_BIN_EDGES) - 1):
            lo, hi = DIST_BIN_EDGES[b], DIST_BIN_EDGES[b+1]
            band_map = {}
            for pid in patches:
                d = dist_df.loc[pid]
                band_map[pid] = [p for p in d[(d > lo) & (d <= hi)].index.tolist() if p in vod_pivot.columns]

            neighbor_resilience = build_neighbor_avg(resilience_pivot, band_map)
            neighbor_controls = {c: build_neighbor_avg(local_pivots[c], band_map) for c in LOCAL_CONTROL_COLS}

            recs = []
            for pid in patches:
                own_res_t = resilience_pivot[pid].values  # B's OWN resilience state (not raw VOD)
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
            main_results.append((lag, lo, hi, mid, coef, pval, ci_low, ci_high, len(panel)))
            sig = "*" if pval < 0.05 else " "
            print(f"  {lo:4d}-{hi:4d}km: coef={coef:+.5f}  [{ci_low:+.5f}, {ci_high:+.5f}]  p={pval:.4f}{sig}  n={len(panel)}")

    main_df = pd.DataFrame(main_results, columns=[
        "lag_months", "dist_lo", "dist_hi", "dist_mid", "coef", "pval", "ci_low", "ci_high", "n_obs"
    ])
    main_df.to_csv(os.path.join(OUT_DIR, "resilience_to_resilience_distance_lag.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {1: 'darkred', 2: 'darkorange', 3: 'darkgreen', 6: 'darkblue'}
    for lag in LAGS:
        sub = main_df[main_df["lag_months"] == lag]
        if len(sub) == 0:
            continue
        ax.errorbar(sub["dist_mid"], sub["coef"],
                     yerr=[sub["coef"]-sub["ci_low"], sub["ci_high"]-sub["coef"]],
                     fmt='o-', capsize=3, color=colors.get(lag, 'gray'), label=f"{lag} month(s)", alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Distance between patches (km)")
    ax.set_ylabel("Effect of neighbor resilience state on B's OWN future resilience\n(after both-side environmental controls + ONI)")
    ax.set_title("Resilience-to-resilience propagation (neighbor exposure -> B's resilience) vs. distance, by lag\n(error bars = 95% CI; CAUTION: lags share overlapping windows - see printed overlap %)")
    ax.legend(title="Lag")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "resilience_to_resilience_by_distance.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/resilience_to_resilience_by_distance.png")

    # ================================================================
    # STEP 3: susceptibility test - does B's own environment modify
    # how strongly neighbor resilience state affects B? (first-order
    # neighbors). Continuous variables STANDARDIZED before building the
    # interaction term, so interaction coefficients are comparable
    # across drivers regardless of raw units.
    # ================================================================
    print(f"\n===== SUSCEPTIBILITY TEST: does B's environment modify the effect? =====")
    print("(first-order/immediate neighbors, all 4 lags, standardized interaction terms)\n")

    neighbor_map = adj.groupby("patch_id")["neighbor_id"].apply(list).to_dict()
    neighbor_resilience_fo = build_neighbor_avg(resilience_pivot, neighbor_map)
    neighbor_controls_fo = {c: build_neighbor_avg(local_pivots[c], neighbor_map) for c in LOCAL_CONTROL_COLS}

    susceptibility_results = []
    for lag in LAGS:
        base_recs = []
        for pid in patches:
            own_res_t = resilience_pivot[pid].values  # B's OWN resilience state (not raw VOD)
            a_resilience_t = neighbor_resilience_fo[pid].values
            oni_t = oni_series.reindex(dates_list).values
            for i in range(len(dates_list) - lag):
                rec = {"patch_id": pid, "own_resilience_t": own_res_t[i], "neighbor_resilience_state": a_resilience_t[i],
                       "own_resilience_future": own_res_t[i + lag], "oni_value": oni_t[i]}
                for c in LOCAL_CONTROL_COLS:
                    rec[f"target_{c}"] = local_pivots[c][pid].values[i]
                    rec[f"neighbor_{c}"] = neighbor_controls_fo[c][pid].values[i]
                base_recs.append(rec)
        base_panel = pd.DataFrame(base_recs).dropna()

        # Standardize the resilience predictor and every target-side driver used in interactions
        base_panel["neighbor_resilience_state_z"] = zscore(base_panel["neighbor_resilience_state"])
        for c in LOCAL_CONTROL_COLS:
            base_panel[f"target_{c}_z"] = zscore(base_panel[f"target_{c}"])

        control_terms = " + ".join([f"target_{c}" for c in LOCAL_CONTROL_COLS] +
                                    [f"neighbor_{c}" for c in LOCAL_CONTROL_COLS] +
                                    GLOBAL_CONTROL_COLS)

        print(f"--- Lag {lag} month(s) (n={len(base_panel)}) ---")
        for driver in LOCAL_CONTROL_COLS:
            formula = (f"own_resilience_future ~ own_resilience_t + neighbor_resilience_state_z + {control_terms} "
                       f"+ neighbor_resilience_state_z:target_{driver}_z")
            m = smf.ols(formula, data=base_panel).fit(cov_type="cluster", cov_kwds={"groups": base_panel["patch_id"]})
            interaction_term = f"neighbor_resilience_state_z:target_{driver}_z"
            coef = m.params[interaction_term]
            pval = m.pvalues[interaction_term]
            ci_low, ci_high = m.conf_int().loc[interaction_term]
            sig = "*" if pval < 0.05 else " "
            print(f"  B's {driver:20s}: interaction coef={coef:+.5f} (standardized)  p={pval:.4f}{sig}")
            susceptibility_results.append((lag, driver, coef, pval, ci_low, ci_high, len(base_panel)))

    susc_df = pd.DataFrame(susceptibility_results, columns=[
        "lag_months", "b_driver", "interaction_coef_standardized", "pval", "ci_low", "ci_high", "n_obs"
    ])
    susc_df.to_csv(os.path.join(OUT_DIR, "resilience_to_resilience_susceptibility.csv"), index=False)

    n_sig_main = (main_df["pval"] < 0.05).sum()
    n_sig_susc = (susc_df["pval"] < 0.05).sum()
    print(f"\n===== SUMMARY =====")
    print(f"Main effect (neighbor resilience state -> B): {n_sig_main} / {len(main_df)} distance-lag "
          f"combinations statistically significant")
    print(f"Susceptibility (B's environment modifying the effect): {n_sig_susc} / {len(susc_df)} "
          f"driver-lag combinations statistically significant")
    if n_sig_susc > 0:
        print("\nSignificant susceptibility factors (B is more/less affected depending on):")
        for _, row in susc_df[susc_df["pval"] < 0.05].iterrows():
            print(f"  Lag {row['lag_months']}mo, B's {row['b_driver']}: "
                  f"coef={row['interaction_coef_standardized']:+.5f} (standardized), p={row['pval']:.4f}")

    print("\n===== WORDING CAUTION =====")
    print("Higher rolling AR(1) indicates greater temporal persistence, which is commonly")
    print("interpreted as a warning signal consistent with reduced resilience - not a direct")
    print("physical measurement of recovery speed. The distance-banded result reflects NEIGHBOR")
    print("RESILIENCE EXPOSURE (average resilience state of patches at that distance), consistent")
    print("with how neighbor effects have been defined since Stage 6 - not a single-pair A->B link.")
    print("This remains a RESIDUAL/CONDITIONAL statistical association after controlling for both")
    print("patches' measured local environments and ENSO - not a proven causal or physical effect.")
    print("\nSENSITIVITY CHECK RECOMMENDED (not run here): repeat the 24-month rolling AR(1) window")
    print("with 36- and 48-month windows to confirm the result is not an artifact of window length -")
    print("a 24-month window gives only 23 lag pairs per estimate, which is fairly short for stable")
    print("autocorrelation estimation. Establish this result first, then test that sensitivity.")

if __name__ == "__main__":
    main()