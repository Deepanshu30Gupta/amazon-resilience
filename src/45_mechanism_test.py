"""
45_mechanism_test.py

Purpose: Stage 44 found a two-regime pattern - near/intermediate
distances (75-650km) show short-lag, positive, largely symmetric
associations (consistent with synchrony/shared local forcing); far
distances (800-1100km) show long-lag, negative, temporally ASYMMETRIC
associations (forward significant, backward/placebo null - more
consistent with a directional component than pure synchrony).

This stage does NOT try to prove the far-distance effect is real. It
tries to ELIMINATE competing explanations, in order of increasing
stringency, for the far-distance (800-1100km) negative association at
lag=3 and lag=6:

  H0 (null/competing): the apparent long-distance reversal is
     explained by shared climate forcing, spatial/geographic
     confounding, or other common drivers - NOT a directional
     resilience relationship.
  H1: the association remains after conditioning on shared climate
     forcing and other plausible confounders.

Four progressively stronger models, in order:
  Step 1: bare (own resilience + neighbor resilience only, no controls)
  Step 2: + existing local both-sided environmental controls (the
    Stage 42/44 model - precip/temp/soil/PDSI/VPD/wind/solar/RZSM/
    deltaT/TWI/disturbance, for BOTH patches, + ONI once)
  Step 3: + REGIONAL (whole-study-area) climate controls - basin-wide
    mean precipitation, temperature, VPD, and PDSI anomaly each month,
    testing whether a broader-than-local shared climate signal (not
    just each patch's own conditions) explains the effect
  Step 4: + explicit LATITUDE/LONGITUDE for both patches - testing
    whether the known north-south resilience gradient (Stage 5) or
    other geographic positioning, not yet explicitly controlled even
    though TWI/disturbance distance are, explains the effect

Finally, the FORWARD vs BACKWARD (placebo) comparison from Stage 44 is
re-run at the most stringent (Step 4) control level - if the
asymmetry (forward significant, backward null) still holds after this
much more demanding specification, that is substantially more
compelling evidence for H1 than the Stage 44 result alone.

WORDING DISCIPLINE: surviving these controls would support H1 being
MORE LIKELY than H0 - it does not prove a physical/causal mechanism,
and unmeasured confounders can never be fully ruled out in an
observational study of this kind.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/patch_rolling_ar1.csv (reused if present)
        data/raw/[all environmental raster files]
        data/raw/oni_index.csv
Output: data/processed/mechanism_test_progressive_controls.csv
        data/processed/mechanism_test_final_placebo.csv
        figures/mechanism_test_progressive_controls.png
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

FAR_BAND = (800, 1100)
FAR_LAGS = [3, 6]  # focusing on the lags where the effect was significant in Stage 42-44
REGIONAL_CLIMATE_VARS = ["precip_anomaly", "temp_anomaly", "vpd_anomaly", "pdsi_anomaly"]

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

    # ---- REGIONAL climate controls: whole-study-area mean of key variables each month ----
    for c in REGIONAL_CLIMATE_VARS:
        regional_mean = merged.groupby("date")[c].transform("mean")
        merged[f"regional_{c}"] = regional_mean
    print("Added regional (whole-area) climate controls:", [f"regional_{c}" for c in REGIONAL_CLIMATE_VARS])

    n = len(loc)
    lats, lons, pids = loc["lat"].values, loc["lon"].values, loc["patch_id"].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine(lats[i], lons[i], lats, lons)
    dist_df = pd.DataFrame(dist_matrix, index=pids, columns=pids)
    latlon_df = loc.set_index("patch_id")[["lat", "lon"]]

    resilience_pivot = merged.pivot(index="date", columns="patch_id", values="resilience_ar1").sort_index()
    local_pivots = {c: merged.pivot(index="date", columns="patch_id", values=c).sort_index() for c in LOCAL_CONTROL_COLS}
    oni_series = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
    regional_series = {c: merged.drop_duplicates("date").set_index("date")[f"regional_{c}"].sort_index()
                        for c in REGIONAL_CLIMATE_VARS}
    dates_list = resilience_pivot.index.to_list()
    patches = resilience_pivot.columns.to_list()

    def build_neighbor_avg(pivot, band_map):
        out = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
        for pid in patches:
            neighbors = [p for p in band_map.get(pid, []) if p in pivot.columns]
            out[pid] = pivot[neighbors].mean(axis=1) if neighbors else np.nan
        return out

    band_map = {}
    lo, hi = FAR_BAND
    for pid in patches:
        d = dist_df.loc[pid]
        band_map[pid] = [p for p in d[(d > lo) & (d <= hi)].index.tolist() if p in resilience_pivot.columns]

    neighbor_resilience = build_neighbor_avg(resilience_pivot, band_map)
    neighbor_controls = {c: build_neighbor_avg(local_pivots[c], band_map) for c in LOCAL_CONTROL_COLS}
    neighbor_latlon = {}
    for coord in ["lat", "lon"]:
        out = pd.Series(index=patches, dtype=float)
        for pid in patches:
            neighbors = band_map.get(pid, [])
            out[pid] = latlon_df.loc[neighbors, coord].mean() if neighbors else np.nan
        neighbor_latlon[coord] = out

    def build_panel(lag, direction):
        recs = []
        for pid in patches:
            own_res = resilience_pivot[pid].values
            neigh_res = neighbor_resilience[pid].values
            oni_vals = oni_series.reindex(dates_list).values
            regional_vals = {c: regional_series[c].reindex(dates_list).values for c in REGIONAL_CLIMATE_VARS}
            for i in range(len(dates_list) - lag):
                if direction == "forward":
                    own_val, neigh_val, own_future, ctrl_idx = own_res[i], neigh_res[i], own_res[i + lag], i
                else:
                    own_val, neigh_val, own_future, ctrl_idx = own_res[i + lag], neigh_res[i + lag], own_res[i], i + lag
                rec = {"patch_id": pid, "own_resilience_t": own_val, "neighbor_resilience_state": neigh_val,
                       "own_resilience_future": own_future, "oni_value": oni_vals[ctrl_idx],
                       "target_lat": latlon_df.loc[pid, "lat"], "target_lon": latlon_df.loc[pid, "lon"],
                       "neighbor_lat": neighbor_latlon["lat"][pid], "neighbor_lon": neighbor_latlon["lon"][pid]}
                for c in LOCAL_CONTROL_COLS:
                    rec[f"target_{c}"] = local_pivots[c][pid].values[ctrl_idx]
                    rec[f"neighbor_{c}"] = neighbor_controls[c][pid].values[ctrl_idx]
                for c in REGIONAL_CLIMATE_VARS:
                    rec[f"regional_{c}"] = regional_vals[c][ctrl_idx]
                recs.append(rec)
        return pd.DataFrame(recs).dropna()

    local_control_terms = " + ".join([f"target_{c}" for c in LOCAL_CONTROL_COLS] +
                                      [f"neighbor_{c}" for c in LOCAL_CONTROL_COLS] + GLOBAL_CONTROL_COLS)
    regional_terms = " + ".join([f"regional_{c}" for c in REGIONAL_CLIMATE_VARS])
    latlon_terms = "target_lat + target_lon + neighbor_lat + neighbor_lon"

    steps = [
        ("Step 1: bare (no controls)", "own_resilience_t + neighbor_resilience_state"),
        ("Step 2: + local both-side environment (Stage 42/44 model)",
         f"own_resilience_t + neighbor_resilience_state + {local_control_terms}"),
        ("Step 3: + regional climate controls",
         f"own_resilience_t + neighbor_resilience_state + {local_control_terms} + {regional_terms}"),
        ("Step 4: + lat/lon (geographic confounding)",
         f"own_resilience_t + neighbor_resilience_state + {local_control_terms} + {regional_terms} + {latlon_terms}"),
    ]

    print(f"\n===== PROGRESSIVE CONTROLS: far band (800-1100km) negative effect =====")
    print("Testing whether the effect survives increasingly stringent controls (H0 = it")
    print("disappears once we control enough = shared forcing/confounding explains it;")
    print("H1 = it survives = more consistent with a real directional component)\n")

    progressive_results = []
    for lag in FAR_LAGS:
        print(f"--- Lag {lag} month(s) ---")
        panel_fwd = build_panel(lag, "forward")
        for label, rhs in steps:
            formula = f"own_resilience_future ~ {rhs}"
            m = smf.ols(formula, data=panel_fwd).fit(cov_type="cluster", cov_kwds={"groups": panel_fwd["patch_id"]})
            coef = m.params["neighbor_resilience_state"]
            pval = m.pvalues["neighbor_resilience_state"]
            ci_low, ci_high = m.conf_int().loc["neighbor_resilience_state"]
            sig = "*" if pval < 0.05 else " "
            print(f"  {label:48s}: coef={coef:+.5f} [{ci_low:+.5f},{ci_high:+.5f}] p={pval:.4f}{sig} n={len(panel_fwd)}")
            progressive_results.append((lag, label, coef, pval, ci_low, ci_high, len(panel_fwd)))

    progressive_df = pd.DataFrame(progressive_results, columns=[
        "lag_months", "step", "coef", "pval", "ci_low", "ci_high", "n_obs"
    ])
    progressive_df.to_csv(os.path.join(OUT_DIR, "mechanism_test_progressive_controls.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {3: 'darkorange', 6: 'darkred'}
    for lag in FAR_LAGS:
        sub = progressive_df[progressive_df["lag_months"] == lag]
        ax.errorbar(range(len(sub)), sub["coef"],
                     yerr=[sub["coef"]-sub["ci_low"], sub["ci_high"]-sub["coef"]],
                     fmt='o-', capsize=4, color=colors.get(lag, 'gray'), label=f"lag={lag}mo")
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0].replace("Step ", "").split(":")[0] for s in steps])
    ax.set_ylabel("Far-band (800-1100km) neighbor resilience effect")
    ax.set_title("Does the far-distance negative effect survive progressively stronger controls?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mechanism_test_progressive_controls.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/mechanism_test_progressive_controls.png")

    # ================================================================
    # Final placebo check at the MOST stringent (Step 4) control level
    # ================================================================
    print(f"\n===== FINAL PLACEBO CHECK at the most stringent control level (Step 4) =====")
    final_rhs = steps[-1][1]
    placebo_results = []
    for lag in FAR_LAGS:
        panel_fwd = build_panel(lag, "forward")
        panel_bwd = build_panel(lag, "backward")
        formula = f"own_resilience_future ~ {final_rhs}"
        m_fwd = smf.ols(formula, data=panel_fwd).fit(cov_type="cluster", cov_kwds={"groups": panel_fwd["patch_id"]})
        m_bwd = smf.ols(formula, data=panel_bwd).fit(cov_type="cluster", cov_kwds={"groups": panel_bwd["patch_id"]})
        f_coef, f_pval = m_fwd.params["neighbor_resilience_state"], m_fwd.pvalues["neighbor_resilience_state"]
        b_coef, b_pval = m_bwd.params["neighbor_resilience_state"], m_bwd.pvalues["neighbor_resilience_state"]
        print(f"  Lag {lag}mo: FORWARD coef={f_coef:+.5f} p={f_pval:.4f}  |  BACKWARD(placebo) coef={b_coef:+.5f} p={b_pval:.4f}")
        placebo_results.append((lag, "forward", f_coef, f_pval, len(panel_fwd)))
        placebo_results.append((lag, "backward", b_coef, b_pval, len(panel_bwd)))

    placebo_df = pd.DataFrame(placebo_results, columns=["lag_months", "direction", "coef", "pval", "n_obs"])
    placebo_df.to_csv(os.path.join(OUT_DIR, "mechanism_test_final_placebo.csv"), index=False)

    print("\n===== OVERALL VERDICT =====")
    step4_results = progressive_df[progressive_df["step"] == steps[-1][0]]
    n_still_sig = (step4_results["pval"] < 0.05).sum()
    fwd_sig = placebo_df[(placebo_df["direction"] == "forward") & (placebo_df["pval"] < 0.05)]
    bwd_sig = placebo_df[(placebo_df["direction"] == "backward") & (placebo_df["pval"] < 0.05)]
    print(f"Effect still significant after ALL controls (Step 4): {n_still_sig} / {len(step4_results)} lags")
    print(f"Forward still significant at Step 4: {len(fwd_sig)} / {len(FAR_LAGS)} lags")
    print(f"Backward(placebo) significant at Step 4: {len(bwd_sig)} / {len(FAR_LAGS)} lags")
    if n_still_sig > 0 and len(bwd_sig) == 0:
        print("\n-> H1 SUPPORTED: the far-distance negative effect survives progressively stronger")
        print("   controls (local+regional climate, ONI, lat/lon), while the backward placebo")
        print("   remains null even at this stringent level. This is meaningfully more compelling")
        print("   evidence for a genuine directional component than Stage 44 alone - though it")
        print("   still does not prove a physical/causal mechanism, and unmeasured confounders")
        print("   can never be fully excluded in an observational design.")
    elif n_still_sig == 0:
        print("\n-> H0 SUPPORTED: the far-distance effect disappears once regional climate and/or")
        print("   geographic controls are added - consistent with shared forcing or spatial")
        print("   confounding explaining the Stage 42-44 pattern, not a directional relationship.")
    else:
        print("\n-> MIXED evidence: the effect partially survives but the placebo asymmetry weakens")
        print("   or disappears - interpret with caution, report both patterns honestly.")

if __name__ == "__main__":
    main()