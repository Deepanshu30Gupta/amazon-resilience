"""
50_moisture_proxy_chain.py

Purpose: A tightly-scoped EXPLORATORY mechanism test (per team
agreement) - NOT a full atmospheric moisture-transport model. Builds a
downwind moisture-availability PROXY entirely from data already in
this project's pipeline (no new downloads, no atmospheric back-
trajectory modeling), and tests two genuinely new links in the
hypothesized chain:

  A resilience -> A moisture contribution -> downwind moisture near B
  -> B precipitation -> B resilience

PROXY CONSTRUCTION (explicit, so its limitations are auditable):
  M_AB(t) = source_resilience_t * wind_alignment(t) * source_rzsm_anomaly(t)
  - source_resilience_t: source patch's vegetation/resilience state
    (proxy for evapotranspiration - healthier vegetation transpires
    more; NOT a real ET measurement, no ET data exists in this project)
  - wind_alignment(t): from Stage 49 - does surface wind blow from
    source toward target (+1) or away (-1)
  - source_rzsm_anomaly(t): source's root-zone soil moisture anomaly
    (proxy for water available to be transpired)
  This is a defensible combination of existing variables representing
  the CONCEPTUAL components of the hypothesis - explicitly NOT a
  validated physical moisture-transport quantity.

PRE-SPECIFIED DECISION TREE (reported honestly regardless of outcome):
  Link 2: does M_AB(t) predict target's FUTURE precipitation?
          (genuinely new test - Stage 49 never touched precipitation)
  Link 3: does target's OWN precipitation predict target's OWN future
          resilience? (genuinely new test, single-patch, not pairwise)
  Mediation check: does adding M_AB to the Stage 48 pairwise model
          attenuate the direct source_resilience_t coefficient?
If all links hold, that is a mechanism-CONSISTENT chain (not proof).
If some or none hold, that is reported exactly as such - the honest
fallback ("A-B association detected, physical pathway not identified
by available data") remains a legitimate, complete result.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/patch_rolling_ar1.csv (reused if present)
        data/raw/[all environmental raster files]
        data/raw/oni_index.csv
Output: data/processed/moisture_proxy_link2_results.csv
        data/processed/moisture_proxy_link3_results.csv
        data/processed/moisture_proxy_mediation_results.csv
        figures/moisture_proxy_chain_summary.png
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
LAG = 3

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), parse_dates=["date"])
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

    # Raw (not anomaly) u/v wind components, needed for actual wind DIRECTION -
    # the existing "wind_anomaly" control is magnitude-only and loses direction
    def to_long_raw(patch_vals, n_months, loc, colname):
        dates = pd.date_range(START_DATE, periods=n_months, freq="MS")
        records = []
        for _, r in loc.iterrows():
            pid, pr, pc = int(r.patch_id), int(r.row), int(r.col)
            for m in range(n_months):
                records.append((pid, dates[m], patch_vals[m, pr, pc]))
        return pd.DataFrame(records, columns=["patch_id", "date", colname]).dropna()

    wind_u_raw_df = to_long_raw(u_vals, n_m, loc, "wind_u_raw")
    wind_v_raw_df = to_long_raw(v_vals, n_m, loc, "wind_v_raw")

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
               .merge(rolling_ar1_df, on=["patch_id", "date"], how="inner") \
               .merge(wind_u_raw_df, on=["patch_id", "date"], how="left") \
               .merge(wind_v_raw_df, on=["patch_id", "date"], how="left")
    merged = merged.dropna(subset=LOCAL_CONTROL_COLS + GLOBAL_CONTROL_COLS + ["resilience_ar1"])
    print(f"Patch-month dataset shape: {merged.shape}")

    # ---- Distance matrix and valid pairs list ----
    n = len(loc)
    lats, lons, pids = loc["lat"].values, loc["lon"].values, loc["patch_id"].values
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine(lats[i], lons[i], lats, lons)
    dist_df = pd.DataFrame(dist_matrix, index=pids, columns=pids)

    lo, hi = FAR_BAND
    pair_list = []
    for target in pids:
        d = dist_df.loc[target]
        sources = d[(d > lo) & (d <= hi)].index.tolist()
        for source in sources:
            pair_list.append((target, source))
    pairs_df = pd.DataFrame(pair_list, columns=["target_patch", "source_patch"])
    print(f"\nValid (target, source) pairs in {lo}-{hi}km band: {len(pairs_df)}")
    print(f"(directed pairs - target A paired with each source B separately, not averaged)")

    # SUBSAMPLE pairs (keeping each sampled pair's FULL time series) - the full
    # pairwise panel would be ~4.5 million rows, which crashes statsmodels' OLS
    # fitting even on a machine with several GB of RAM. Sampling a large, random
    # subset of pairs preserves genuine pair-level (non-averaged) structure while
    # keeping the computation tractable. Fixed seed for reproducibility.
    MAX_PAIRS = 3000
    if len(pairs_df) > MAX_PAIRS:
        print(f"Subsampling {MAX_PAIRS} of {len(pairs_df)} pairs (fixed seed=42) to keep the")
        print(f"regression computationally tractable - each sampled pair keeps its FULL")
        print(f"monthly time series, so this is a random sample of PAIRS, not of months.")
        pairs_df = pairs_df.sample(n=MAX_PAIRS, random_state=42).reset_index(drop=True)

    # ---- VECTORIZED pairwise panel construction (no nested Python loops) ----
    print("\nBuilding pairwise panel via vectorized merges (this dataset is much larger")
    print("than previous stages - expect this step to take real time)...")

    target_cols = ["patch_id", "date", "resilience_ar1"] + LOCAL_CONTROL_COLS
    target_data = merged[target_cols].rename(columns={
        "patch_id": "target_patch", "resilience_ar1": "target_resilience_t",
        **{c: f"target_{c}" for c in LOCAL_CONTROL_COLS}
    })
    # target's FUTURE resilience (the outcome) - shift by -LAG within each patch
    target_future = merged[["patch_id", "date", "resilience_ar1"]].copy()
    target_future["date"] = target_future["date"] - pd.DateOffset(months=LAG)
    target_future = target_future.rename(columns={
        "patch_id": "target_patch", "resilience_ar1": "target_resilience_future"
    })
    target_data = target_data.merge(target_future, on=["target_patch", "date"], how="inner")

    source_cols = ["patch_id", "date", "resilience_ar1", "wind_u_raw", "wind_v_raw"] + LOCAL_CONTROL_COLS
    source_data = merged[source_cols].rename(columns={
        "patch_id": "source_patch", "resilience_ar1": "source_resilience_t",
        **{c: f"source_{c}" for c in LOCAL_CONTROL_COLS}
    })

    oni_data = merged[["date", "oni_value"]].drop_duplicates()

    # Merge: pairs -> target's data (at every date) -> source's data (SAME date t,
    # since target_data's "date" column already represents time t after the
    # future-shift merge above) -> ONI
    panel = pairs_df.merge(target_data, on="target_patch", how="inner")
    panel = panel.merge(source_data, on=["source_patch", "date"], how="inner")
    panel = panel.merge(oni_data, on="date", how="inner")
    panel = panel.dropna()
    print(f"\nFinal pairwise panel shape: {panel.shape}")

    # ================================================================
    # WIND ALIGNMENT: bearing from source to target, wind direction at
    # the source, and their alignment (+1 = wind blows directly toward
    # target, -1 = directly away)
    # ================================================================
    print("\nComputing wind alignment (bearing from source to target vs. actual wind direction)...")
    latlon_df = loc.set_index("patch_id")[["lat", "lon"]]

    def compass_bearing(lat1, lon1, lat2, lon2):
        lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2r - lon1r
        x = np.sin(dlon) * np.cos(lat2r)
        y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    # Static per-pair bearing (source -> target), computed once per unique pair
    unique_pairs = panel[["target_patch", "source_patch"]].drop_duplicates()
    unique_pairs["bearing_source_to_target"] = compass_bearing(
        latlon_df.loc[unique_pairs["source_patch"], "lat"].values,
        latlon_df.loc[unique_pairs["source_patch"], "lon"].values,
        latlon_df.loc[unique_pairs["target_patch"], "lat"].values,
        latlon_df.loc[unique_pairs["target_patch"], "lon"].values,
    )
    panel = panel.merge(unique_pairs, on=["target_patch", "source_patch"], how="left")

    # Time-varying wind direction AT THE SOURCE (the direction the wind is
    # blowing TOWARD, from the source's own raw u/v components)
    panel["wind_bearing_source"] = (np.degrees(np.arctan2(panel["wind_u_raw"], panel["wind_v_raw"])) + 360) % 360

    # Alignment: +1 = wind blows directly toward target, -1 = directly away
    angle_diff = np.radians(panel["wind_bearing_source"] - panel["bearing_source_to_target"])
    panel["wind_alignment"] = np.cos(angle_diff)
    print(f"Wind alignment computed: mean={panel['wind_alignment'].mean():.3f}, "
          f"std={panel['wind_alignment'].std():.3f} (range -1 to +1)")

    # ================================================================
    # M_AB PROXY CONSTRUCTION
    # ================================================================
    panel["M_AB"] = panel["source_resilience_t"] * panel["wind_alignment"] * panel["source_rzsm_anomaly"]
    print(f"\nM_AB proxy constructed: mean={panel['M_AB'].mean():.5f}, std={panel['M_AB'].std():.5f}")

    local_terms_all = " + ".join([f"target_{c}" for c in LOCAL_CONTROL_COLS] +
                                  [f"source_{c}" for c in LOCAL_CONTROL_COLS] + GLOBAL_CONTROL_COLS)
    groups_2way = panel[["target_patch", "source_patch"]].apply(lambda col: pd.factorize(col)[0])

    # ================================================================
    # LINK 2: does M_AB(t) predict target's FUTURE precipitation?
    # (genuinely new test - Stage 49 never touched precipitation)
    # ================================================================
    print(f"\n===== LINK 2: M_AB(t) -> target's future precipitation =====")
    target_precip_future = merged[["patch_id", "date", "precip_anomaly"]].copy()
    target_precip_future["date"] = target_precip_future["date"] - pd.DateOffset(months=LAG)
    target_precip_future = target_precip_future.rename(columns={
        "patch_id": "target_patch", "precip_anomaly": "target_precip_future"
    })
    panel_link2 = panel.merge(target_precip_future, on=["target_patch", "date"], how="inner").dropna(
        subset=["target_precip_future", "M_AB"])
    print(f"Link 2 panel: {panel_link2.shape[0]} rows")

    # exclude target_precip_anomaly from controls here since precip is now the outcome
    link2_controls = [c for c in LOCAL_CONTROL_COLS if c != "precip_anomaly"]
    link2_terms = " + ".join([f"target_{c}" for c in link2_controls] +
                              [f"source_{c}" for c in link2_controls] + GLOBAL_CONTROL_COLS)
    formula_link2 = f"target_precip_future ~ M_AB + target_resilience_t + {link2_terms}"
    groups_link2 = panel_link2[["target_patch", "source_patch"]].apply(lambda col: pd.factorize(col)[0])
    m_link2 = smf.ols(formula_link2, data=panel_link2).fit(cov_type="cluster", cov_kwds={"groups": groups_link2})
    coef2, pval2 = m_link2.params["M_AB"], m_link2.pvalues["M_AB"]
    ci2 = m_link2.conf_int().loc["M_AB"]
    print(f"M_AB -> target future precip: coef={coef2:+.5f}  95% CI=[{ci2[0]:+.5f},{ci2[1]:+.5f}]  p={pval2:.4f}")
    link2_holds = pval2 < 0.05
    print(f"LINK 2 {'HOLDS' if link2_holds else 'DOES NOT HOLD'} (p<0.05: {link2_holds})")

    pd.DataFrame([{"link": "M_AB -> target future precip", "coef": coef2, "pval": pval2,
                    "ci_low": ci2[0], "ci_high": ci2[1], "holds": link2_holds}]).to_csv(
        os.path.join(OUT_DIR, "moisture_proxy_link2_results.csv"), index=False)

    # ================================================================
    # LINK 3: does target's OWN precipitation predict target's OWN
    # future resilience? Single-patch test, NOT pairwise.
    # ================================================================
    print(f"\n===== LINK 3: target's own precipitation -> target's own future resilience =====")
    single_patch_recs = []
    patches_list = merged["patch_id"].unique()
    resilience_pivot_single = merged.pivot(index="date", columns="patch_id", values="resilience_ar1").sort_index()
    precip_pivot_single = merged.pivot(index="date", columns="patch_id", values="precip_anomaly").sort_index()
    other_pivots_single = {c: merged.pivot(index="date", columns="patch_id", values=c).sort_index()
                            for c in LOCAL_CONTROL_COLS if c != "precip_anomaly"}
    oni_series_single = merged.drop_duplicates("date").set_index("date")["oni_value"].sort_index()
    dates_single = resilience_pivot_single.index.to_list()

    for pid in patches_list:
        if pid not in resilience_pivot_single.columns:
            continue
        res_t = resilience_pivot_single[pid].values
        precip_t = precip_pivot_single[pid].values
        oni_t = oni_series_single.reindex(dates_single).values
        for i in range(len(dates_single) - LAG):
            rec = {"patch_id": pid, "own_resilience_t": res_t[i], "own_precip_t": precip_t[i],
                   "own_resilience_future": res_t[i + LAG], "oni_value": oni_t[i]}
            for c in other_pivots_single:
                rec[f"own_{c}"] = other_pivots_single[c][pid].values[i]
            single_patch_recs.append(rec)
    single_panel = pd.DataFrame(single_patch_recs).dropna()
    print(f"Link 3 panel: {single_panel.shape[0]} rows, {single_panel['patch_id'].nunique()} patches")

    own_control_terms = " + ".join([f"own_{c}" for c in other_pivots_single] + GLOBAL_CONTROL_COLS)
    formula_link3 = f"own_resilience_future ~ own_precip_t + own_resilience_t + {own_control_terms}"
    m_link3 = smf.ols(formula_link3, data=single_panel).fit(cov_type="cluster", cov_kwds={"groups": single_panel["patch_id"]})
    coef3, pval3 = m_link3.params["own_precip_t"], m_link3.pvalues["own_precip_t"]
    ci3 = m_link3.conf_int().loc["own_precip_t"]
    print(f"Own precip -> own future resilience: coef={coef3:+.5f}  95% CI=[{ci3[0]:+.5f},{ci3[1]:+.5f}]  p={pval3:.4f}")
    link3_holds = pval3 < 0.05
    print(f"LINK 3 {'HOLDS' if link3_holds else 'DOES NOT HOLD'} (p<0.05: {link3_holds})")

    pd.DataFrame([{"link": "own precip -> own future resilience", "coef": coef3, "pval": pval3,
                    "ci_low": ci3[0], "ci_high": ci3[1], "holds": link3_holds}]).to_csv(
        os.path.join(OUT_DIR, "moisture_proxy_link3_results.csv"), index=False)

    # ================================================================
    # MEDIATION CHECK: does adding M_AB to the Stage 48 pairwise model
    # attenuate the direct source_resilience_t coefficient?
    # ================================================================
    print(f"\n===== MEDIATION CHECK: does M_AB attenuate the direct source-target effect? =====")
    formula_before = f"target_resilience_future ~ target_resilience_t + source_resilience_t + {local_terms_all}"
    formula_after = f"target_resilience_future ~ target_resilience_t + source_resilience_t + M_AB + {local_terms_all}"

    m_before = smf.ols(formula_before, data=panel).fit(cov_type="cluster", cov_kwds={"groups": groups_2way})
    coef_before = m_before.params["source_resilience_t"]
    pval_before = m_before.pvalues["source_resilience_t"]

    m_after = smf.ols(formula_after, data=panel).fit(cov_type="cluster", cov_kwds={"groups": groups_2way})
    coef_after = m_after.params["source_resilience_t"]
    pval_after = m_after.pvalues["source_resilience_t"]
    coef_mab = m_after.params["M_AB"]
    pval_mab = m_after.pvalues["M_AB"]

    print(f"BEFORE (no M_AB): source_resilience_t coef={coef_before:+.5f}  p={pval_before:.4f}")
    print(f"AFTER  (+M_AB):   source_resilience_t coef={coef_after:+.5f}  p={pval_after:.4f}")
    print(f"                  M_AB coef={coef_mab:+.5f}  p={pval_mab:.4f}")
    attenuation_pct = 100 * (abs(coef_before) - abs(coef_after)) / abs(coef_before) if coef_before != 0 else np.nan
    print(f"Attenuation of direct effect: {attenuation_pct:+.1f}%")

    pd.DataFrame([
        {"model": "before (no M_AB)", "coef": coef_before, "pval": pval_before},
        {"model": "after (+M_AB)", "coef": coef_after, "pval": pval_after},
        {"model": "M_AB term itself", "coef": coef_mab, "pval": pval_mab},
    ]).to_csv(os.path.join(OUT_DIR, "moisture_proxy_mediation_results.csv"), index=False)

    # ---- Plot summary ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    links = ["Link 2\n(M_AB->precip)", "Link 3\n(precip->resilience)"]
    coefs_plot = [coef2, coef3]
    pvals_plot = [pval2, pval3]
    colors_plot = ['darkred' if p < 0.05 else 'gray' for p in pvals_plot]
    ax.bar(links, coefs_plot, color=colors_plot)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title("Decision tree links\n(red=significant, gray=not)")

    ax2 = axes[1]
    ax2.bar(["Before M_AB", "After M_AB"], [coef_before, coef_after],
            color=['steelblue', 'darkorange'])
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title(f"Mediation check\n(direct effect attenuation: {attenuation_pct:+.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "moisture_proxy_chain_summary.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/moisture_proxy_chain_summary.png")

    # ---- Final honest verdict ----
    print(f"\n===== FINAL DECISION TREE VERDICT =====")
    n_holds = sum([link2_holds, link3_holds])
    print(f"Link 2 (M_AB -> target future precip): {'HOLDS' if link2_holds else 'does not hold'}")
    print(f"Link 3 (own precip -> own future resilience): {'HOLDS' if link3_holds else 'does not hold'}")
    if n_holds == 2:
        print("\n-> BOTH links hold: this is a mechanism-CONSISTENT chain using the available proxy")
        print("   data. This does NOT prove the physical moisture-transport mechanism - M_AB is an")
        print("   exploratory proxy, not a validated atmospheric transport quantity - but it is a")
        print("   meaningfully more complete, internally consistent result than Stage 49 alone.")
    elif n_holds == 1:
        print("\n-> ONE of two links holds. Report exactly which one, honestly, rather than treating")
        print("   this as a complete chain. The mechanism remains only partially supported.")
    else:
        print("\n-> NEITHER link holds under this proxy construction. The honest conclusion: the A-B")
        print("   statistical association (Stage 48) is real and robust, but the available data do")
        print("   NOT identify the physical transmission mechanism through this moisture-proxy")
        print("   pathway. That remains a complete, legitimate research result - the mechanism")
        print("   should be stated as an open question for future work with dedicated atmospheric")
        print("   moisture-tracking data, not something this project's data could resolve.")

if __name__ == "__main__":
    main()