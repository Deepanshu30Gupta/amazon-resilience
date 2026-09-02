"""
49_wind_alignment.py

Purpose: The smallest tractable, honest test of first-order physical
plausibility for the moisture-transport hypothesis (deforestation/
resilience loss in A -> reduced evapotranspiration -> reduced
atmospheric moisture transport toward B -> B's precipitation/
resilience), using ONLY data already collected in this project (ERA5
2m wind u/v components) - NOT the full multi-level atmospheric
moisture-tracking model that would require substantial new data
acquisition and specialized methodology (explicitly out of scope for
this project's timeline, per team discussion).

LOGIC: if literal atmospheric moisture transport connects source patch
B to target patch A, the pairwise resilience association found in
Stage 48 should be STRONGER when the prevailing wind actually blows
FROM B TOWARD A, and weaker/absent when the wind blows the "wrong"
way (opposing or perpendicular to the B->A direction). This is a
genuine, physically-motivated test - if wind alignment doesn't matter
at all, that is real evidence AGAINST direct atmospheric transport
being the mechanism; if it does matter, that is suggestive (not
proof) supporting evidence.

For each (target A, source B, month t) observation in the Stage 48
pairwise dataset:
  1. bearing_BA = compass bearing from source B to target A (static
     per pair - the geographic direction moisture would need to travel)
  2. wind_bearing(t) = compass bearing the wind is blowing TOWARD,
     computed from source patch B's ERA5 u/v wind components at time t
  3. alignment(t) = cos(wind_bearing(t) - bearing_BA), ranging from
     +1 (wind blows directly toward target) to -1 (directly away)

Tests the interaction: source_resilience_t : alignment - does the
source-to-target coefficient become more negative (stronger) when wind
alignment is high? Also reports a simple aligned-vs-opposed subgroup
split for interpretability.

WORDING DISCIPLINE: this tests ONE piece of first-order plausibility
(does wind direction modulate the association) - NOT the full ET ->
moisture -> precipitation -> resilience causal chain, which would
require substantially more data and modeling than attempted here.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/patch_rolling_ar1.csv (reused if present)
        data/raw/[all environmental raster files]
        data/raw/oni_index.csv
Output: data/processed/wind_alignment_results.csv
        figures/wind_alignment_interaction.png
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

    local_terms = " + ".join([f"target_{c}" for c in LOCAL_CONTROL_COLS] +
                              [f"source_{c}" for c in LOCAL_CONTROL_COLS] + GLOBAL_CONTROL_COLS)
    formula = f"target_resilience_future ~ target_resilience_t + source_resilience_t + {local_terms}"

    print(f"\n===== BASELINE (from Stage 48, for reference): pairwise model, no wind info =====")
    print(f"Sample: {panel['target_patch'].nunique()} unique target patches, "
          f"{panel['source_patch'].nunique()} unique source patches, "
          f"{panel[['target_patch','source_patch']].drop_duplicates().shape[0]} unique pairs")

    panel["pair_id"] = panel["target_patch"].astype(str) + "_" + panel["source_patch"].astype(str)
    groups_2way = panel[["target_patch", "source_patch"]].apply(lambda col: pd.factorize(col)[0])

    m_base = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": groups_2way})
    coef_base = m_base.params["source_resilience_t"]
    pval_base = m_base.pvalues["source_resilience_t"]
    print(f"  beta={coef_base:+.5f}  p={pval_base:.4f} (two-way clustered, matches Stage 48)")

    # ================================================================
    # WIND ALIGNMENT INTERACTION TEST
    # ================================================================
    print(f"\n===== WIND ALIGNMENT TEST =====")
    print("Does the source-to-target effect get STRONGER when wind actually blows")
    print("from source toward target? (interaction term, two-way clustered)\n")

    formula_interaction = f"{formula} + source_resilience_t:wind_alignment"
    m_interact = smf.ols(formula_interaction, data=panel).fit(cov_type="cluster", cov_kwds={"groups": groups_2way})
    interaction_coef = m_interact.params["source_resilience_t:wind_alignment"]
    interaction_pval = m_interact.pvalues["source_resilience_t:wind_alignment"]
    interaction_ci = m_interact.conf_int().loc["source_resilience_t:wind_alignment"]
    print(f"Interaction (source_resilience_t x wind_alignment):")
    print(f"  coef={interaction_coef:+.6f}  95% CI=[{interaction_ci[0]:+.6f},{interaction_ci[1]:+.6f}]  p={interaction_pval:.4f}")
    if interaction_pval < 0.05 and interaction_coef < 0:
        print("  -> SUPPORTS wind-transport plausibility: the (negative) source effect becomes")
        print("     MORE negative (stronger) as wind alignment increases toward the target.")
    elif interaction_pval < 0.05:
        print("  -> Significant interaction, but in the OPPOSITE direction from what simple")
        print("     wind-transport would predict - worth noting honestly, not discarding.")
    else:
        print("  -> NOT significant: wind alignment does not detectably modulate the source-to-")
        print("     target effect. This is real evidence AGAINST simple direct wind-transport")
        print("     being the (or a) mechanism - though it doesn't rule out more complex")
        print("     atmospheric pathways (e.g. multi-day transport, upper-level winds) that")
        print("     this surface-wind-only test cannot capture.")

    # ---- Simple aligned vs opposed subgroup split, for interpretability ----
    print(f"\n===== SUBGROUP SPLIT: wind-aligned vs wind-opposed pairs =====")
    aligned = panel[panel["wind_alignment"] > 0]
    opposed = panel[panel["wind_alignment"] <= 0]
    print(f"Aligned (wind blowing toward target, n={len(aligned)}):")
    if len(aligned) > 100:
        groups_a = aligned[["target_patch", "source_patch"]].apply(lambda col: pd.factorize(col)[0])
        m_a = smf.ols(formula, data=aligned).fit(cov_type="cluster", cov_kwds={"groups": groups_a})
        coef_a, pval_a = m_a.params["source_resilience_t"], m_a.pvalues["source_resilience_t"]
        print(f"  beta={coef_a:+.5f}  p={pval_a:.4f}")
    else:
        coef_a, pval_a = np.nan, np.nan
        print("  (too few observations)")

    print(f"Opposed (wind blowing away from target, n={len(opposed)}):")
    if len(opposed) > 100:
        groups_o = opposed[["target_patch", "source_patch"]].apply(lambda col: pd.factorize(col)[0])
        m_o = smf.ols(formula, data=opposed).fit(cov_type="cluster", cov_kwds={"groups": groups_o})
        coef_o, pval_o = m_o.params["source_resilience_t"], m_o.pvalues["source_resilience_t"]
        print(f"  beta={coef_o:+.5f}  p={pval_o:.4f}")
    else:
        coef_o, pval_o = np.nan, np.nan
        print("  (too few observations)")

    results_df = pd.DataFrame([
        {"test": "baseline (no wind info, from Stage 48)", "coef": coef_base, "pval": pval_base},
        {"test": "interaction (source_resilience x wind_alignment)", "coef": interaction_coef, "pval": interaction_pval},
        {"test": "aligned subgroup", "coef": coef_a, "pval": pval_a},
        {"test": "opposed subgroup", "coef": coef_o, "pval": pval_o},
    ])
    results_df["band_lo"], results_df["band_hi"], results_df["lag"] = FAR_BAND[0], FAR_BAND[1], LAG
    results_df.to_csv(os.path.join(OUT_DIR, "wind_alignment_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Opposed\n(wind away)", "Aligned\n(wind toward)"]
    coefs = [coef_o, coef_a]
    colors = ['gray' if np.isnan(p) or p >= 0.05 else 'darkred' for p in [pval_o, pval_a]]
    ax.bar(labels, coefs, color=colors)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel("Source-to-target coefficient")
    ax.set_title("Pairwise effect: wind-aligned vs wind-opposed pairs\n(red = significant, gray = not)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "wind_alignment_interaction.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/wind_alignment_interaction.png")

    print("\n===== IMPORTANT LIMITATION =====")
    print("This tests only whether SURFACE wind direction modulates the association - a")
    print("first-order plausibility check using data already collected, NOT the full ET ->")
    print("atmospheric moisture transport -> precipitation -> resilience pathway, which would")
    print("require substantially more data (multi-level humidity/wind, evapotranspiration")
    print("fields) and specialized moisture-tracking methodology beyond this project's scope.")
    print("A null result here does not rule out more complex atmospheric pathways; a positive")
    print("result is suggestive supporting evidence, not proof of the mechanism.")

if __name__ == "__main__":
    main()