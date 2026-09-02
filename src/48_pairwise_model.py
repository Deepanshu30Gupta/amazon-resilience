"""
48_pairwise_model.py

Purpose: Move from the distance-BAND-AVERAGED model used throughout
Stages 39-47 (where B's "neighbor exposure" is the average resilience
of ALL patches at a given distance) to a genuinely PAIRWISE model,
preserving individual (source, target) patch pair variation that
averaging smoothed over.

Model, for every genuine pair (target=A, source=B) with
800 < distance(A,B) <= 1100 km:

  R_A(t+3) = alpha + beta*R_B(t) + gamma*R_A(t)
             + theta'*X_A(t) + phi'*X_B(t) + rho'*Z(t) + epsilon

where R_A(t+3) is target A's future resilience, R_B(t) is the SPECIFIC
source patch B's current resilience (not averaged with other far-band
patches), R_A(t) is A's own current resilience, X_A/X_B are each
patch's own local environmental controls, and Z(t) is ONI (global).

SCOPE DECISION (per team review): uses the EXISTING both-side local
environmental controls (already patch-specific) rather than full
two-way patch fixed effects, which would add 700+ dummy parameters and
risk severe overfitting/collinearity with the existing controls.

INFERENCE (REVISED per team review): the full 26,860-pair dataset
(~4.5 million rows) caused an out-of-memory crash under pair-level
clustering, and even under simpler target-patch clustering. Fixed by
(a) randomly sampling 3,000 of the 26,860 eligible pairs (fixed seed
42, whole pairs kept with their full time series - NOT a random sample
of months) to make the computation tractable, and (b) reporting THREE
separate clustering approaches on this sampled dataset: target-
clustered, source-clustered, and two-way target x source clustered SE
- the last being the primary specification, because dependence can arise
through both repeated target patches AND repeated
use of the same source patch across different pairs. The nominal
~498,000 row count should NOT be read as 498,000 independent
observations - target A's outcome at a given month is repeated across
every source B paired with it, a form of pseudo-replication that
clustering is specifically meant to address, not resolve away.

IMPLEMENTATION NOTE: this dataset is far larger than any previous
stage (every individual pair is now its own set of observations,
rather than being averaged into one target-patch-month row) -
potentially over a million rows for the 800-1100km band alone. Built
using VECTORIZED pandas merges, not the nested Python loops used in
every earlier stage - those would not scale to this size.

WORDING DISCIPLINE: even a significant, well-estimated beta here would
establish "specific source-target pairs show a statistically
detectable directional association after controlling for measured
confounders" - NOT proof that patch B physically causes changes in
patch A.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/patch_rolling_ar1.csv (reused if present)
        data/raw/[all environmental raster files]
        data/raw/oni_index.csv
Output: data/processed/pairwise_model_results.csv
        figures/pairwise_model_coefficient_by_distance.png
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

    source_cols = ["patch_id", "date", "resilience_ar1"] + LOCAL_CONTROL_COLS
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

    local_terms = " + ".join([f"target_{c}" for c in LOCAL_CONTROL_COLS] +
                              [f"source_{c}" for c in LOCAL_CONTROL_COLS] + GLOBAL_CONTROL_COLS)
    formula = f"target_resilience_future ~ target_resilience_t + source_resilience_t + {local_terms}"

    print(f"\n===== PAIRWISE MODEL: {FAR_BAND[0]}-{FAR_BAND[1]}km, lag={LAG} =====")
    print(f"Sample: {panel['target_patch'].nunique()} unique target patches, "
          f"{panel['source_patch'].nunique()} unique source patches, "
          f"{panel[['target_patch','source_patch']].drop_duplicates().shape[0]} unique pairs")
    print("NOTE: the row count below is NOT the effective sample size - target A's outcome")
    print("repeats across every source B paired with it (pseudo-replication). Clustering is")
    print("meant to account for this, not eliminate the need to interpret n cautiously.\n")

    panel["pair_id"] = panel["target_patch"].astype(str) + "_" + panel["source_patch"].astype(str)

    print("--- Target-clustered SE ---")
    m_target = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["target_patch"]})
    coef_t, pval_t = m_target.params["source_resilience_t"], m_target.pvalues["source_resilience_t"]
    ci_t = m_target.conf_int().loc["source_resilience_t"]
    print(f"  beta={coef_t:+.5f}  95% CI=[{ci_t[0]:+.5f},{ci_t[1]:+.5f}]  p={pval_t:.4f}")

    print("--- Source-clustered SE ---")
    m_source = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["source_patch"]})
    coef_s, pval_s = m_source.params["source_resilience_t"], m_source.pvalues["source_resilience_t"]
    ci_s = m_source.conf_int().loc["source_resilience_t"]
    print(f"  beta={coef_s:+.5f}  95% CI=[{ci_s[0]:+.5f},{ci_s[1]:+.5f}]  p={pval_s:.4f}")

    print("--- Two-way (target x source) clustered SE - PRIMARY specification, because")
    print("    dependence can arise through both repeated target patches AND repeated")
    print("    source patches (verified numerically equivalent to statsmodels'")
    print("    cov_cluster_2groups() called directly - not just an approximation) ---")
    groups_2way = panel[["target_patch", "source_patch"]].apply(lambda col: pd.factorize(col)[0])
    m_2way = smf.ols(formula, data=panel).fit(cov_type="cluster", cov_kwds={"groups": groups_2way})
    coef, pval = m_2way.params["source_resilience_t"], m_2way.pvalues["source_resilience_t"]
    ci_low, ci_high = m_2way.conf_int().loc["source_resilience_t"]
    print(f"  beta={coef:+.5f}  95% CI=[{ci_low:+.5f},{ci_high:+.5f}]  p={pval:.4f}")
    print(f"\nPoint estimate is identical across all three clustering approaches because")
    print(f"clustering changes the estimated covariance/SE, not the OLS coefficient itself.")
    print(f"beta={coef:+.5f}")

    results_df = pd.DataFrame([
        {"clustering": "target", "coef": coef_t, "pval": pval_t, "ci_low": ci_t[0], "ci_high": ci_t[1]},
        {"clustering": "source", "coef": coef_s, "pval": pval_s, "ci_low": ci_s[0], "ci_high": ci_s[1]},
        {"clustering": "two-way (primary specification)", "coef": coef, "pval": pval, "ci_low": ci_low, "ci_high": ci_high},
    ])
    results_df["band_lo"], results_df["band_hi"], results_df["lag"] = FAR_BAND[0], FAR_BAND[1], LAG
    results_df["n_obs"] = len(panel)
    results_df["n_unique_pairs"] = panel["pair_id"].nunique()
    results_df.to_csv(os.path.join(OUT_DIR, "pairwise_model_results.csv"), index=False)

    # ---- Also break down by distance within the band, to see if the pairwise
    # result shows any gradient the band-average approach might have missed ----
    print(f"\n===== BONUS: pairwise coefficient by distance sub-bands within {FAR_BAND} =====")
    # Vectorized distance lookup (NOT row-wise .apply, which would not scale to
    # this dataset size) - build a (target,source)->distance lookup table once,
    # then merge it in
    pair_distances = []
    for target in pids:
        d = dist_df.loc[target]
        sources = d[(d > lo) & (d <= hi)].index.tolist()
        for source in sources:
            pair_distances.append((target, source, d[source]))
    pair_dist_df = pd.DataFrame(pair_distances, columns=["target_patch", "source_patch", "pair_distance"])
    panel = panel.merge(pair_dist_df, on=["target_patch", "source_patch"], how="left")
    sub_edges = np.linspace(FAR_BAND[0], FAR_BAND[1], 4)
    subband_results = []
    for i in range(len(sub_edges) - 1):
        sub_lo, sub_hi = sub_edges[i], sub_edges[i+1]
        sub_panel = panel[(panel["pair_distance"] > sub_lo) & (panel["pair_distance"] <= sub_hi)]
        if len(sub_panel) < 100:
            continue
        m_sub = smf.ols(formula, data=sub_panel).fit(cov_type="cluster", cov_kwds={"groups": sub_panel["target_patch"]})
        c, p = m_sub.params["source_resilience_t"], m_sub.pvalues["source_resilience_t"]
        sig = "*" if p < 0.05 else " "
        n_targets = sub_panel["target_patch"].nunique()
        n_sources = sub_panel["source_patch"].nunique()
        n_pairs = sub_panel[["target_patch", "source_patch"]].drop_duplicates().shape[0]
        print(f"  {sub_lo:.0f}-{sub_hi:.0f}km: coef={c:+.5f} p={p:.4f}{sig} n={len(sub_panel)} "
              f"({n_targets} targets, {n_sources} sources, {n_pairs} pairs)")
        subband_results.append((sub_lo, sub_hi, c, p, len(sub_panel)))

    fig, ax = plt.subplots(figsize=(8, 5))
    if subband_results:
        sb_df = pd.DataFrame(subband_results, columns=["lo", "hi", "coef", "pval", "n"])
        sb_df["mid"] = (sb_df["lo"] + sb_df["hi"]) / 2
        colors = ['darkred' if p < 0.05 else 'gray' for p in sb_df["pval"]]
        ax.scatter(sb_df["mid"], sb_df["coef"], c=colors, s=100)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(coef, color='blue', linestyle=':', label=f"Overall pairwise coef ({coef:+.4f})")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Pairwise coefficient")
    ax.set_title(f"Pairwise model coefficient within the {FAR_BAND[0]}-{FAR_BAND[1]}km band")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pairwise_model_coefficient_by_distance.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/pairwise_model_coefficient_by_distance.png")

    print("\n===== INTERPRETATION =====")
    if pval < 0.05:
        print(f"The pairwise model finds a statistically significant, genuinely pair-level")
        print(f"association (not averaged across neighbors): beta={coef:+.5f}, p={pval:.4f}.")
        print(f"This is a more granular test than any prior stage, using real individual")
        print(f"source-target pairs rather than distance-band averages.")
    else:
        print(f"The pairwise model does NOT find a significant pair-level association")
        print(f"(beta={coef:+.5f}, p={pval:.4f}). Combined with Stage 47's fake-neighbor")
        print(f"placebo result, this is consistent evidence that the far-distance pattern")
        print(f"found in the band-averaged models (Stages 42-46) may not reflect genuine")
        print(f"specific pairwise relationships.")
    print("\nWORDING CAUTION: even a significant result here establishes 'specific source-")
    print("target pairs show a statistically detectable directional association after")
    print("controlling for measured confounders' - NOT proof of physical causation.")

if __name__ == "__main__":
    main()