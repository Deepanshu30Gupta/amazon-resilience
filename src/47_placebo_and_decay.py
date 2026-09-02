"""
47_placebo_and_decay.py

Purpose: Two final, scoped additions to the lag=3 far-distance finding
(Stages 42-46), explicitly NOT attempting a mediation/mechanism
analysis (which would require new data on a candidate transmission
variable - stated as a limitation/future work, not built here).

PART A: Fake-neighbor (spatial permutation) placebo test. Tests: is
the effect specific to genuine spatial neighbor relationships, or
would ARBITRARY patch pairings at the same nominal "exposure" produce
a similar result? For each of N random permutations, the mapping from
target patch to its "neighbor resilience exposure" value is randomly
shuffled across patches (breaking the true spatial correspondence
while preserving the overall distribution of exposure values and the
correct own-patch/environmental-control alignment), and the far-band
lag=3 model (Stage 45's full Step 4 specification) is re-fit. This
builds a null distribution of coefficients under "fake" spatial
matching; the REAL coefficient's position within that null
distribution gives a permutation-based p-value - a genuinely different
and complementary test to the parametric p-values used throughout the
project so far.

PART B: Distance-decay curve fit. Fits beta(d) = beta_0 * exp(-d/lambda)
to the fully-controlled (Stage 45 Step 4 specification) lag=3
coefficient estimated separately for EVERY distance band (not just the
far band), giving a single interpretable parameter (lambda, the
characteristic spatial decay distance) for the paper, plus beta_0 (the
extrapolated effect at distance -> 0).

WORDING DISCIPLINE: even if the fake-neighbor placebo comes back null
and the decay curve fits well, this supports the finding being
consistent with genuine spatial structure - it does not prove a
physical/causal mechanism. The transmission mechanism itself remains
unidentified and is explicitly noted as a direction for future work
requiring additional data.

Input:  data/processed/patch_timeseries_anomaly.csv
        data/processed/patch_adjacency.csv
        data/processed/patch_locations.csv
        data/processed/patch_twi.csv
        data/processed/patch_disturbance_distance.csv
        data/processed/patch_rolling_ar1.csv (reused if present)
        data/raw/[all environmental raster files]
        data/raw/oni_index.csv
Output: data/processed/fake_neighbor_placebo_results.csv
        data/processed/distance_decay_curve_results.csv
        figures/fake_neighbor_placebo_distribution.png
        figures/distance_decay_curve_fit.png
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

    def fit_far_band_lag3(merged, band, extra_terms="", cluster_two_way=False, shuffle_seed=None):
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
        if shuffle_seed is not None:
            # FAKE-NEIGHBOR PLACEBO: randomly permute which patch's exposure series
            # gets assigned to which target patch, breaking the true spatial
            # correspondence while preserving the overall distribution of values
            # and each patch's own correct own-state/environmental alignment
            rng = np.random.default_rng(shuffle_seed)
            shuffled_cols = rng.permutation(neighbor_resilience.columns.tolist())
            neighbor_resilience = neighbor_resilience[shuffled_cols]
            neighbor_resilience.columns = patches  # reassign to original patch identities
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

    ALL_BANDS = [(75, 150), (150, 225), (225, 300), (300, 375), (375, 450),
                 (450, 550), (550, 650), (650, 800), (800, 1100)]
    N_PERMUTATIONS = 100

    # ================================================================
    # PART A: Fake-neighbor (spatial permutation) placebo test
    # ================================================================
    print(f"\n===== PART A: Fake-neighbor placebo test (far band, lag=3) =====")
    print(f"Real spatial assignment vs {N_PERMUTATIONS} random permutations of neighbor identity\n")

    real_coef, real_pval, real_ci, real_n = fit_far_band_lag3(dataset_24, ORIGINAL_BAND)
    print(f"REAL (true spatial neighbors): coef={real_coef:+.5f} p={real_pval:.4f} n={real_n}")

    perm_coefs = []
    for i in range(N_PERMUTATIONS):
        coef, pval, ci, n_obs = fit_far_band_lag3(dataset_24, ORIGINAL_BAND, shuffle_seed=i)
        perm_coefs.append(coef)
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{N_PERMUTATIONS} permutations done")

    perm_coefs = np.array(perm_coefs)
    perm_mean, perm_std = perm_coefs.mean(), perm_coefs.std()
    # two-sided permutation p-value: how often does a random permutation produce
    # a coefficient at least as extreme (in absolute value) as the real one?
    perm_pvalue = (np.sum(np.abs(perm_coefs) >= np.abs(real_coef)) + 1) / (N_PERMUTATIONS + 1)

    print(f"\nPermutation null distribution: mean={perm_mean:+.5f}, std={perm_std:.5f}")
    print(f"Real coefficient: {real_coef:+.5f}")
    print(f"Permutation-based p-value: {perm_pvalue:.4f}")
    if perm_pvalue < 0.05:
        print("-> The real (spatially-correct) coefficient is significantly more extreme than")
        print("   what random patch-pairing produces. Genuine spatial matching matters -")
        print("   arbitrary/fake neighbor assignments do NOT reproduce this effect.")
    else:
        print("-> The real coefficient is NOT significantly different from what random")
        print("   patch-pairing produces - this weakens confidence that the specific spatial")
        print("   relationship (vs. some other shared factor affecting all far-apart patches)")
        print("   is what matters.")

    placebo_df = pd.DataFrame({"permutation": range(N_PERMUTATIONS), "coef": perm_coefs})
    placebo_df.to_csv(os.path.join(OUT_DIR, "fake_neighbor_placebo_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_coefs, bins=25, color='lightgray', edgecolor='black', label="Fake-neighbor permutations")
    ax.axvline(real_coef, color='darkred', linewidth=2.5, label=f"Real spatial neighbors ({real_coef:+.4f})")
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Neighbor resilience effect coefficient")
    ax.set_ylabel("Count (out of 100 permutations)")
    ax.set_title(f"Fake-neighbor placebo: real vs. randomly-shuffled patch pairing\npermutation p={perm_pvalue:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fake_neighbor_placebo_distribution.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/fake_neighbor_placebo_distribution.png")

    # ================================================================
    # PART B: Distance-decay curve fit
    # ================================================================
    print(f"\n===== PART B: Distance-decay curve fit (fully-controlled lag=3 coefficient per band) =====")
    band_results = []
    for band in ALL_BANDS:
        coef, pval, ci, n_obs = fit_far_band_lag3(dataset_24, band)
        mid = (band[0] + band[1]) / 2
        sig = "*" if pval < 0.05 else " "
        print(f"  {band[0]:4d}-{band[1]:4d}km (mid={mid:5.0f}): coef={coef:+.5f} p={pval:.4f}{sig} n={n_obs}")
        band_results.append((band[0], band[1], mid, coef, pval, ci[0], ci[1], n_obs))

    band_df = pd.DataFrame(band_results, columns=["dist_lo", "dist_hi", "dist_mid", "coef", "pval", "ci_low", "ci_high", "n_obs"])

    from scipy.optimize import curve_fit

    def decay_func(d, beta0, lam):
        return beta0 * np.exp(-d / lam)

    try:
        popt, pcov = curve_fit(decay_func, band_df["dist_mid"], band_df["coef"],
                                p0=[0.01, 500], maxfev=5000)
        beta0_fit, lambda_fit = popt
        perr = np.sqrt(np.diag(pcov))
        print(f"\nFitted curve: beta(d) = {beta0_fit:+.5f} * exp(-d / {lambda_fit:.1f})")
        print(f"  beta_0 = {beta0_fit:+.5f} (+/- {perr[0]:.5f})")
        print(f"  lambda = {lambda_fit:.1f} km (+/- {perr[1]:.1f}) -- characteristic decay distance")

        residuals = band_df["coef"] - decay_func(band_df["dist_mid"], *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((band_df["coef"] - band_df["coef"].mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        print(f"  R-squared of fit: {r_squared:.3f}")

        band_df["beta0_fit"] = beta0_fit
        band_df["lambda_fit"] = lambda_fit
        band_df["r_squared"] = r_squared
        fit_success = True
    except RuntimeError:
        print("\nCurve fit did NOT converge - the coefficients likely don't follow a simple")
        print("monotonic exponential decay shape (consistent with the two-regime pattern")
        print("found in Stage 42-44: positive near, negative far, not a single smooth decay).")
        fit_success = False

    band_df.to_csv(os.path.join(OUT_DIR, "distance_decay_curve_results.csv"), index=False)

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.errorbar(band_df["dist_mid"], band_df["coef"],
                 yerr=[band_df["coef"]-band_df["ci_low"], band_df["ci_high"]-band_df["coef"]],
                 fmt='o', capsize=4, color='darkblue', label="Estimated coefficient (fully controlled)")
    if fit_success:
        d_smooth = np.linspace(band_df["dist_mid"].min(), band_df["dist_mid"].max(), 200)
        ax2.plot(d_smooth, decay_func(d_smooth, *popt), 'r--',
                  label=f"Fit: beta0*exp(-d/{lambda_fit:.0f}), R2={r_squared:.2f}")
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Neighbor resilience effect (lag=3, fully controlled)")
    ax2.set_title("Distance-decay pattern: fully-controlled lag=3 coefficient vs. distance")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "distance_decay_curve_fit.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/distance_decay_curve_fit.png")

    print("\n===== IMPORTANT LIMITATION =====")
    print("Neither of these tests identifies a physical transmission mechanism (e.g. atmospheric")
    print("moisture transport). Establishing the actual physical pathway would require additional")
    print("data (e.g. wind-direction-resolved moisture flux estimates between specific patch")
    print("pairs) and is explicitly noted here as a direction for future work, not attempted in")
    print("this project. These results support the finding being consistent with genuine spatial")
    print("structure (not arbitrary pairing) and provide a quantified spatial scale - they do not")
    print("prove causation.")

if __name__ == "__main__":
    main()