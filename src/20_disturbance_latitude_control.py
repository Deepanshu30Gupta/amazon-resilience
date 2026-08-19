"""
20_disturbance_latitude_control.py

Purpose: Test whether Stage 18's surprising finding (distance-to-
disturbance positively correlates with resilience loss, r=0.58)
survives controlling for latitude - since the Arc of Deforestation
clusters geographically, "far from disturbance" may partly just mean
"farther north," and Stage 5's resilience map already showed its own
north-south gradient. If both variables are independently driven by
latitude, the Stage 18 correlation could be confounded rather than a
genuine disturbance-related pattern.

Three models:
  Model 1: kendall_tau ~ dist_to_disturbance_km                (Stage 18's original test)
  Model 2: kendall_tau ~ dist_to_disturbance_km + lat            (latitude-controlled)
  Model 3: kendall_tau ~ dist_to_disturbance_km + lat + lon      (fully geography-controlled)

Also reports the raw correlation between distance-to-disturbance and
latitude itself, to check how much overlap (collinearity) exists
between the two explanations in the first place.

Input:  data/processed/patch_locations.csv
        data/processed/patch_resilience_trend.csv
        data/processed/patch_disturbance_distance.csv
Output: data/processed/disturbance_latitude_control_results.csv
        printed comparison across the three models
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import os

OUT_DIR = "data/processed"

def main():
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    trend = pd.read_csv(os.path.join(OUT_DIR, "patch_resilience_trend.csv")).dropna(subset=["kendall_tau"])
    dist_df = pd.read_csv(os.path.join(OUT_DIR, "patch_disturbance_distance.csv"))

    merged = trend.merge(loc[["patch_id", "lat", "lon"]], on="patch_id").merge(dist_df, on="patch_id")
    print(f"Merged dataset: {len(merged)} patches\n")

    # First: how collinear are distance-to-disturbance and latitude?
    collin_r, collin_p = stats.pearsonr(merged["dist_to_disturbance_km"], merged["lat"])
    print(f"Collinearity check: correlation between distance-to-disturbance and latitude")
    print(f"  r={collin_r:.4f}, p={collin_p:.4f}")
    if abs(collin_r) > 0.5:
        print("  -> Substantial overlap: distance-to-disturbance and latitude are strongly related,")
        print("     meaning the latitude confound is a real concern worth taking seriously.\n")
    else:
        print("  -> Limited overlap: distance-to-disturbance is not simply a proxy for latitude.\n")

    specs = [
        ("Model 1: distance only",
         "kendall_tau ~ dist_to_disturbance_km"),
        ("Model 2: distance + latitude",
         "kendall_tau ~ dist_to_disturbance_km + lat"),
        ("Model 3: distance + lat + lon",
         "kendall_tau ~ dist_to_disturbance_km + lat + lon"),
    ]

    results = []
    print("===== Regression results =====")
    for label, formula in specs:
        m = smf.ols(formula, data=merged).fit()
        coef = m.params["dist_to_disturbance_km"]
        pval = m.pvalues["dist_to_disturbance_km"]
        r2 = m.rsquared
        print(f"\n{label}")
        print(f"  distance coefficient: {coef:.6f}, p={pval:.4f}, model R^2={r2:.3f}")
        results.append((label, coef, pval, r2))

    results_df = pd.DataFrame(results, columns=["model", "distance_coef", "distance_pval", "r_squared"])
    results_df.to_csv(os.path.join(OUT_DIR, "disturbance_latitude_control_results.csv"), index=False)

    first_p, last_p = results_df.iloc[0]["distance_pval"], results_df.iloc[-1]["distance_pval"]
    first_coef, last_coef = results_df.iloc[0]["distance_coef"], results_df.iloc[-1]["distance_coef"]
    pct_change = 100 * (first_coef - last_coef) / first_coef if first_coef != 0 else float('nan')

    print(f"\n===== SUMMARY =====")
    print(f"Distance coefficient: {first_coef:.6f} (alone) -> {last_coef:.6f} (with lat+lon), "
          f"{pct_change:.1f}% change")
    print(f"Significance: p={first_p:.4f} (alone) -> p={last_p:.4f} (with lat+lon)")
    if last_p >= 0.05 and first_p < 0.05:
        print("\n-> OUTCOME 1: The distance effect DISAPPEARS once geography is controlled for.")
        print("   The Stage 18 finding was likely largely a latitude/geographic artifact.")
    elif last_p < 0.05 and abs(pct_change) > 30:
        print("\n-> OUTCOME 2: The distance effect SHRINKS but remains significant.")
        print("   Latitude explains some of the pattern, but disturbance distance still")
        print("   carries independent information.")
    elif last_p < 0.05:
        print("\n-> OUTCOME 3: The distance effect REMAINS STRONG even after controlling for")
        print("   latitude and longitude. This is a robust, non-confounded, genuinely")
        print("   unexpected association - worth treating as a real finding in the write-up,")
        print("   with the standard caution that association is not proof of causation.")
    else:
        print("\n-> Result unclear - review the full model outputs above.")

if __name__ == "__main__":
    main()