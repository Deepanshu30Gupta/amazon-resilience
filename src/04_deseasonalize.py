"""
04_deseasonalize.py

Purpose: Remove the seasonal (wet/dry season) cycle from both VOD and
precipitation, for every patch. This matters because raw VOD has a
strong recurring seasonal pattern - without removing it first, any
autocorrelation we compute later would mostly reflect "this month looks
like last year's same month" rather than genuine resilience-loss signal.

Method: for each patch, compute the average value for each calendar
month (Jan average, Feb average, ...) across all years, then subtract
that month's average from each observation. What's left is the anomaly
- how far above/below the normal seasonal pattern that month was.

Input:  data/processed/patch_timeseries.csv
Output: data/processed/patch_timeseries_anomaly.csv
        (adds vod_anomaly and precip_anomaly columns)
"""

import pandas as pd
import os

OUT_DIR = "data/processed"

def main():
    ts = pd.read_csv(os.path.join(OUT_DIR, "patch_timeseries.csv"), parse_dates=["date"])
    ts["month"] = ts["date"].dt.month

    # Seasonal climatology per patch per calendar month, then subtract
    seasonal_vod = ts.groupby(["patch_id", "month"])["vod"].transform("mean")
    ts["vod_anomaly"] = ts["vod"] - seasonal_vod

    seasonal_precip = ts.groupby(["patch_id", "month"])["precip_mm"].transform("mean")
    ts["precip_anomaly"] = ts["precip_mm"] - seasonal_precip

    # Sanity check: anomaly means should be ~0 (that's what "removing the
    # average seasonal pattern" means, by construction)
    print("VOD anomaly mean (should be ~0):", ts["vod_anomaly"].mean())
    print("VOD anomaly std:", ts["vod_anomaly"].std())
    print("Precip anomaly mean (should be ~0):", ts["precip_anomaly"].mean())
    print("Precip anomaly std:", ts["precip_anomaly"].std())

    ts.to_csv(os.path.join(OUT_DIR, "patch_timeseries_anomaly.csv"), index=False)
    print("\nSaved patch_timeseries_anomaly.csv - shape", ts.shape)
    print(ts[["patch_id", "date", "vod", "vod_anomaly", "precip_mm", "precip_anomaly"]].head())

if __name__ == "__main__":
    main()