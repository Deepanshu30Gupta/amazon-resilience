"""
01_inspect_data.py

Purpose: Load the raw VODCA and CHIRPS GeoTIFFs and report their basic
properties (dimensions, resolution, bounds, value ranges, missing data)
so we know exactly what we're working with before doing anything else.

Input:  data/raw/vodca_amazon_cerrado_monthly.tif
        data/raw/chirps_amazon_cerrado_monthly.tif
Output: printed report to console (no files written at this stage)
"""

import rasterio
import numpy as np

FILES = {
    "VODCA": "data/raw/vodca_amazon_cerrado_monthly.tif",
    "CHIRPS": "data/raw/chirps_amazon_cerrado_monthly.tif",
}

def inspect(name, path):
    print(f"\n===== {name} =====")
    with rasterio.open(path) as src:
        print("Bands (months):", src.count)
        print("Height x Width (pixels):", src.height, "x", src.width)
        print("CRS:", src.crs)
        print("Bounds:", src.bounds)
        print("Resolution (degrees):", src.res)
        print("First band name:", src.descriptions[0])
        print("Last band name:", src.descriptions[-1])

        data = src.read()  # read ALL bands, not just band 1
        print("Data type:", data.dtype)
        print("Min / Max / Mean (all bands):",
              np.nanmin(data), np.nanmax(data), np.nanmean(data))
        n_nan = np.isnan(data).sum()
        print(f"Missing values (NaN): {n_nan} / {data.size} "
              f"({100*n_nan/data.size:.3f}%)")

if __name__ == "__main__":
    for name, path in FILES.items():
        inspect(name, path)