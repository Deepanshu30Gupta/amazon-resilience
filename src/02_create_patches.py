"""
02_create_patches.py

Purpose: Aggregate the VODCA pixel grid into 2x2-pixel patches (252
patches total). Each patch becomes one "node" in our spatial network.
We use VODCA's grid (coarser than CHIRPS) as the reference, since patch
size can't be finer than the coarsest input dataset.

Why 2x2: 1x1 (1,036 patches) is very fine-grained with many
near-redundant neighbors; 3x3 (108 patches) loses spatial detail.
2x2 (252 patches) is the working balance chosen for this project.

Input:  data/raw/vodca_amazon_cerrado_monthly.tif
Output: data/processed/patch_locations.csv (patch_id, row, col, lon, lat)
        data/processed/patch_vod.npy (raw patch-level VOD array, months x rows x cols)
"""

import rasterio
import numpy as np
import pandas as pd
import os

VODCA_PATH = "data/raw/vodca_amazon_cerrado_monthly.tif"
OUT_DIR = "data/processed"
PATCH_SIZE = 4  # n x n pixel blocks (bumped from 2 to 4 for the expanded v2 region,
                # to keep patch count manageable - was 252 patches at size 2 on the
                # smaller v1 region; size 4 on this larger region gives a similar count)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with rasterio.open(VODCA_PATH) as src:
        vod_data = src.read()          # (months, height, width)
        bounds = src.bounds

    n_months, h, w = vod_data.shape
    n = PATCH_SIZE

    # Trim to a multiple of the patch size, then average within each block
    h_trim = (h // n) * n
    w_trim = (w // n) * n
    vod_trim = vod_data[:, :h_trim, :w_trim]
    reshaped = vod_trim.reshape(n_months, h_trim // n, n, w_trim // n, n)
    patch_vod = np.nanmean(reshaped, axis=(2, 4))  # (months, patch_rows, patch_cols)

    n_patch_rows, n_patch_cols = patch_vod.shape[1], patch_vod.shape[2]
    print(f"Patch grid: {n_patch_rows} x {n_patch_cols} = "
          f"{n_patch_rows * n_patch_cols} patches")

    # Compute each patch's center lat/lon
    lon_step = (bounds.right - bounds.left) / w
    lat_step = (bounds.top - bounds.bottom) / h

    rows, cols, lons, lats, pids = [], [], [], [], []
    pid = 0
    for pr in range(n_patch_rows):
        for pc in range(n_patch_cols):
            row_start, row_end = pr * n, pr * n + n
            col_start, col_end = pc * n, pc * n + n
            lon_center = bounds.left + (col_start + col_end) / 2 * lon_step
            lat_center = bounds.top - (row_start + row_end) / 2 * lat_step
            pids.append(pid); rows.append(pr); cols.append(pc)
            lons.append(lon_center); lats.append(lat_center)
            pid += 1

    patch_meta = pd.DataFrame({
        "patch_id": pids, "row": rows, "col": cols, "lon": lons, "lat": lats
    })
    patch_meta.to_csv(os.path.join(OUT_DIR, "patch_locations.csv"), index=False)
    np.save(os.path.join(OUT_DIR, "patch_vod.npy"), patch_vod)

    # Check how many patches have any missing months after aggregation -
    # relevant now since the expanded region has more raw NaN pixels than before
    n_missing_patches = np.isnan(patch_vod).any(axis=0).sum()
    print(f"\nPatches with at least one missing month: {n_missing_patches} / "
          f"{n_patch_rows * n_patch_cols}")
    if n_missing_patches > 0:
        print("(These will need to be handled - likely dropped - in later stages")
        print("if the missing months are frequent for any given patch.)")

    print("Saved patch_locations.csv and patch_vod.npy to", OUT_DIR)
    print(patch_meta.head())

if __name__ == "__main__":
    main()