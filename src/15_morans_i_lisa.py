"""
15_morans_i_lisa.py

Purpose: Independently measure how spatially clustered the
resilience-loss pattern actually is, using the AR(1) trend (Kendall's
tau, from Stage 5) as the variable of interest. This doesn't depend on
the regression approach used in Stages 6-14 - it's a separate,
well-established spatial statistics method, giving an independent line
of evidence for spatial structure.

Global Moran's I: a single number summarizing whether nearby patches'
resilience trends are more similar than random chance would produce
(positive = clustering, ~0 = random, negative = checkerboard pattern).
Tested for significance via a permutation test (shuffle the values many
times, see how extreme the real Moran's I is compared to random shuffles).

Local Moran's I (LISA): the same idea computed per-patch, classifying
each patch into:
  HH = High value patch surrounded by High-value neighbors (resilience-
       loss hotspot)
  LL = Low value patch surrounded by Low-value neighbors (resilience-
       gain "coldspot")
  HL = High value patch surrounded by Low-value neighbors (spatial
       outlier)
  LH = Low value patch surrounded by High-value neighbors (spatial
       outlier)

Uses the existing first-order adjacency (patch_adjacency.csv) as the
spatial weights matrix, row-standardized.

Input:  data/processed/patch_resilience_trend.csv
        data/processed/patch_locations.csv
        data/processed/patch_adjacency.csv
Output: data/processed/morans_i_results.csv (global result)
        data/processed/lisa_results.csv (per-patch local results)
        figures/lisa_cluster_map.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = "data/processed"
FIG_DIR = "figures"
N_PERMUTATIONS = 999
RANDOM_SEED = 42

def build_weight_matrix(adj_df, patch_ids):
    """Row-standardized binary spatial weights matrix from adjacency edges."""
    n = len(patch_ids)
    idx = {pid: i for i, pid in enumerate(patch_ids)}
    W = np.zeros((n, n))
    for _, row in adj_df.iterrows():
        i, j = idx[row["patch_id"]], idx[row["neighbor_id"]]
        W[i, j] = 1
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero for isolated patches
    W_std = W / row_sums
    return W_std

def global_morans_i(x, W):
    n = len(x)
    x_bar = x.mean()
    z = x - x_bar
    num = np.sum(W * np.outer(z, z))
    den = np.sum(z ** 2)
    S0 = W.sum()
    I = (n / S0) * (num / den)
    return I

def local_morans_i(x, W):
    n = len(x)
    x_bar = x.mean()
    z = x - x_bar
    S2 = np.sum(z ** 2) / n
    local_I = np.zeros(n)
    for i in range(n):
        local_I[i] = (z[i] / S2) * np.sum(W[i] * z)
    return local_I

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    trend = pd.read_csv(os.path.join(OUT_DIR, "patch_resilience_trend.csv")).dropna(subset=["kendall_tau"])
    loc = pd.read_csv(os.path.join(OUT_DIR, "patch_locations.csv"))
    adj = pd.read_csv(os.path.join(OUT_DIR, "patch_adjacency.csv"))

    merged = trend.merge(loc, on="patch_id").sort_values("patch_id").reset_index(drop=True)
    patch_ids = merged["patch_id"].tolist()
    x = merged["kendall_tau"].values

    W = build_weight_matrix(adj, patch_ids)

    # ---- Global Moran's I ----
    I_observed = global_morans_i(x, W)
    print(f"Global Moran's I (observed): {I_observed:.4f}")

    rng = np.random.default_rng(RANDOM_SEED)
    perm_Is = np.zeros(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        x_shuffled = rng.permutation(x)
        perm_Is[p] = global_morans_i(x_shuffled, W)
    p_value = (np.sum(perm_Is >= I_observed) + 1) / (N_PERMUTATIONS + 1) if I_observed > 0 else \
              (np.sum(perm_Is <= I_observed) + 1) / (N_PERMUTATIONS + 1)

    print(f"Permutation test p-value: {p_value:.4f} (based on {N_PERMUTATIONS} shuffles)")
    print(f"Expected Moran's I under randomness: ~{-1/(len(x)-1):.4f}")
    if p_value < 0.05:
        print("-> Resilience trends are SIGNIFICANTLY spatially clustered (not random).")
    else:
        print("-> Cannot reject the null of spatial randomness.")

    pd.DataFrame([{
        "morans_i": I_observed, "p_value": p_value, "n_patches": len(x),
        "n_permutations": N_PERMUTATIONS
    }]).to_csv(os.path.join(OUT_DIR, "morans_i_results.csv"), index=False)

    # ---- Local Moran's I (LISA) ----
    local_I = local_morans_i(x, W)
    x_bar = x.mean()
    lag_x = W @ x  # spatial lag: weighted average of each patch's neighbors

    cluster_type = []
    for i in range(len(x)):
        if x[i] >= x_bar and lag_x[i] >= x_bar:
            cluster_type.append("HH")  # resilience-loss hotspot
        elif x[i] < x_bar and lag_x[i] < x_bar:
            cluster_type.append("LL")  # resilience-gain coldspot
        elif x[i] >= x_bar and lag_x[i] < x_bar:
            cluster_type.append("HL")  # outlier
        else:
            cluster_type.append("LH")  # outlier

    merged["local_morans_i"] = local_I
    merged["cluster_type"] = cluster_type
    merged.to_csv(os.path.join(OUT_DIR, "lisa_results.csv"), index=False)

    print("\n===== LISA cluster counts =====")
    print(merged["cluster_type"].value_counts())

    # ---- Map ----
    color_map = {"HH": "darkred", "LL": "darkblue", "HL": "orange", "LH": "lightblue"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for ctype, color in color_map.items():
        sub = merged[merged["cluster_type"] == ctype]
        ax.scatter(sub["lon"], sub["lat"], c=color, s=80, marker='s', label=ctype)
    ax.set_title(f"LISA Cluster Map (Global Moran's I = {I_observed:.3f}, p = {p_value:.4f})\n"
                 "HH = resilience-loss hotspot, LL = resilience-gain coldspot, HL/LH = outliers")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "lisa_cluster_map.png"), dpi=130, bbox_inches="tight")
    print("\nSaved figures/lisa_cluster_map.png")

if __name__ == "__main__":
    main()