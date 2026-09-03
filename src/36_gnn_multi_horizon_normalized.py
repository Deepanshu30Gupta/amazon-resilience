"""
36_gnn_multi_horizon_normalized.py

NORMALIZED RERUN of 36_gnn_multi_horizon.py: identical model code and
procedure, but loads the feature-standardized tensors
(gnn_tensors_expanded_normalized.pt, from 34_gnn_data_prep_normalized.py)
instead of the pre-normalization ones, and writes
gnn_multi_horizon_results_normalized.csv. This is the corrected-pipeline
counterpart to 35_gnn_four_model_comparison_normalized.py.

Purpose: Extend the Stage 35 four-model comparison (baseline, fixed
geographic GNN, adaptive/learned GNN, attention-based GNN) across all
three forecast horizons (1, 3, 6 months), to test whether spatial
information becomes useful at longer timescales even though it did not
help at 1 month.

Same architectures, same train/val/test split, same training procedure
as Stage 35 - only the target horizon changes. Each of the 12
combinations (4 models x 3 horizons) is trained and evaluated
independently, all against the same held-out test set for that horizon.

HONEST NOTE: as in Stage 35, the training/val/test sequence counts are
small (roughly 130/24/17-19 depending on horizon, since longer horizons
need slightly more trailing months held back), so results should be
read as indicative rather than definitive on their own.

Input:  data/processed/gnn_tensors_expanded_normalized.pt
Output: data/processed/gnn_multi_horizon_results_normalized.csv
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time

OUT_DIR = "data/processed"
HIDDEN_DIM = 24
EPOCHS = 100
LR = 0.005
PATIENCE = 12
HORIZONS = [1, 3, 6]

torch.manual_seed(42)


class TemporalGRU(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(n_features, hidden_dim)

    def forward(self, x_seq):
        batch, seq_len, n_nodes, n_feat = x_seq.shape
        h = torch.zeros(batch * n_nodes, self.hidden_dim, device=x_seq.device)
        for t in range(seq_len):
            xt = x_seq[:, t, :, :].reshape(batch * n_nodes, n_feat)
            h = self.gru_cell(xt, h)
        return h.reshape(batch, n_nodes, self.hidden_dim)


class BaselineModel(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.gru = TemporalGRU(n_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq):
        h = self.gru(x_seq)
        return self.head(h).squeeze(-1)


class DiffusionConvGRU(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(n_features * 2, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x_seq, A):
        batch, seq_len, n_nodes, n_feat = x_seq.shape
        h = torch.zeros(batch * n_nodes, self.hidden_dim, device=x_seq.device)
        for t in range(seq_len):
            xt = x_seq[:, t, :, :]
            diffused = torch.einsum('ij,bjf->bif', A, xt)
            combined = torch.cat([xt, diffused], dim=-1).reshape(batch * n_nodes, -1)
            proj = self.input_proj(combined)
            h = self.gru_cell(proj, h)
        return h.reshape(batch, n_nodes, self.hidden_dim)


class FixedGraphModel(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.dcgru = DiffusionConvGRU(n_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, A):
        h = self.dcgru(x_seq, A)
        return self.head(h).squeeze(-1)


class AdaptiveGraphModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_nodes, embed_dim=10):
        super().__init__()
        self.node_emb1 = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(n_nodes, embed_dim) * 0.1)
        self.dcgru = DiffusionConvGRU(n_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def learned_adjacency(self):
        scores = torch.relu(self.node_emb1 @ self.node_emb2.T)
        return torch.softmax(scores, dim=-1)

    def forward(self, x_seq):
        A_learned = self.learned_adjacency()
        h = self.dcgru(x_seq, A_learned)
        return self.head(h).squeeze(-1)


class AttentionGraphModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_nodes, neighbor_idx, neighbor_mask):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.max_degree = neighbor_idx.shape[1]
        self.register_buffer('neighbor_idx', neighbor_idx)
        self.register_buffer('neighbor_mask', neighbor_mask)
        self.attn_proj = nn.Linear(n_features, hidden_dim)
        self.attn_score = nn.Linear(2 * hidden_dim, 1)
        self.input_proj = nn.Linear(n_features * 2, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, return_attention=False):
        batch, seq_len, n_nodes, n_feat = x_seq.shape
        h = torch.zeros(batch, n_nodes, self.hidden_dim, device=x_seq.device)
        last_attn = None
        for t in range(seq_len):
            xt = x_seq[:, t, :, :]
            proj = self.attn_proj(xt)
            neighbor_feat = proj[:, self.neighbor_idx, :]
            neighbor_raw = xt[:, self.neighbor_idx, :]
            proj_self = proj.unsqueeze(2).expand(-1, -1, self.max_degree, -1)
            scores = self.attn_score(torch.cat([proj_self, neighbor_feat], dim=-1)).squeeze(-1)
            mask = self.neighbor_mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            aggregated = (attn.unsqueeze(-1) * neighbor_raw).sum(dim=2)
            combined = torch.cat([xt, aggregated], dim=-1).reshape(batch * n_nodes, -1)
            proj_in = self.input_proj(combined)
            h_flat = self.gru_cell(proj_in, h.reshape(batch * n_nodes, self.hidden_dim))
            h = h_flat.reshape(batch, n_nodes, self.hidden_dim)
            last_attn = attn
        out = self.head(h).squeeze(-1)
        if return_attention:
            return out, last_attn
        return out


def build_padded_neighbor_lists(A):
    n_nodes = A.shape[0]
    mask_bin = (A > 0)
    degrees = mask_bin.sum(dim=1)
    max_degree = int(degrees.max().item())
    neighbor_idx = torch.zeros((n_nodes, max_degree), dtype=torch.long)
    neighbor_mask = torch.zeros((n_nodes, max_degree), dtype=torch.float32)
    for i in range(n_nodes):
        neighbors = torch.where(mask_bin[i])[0]
        k = len(neighbors)
        neighbor_idx[i, :k] = neighbors
        neighbor_mask[i, :k] = 1.0
        if k < max_degree:
            neighbor_idx[i, k:] = i
    return neighbor_idx, neighbor_mask


def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def train_model(model, X_fit, y_fit, X_val, y_val, forward_fn, label=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = forward_fn(model, X_fit)
        loss = loss_fn(pred, y_fit)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_rmse = rmse(forward_fn(model, X_val), y_val)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_val_rmse


def main():
    tensors = torch.load(os.path.join(OUT_DIR, "gnn_tensors_expanded_normalized.pt"), weights_only=False)
    A = tensors["adjacency"]
    n_nodes = tensors["n_nodes"]
    n_features = tensors["n_features"]
    neighbor_idx, neighbor_mask = build_padded_neighbor_lists(A)

    all_results = []
    t_start = time.time()

    for horizon in HORIZONS:
        print(f"\n{'='*60}\nHORIZON: {horizon} month(s)\n{'='*60}")
        X_train, y_train = tensors["X_train"], tensors[f"y_train_{horizon}"]
        X_val, y_val = tensors["X_val"], tensors[f"y_val_{horizon}"]
        X_test, y_test = tensors["X_test"], tensors[f"y_test_{horizon}"]
        print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

        m1 = BaselineModel(n_features, HIDDEN_DIM)
        fwd1 = lambda model, X: model(X)
        m1, _ = train_model(m1, X_train, y_train, X_val, y_val, fwd1, "Baseline")
        with torch.no_grad():
            r1 = rmse(fwd1(m1, X_test), y_test)
        print(f"  Baseline (no spatial):    RMSE={r1:.5f}")
        all_results.append((horizon, "1. Baseline", r1))

        m2 = FixedGraphModel(n_features, HIDDEN_DIM)
        fwd2 = lambda model, X: model(X, A)
        m2, _ = train_model(m2, X_train, y_train, X_val, y_val, fwd2, "FixedGraph")
        with torch.no_grad():
            r2 = rmse(fwd2(m2, X_test), y_test)
        print(f"  Fixed geographic GNN:     RMSE={r2:.5f}")
        all_results.append((horizon, "2. Fixed geographic GNN", r2))

        m3 = AdaptiveGraphModel(n_features, HIDDEN_DIM, n_nodes)
        fwd3 = lambda model, X: model(X)
        m3, _ = train_model(m3, X_train, y_train, X_val, y_val, fwd3, "Adaptive")
        with torch.no_grad():
            r3 = rmse(fwd3(m3, X_test), y_test)
        print(f"  Adaptive/learned GNN:     RMSE={r3:.5f}")
        all_results.append((horizon, "3. Adaptive/learned GNN", r3))

        m4 = AttentionGraphModel(n_features, HIDDEN_DIM, n_nodes, neighbor_idx, neighbor_mask)
        fwd4 = lambda model, X: model(X)
        m4, _ = train_model(m4, X_train, y_train, X_val, y_val, fwd4, "Attention")
        with torch.no_grad():
            r4 = rmse(fwd4(m4, X_test), y_test)
        print(f"  Attention-based GNN:      RMSE={r4:.5f}")
        all_results.append((horizon, "4. Attention-based GNN", r4))

        best_spatial = min(r2, r3, r4)
        improvement = 100 * (r1 - best_spatial) / r1
        print(f"  -> Best spatial vs baseline: {improvement:+.2f}%")

    results_df = pd.DataFrame(all_results, columns=["horizon_months", "model", "test_rmse"])
    results_df.to_csv(os.path.join(OUT_DIR, "gnn_multi_horizon_results_normalized.csv"), index=False)

    print(f"\n{'='*60}\nFULL RESULTS TABLE\n{'='*60}")
    pivot = results_df.pivot(index="model", columns="horizon_months", values="test_rmse")
    print(pivot.to_string())

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} minutes")

    print("\n===== SUMMARY ACROSS ALL HORIZONS =====")
    any_improvement = False
    for h in HORIZONS:
        sub = results_df[results_df["horizon_months"] == h]
        baseline_rmse = sub[sub["model"] == "1. Baseline"]["test_rmse"].values[0]
        best_spatial = sub[sub["model"] != "1. Baseline"]["test_rmse"].min()
        imp = 100 * (baseline_rmse - best_spatial) / baseline_rmse
        print(f"Horizon {h} month(s): best spatial model improvement over baseline = {imp:+.2f}%")
        if imp > 5:
            any_improvement = True

    if any_improvement:
        print("\n-> At least one horizon shows a meaningful spatial improvement - investigate")
        print("   which horizon and why before drawing a final conclusion.")
    else:
        print("\n-> NO horizon (1, 3, or 6 months) shows a meaningful improvement from any")
        print("   spatial mechanism. Combined with the large-sample linear regression tests")
        print("   (Stages 22-24) finding the same pattern, this is now strong, well-")
        print("   triangulated evidence: the statistically robust spatial synchrony in this")
        print("   data does not translate into exploitable predictive information, at any")
        print("   timescale tested, by any spatial mechanism tested (geographic, learned,")
        print("   or attention-based).")

if __name__ == "__main__":
    main()