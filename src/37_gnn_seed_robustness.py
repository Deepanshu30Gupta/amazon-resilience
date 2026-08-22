"""
37_gnn_seed_robustness.py

Purpose: Test whether the Stage 35/36 result (baseline beats all
spatial mechanisms) is robust to random initialization, or whether it
could be an artifact of one particular (unlucky, for the spatial
models) training run. This directly addresses the strongest remaining
criticism of the GNN comparison: "your dataset is small, maybe the
result depends on the seed."

Trains all 4 models (baseline, fixed geographic GNN, adaptive/learned
GNN, attention-based GNN) across 5 different random seeds, at the
1-month horizon (the horizon with the most training data available and
the primary comparison point), and reports mean, standard deviation,
best, and worst RMSE for each model across the 5 runs.

If the baseline consistently outperforms every spatial mechanism
across ALL 5 seeds (not just on average), that's much stronger evidence
than a single run - it means no reasonable random initialization
rescues the spatial models' performance.

Input:  data/processed/gnn_tensors_expanded.pt
Output: data/processed/gnn_seed_robustness_results.csv
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
HORIZON = 1
SEEDS = [42, 123, 456, 789, 2024]


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


def train_model(model, X_fit, y_fit, X_val, y_val, forward_fn):
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
    return model


def main():
    tensors = torch.load(os.path.join(OUT_DIR, "gnn_tensors_expanded.pt"), weights_only=False)
    A = tensors["adjacency"]
    n_nodes = tensors["n_nodes"]
    n_features = tensors["n_features"]
    neighbor_idx, neighbor_mask = build_padded_neighbor_lists(A)

    X_train, y_train = tensors["X_train"], tensors[f"y_train_{HORIZON}"]
    X_val, y_val = tensors["X_val"], tensors[f"y_val_{HORIZON}"]
    X_test, y_test = tensors["X_test"], tensors[f"y_test_{HORIZON}"]

    print(f"Horizon: {HORIZON} month(s) | Seeds: {SEEDS}")
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}\n")

    t_start = time.time()
    all_results = []

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        torch.manual_seed(seed)

        m1 = BaselineModel(n_features, HIDDEN_DIM)
        fwd1 = lambda model, X: model(X)
        m1 = train_model(m1, X_train, y_train, X_val, y_val, fwd1)
        with torch.no_grad():
            r1 = rmse(fwd1(m1, X_test), y_test)

        torch.manual_seed(seed)
        m2 = FixedGraphModel(n_features, HIDDEN_DIM)
        fwd2 = lambda model, X: model(X, A)
        m2 = train_model(m2, X_train, y_train, X_val, y_val, fwd2)
        with torch.no_grad():
            r2 = rmse(fwd2(m2, X_test), y_test)

        torch.manual_seed(seed)
        m3 = AdaptiveGraphModel(n_features, HIDDEN_DIM, n_nodes)
        fwd3 = lambda model, X: model(X)
        m3 = train_model(m3, X_train, y_train, X_val, y_val, fwd3)
        with torch.no_grad():
            r3 = rmse(fwd3(m3, X_test), y_test)

        torch.manual_seed(seed)
        m4 = AttentionGraphModel(n_features, HIDDEN_DIM, n_nodes, neighbor_idx, neighbor_mask)
        fwd4 = lambda model, X: model(X)
        m4 = train_model(m4, X_train, y_train, X_val, y_val, fwd4)
        with torch.no_grad():
            r4 = rmse(fwd4(m4, X_test), y_test)

        print(f"  Baseline={r1:.5f}  Fixed={r2:.5f}  Adaptive={r3:.5f}  Attention={r4:.5f}")
        all_results.append((seed, r1, r2, r3, r4))

    results_df = pd.DataFrame(all_results, columns=["seed", "baseline", "fixed_graph", "adaptive", "attention"])
    results_df.to_csv(os.path.join(OUT_DIR, "gnn_seed_robustness_results.csv"), index=False)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} minutes\n")
    print("===== SUMMARY ACROSS 5 SEEDS =====")
    summary = []
    for col in ["baseline", "fixed_graph", "adaptive", "attention"]:
        vals = results_df[col].values
        print(f"{col:12s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"best={vals.min():.4f}  worst={vals.max():.4f}")
        summary.append((col, vals.mean(), vals.std(), vals.min(), vals.max()))

    baseline_mean = results_df["baseline"].mean()
    n_baseline_wins = sum(
        results_df.loc[i, "baseline"] < min(results_df.loc[i, "fixed_graph"],
                                              results_df.loc[i, "adaptive"],
                                              results_df.loc[i, "attention"])
        for i in range(len(results_df))
    )
    print(f"\nBaseline beat ALL spatial models in {n_baseline_wins} / {len(SEEDS)} seeds")
    if n_baseline_wins == len(SEEDS):
        print("-> The baseline wins in EVERY seed tested - this is not a fluke of one")
        print("   unlucky initialization. The negative result is robust to random")
        print("   initialization, on top of already being robust across model")
        print("   architecture and forecast horizon.")
    else:
        print("-> The result varies by seed - worth examining which seeds favor spatial")
        print("   models and why.")

if __name__ == "__main__":
    main()