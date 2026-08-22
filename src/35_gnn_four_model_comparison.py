"""
35_gnn_four_model_comparison.py

Purpose: Train and compare FOUR models on the expanded Stage 34 tensors,
for the 1-month-ahead horizon (3- and 6-month horizons follow in a
later stage once this comparison is verified working):

  1. Baseline (non-spatial): own 12-month history only, no graph
     structure at all - a GRU processing each node's own features
     independently.
  2. Fixed geographic GNN (DCRNN-style): graph convolution using your
     actual geographic adjacency matrix, at every timestep.
  3. Adaptive/learned GNN (Graph WaveNet-style): learns its own
     adjacency matrix from data via trainable node embeddings, instead
     of assuming geographic adjacency is correct.
  4. Attention-based GNN (GAT-style): for each patch's ACTUAL geographic
     neighbors, learns how much attention/weight to give each one,
     rather than a simple fixed average - lets us inspect which
     neighbors the model finds most informative.

All four share the same GRU temporal backbone and the same 15-feature
input, so any difference in performance reflects the SPATIAL mechanism,
not other architectural differences - a fair comparison.

Uses a proper held-out validation set (Stage 34's chronological split)
for early stopping, and reports final performance on the untouched
test set.

HONEST SCOPING NOTE: given the real training time each architecture
requires (~5-10 min based on Stage 27), this script covers only the
1-month horizon. 3- and 6-month horizons are a planned follow-up once
this comparison is confirmed working correctly.

Input:  data/processed/gnn_tensors_expanded.pt
Output: data/processed/gnn_four_model_results.csv
        data/processed/learned_adjacency_weights_v2.csv (Model 3)
        data/processed/attention_weights.csv (Model 4)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

OUT_DIR = "data/processed"
HIDDEN_DIM = 24
EPOCHS = 100
LR = 0.005
PATIENCE = 12
HORIZON = 1

torch.manual_seed(42)


class TemporalGRU(nn.Module):
    """Shared temporal backbone: processes each node's own feature
    sequence independently (no graph mixing) - used as-is for the
    baseline, and as a building block inside the spatial models."""
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(n_features, hidden_dim)

    def forward(self, x_seq):
        # x_seq: (batch, seq_len, n_nodes, n_features)
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

    def forward(self, x_seq, A=None):
        h = self.gru(x_seq)
        return self.head(h).squeeze(-1)


class DiffusionConvGRU(nn.Module):
    """Graph convolution + GRU, shared design for both fixed-graph and
    adaptive-graph models (only the adjacency source differs)."""
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
    """GAT-style: attention restricted to REAL geographic neighbors
    only, using a padded neighbor-list representation (not a dense
    NxN attention matrix, which is far too memory-hungry for N=352 -
    an earlier version of this script hit an out-of-memory crash for
    exactly this reason). Each node attends only to its actual (small
    number of) geographic neighbors, with the WEIGHT given to each
    learned per-timestep from node features, rather than a fixed
    average."""
    def __init__(self, n_features, hidden_dim, n_nodes, neighbor_idx, neighbor_mask):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.max_degree = neighbor_idx.shape[1]
        self.register_buffer('neighbor_idx', neighbor_idx)   # (n_nodes, max_degree)
        self.register_buffer('neighbor_mask', neighbor_mask)  # (n_nodes, max_degree), 1=real, 0=padding
        self.attn_proj = nn.Linear(n_features, hidden_dim)
        self.attn_score = nn.Linear(2 * hidden_dim, 1)
        self.input_proj = nn.Linear(n_features * 2, hidden_dim)  # xt (n_feat) + aggregated raw neighbor feats (n_feat)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, adj_mask=None, return_attention=False):
        # adj_mask kept as an unused arg for call-signature compatibility
        # with the other models' fwd_fn pattern
        batch, seq_len, n_nodes, n_feat = x_seq.shape
        h = torch.zeros(batch, n_nodes, self.hidden_dim, device=x_seq.device)
        last_attn = None
        for t in range(seq_len):
            xt = x_seq[:, t, :, :]  # (batch, n_nodes, n_feat)
            proj = self.attn_proj(xt)  # (batch, n_nodes, hidden_dim)

            # Gather each node's (padded) neighbors' projected features:
            # (batch, n_nodes, max_degree, hidden_dim)
            neighbor_feat = proj[:, self.neighbor_idx, :]  # advanced indexing over the node dim
            neighbor_raw = xt[:, self.neighbor_idx, :]      # (batch, n_nodes, max_degree, n_feat)

            proj_self = proj.unsqueeze(2).expand(-1, -1, self.max_degree, -1)
            scores = self.attn_score(torch.cat([proj_self, neighbor_feat], dim=-1)).squeeze(-1)
            # (batch, n_nodes, max_degree)
            mask = self.neighbor_mask.unsqueeze(0)  # (1, n_nodes, max_degree)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)

            aggregated = (attn.unsqueeze(-1) * neighbor_raw).sum(dim=2)  # (batch, n_nodes, n_feat)
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
    """Convert a dense NxN adjacency matrix into a padded (N, max_degree)
    neighbor-index tensor plus a matching validity mask - avoids ever
    materializing an NxN attention tensor."""
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
        # padding slots: point at self (harmless, always masked out anyway)
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
            val_pred = forward_fn(model, X_val)
            val_rmse = rmse(val_pred, y_val)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [{label}] Early stopping at epoch {epoch+1} (best val RMSE={best_val_rmse:.5f})")
                break
        if (epoch + 1) % 25 == 0:
            print(f"  [{label}] Epoch {epoch+1}: train_loss={loss.item():.5f}  val_rmse={val_rmse:.5f}")

    model.load_state_dict(best_state)
    return model, best_val_rmse


def main():
    tensors = torch.load(os.path.join(OUT_DIR, "gnn_tensors_expanded.pt"), weights_only=False)
    X_train, y_train = tensors["X_train"], tensors[f"y_train_{HORIZON}"]
    X_val, y_val = tensors["X_val"], tensors[f"y_val_{HORIZON}"]
    X_test, y_test = tensors["X_test"], tensors[f"y_test_{HORIZON}"]
    A = tensors["adjacency"]
    A_mask = (A > 0).float()  # binary version for attention masking
    n_nodes = tensors["n_nodes"]
    n_features = tensors["n_features"]
    patch_ids = tensors["patch_ids"]

    print(f"Data: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
    print(f"Horizon: {HORIZON} month(s)\n")

    results = []

    print("===== Model 1: Baseline (no spatial info) =====")
    m1 = BaselineModel(n_features, HIDDEN_DIM)
    fwd1 = lambda model, X: model(X)
    m1, _ = train_model(m1, X_train, y_train, X_val, y_val, fwd1, "Baseline")
    with torch.no_grad():
        test_rmse_1 = rmse(fwd1(m1, X_test), y_test)
    print(f"Test RMSE: {test_rmse_1:.5f}\n")
    results.append(("1. Baseline (no spatial)", test_rmse_1))

    print("===== Model 2: Fixed geographic GNN (DCRNN-style) =====")
    m2 = FixedGraphModel(n_features, HIDDEN_DIM)
    fwd2 = lambda model, X: model(X, A)
    m2, _ = train_model(m2, X_train, y_train, X_val, y_val, fwd2, "FixedGraph")
    with torch.no_grad():
        test_rmse_2 = rmse(fwd2(m2, X_test), y_test)
    print(f"Test RMSE: {test_rmse_2:.5f}\n")
    results.append(("2. Fixed geographic GNN", test_rmse_2))

    print("===== Model 3: Adaptive/learned GNN (Graph WaveNet-style) =====")
    m3 = AdaptiveGraphModel(n_features, HIDDEN_DIM, n_nodes)
    fwd3 = lambda model, X: model(X)
    m3, _ = train_model(m3, X_train, y_train, X_val, y_val, fwd3, "Adaptive")
    with torch.no_grad():
        test_rmse_3 = rmse(fwd3(m3, X_test), y_test)
    print(f"Test RMSE: {test_rmse_3:.5f}\n")
    results.append(("3. Adaptive/learned GNN", test_rmse_3))

    print("===== Model 4: Attention-based GNN (GAT-style, real neighbors only) =====")
    neighbor_idx, neighbor_mask = build_padded_neighbor_lists(A)
    print(f"  Max degree: {neighbor_idx.shape[1]} (padded neighbor-list representation, avoids O(N^2) memory)")
    m4 = AttentionGraphModel(n_features, HIDDEN_DIM, n_nodes, neighbor_idx, neighbor_mask)
    fwd4 = lambda model, X: model(X)
    m4, _ = train_model(m4, X_train, y_train, X_val, y_val, fwd4, "Attention")
    with torch.no_grad():
        test_rmse_4 = rmse(fwd4(m4, X_test), y_test)
    print(f"Test RMSE: {test_rmse_4:.5f}\n")
    results.append(("4. Attention-based GNN", test_rmse_4))

    results_df = pd.DataFrame(results, columns=["model", "test_rmse"]).sort_values("test_rmse")
    results_df.to_csv(os.path.join(OUT_DIR, "gnn_four_model_results.csv"), index=False)

    print("===== FINAL COMPARISON (1-month horizon, same held-out test set) =====")
    print(results_df.to_string(index=False))

    baseline_rmse = test_rmse_1
    best_spatial_rmse = min(test_rmse_2, test_rmse_3, test_rmse_4)
    improvement = 100 * (baseline_rmse - best_spatial_rmse) / baseline_rmse
    print(f"\nBest spatial model's improvement over baseline: {improvement:.2f}%")
    if improvement > 5:
        print("-> A GNN provides a MEANINGFUL improvement over the non-spatial baseline.")
    elif improvement > 0:
        print("-> A GNN provides a SMALL improvement - modest but real.")
    else:
        print("-> No spatial mechanism tested (fixed, adaptive, or attention) improves on")
        print("   the simple baseline. Combined with everything else found in this project,")
        print("   this is strong, comprehensive evidence that the real spatial synchrony,")
        print("   however statistically robust, is not exploitable for prediction by any")
        print("   graph-based mechanism tested.")

    # Save learned adjacency (Model 3) and attention weights (Model 4) for inspection
    with torch.no_grad():
        learned_A = m3.learned_adjacency().numpy()
    top_connections = []
    for i, pid in enumerate(patch_ids):
        top_j = np.argsort(-learned_A[i])[:3]
        for j in top_j:
            top_connections.append((pid, patch_ids[j], learned_A[i, j]))
    pd.DataFrame(top_connections, columns=["patch_id", "learned_neighbor_id", "weight"]).to_csv(
        os.path.join(OUT_DIR, "learned_adjacency_weights_v2.csv"), index=False)

    with torch.no_grad():
        _, attn = m4(X_test, return_attention=True)
    attn_mean = attn.mean(dim=0).numpy()  # (n_nodes, max_degree), averaged over the test set
    attn_records = []
    for i, pid in enumerate(patch_ids):
        for k in range(neighbor_idx.shape[1]):
            if neighbor_mask[i, k] > 0:
                neighbor_pid = patch_ids[neighbor_idx[i, k].item()]
                attn_records.append((pid, neighbor_pid, attn_mean[i, k]))
    pd.DataFrame(attn_records, columns=["patch_id", "neighbor_id", "attention_weight"]).to_csv(
        os.path.join(OUT_DIR, "attention_weights.csv"), index=False)

    print("\nSaved learned_adjacency_weights_v2.csv (Model 3) and attention_weights.csv (Model 4)")

if __name__ == "__main__":
    main()