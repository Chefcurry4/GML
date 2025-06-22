import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from scipy.sparse import coo_matrix
import networkx as nx
import community as community_louvain

from utils.utils import log_train_results

# ----------- Graph Convolution Layer --------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        B, N, _ = x.shape
        support = torch.matmul(x, self.weight)
        out = torch.zeros_like(support)
        row, col = edge_index

        for b in range(B):
            out[b].index_add_(0, col, support[b][row])

        if self.bias is not None:
            out += self.bias
        return out

# ----------- Cluster-GCN Model --------------
class ClusterGCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, 1)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.gc1(x, edge_index)
        B, N, f = h.shape
        h = self.bn1(h.view(B * N, f)).view(B, N, f)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.gc2(h, edge_index)
        B, N, f = h.shape
        h = self.bn2(h.view(B * N, f)).view(B, N, f)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        out = self.gc3(h, edge_index)
        return out.squeeze(-1)

# ----------- Louvain Graph Clustering --------------
def louvain_cluster(edge_index):
    import networkx as nx
    import community as community_louvain

    G = nx.Graph()
    edges = edge_index.T.cpu().numpy()
    G.add_edges_from(edges)

    partition = community_louvain.best_partition(G)

    clusters = {}
    for node, cid in partition.items():
        clusters.setdefault(cid, []).append(node)

    sizes = [len(v) for v in clusters.values()]
    print(f"Louvain clusters: {len(clusters)} | min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
    return clusters

# ----------- Build per-cluster adjacency matrix --------------
def build_cluster_adj(edge_index, cluster_nodes, device="cpu"):
    node_map = {n: i for i, n in enumerate(cluster_nodes)}
    edges = [(node_map[u], node_map[v]) for u, v in edge_index.T.tolist() if u in node_map and v in node_map]
    if not edges:
        adj = torch.eye(len(cluster_nodes))
        # Move to right device and return
        return adj.to(device)

    rows, cols = zip(*edges)
    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(len(cluster_nodes), len(cluster_nodes)))

    A = A + A.T.multiply(A.T > A)
    D = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = np.power(D + 1e-6, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat = coo_matrix((D_inv_sqrt, (np.arange(len(D)), np.arange(len(D)))), shape=A.shape)
    A_norm = D_mat @ A @ D_mat

    adj = torch.tensor(A_norm.toarray(), dtype=torch.float32)
    # Move to right device and return
    return adj.to(device)

# ----------- Training Cluster-GCN --------------
def train_clustergcn_from_arrays(
    X_train, Y_train, X_val, Y_val, edge_index,
    hidden=64, dropout=0.1, epochs=20,
    patience=10, batch_size=32, lr=0.01, device="cuda" if torch.cuda.is_available() else "cpu", args=None
):
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    # Original Y_train: [B, num_turbines]
    # Expand to: [B, num_turbines × T]
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    # Expand to full product graph size
    T = X_train.shape[1] // 134  # number of time steps (e.g., 12)
    Y_train = Y_train.repeat(1, T)  # shape: [batch, 134*T] == [5635, 1608]
    print(f"Y_train shape after expansion: {Y_train.shape}")
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    print(f"X_val shape: {X_val.shape}")
    Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
    print(f"Y_val shape: {Y_val.shape}")
    # Expand to full product graph size

    # Ensure that edge_indes is on the right device
    edge_index = edge_index.to(device)


    print("Clustering nodes using Louvain algorithm...")
    clusters = louvain_cluster(edge_index)
    print(f"Clusters found: {len(clusters)}")

    model = ClusterGCN(nfeat=X_train.shape[2], nhid=hidden, dropout=dropout).to(device)

    print(f"\n[ClusterGCN Architecture]")
    print(f"Hidden dimension: {hidden}")
    print(f"Dropout rate: {dropout}")
    print(f"Batch size: {batch_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    training_time_seconds = 0
    actual_epochs_trained = 0

    for epoch in range(epochs):
        t_start = time.time()
        model.train()
        total_loss = 0
        num_batches = 0

        for cluster_id, nodes in clusters.items():
            if len(nodes) < 2:
                continue

            adj = build_cluster_adj(edge_index, nodes, device=device)
            edge_idx = adj.nonzero().T

            x_cluster = X_train[:, nodes, :]
            # print(f"x_cluster shape:", x_cluster.shape)
            y_cluster = Y_train[:, nodes]
            # print(f"y_cluster shape:", y_cluster.shape)

            for i in range(0, x_cluster.shape[0], batch_size):
                x_batch = x_cluster[i:i+batch_size]
                y_batch = y_cluster[i:i+batch_size]

                out = model(x_batch, edge_idx)
                loss = loss_fn(out, y_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                x = X_val[i:i+batch_size]
                y = Y_val[i:i+batch_size]

                out = model(x, edge_index)  # Use full graph input (1608 nodes)
                loss = loss_fn(out, y.repeat(1, T))  # Match y to full length if needed
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - t_start
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"Train RMSE: {np.sqrt(avg_train_loss):.4f} | Val RMSE: {np.sqrt(avg_val_loss):.4f}")

        training_time_seconds += epoch_time
        actual_epochs_trained += 1

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break

    print("Cluster-GCN training complete.")

    # Log training results
    log_train_results(
        args=args,
        num_epochs=actual_epochs_trained,
        total_time=training_time_seconds,
        best_val_loss=best_val_loss,
    )

    return model, train_losses, val_losses

def forecast(model, X, edge_index, num_turbines):
    """
    Forecast the power output for a single sample using the trained ClusterGCN model.
    Args:
        model: Trained ClusterGCN model
        X: Input features for the sample, shape (num_nodes, num_features)
        edge_index: Edge indices for the product graph
        num_turbines: Number of turbines (for extracting last time step)
    Returns:
        y_pred: Predicted power output for each turbine at the last time step, shape (num_turbines,)
    """
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        x_sample = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)  # (1, num_nodes, num_features)
        edge_index = edge_index.to(device)

        # Forward pass
        out = model(x_sample, edge_index)  # (1, num_nodes)
        out = out.squeeze(0)  # (num_nodes,)

        # Extract predictions for the last time step
        from config import INPUT_SEQUENCE_LENGTH
        last_time_start = (INPUT_SEQUENCE_LENGTH - 1) * num_turbines
        last_time_end = INPUT_SEQUENCE_LENGTH * num_turbines
        y_pred = out[last_time_start:last_time_end].cpu().numpy() # (num_turbines,)
        return y_pred