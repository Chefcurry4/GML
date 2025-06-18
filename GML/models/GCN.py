import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# ----------- Graph Convolution Layer --------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support)
        if self.bias is not None:
            out += self.bias
        return out

# ----------- Standard GCN Model --------------
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        B, N, Fin = x.shape
        h = self.gc1(x, adj)
        h = h.view(B * N, -1)
        h = self.bn1(h)
        h = h.view(B, N, -1)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.gc2(h, adj)
        h = h.view(B * N, -1)
        h = self.bn2(h)
        h = h.view(B, N, -1)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        out = self.gc3(h, adj)
        return out

# ----------- Normalize Adjacency Matrix --------------
def normalize_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = (adj + adj.T) / 2  # Make symmetric
    deg = adj.sum(1)
    deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

# ----------- Train Standard GCN --------------
def train_gcn_from_arrays(
    X_train, Y_train, X_val, Y_val, edge_index,
    hidden=32, dropout=0.3, epochs=20,
    patience=10, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"
):
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)

    B, N, Fin = X_train.shape
    N_out = Y_train.shape[1]
    adj = normalize_adj(edge_index, N).to(device)
    model = GCN(nfeat=Fin, nhid=hidden, nout=1, dropout=dropout).to(device)  # Predict 1 value per node

    print("\n[Standard GCN Architecture]")
    print(f"Input features: {Fin}")
    print(f"Hidden dimension: {hidden}")
    print(f"Dropout rate: {dropout}")
    print(f"Batch size: {batch_size}")
    print(f"Output nodes: {N_out}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        t_start = time.time()
        model.train()
        total_loss = 0
        num_batches = 0

        for i in range(0, B, batch_size):
            x_batch = X_train[i:i+batch_size]  # shape (B, N, F)
            y_batch = Y_train[i:i+batch_size]  # shape (B, N)
            adj_batch = adj.unsqueeze(0).expand(x_batch.shape[0], -1, -1)

            out = model(x_batch, adj_batch).squeeze(-1)  # shape (B, N)
            loss = loss_fn(out, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                x = X_val[i:i+batch_size]
                y = Y_val[i:i+batch_size]
                adj_val = adj.unsqueeze(0).expand(x.shape[0], -1, -1)

                out = model(x, adj_val).squeeze(-1)  # shape (B, N)
                val_loss += loss_fn(out, y).item()

        avg_val_loss = val_loss / (X_val.shape[0] // batch_size)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - t_start
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"Train RMSE: {np.sqrt(avg_train_loss):.4f} | Val RMSE: {np.sqrt(avg_val_loss):.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping triggered after {epoch+1} epochs!")
                break

    print("✅ Standard GCN training complete.")
    return model, train_losses, val_losses
