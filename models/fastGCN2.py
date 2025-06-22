# fastGCN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
import numpy as np
import time

# ----------- FastGCN Sampler --------------
def fastgcn_sampler(edge_index, num_nodes, sample_size):
    row, _ = edge_index
    deg = degree(row, num_nodes=num_nodes)
    prob = deg / deg.sum()
    sampled_nodes = torch.multinomial(prob, sample_size, replacement=True)

    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[sampled_nodes] = True
    sub_edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    sub_edge_index = edge_index[:, sub_edge_mask]

    return sampled_nodes, sub_edge_index

# ----------- Graph Convolution Layer --------------
class FastGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, node_idx=None, edge_weight=None):
        # x: [num_sampled_nodes, in_features]
        # edge_index: [2, num_edges] (indices in original graph)
        # node_idx: tensor of original node indices for sampled nodes
        support = torch.matmul(x, self.weight)
        out = torch.zeros_like(support)
        row, col = edge_index
        if node_idx is not None:
            # Map original node indices to new indices in sampled set
            idx_map = {orig.item(): i for i, orig in enumerate(node_idx)}
            mask = [(u.item() in idx_map and v.item() in idx_map) for u, v in zip(row, col)]
            row = row[mask]
            col = col[mask]
            # Remap
            row = torch.tensor([idx_map[u.item()] for u in row], device=x.device)
            col = torch.tensor([idx_map[v.item()] for v in col], device=x.device)
        edge_weight = torch.ones(row.size(0), device=x.device) if edge_weight is None else edge_weight
        edge_weight = edge_weight.view(-1, 1)
        out.scatter_add_(0, col.view(-1, 1).expand(-1, support.size(1)), support[row] * edge_weight)
        if self.bias is not None:
            out += self.bias
        return out

# ----------- FastGCN Model --------------
class TrueFastGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_nodes, dropout=0.2, sample_size=128):
        super().__init__()
        self.sample_size = sample_size
        self.num_nodes = num_nodes
        self.gc1 = FastGraphConvolution(in_features, hidden_dim)
        self.gc2 = FastGraphConvolution(hidden_dim, hidden_dim)
        self.gc3 = FastGraphConvolution(hidden_dim, 1)
        self.dropout = dropout
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        sampled_nodes, sampled_edges = fastgcn_sampler(edge_index, self.num_nodes, self.sample_size)
        x_sampled = x[sampled_nodes]
        # Pass node_idx to conv layers for remapping
        h = self.gc1(x_sampled, sampled_edges, node_idx=sampled_nodes)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, sampled_edges, node_idx=sampled_nodes)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        out = self.gc3(h, sampled_edges, node_idx=sampled_nodes)
        return out, sampled_nodes

# ----------- Training Function --------------
def train_fastgcn_from_arrays(
    X_train, Y_train, X_val, Y_val, edge_index,
    hidden=64, epochs=100, lr=0.001, sample_size=128,
    patience=10, device='cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    num_nodes = X_train.shape[1]
    in_features = X_train.shape[2]

    # Assume edge_index is already local (0 to num_nodes-1)
    model = TrueFastGCN(in_features, hidden, num_nodes, sample_size=sample_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        t_start = time.time()
        model.train()
        total_loss = 0.0
        
        # Training loop
        for i in range(len(X_train)):
            x = torch.tensor(X_train[i], dtype=torch.float32, device=device)
            y = torch.tensor(Y_train[i], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            out, sampled_nodes = model(x, edge_index.to(device))
            y_sampled = y[sampled_nodes]  # match target with sampled nodes
            loss = F.mse_loss(out.squeeze(), y_sampled)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(X_train)
        train_losses.append(avg_train_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(len(X_val)):
                x = torch.tensor(X_val[i], dtype=torch.float32, device=device)
                y = torch.tensor(Y_val[i], dtype=torch.float32, device=device)
                out, sampled_nodes = model(x, edge_index.to(device))
                y_sampled = y[sampled_nodes]
                val_loss += F.mse_loss(out.squeeze(), y_sampled).item()
                
        avg_val_loss = val_loss / len(X_val)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        epoch_time = time.time() - t_start
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"Train RMSE: {np.sqrt(avg_train_loss):.4f} | Val RMSE: {np.sqrt(avg_val_loss):.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    return model
