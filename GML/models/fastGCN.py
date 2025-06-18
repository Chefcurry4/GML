# fastGCN.py
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler  
from torch_sparse import coalesce 
import os
import time
from torch_geometric.utils import degree
from config import INPUT_SEQUENCE_LENGTH

# ----------- Normalize Sparse Adjacency Matrix --------------
def normalize_sparse_adj(edge_index, edge_weight=None, num_nodes=None):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, edge_weight)
    deg.scatter_add_(0, col, edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_weight = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, norm_weight

# ----------- Graph Convolution Layer (SPARSE) --------------
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

    def forward(self, x, edge_index, edge_weight=None):
        # Compute XW first (more efficient for sparse ops)
        support = torch.matmul(x, self.weight)  # [N, out_features]
        
        # Use efficient scatter operations
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Aggregate using scatter_add_ (much more memory efficient)
        out = torch.zeros_like(support)
        row, col = edge_index
        edge_weight = edge_weight.view(-1, 1)
        out.scatter_add_(0, col.view(-1, 1).expand(-1, self.out_features), 
                        support[row] * edge_weight)
        
        if self.bias is not None:
            out = out + self.bias
        return out

# ----------- FastGCN Model --------------
class FastGCN(nn.Module):  
    def __init__(self, nfeat, nhid=512, dropout=0.35, edge_index=None, n_nodes=None, batch_size=128): 
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, bias=True)
        self.gc2 = GraphConvolution(nhid, nhid, bias=True)
        self.gc3 = GraphConvolution(nhid, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.dropout = dropout
        self.batch_size = batch_size

        if n_nodes is not None:
            self.num_turbines = 32  # Will be overridden later
        if edge_index is not None and n_nodes is not None:
            self.edge_index = edge_index
            self.n_nodes = n_nodes
            self.norm_edge_index = edge_index
            self.norm_edge_weight = None

        # L2 normalization weight
        self.l2_lambda = 0.01

        print(f"\n[FastGCN Architecture]")
        print(f"Input features: {nfeat}")
        print(f"Hidden dimension: {nhid}")
        print(f"Dropout rate: {dropout}")
        print(f"Batch size: {batch_size}")
        print(f"Parameters: {sum(p.numel() for p in self.parameters())}")

    def l2_regularization(self):
        """Compute L2 regularization loss"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_lambda * l2_loss

    def forward(self, x, edge_info=None):
        batch_size = x.size(0)
        outputs = []
        edge_index, edge_weight = (self.norm_edge_index, self.norm_edge_weight) if edge_info is None else edge_info
        for i in range(batch_size):
            x_i = x[i]
            h = self.gc1(x_i, edge_index, edge_weight)
            h = self.bn1(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.gc2(h, edge_index, edge_weight)
            h = self.bn2(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            out = self.gc3(h, edge_index, edge_weight)
        
            
            outputs.append(out[:self.num_turbines, 0])
        return torch.stack(outputs)

# ----------- FastGCN Training Function --------------
def train_fastgcn_from_arrays(
    X_train, Y_train, X_val, Y_val, edge_index,
    hidden=256, samples=512, dropout=0.2, epochs=40, lr=0.0001
):
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    edge_index = edge_index.to(device)
    num_nodes = X_train.shape[1]
    num_features = X_train.shape[2]
    num_turbines = Y_train.shape[1]
    print(f"\nDataset stats:")
    print(f"- Number of nodes per graph: {num_nodes}")
    print(f"- Number of features per node: {num_features}")
    print(f"- Number of turbines (output dim): {num_turbines}")
    print(f"- Number of training samples: {len(X_train)}")
    print(f"- Number of validation samples: {len(X_val)}")

    model = FastGCN(nfeat=num_features, nhid=hidden, dropout=dropout, edge_index=edge_index, n_nodes=num_nodes).to(device)
    model.num_turbines = num_turbines
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        t_start = time.time()
        model.train()
        total_loss = 0
        num_batches = 0
        for i in range(0, len(X_train), model.batch_size):
            batch_end = min(i + model.batch_size, len(X_train))
            x = torch.tensor(X_train[i:batch_end], dtype=torch.float32, device=device)
            y = torch.tensor(Y_train[i:batch_end], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            out = model(x, (edge_index, None))
            loss = F.mse_loss(out, y)
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
            for i in range(0, len(X_val), model.batch_size):
                batch_end = min(i + model.batch_size, len(X_val))
                x = torch.tensor(X_val[i:batch_end], dtype=torch.float32, device=device)
                y = torch.tensor(Y_val[i:batch_end], dtype=torch.float32, device=device)
                out = model(x, (edge_index, None))
                val_loss += F.mse_loss(out, y).item()
                num_val_batches += 1
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        epoch_time = time.time() - t_start
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"Train RMSE: {np.sqrt(avg_train_loss):.4f}| Val RMSE: {np.sqrt(avg_val_loss):.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses

# ======= Example Entry Point =======
if __name__ == "__main__":  
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    scada_path = os.path.join("data", "wind_power_sdwpf.csv") 
    train_fastgcn_from_arrays(scada_path, edge_index)
