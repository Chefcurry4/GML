import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from config import INPUT_SEQUENCE_LENGTH

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
        """
        x: (batch_size, num_nodes, in_features)
        adj: (num_nodes, num_nodes) normalized adjacency matrix
        """
        support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
        out = torch.matmul(adj, support)        # (batch, nodes, out_features)
        if self.bias is not None:
            out += self.bias
        return out

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
        """
        x: (batch_size, T*N, features) - product graph nodes
        adj: (T*N, T*N) - product graph adjacency
        Returns: (batch_size, N) - power prediction for N turbines
        """
        B, TN, num_feat = x.shape  # Changed F to num_feat
        N = TN // INPUT_SEQUENCE_LENGTH  # number of turbines
        
        # First GCN layer
        h = self.gc1(x, adj)
        h = h.view(B * TN, -1)
        h = self.bn1(h)
        h = h.view(B, TN, -1)
        h = F.relu(h)  # Now F refers to torch.nn.functional
        h = F.dropout(h, self.dropout, training=self.training)

        # Second GCN layer
        h = self.gc2(h, adj)
        h = h.view(B * TN, -1)
        h = self.bn2(h)
        h = h.view(B, TN, -1)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)

        # Output layer
        out = self.gc3(h, adj)  # (B, T*N, 1)
        
        # Extract predictions for the last time step of each turbine
        # Nodes are ordered as: (t0,n0), (t0,n1), ..., (t1,n0), ..., (tT-1,nN-1)
        last_time_indices = torch.arange((INPUT_SEQUENCE_LENGTH-1)*N, TN, device=x.device)
        predictions = out[:, last_time_indices, 0]  # (B, N)
        
        return predictions

def normalize_adj(edge_index, num_nodes):
    """Create normalized adjacency matrix from edge_index"""
    row, col = edge_index
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    
    # Degree computation
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.scatter_add_(0, row, edge_weight)
    deg.scatter_add_(0, col, edge_weight)
    
    # Symmetric normalization D^(-1/2) A D^(-1/2)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # Create normalized adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[row, col] = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
    return adj

def train_gcn(
    X_train, Y_train, X_val, Y_val, edge_index,
    hidden, samples, dropout, epochs, lr, patience, batch_size
):
    """
    Train GCN for power output prediction
    Args:
        X_train: (num_samples, T*N, F) - input features for product graph
        Y_train: (num_samples, N) - target power output for N turbines
        X_val: (num_samples, T*N, F) - validation features
        Y_val: (num_samples, N) - validation targets
        edge_index: (2, E) - product graph edges
        hidden: hidden dimension
        samples: not used for standard GCN
        dropout: dropout rate
        epochs: number of epochs
        lr: learning rate
        patience: early stopping patience
        batch_size: batch size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)

    # Get dimensions
    num_samples, num_nodes, num_features = X_train.shape
    num_turbines = Y_train.shape[1]
    
    # Create normalized adjacency matrix
    adj = normalize_adj(edge_index, num_nodes).to(device)
    
    # Initialize model
    model = GCN(nfeat=num_features, nhid=hidden, nout=1, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    print(f"\n[GCN Architecture]")
    print(f"Input features: {num_features}")
    print(f"Hidden dimension: {hidden}")
    print(f"Dropout rate: {dropout}")
    print(f"Output turbines: {num_turbines}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        t_start = time.time()
        model.train()
        total_loss = 0
        num_batches = 0

        # Training loop
        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            x_batch = X_train[i:batch_end]
            y_batch = Y_train[i:batch_end]
            
            # Expand adjacency for batch
            adj_batch = adj.unsqueeze(0).expand(x_batch.shape[0], -1, -1)

            optimizer.zero_grad()
            out = model(x_batch, adj_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                batch_end = min(i + batch_size, X_val.shape[0])
                x_batch = X_val[i:batch_end]
                y_batch = Y_val[i:batch_end]
                adj_batch = adj.unsqueeze(0).expand(x_batch.shape[0], -1, -1)

                out = model(x_batch, adj_batch)
                val_loss += criterion(out, y_batch).item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        epoch_time = time.time() - t_start
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"Train RMSE: {np.sqrt(avg_train_loss):.4f} | Val RMSE: {np.sqrt(avg_val_loss):.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break

    print("GCN training complete.")
    return model, train_losses, val_losses