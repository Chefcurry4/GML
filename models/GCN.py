import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
from config import INPUT_SEQUENCE_LENGTH

class OptimizedGCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super().__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid) 
        self.conv3 = GCNConv(nhid, nout)
        self.dropout = dropout

    def forward(self, x, edge_index, batch_info):
        # x: (total_nodes, features) - flattened across all batches
        # edge_index: (2, E) - edge indices
        # batch_info: which nodes belong to which batch
        
        # * Note: Use 2 GCN layers. 1 layer might not capture enough spatial dependencies. 3 layers might cause over-smoothing.

        # First GCN layer
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        
        # Second GCN layer
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        
        # Output layer
        out = self.conv3(h, edge_index)
        
        # Convert back to batch format
        out_batched, mask = to_dense_batch(out, batch_info)
        return out_batched


def train_gcn(
    X_train, Y_train, X_val, Y_val, edge_index,
    hidden, dropout, epochs, lr, patience, batch_size
):
    """
    Train GCN for power output prediction
    Args:
        X_train: (num_samples, T*N, F) - input features for product graph
        Y_train: (num_samples, N) - target power output for N turbines
        X_val: (num_samples, T*N, F) - validation features
        Y_val: (num_samples, N) - validation targets
        edge_index: (2, E) - product graph edges
        hidden: number of hidden units in GCN layers
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
    
    # Initialize model
    model = OptimizedGCN(nfeat=num_features, nhid=hidden, nout=1, dropout=dropout).to(device)
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

            # Flatten batch for PyG
            B, N, F = x_batch.shape
            x_flat = x_batch.view(-1, F)  # (B*N, F)
            
            # Create batch info
            batch_info = torch.arange(B, device=device).repeat_interleave(N)
            
            optimizer.zero_grad()
            out = model(x_flat, edge_index, batch_info)  # Shape: (B, N, 1)

            # Extract predictions for the last time step of each turbine
            # For product graph: nodes are ordered as (t0,n0), (t0,n1), ..., (t1,n0), ..., (tT-1,nN-1)
            # We want the last time step for each turbine: (tT-1,n0), (tT-1,n1), ..., (tT-1,nN-1)
            
            # Calculate indices for last time step nodes
            last_time_start = (INPUT_SEQUENCE_LENGTH - 1) * num_turbines
            last_time_end = INPUT_SEQUENCE_LENGTH * num_turbines
            last_indices = torch.arange(last_time_start, last_time_end, device=device)
            
            # Extract predictions: out shape is (B, N, 1), we want (B, num_turbines)
            predictions = out[:, last_indices, 0]  # Shape: (B, num_turbines)

            # Use predictions for loss calculation
            loss = criterion(predictions, y_batch)
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

                # Same processing for validation
                B, N, F = x_batch.shape
                x_flat = x_batch.view(-1, F)
                batch_info = torch.arange(B, device=device).repeat_interleave(N)
                
                out = model(x_flat, edge_index, batch_info)
                
                # Extract last time step predictions
                last_time_start = (INPUT_SEQUENCE_LENGTH - 1) * num_turbines
                last_time_end = INPUT_SEQUENCE_LENGTH * num_turbines
                last_indices = torch.arange(last_time_start, last_time_end, device=device)
                predictions = out[:, last_indices, 0]
                
                # Use predictions for validation loss
                val_loss += criterion(predictions, y_batch).item()
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