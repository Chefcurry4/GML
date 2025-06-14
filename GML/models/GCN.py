import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.sparse import coo_matrix

# ----------- Graph Convolution Layer --------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        return torch.matmul(adj, torch.matmul(x, self.weight))

# ----------- Standard GCN --------------
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x.squeeze()

# ----------- Load and process SCADA data --------------
def load_sdwpf_data(scada_csv):
     
    df = pd.read_csv(scada_csv)  

    df = df.sort_values(['TurbID', 'Tmstamp']) 

    df['time_idx'] = df.groupby('TurbID').cumcount()  # Assign time index per turbine
    
    max_time = df['time_idx'].max() 

    df['node_idx'] = df['TurbID'] * (max_time + 1) + df['time_idx']  # Unique node index for each turbine-time pair

    feature_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3']  # Features
    target_col = 'Patv'  # Target 

    df = df.dropna(subset=feature_cols + [target_col])  # Drop rows with missing values (NOT SURE)
    scaler = StandardScaler()  
    # X = Feature matrix: describes each node (turbine-time pair)
    X = scaler.fit_transform(df[feature_cols])  # Normalize features
    # y = Target vector: Patv for each turbine-time pair
    y = df[target_col].values  # Extract target values
    # node_idx = Unique index for each turbine-time pair
    # This will be used to map features and targets back to the original node indices
    node_idx = df['node_idx'].values  # Extract node indices

    num_nodes = node_idx.max() + 1  # Total number of nodes
    X_tensor = torch.zeros((num_nodes, len(feature_cols)))  
    y_tensor = torch.zeros(num_nodes)  
    X_tensor[node_idx] = torch.tensor(X, dtype=torch.float32)  # Assign features to tensor
    y_tensor[node_idx] = torch.tensor(y, dtype=torch.float32)  # Assign targets to tensor

    return X_tensor, y_tensor, num_nodes 

# ----------- Build dense adjacency matrix from edge_index --------------
def build_dense_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = (adj + adj.T) / 2
    adj = adj / (adj.sum(1, keepdim=True) + 1e-6) # Row-normalize adjacency
    return adj

# ----------- Train the GCN model --------------
def train_gcn(scada_csv, edge_index, hidden=64, dropout=0.3, epochs=20):
    print("\U0001F4CA Loading SCADA data...")
    X, y, num_nodes = load_sdwpf_data(scada_csv)
    print("\U0001F517 Building adjacency matrix...")
    adj = build_dense_adj(edge_index, num_nodes)
    print("\U0001F680 Initializing GCN...")
    model = GCN(nfeat=X.shape[1], nhid=hidden, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print("\U0001F3CB\uFE0F Training...")
    model.train()
    for epoch in range(epochs):
        output = model(X, adj)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch+1:02d} | MSE Loss: {loss.item():.4f}")
    print("\u2705 GCN training complete!")

# ======= Example Usage =======
if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)  # Example edge index
    scada_path = os.path.join("data", "wind_power_sdwpf.csv")
    train_gcn(scada_path, edge_index)
