import torch  
import torch.nn as nn  
import torch.nn.functional as F
import pandas as pd  
import numpy as np  
import os  
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix  
import networkx as nx  


# ----------- Graph Convolution Layer --------------
class GraphConvolution(nn.Module):  
    def __init__(self, in_features, out_features):  
        super().__init__()  
        self.weight = nn.Parameter(torch.empty(in_features, out_features)) 
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):  
        return torch.matmul(adj, torch.matmul(x, self.weight))  


# ----------- Cluster-GCN Model --------------
class ClusterGCN(nn.Module):  
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


# ----------- Load SCADA Dataset --------------
def load_sdwpf_data(scada_csv):  

    df = pd.read_csv(scada_csv) 
  
    df = df.sort_values(['TurbID', 'Tmstamp'])  
 
    df['time_idx'] = df.groupby('TurbID').cumcount()   

    max_time = df['time_idx'].max()  

    df['node_idx'] = df['TurbID'] * (max_time + 1) + df['time_idx']  

    feature_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3']  
    target_col = 'Patv' 

    df = df.dropna(subset=feature_cols + [target_col])  
    scaler = StandardScaler() 
    X = scaler.fit_transform(df[feature_cols]) 
    y = df[target_col].values  
    node_idx = df['node_idx'].values  

    num_nodes = node_idx.max() + 1  
    X_tensor = torch.zeros((num_nodes, len(feature_cols)))  
    y_tensor = torch.zeros(num_nodes)  
    X_tensor[node_idx] = torch.tensor(X, dtype=torch.float32)  
    y_tensor[node_idx] = torch.tensor(y, dtype=torch.float32)  

    return X_tensor, y_tensor, num_nodes 


# ----------- Partition graph using KMeans over node features --------------
def cluster_graph(X, num_clusters):
    """
    Partitions the graph using KMeans clustering over the node features into `num_clusters` disjoint groups.
    Returns a dictionary: {cluster_id: [node_indices]}
    """
    X_np = X.cpu().numpy()
    km = KMeans(n_clusters=num_clusters, random_state=42)
    labels = km.fit_predict(X_np)

    clusters = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return clusters


# ----------- Build per-cluster adjacency matrix --------------
def build_cluster_adj(edge_index, cluster_nodes): 
    node_map = {n: i for i, n in enumerate(cluster_nodes)}  
    edges = [(node_map[u], node_map[v]) for u, v in edge_index.T.tolist()
     
    # Filter edges within the cluster; if no edges, return identity matrix       
             if u in node_map and v in node_map]  
    if not edges: 
        return torch.eye(len(cluster_nodes))
    
    rows, cols = zip(*edges)  # Separate edge endpoints
    data = np.ones(len(rows))  # Edge weights (all 1)
    size = len(cluster_nodes)  
    A = coo_matrix((data, (rows, cols)), shape=(size, size))  
    A = A + A.T  # Make adjacency symmetric
    A = A.tocsr()  # Convert to CSR format (Compressed Sparse Row), a memory-efficient way to store sparse matrices.
    A = A.multiply(1 / (A.sum(axis=1) + 1e-6))  
    return torch.tensor(A.toarray(), dtype=torch.float32)  


# ----------- Train Cluster-GCN --------------
def train_cluster_gcn(scada_csv, edge_index, hidden=64, num_clusters=20, dropout=0.3, epochs=10):  
    print("\U0001F4CA Loading SCADA data...") 
    X, y, num_nodes = load_sdwpf_data(scada_csv) 
    print("\U0001F517 Clustering graph...")  
    clusters = cluster_graph(X, num_clusters)  

    model = ClusterGCN(nfeat=X.shape[1], nhid=hidden, dropout=dropout)  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  

    print("\U0001F3CB\uFE0F Training...") 
    model.train()  
    for epoch in range(epochs):  
        total_loss = 0 
        optimizer.zero_grad()  # Zero gradients at the start of each epoch for gradient accumulation
        batch_count = 0  
        batch_size = 4  # Number of clusters to accumulate gradients over (tune as needed)

        # Iterate over clusters in sorted order for reproducibility
        for cluster_id, nodes in sorted(clusters.items()):  
            if len(nodes) < 2:  # Skip clusters with fewer than 2 nodes
                continue
            x_sub = X[nodes] 
            y_sub = y[nodes]  
            adj_sub = build_cluster_adj(edge_index, nodes)  
            out = model(x_sub, adj_sub)  
            loss = F.mse_loss(out, y_sub)  
            loss = loss / batch_size  
            loss.backward()  # Backpropagate loss (accumulate gradients)
            total_loss += loss.item() * batch_size  # Accumulate (unnormalized) loss for reporting
            batch_count += 1
            # Perform optimizer step after accumulating gradients over batch_size clusters
            if batch_count % batch_size == 0:
                optimizer.step()  # Update model parameters
                optimizer.zero_grad()  # Reset gradients

        # Final optimizer step for any remaining clusters in the epoch
        if batch_count % batch_size != 0:
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/num_clusters:.4f}")  
    print("\u2705 Cluster-GCN training complete!") 


# ====== Example ======
if __name__ == "__main__":  
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long) 
    scada_path = os.path.join("data", "wind_power_sdwpf.csv")  
    train_cluster_gcn(scada_path, edge_index)  

# Note: We accumulate gradients over several clusters before updating the model parameters (optimizer.step()).
# This simulates a larger batch size, which can improve training stability and efficiency for large graphs.
# We also iterate over clusters in sorted order for reproducibility and clarity.

