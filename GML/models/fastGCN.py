# fastGCN.py
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler  
from torch_sparse import coalesce 
import os  

# ----------- Graph Convolution Layer --------------
class GraphConvolution(nn.Module):  
    def __init__(self, in_features, out_features): 
        super().__init__()  
        self.weight = nn.Parameter(torch.empty(in_features, out_features)) 
        nn.init.xavier_uniform_(self.weight)  

    def forward(self, x, adj):  
        return torch.matmul(adj, torch.matmul(x, self.weight))  # Graph convolution operation


# ----------- FastGCN Model (regression task to predict Patv) --------------
class FastGCN(nn.Module):  
    def __init__(self, nfeat, nhid, dropout, sampler): 
        super().__init__()  
        self.gc1 = GraphConvolution(nfeat, nhid) 
        self.gc2 = GraphConvolution(nhid, 1)  
        self.dropout = dropout # prevent overfitting by setting to zero a fraction of the neurons' outputs in a layer on each forward pass.
        self.sampler = sampler 

    def forward(self, x, adjs): 
        x = F.relu(self.gc1(x, adjs[0]))  # Apply first GCN layer and ReLU activation
        x = F.dropout(x, self.dropout, training=self.training)  # Apply dropout
        x = self.gc2(x, adjs[1])  # Apply second GCN layer
        return x.squeeze()  # Remove single-dimensional entries from output

    def sampling(self, features, adj):  # Wrapper for sampler's sampling method
        return self.sampler.sampling(features, adj)


# ----------- Importance Sampler --------------
class ImportanceSampler:  
    def __init__(self, num_samples):  
        self.num_samples = num_samples  

    def sampling(self, features, adj):  # Perform importance sampling
        num_nodes = features.shape[0]  
        prob = adj.sum(dim=1).numpy()  # Compute node degrees as sampling probabilities
        prob = prob / prob.sum()  # Normalize probabilities
        sampled_idx = np.random.choice(num_nodes, self.num_samples, p=prob, replace=False)  # Sample node indices
        sampled_idx = torch.tensor(sampled_idx, dtype=torch.long)  # Convert indices to tensor

        sampled_features = features[sampled_idx]  # Select sampled node features
        sampled_adj = adj[sampled_idx][:, sampled_idx]  # Select sub-adjacency matrix for sampled nodes

        return sampled_features, [sampled_adj, sampled_adj], sampled_idx  


# ----------- Build dense adj from edge_index --------------
# in FastGCN use a list of adjacency matrices—one for each layer—because Each layer samples a different subgraph from the full graph.

def build_dense_adj(edge_index, num_nodes): 
    edge_index, _ = coalesce(edge_index, torch.ones(edge_index.shape[1]), num_nodes, num_nodes)  # Coalesce edges (Ensures the edge index is sorted and unique.)
    adj = torch.zeros((num_nodes, num_nodes))  
    adj[edge_index[0], edge_index[1]] = 1.0  
    adj = (adj + adj.T) / 2  
    adj = adj / (adj.sum(1, keepdim=True) + 1e-6)  # Row-normalize adjacency
    return adj  


# ----------- Load and process SDWPF .csv into X, y --------------
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


# ----------- Train the model --------------
def train_fastgcn(scada_csv, edge_index, hidden=64, samples=512, dropout=0.3): 
    print("\U0001F4CA Loading SCADA data...")  
    X, y, num_nodes = load_sdwpf_data(scada_csv) 

    print("\U0001F517 Building adjacency matrix...") 
    adj = build_dense_adj(edge_index, num_nodes) 

    print("\U0001F680 Initializing FastGCN...") 
    sampler = ImportanceSampler(samples)  
    model = FastGCN(nfeat=X.shape[1], nhid=hidden, dropout=dropout, sampler=sampler) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

    print("\U0001F3CB\uFE0F Training...")  
    model.train()  
    for epoch in range(1, 21):  
        sampled_X, sampled_adjs, idx = model.sampling(X, adj)  
        output = model(sampled_X, sampled_adjs) 
        loss = F.mse_loss(output, y[idx])  
        loss.backward() 
        optimizer.step()  
        optimizer.zero_grad()  
        print(f"Epoch {epoch:02d} | MSE Loss: {loss.item():.4f}")  

    print("\u2705 FASTGCN TRAINING is complete!")  


# ======= Example =======
if __name__ == "__main__":  
   
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)  # change here

  
    scada_path = os.path.join("data", "wind_power_sdwpf.csv") 
    train_fastgcn(scada_path, edge_index)  
