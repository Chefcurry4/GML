# wind_power_forecasting/graph_construction.py

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from config import SPATIAL_GRAPH_TYPE, SPATIAL_RADIUS, K_NEIGHBORS, TEMPORAL_GRAPH_TYPE

def build_spatial_graph(location_df, graph_type='e_ball', radius=1000, k=5):
    """
    Builds a spatial graph based on turbine locations.
    Args:
        location_df (pd.DataFrame): DataFrame with 'TurbID', 'X', 'Y'.
        graph_type (str): 'e_ball' or 'knn'.
        radius (float): Radius for e-ball graph.
        k (int): Number of neighbors for k-NN graph.
    Returns:
        torch_geometric.data.Data: Graph data object with edge_index and edge_attr (distances).
        np.ndarray: Array of node features (locations) sorted by TurbID.
    """
    # Ensure locations are sorted by TurbID to match data ordering
    location_df = location_df.sort_values('TurbID').reset_index(drop=True)
    locations = location_df[['X', 'Y']].values # Shape (num_turbines, 2)
    num_turbines = locations.shape[0]

    if num_turbines == 0:
         print("Warning: No turbines found in location data. Cannot build spatial graph.")
         return Data(edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 1)), x=torch.empty((0, 2))), locations


    adj_matrix = np.zeros((num_turbines, num_turbines))
    distances = cdist(locations, locations) # Pairwise distances

    if graph_type == 'e_ball':
        # Connect nodes within the specified radius
        adj_matrix[distances <= radius] = 1
        np.fill_diagonal(adj_matrix, 0) # No self-loops
    elif graph_type == 'knn':
        # Connect each node to its k nearest neighbors
        if k >= num_turbines -1:
             print(f"Warning: k ({k}) is >= number of other turbines ({num_turbines-1}). Using k = {num_turbines-1}.")
             k = num_turbines - 1
        if k <= 0:
             print("Warning: k is <= 0. Building empty spatial graph.")
             adj_matrix = np.zeros((num_turbines, num_turbines))
        else:
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(locations) # k+1 includes self
            knn_distances, indices = nn.kneighbors(locations) # indices shape (num_turbines, k+1)

            for i in range(num_turbines):
                # Connect i to its k nearest neighbors (excluding self)
                neighbors = indices[i, 1:] # Exclude self index
                adj_matrix[i, neighbors] = 1
                # For undirected: Add reverse edges
                adj_matrix[neighbors, i] = 1

            np.fill_diagonal(adj_matrix, 0) # No self-loops
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Convert adjacency matrix to edge_index (sparse format)
    # Ensure matrix is binary before converting
    adj_matrix[adj_matrix > 0] = 1
    adj_tensor = torch.FloatTensor(adj_matrix)
    edge_index, _ = dense_to_sparse(adj_tensor)

    # Get edge attributes (distances for connected pairs)
    # Need to get distances corresponding to edge_index
    if edge_index.shape[1] > 0:
        row, col = edge_index
        edge_attr = torch.FloatTensor(distances[row.numpy(), col.numpy()]).unsqueeze(1) # Shape (num_edges, 1)
    else:
         edge_attr = torch.empty((0, 1), dtype=torch.float)


    graph_data = Data(edge_index=edge_index, edge_attr=edge_attr, x=torch.FloatTensor(locations)) # Node features can be locations or other static turbine properties

    return graph_data, locations # Return locations for product graph construction


def build_temporal_graph(num_time_steps, graph_type='line'):
    """
    Builds a temporal graph.
    Args:
        num_time_steps (int): Total number of time steps in the sequence.
        graph_type (str): 'line' for connecting consecutive steps.
    Returns:
        torch_geometric.data.Data: Temporal graph data object.
    """
    if num_time_steps <= 1:
         print(f"Warning: Cannot build temporal graph with {num_time_steps} time steps.")
         return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_time_steps)

    if graph_type == 'line':
        # Edges (0,1), (1,2), ..., (N-2, N-1)
        # And their reverses (1,0), (2,1), ..., (N-1, N-2)
        rows = torch.arange(num_time_steps - 1)
        cols = torch.arange(1, num_time_steps)
        edge_index_fwd = torch.stack([rows, cols], dim=0)
        edge_index_bwd = torch.stack([cols, rows], dim=0)
        temporal_edge_index = torch.cat([edge_index_fwd, edge_index_bwd], dim=1)

        temporal_graph = Data(edge_index=temporal_edge_index, num_nodes=num_time_steps) # num_nodes is important for PyG
        return temporal_graph
    else:
        raise ValueError(f"Unknown temporal graph type: {graph_type}")


def build_product_graph(spatial_graph, temporal_graph, num_turbines, window_size):
    """
    Builds the product graph (Kronecker product) of spatial and temporal graphs
    for a given sliding window.
    A node in the product graph represents (turbine_id, time_step_in_window).
    The node index is time_step_in_window * num_turbines + turbine_id. (Corrected Indexing)

    Args:
        spatial_graph (torch_geometric.data.Data): Spatial graph data (edge_index, edge_attr).
        temporal_graph (torch_geometric.data.Data): Temporal graph data (edge_index) for the window length.
        num_turbines (int): Number of turbines.
        window_size (int): Length of the time window (input_sequence_length).

    Returns:
        torch.Tensor: edge_index for the product graph.
        torch.Tensor: edge_attr for the product graph (indicator: 1 for spatial, 0 for temporal).
    """
    if temporal_graph.num_nodes != window_size:
         raise ValueError(f"Temporal graph must have {window_size} nodes, but has {temporal_graph.num_nodes}")

    spatial_edge_index = spatial_graph.edge_index # (2, num_spatial_edges)
    temporal_edge_index = temporal_graph.edge_index # (2, num_temporal_edges)

    num_spatial_nodes = num_turbines
    num_temporal_nodes = window_size
    num_product_nodes = num_spatial_nodes * num_temporal_nodes

    if num_product_nodes == 0:
         return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)


    # Product Graph Node Indexing: (t, u) -> t * num_spatial_nodes + u
    # t is temporal node index (0 to window_size-1)
    # u is spatial node index (0 to num_turbines-1)

    # Edges in Product Graph:
    # 1. Spatial edges at the same time step t: ((t, u), (t, v)) if (u, v) is spatial edge
    #    For spatial edge (u, v) and time step t: add edge from `t * num_spatial_nodes + u` to `t * num_spatial_nodes + v`
    spatial_product_edges = []
    spatial_product_attr_indicator = [] # Attribute will be 1 for spatial

    spatial_edges_uv = spatial_edge_index.t().tolist()
    temporal_edges_ts = temporal_edge_index.t().tolist()

    for u, v in spatial_edges_uv:
        for t in range(num_temporal_nodes):
            node1_idx = t * num_spatial_nodes + u
            node2_idx = t * num_spatial_nodes + v
            spatial_product_edges.append([node1_idx, node2_idx])
            spatial_product_attr_indicator.append(1.0) # Spatial indicator

    # 2. Temporal edges for the same turbine u: ((t, u), (s, u)) if (t, s) is temporal edge
    #    For temporal edge (t, s) and turbine u: add edge from `t * num_spatial_nodes + u` to `s * num_spatial_nodes + u`
    temporal_product_edges = []
    temporal_product_attr_indicator = [] # Attribute will be 0 for temporal

    for t, s in temporal_edges_ts:
        for u in range(num_spatial_nodes):
            node1_idx = t * num_spatial_nodes + u
            node2_idx = s * num_spatial_nodes + u
            temporal_product_edges.append([node1_idx, node2_idx])
            temporal_product_attr_indicator.append(0.0) # Temporal indicator

    # Combine edges and attributes
    product_edge_index = torch.tensor(spatial_product_edges + temporal_product_edges, dtype=torch.long).t()
    product_edge_attr_indicator = torch.tensor(spatial_product_attr_indicator + temporal_product_attr_indicator, dtype=torch.float).unsqueeze(1) # Shape (num_edges, 1)

    return product_edge_index, product_edge_attr_indicator 