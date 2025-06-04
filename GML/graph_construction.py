# wind_power_forecasting/graph_construction.py

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from config import SPATIAL_GRAPH_TYPE, SPATIAL_RADIUS, K_NEIGHBORS, TEMPORAL_GRAPH_TYPE
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops

def build_spatial_graph(location_df, graph_type='radius', radius=None, k=None):
    """
    Builds a spatial, undirected, self‐loop‐free graph:
      • radius: connect all pairs with dist ≤ radius
      • knn:    connect each node to its k nearest neighbors
    Returns:
      edge_index: LongTensor[2, E]
      edge_attr:  FloatTensor[E, 1] (distance)
      locations:  np.ndarray[N,2]
    """
    # 1) load coords & distances
    locations = location_df[['x','y']].values
    N = len(locations)
    D = cdist(locations, locations).astype(np.float32)

    # 2) build raw row/col lists
    if graph_type == 'radius':
        if radius is None:
            raise ValueError("radius required for radius graph")
        rows, cols = np.where(D <= radius)

    elif graph_type == 'knn':
        if k is None:
            raise ValueError("k required for kNN graph")
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(locations)
        _, idx = nbrs.kneighbors(locations)
        # drop self and flatten
        rows = np.repeat(np.arange(N), k)
        cols = idx[:, 1:k+1].reshape(-1)

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # 3) remove self‐loops
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]

    # 4) enforce undirected & unique edges
    unique = set()
    for u, v in zip(rows, cols):
        a, b = (u, v) if u < v else (v, u)
        unique.add((a, b))
    rows, cols = zip(*unique) if unique else ([], [])

    # 5) assemble numpy arrays
    edge_index_np = np.vstack((rows, cols))
    edge_attr_np  = D[rows, cols][:, None]

    # 6) convert to torch
    edge_index = torch.from_numpy(edge_index_np).long()
    edge_attr  = torch.from_numpy(edge_attr_np).float()

    return edge_index, edge_attr, locations


def build_temporal_graph(window_size, graph_type='sequential'):
    """Builds a temporal graph for a sliding window.
    Args:
        window_size (int): Length of the time window
        graph_type (str): Type of temporal graph ('sequential')
    Returns:
        torch.Tensor: Edge index for temporal graph
    """
    if graph_type == 'sequential':
        # Create edges between consecutive time steps
        src = torch.arange(window_size - 1)
        dst = torch.arange(1, window_size)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index
    else:
        raise ValueError(f"Unknown temporal graph type: {graph_type}")


def build_product_graph(edge_index, edge_attr, temporal_graph_template, num_turbines, input_seq_len):
    """Builds a product graph from spatial and temporal graphs.
    Args:
        edge_index (torch.Tensor): Spatial graph edge index
        edge_attr (torch.Tensor): Spatial graph edge attributes
        temporal_graph_template (torch.Tensor): Temporal graph edge index template
        num_turbines (int): Number of turbines
        input_seq_len (int): Input sequence length
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Product graph edge index and edge attributes
    """
    # Convert temporal template to actual temporal graph for this window
    temporal_edge_index = temporal_graph_template.clone()
    
    # Create product graph edges
    # For each temporal edge (t1, t2), create spatial edges between corresponding nodes
    product_edges = []
    product_attrs = []
    
    # Add spatial edges for each time step
    for t in range(input_seq_len):
        # Offset the spatial edge indices by (t * num_turbines)
        offset = t * num_turbines
        time_edge_index = edge_index + offset
        product_edges.append(time_edge_index)
        product_attrs.append(edge_attr)
    
    # Add temporal edges for each node
    for n in range(num_turbines):
        for t1, t2 in temporal_edge_index.t():
            # Connect node n at time t1 to node n at time t2
            src = t1.item() * num_turbines + n
            dst = t2.item() * num_turbines + n
            product_edges.append(torch.tensor([[src], [dst]], dtype=torch.long))
            # Use a default attribute for temporal edges (e.g., 1.0)
            product_attrs.append(torch.tensor([[1.0]], dtype=torch.float))
    
    # Combine all edges
    product_edge_index = torch.cat(product_edges, dim=1)
    product_edge_attr = torch.cat(product_attrs, dim=0)
    
    return product_edge_index, product_edge_attr 