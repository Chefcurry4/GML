# wind_power_forecasting/graph_construction.py

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from utils.static_graphs import build_wake_graph
from utils.args_interface import GRAPH_TYPE, Args
from utils.utils import visualize_spatial_graph, visualize_spatio_temporal_graph, visualize_temporal_graph
from config import DOMDIR_ANGLE_THRESHOLD, DOMDIR_DECAY_LENGTH, DOMDIR_INCLUDE_WEIGHTS, DOMDIR_MAX_DISTANCE, DOMDIR_WIND_DIR, INPUT_SEQUENCE_LENGTH, SPATIAL_GRAPH_TYPE, SPATIAL_RADIUS, K_NEIGHBORS, TEMPORAL_GRAPH_TYPE
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
import datetime
import os

def build_spatial_graph(location_df, args):
    """
    Builds a spatial, undirected, self‐loop‐free graph:
      • radius: connect all pairs with dist ≤ radius
      • knn:    connect each node to its k nearest neighbors
    Returns:
      edge_index: LongTensor[2, E]
      locations:  np.ndarray[N,2]
    """
    # 0) load data from config
    graph_type = args.spatial_graph_type

    # 1) load coords & distances
    locations = location_df[['x','y']].values
    N = len(locations)
    D = cdist(locations, locations).astype(np.float32)

    # 2) build raw row/col lists
    if graph_type == GRAPH_TYPE.RADIUS:
        radius = SPATIAL_RADIUS
        if radius is None:
            raise ValueError("radius required for radius graph")
        rows, cols = np.where(D <= radius)

    elif graph_type == GRAPH_TYPE.KNN:
        # * Note that kNN calculates the k nearest neighbors per node. In an undirected graph, this means that nodes might have more than k neighbors. However, each node has at least k neighbors.
        k = K_NEIGHBORS
        if k is None:
            raise ValueError("k required for kNN graph")
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(locations)
        _, idx = nbrs.kneighbors(locations)
        # drop self and flatten
        rows = np.repeat(np.arange(N), k)
        cols = idx[:, 1:k+1].reshape(-1)
    elif graph_type == GRAPH_TYPE.DOMDIR:
        data = build_wake_graph(locations, DOMDIR_WIND_DIR, DOMDIR_INCLUDE_WEIGHTS, DOMDIR_DECAY_LENGTH, DOMDIR_ANGLE_THRESHOLD, DOMDIR_MAX_DISTANCE)

        return data.edge_index, locations

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

    # 6) convert to torch
    edge_index = torch.from_numpy(edge_index_np).long()

    return edge_index, locations


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


def build_spatio_temporal_product(
    spatial_edge_index: torch.LongTensor,   # [2, E_s]
    N: int,                                 # number of spatial nodes
    temporal_edge_index: torch.LongTensor,  # [2, E_t], indices in [0..T-1]
    T: int                                  # window size (number of time layers)
):
    """
    Builds the Cartesian (spatio-temporal) product of:
      • a spatial, undirected graph on N nodes (edge_index),
      • a temporal (directed) chain on T time steps (0→1→…→T-1).
    
    Output graph has N*T nodes, indexed as (t * N + i).  Edges:
      (1) Spatial edges: for each t, copy (u,v) → (u + tN, v + tN), with same weight.
      (2) Temporal edges: for each spatial index i, connect (i + tN) → (i + (t+1)N)
          with a weight = 1.0 (default).

    Args:
        spatial_edge_index: LongTensor of shape [2, E_s], containing one undirected
                            edge per pair (u < v).  (No self-loops, no duplicates.)
        N:                  Number of spatial nodes. 
        temporal_edge_index: LongTensor of shape [2, E_t], where each edge is (t->t+1).
        T:                  Number of time layers (window size).

    Returns:
        st_edge_index: LongTensor of shape [2, E_total] for the product graph.
    """

    # 1) Prepare lists to accumulate edges
    spatial_rows, spatial_cols = spatial_edge_index  # both are length E_s
    E_s = spatial_rows.size(0)
    E_t = temporal_edge_index.size(1)

    # --- (A) Build all spatial edges at each time t ---
    # We will end up with T * E_s spatial edges in the product.
    # For time t, the offset to add is (t * N).
    # New_u = u + t*N,  New_v = v + t*N.

    # Repeat each spatial-edge index T times, then add offsets
    # E.g. if spatial_edge_index = [ [u0, u1, …], [v0, v1, …] ] of length E_s,
    # then stacked as [2, E_s*T] after repeating for each t.

    # 1a) Expand spatial edge‐index tensors from [E_s] → [T, E_s]
    spatial_u = spatial_rows.unsqueeze(0).repeat(T, 1)  # [T, E_s]
    spatial_v = spatial_cols.unsqueeze(0).repeat(T, 1)  # [T, E_s]

    # 1b) Create a ([T] × [E_s])-shaped offset for each time‐slice
    # offsets[t] = t * N, broadcasted to length E_s
    time_offsets = (torch.arange(T, dtype=torch.long) * N).unsqueeze(1)  # [T, 1]
    time_offsets = time_offsets.repeat(1, E_s)                           # [T, E_s]

    # 1c) Add offsets and flatten:
    #   spatial_u_flat = (spatial_u + time_offsets).view(-1)
    #   spatial_v_flat = (spatial_v + time_offsets).view(-1)
    spatial_u_flat = (spatial_u + time_offsets).reshape(-1)  # length = T*E_s
    spatial_v_flat = (spatial_v + time_offsets).reshape(-1)  # length = T*E_s

    # --- (B) Build all temporal edges for each spatial node i ---
    # Each temporal edge is (t -> t+1).  We want, for all i in [0..N-1],
    # ( i + t*N ) → ( i + (t+1)*N ).  If there are E_t = (T-1) such edges,
    # then we create N * (T-1) product edges.

    #  B.1: temporal_edge_index is [2, E_t], where E_t = T-1
    temp_src = temporal_edge_index[0]  # [E_t]
    temp_dst = temporal_edge_index[1]  # [E_t]

    #  B.2: We want to repeat each (t_src -> t_dst) N times, once for each spatial i.
    #       Let E_t = T-1.  Then we build two tensors of shape [N * E_t]:
    #
    #         for each e in 0..E_t-1:
    #             all_i = [0..N-1]
    #             prod_src = all_i + temp_src[e] * N
    #             prod_dst = all_i + temp_dst[e] * N
    #
    #       We can vectorize by:
    #         - temp_src.unsqueeze(1).repeat(1, N) → [E_t, N]
    #         - torch.arange(N).unsqueeze(0).repeat(E_t, 1) → [E_t, N]
    #         then add and flatten.

    all_i = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(E_t, 1)    # [E_t, N]
    t_src_rep = temp_src.unsqueeze(1).repeat(1, N)  # [E_t, N]
    t_dst_rep = temp_dst.unsqueeze(1).repeat(1, N)  # [E_t, N]

    temporal_u_flat = (all_i + t_src_rep * N).reshape(-1)  # [N * E_t]
    temporal_v_flat = (all_i + t_dst_rep * N).reshape(-1)  # [N * E_t]

    # --- (C) Concatenate spatial and temporal pieces ---
    # First stack row‐indices, then col‐indices, along dim=0.
    st_u = torch.cat([spatial_u_flat, temporal_u_flat], dim=0)  # [T*E_s + N*E_t]
    st_v = torch.cat([spatial_v_flat, temporal_v_flat], dim=0)  # [T*E_s + N*E_t]

    st_edge_index = torch.stack([st_u, st_v], dim=0)  # [2, E_total]

    return st_edge_index


def build_graph(locations_df, args: Args):
    print("\n=========================================================")
    print("Building spatio-temporal graphs")

    # Make sure directory to save images exists
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    images_path = f"GML/images/{timestamp_str}"
    os.makedirs(images_path, exist_ok=True)

    # Build the spatial graph
    edge_index_spatial, locations = build_spatial_graph(locations_df, args)

    # Visualize the spatial graph and store the image
    visualize_spatial_graph(
        edge_index_spatial,
        locations,
        save_path=f"{images_path}/spatial_graph_gnn.png",
    )

    # Build temporal graph template for the window size (needed by collate_fn)
    temp_edge_index = build_temporal_graph(INPUT_SEQUENCE_LENGTH, TEMPORAL_GRAPH_TYPE)

    # Visualize the temporal graph
    visualize_temporal_graph(
        temp_edge_index,
        num_time_steps=INPUT_SEQUENCE_LENGTH,
        save_path=f"{images_path}/temporal_graph_gnn.png"
    )

    # Get the number of turbines
    num_turbines = locations_df.shape[0]

    # Build the spatio-temporal product graph 
    spatio_temporal_edge_index = build_spatio_temporal_product(
        spatial_edge_index=edge_index_spatial,
        N=num_turbines,
        temporal_edge_index=temp_edge_index,
        T=INPUT_SEQUENCE_LENGTH
    )
    # Visualize the spatio-temporal product graph
    visualize_spatio_temporal_graph(
        st_edge_index=spatio_temporal_edge_index,
        locations=locations,
        N=num_turbines,
        T=INPUT_SEQUENCE_LENGTH,
        time_offset=4*1800,            # horizontal separation between layers
        save_path=f"{images_path}/spatio_temporal_product_graph_gnn.png",
        node_size=5
    )

    if spatio_temporal_edge_index.numel() == 0 and num_turbines > 0 and INPUT_SEQUENCE_LENGTH > 0:
        raise RuntimeError("Warning: Product graph template is empty, but data exists. Check graph construction parameters.")
    
    print("==========================================================\n")

    return spatio_temporal_edge_index


# TODO: remove old function
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
    raise DeprecationWarning("This function is deprecated. Use build_spatio_temporal_product instead.")
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