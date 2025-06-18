# combined_wake_graph_pipeline.py

# --- Imports ---
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_undirected, coalesce, remove_self_loops
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *

# --- Wake Edge Computation ---
def compute_wake_edges(coords: np.ndarray,
                       wind_dir: float,
                       decay_length: float = 1000.0,
                       angle_threshold: float = 20.0,
                       max_distance: float = 1500.0) -> (np.ndarray, np.ndarray):
    """
    Determine wake interactions based on turbine locations and wind direction.

    Parameters:
        coords (np.ndarray): Array of shape (N, 2) with turbine x/y coordinates.
        wind_dir (float): Wind "from" direction in degrees.
        decay_length (float): Decay length for distance attenuation.
        angle_threshold (float): Angular half-width of the wake cone in degrees.
        max_distance (float): Maximum distance for establishing an edge.

    Returns:
        edge_index (np.ndarray): Array of shape (2, E) with source and target indices.
        edge_weights (np.ndarray): Array of shape (E,) with wake weights.
    """
    N = coords.shape[0]
    # The wind flow is the opposite of the wind "from" direction.
    flow_dir = (wind_dir + 180.0) % 360.0

    rows, cols, weights = [], [], []

    for i in range(N):
        xi, yi = coords[i]
        for j in range(N):
            if i == j:
                continue
            xj, yj = coords[j]
            dx, dy = xj - xi, yj - yi
            d = np.hypot(dx, dy)
            if d > max_distance or d == 0:
                continue

            # Bearing from turbine i to j in compass degrees
            ang_rad = np.arctan2(dy, dx)
            ang_deg = np.degrees(ang_rad)
            compass_ij = (90.0 - ang_deg) % 360.0

            # Compute smallest signed angular difference
            diff = abs((compass_ij - flow_dir + 180.0) % 360.0 - 180.0)

            if diff <= angle_threshold:
                w = np.cos(np.radians(diff)) * np.exp(-d / decay_length)
                if w > 0:
                    rows.append(i)
                    cols.append(j)
                    weights.append(w)

    if rows:
        edge_index = np.vstack((np.array(rows), np.array(cols)))
        edge_weights = np.array(weights)
    else:
        edge_index = np.zeros((2, 0), dtype=int)
        edge_weights = np.array([])

    return edge_index, edge_weights

# --- Graph Generation ---
def build_wake_graph(coords: np.ndarray,
                     wind_dir: float,
                     include_weights: bool = True,
                     decay_length: float = 1000.0,
                     angle_threshold: float = 20.0,
                     max_distance: float = 1500.0) -> Data:
    """
    Build a PyG graph for a single wind direction.
    """
    edge_index, edge_weights = compute_wake_edges(coords, wind_dir,
                                                  decay_length, angle_threshold,
                                                  max_distance)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = (torch.tensor(edge_weights, dtype=torch.float)
                 if include_weights and edge_weights.size else None)
    x = torch.tensor(coords, dtype=torch.float)
    return Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr)

def build_wake_graphs(coords: np.ndarray,
                      wind_directions: list,
                      include_weights: bool = True,
                      **wake_kwargs) -> dict:
    """
    Generate wake graphs for multiple wind directions.
    """
    graphs = {}
    for wd in wind_directions:
        graphs[wd] = build_wake_graph(coords, wd,
                                      include_weights,
                                      **wake_kwargs)
    return graphs

# --- Visualization ---
def plot_wake_graph(graph: Data, show: bool = True) -> None:
    """
    Plot the spatial wake graph structure for a single direction.
    """
    coords = graph.x.numpy()
    edge_index = graph.edge_index.numpy()
    weights = graph.edge_attr.numpy() if graph.edge_attr is not None else None

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, zorder=2)

    for idx, (u, v) in enumerate(edge_index.T):
        lw = 1 + 3 * weights[idx] if weights is not None else 1
        plt.plot([coords[u, 0], coords[v, 0]],
                 [coords[u, 1], coords[v, 1]],
                 color='gray',
                 linewidth=lw,
                 alpha=0.7,
                 zorder=1)

    for i, (x, y) in enumerate(coords):
        plt.text(x, y + 50, str(i + 1), fontsize=12,
                 ha='center', va='bottom', color='red')

    plt.title("Wake Interaction Graph Based on Wind Direction")
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.axis("equal")
    plt.grid(linestyle="--", alpha=0.5)
    if show:
        plt.show()

def plot_wake_graphs(graphs: dict,
                     cols: int = 3,
                     show: bool = True) -> None:
    """
    Plot multiple wake graphs in a grid layout.
    """
    directions = list(graphs.keys())
    n = len(directions)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).reshape(-1)

    for i, wd in enumerate(directions):
        ax = axes[i]
        graph = graphs[wd]
        coords = graph.x.numpy()
        edge_index = graph.edge_index.numpy()
        weights = graph.edge_attr.numpy() if graph.edge_attr is not None else None

        ax.scatter(coords[:, 0], coords[:, 1], s=50)
        for idx, (u, v) in enumerate(edge_index.T):
            lw = 1 + 3 * weights[idx] if weights is not None else 1
            ax.plot([coords[u, 0], coords[v, 0]],
                    [coords[u, 1], coords[v, 1]],
                    linewidth=lw,
                    alpha=0.7)
        ax.set_title(f"Dir: {wd}°")
        ax.axis('equal')

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if show:
        plt.show()

# --- Main Script ---
if __name__ == '__main__':
    # --- First example from graph_construction.py ---
    df_loc = pd.read_csv('GML/data/turbine_location.csv')
    print("Example turbine locations:")
    print(df_loc.head())

    coords = df_loc[['x', 'y']].values

    wind_dir = 45.0            # Wind coming from 45°
    decay_length = 1000.0      # Distance decay length
    angle_threshold = 20.0     # ±20° wake cone
    max_distance = 2000.0      # Maximum distance

    edge_index, edge_weights = compute_wake_edges(
        coords, wind_dir,
        decay_length, angle_threshold,
        max_distance
    )

    print("Edge indices (source -> target):")
    print(edge_index)
    print("Edge weights:")
    print(edge_weights)

    plot_wake_graph(
        build_wake_graph(coords, wind_dir,
                         include_weights=True,
                         decay_length=decay_length,
                         angle_threshold=angle_threshold,
                         max_distance=max_distance)
    )

    for src, tgt, w in zip(edge_index[0], edge_index[1], edge_weights):
        print(f"Edge from node {src} to node {tgt} with weight {w:.3f}")

    # --- Second example from wake_graph.ipynb integration ---
    directions = [0, 60, 120, 180, 240, 300]
    graphs = build_wake_graphs(
        coords,
        directions,
        include_weights=False,
        decay_length=1000.0,
        angle_threshold=20.0,
        max_distance=2000.0
    )

    # Plot for a sanity check
    plot_wake_graph(graphs[directions[0]])
    plot_wake_graphs(graphs, cols=3)

    # Print edges for the first direction
    e_idx = graphs[directions[0]].edge_index.numpy()
    e_wts = (graphs[directions[0]].edge_attr.numpy()
             if graphs[directions[0]].edge_attr is not None else None)

    print(f"Edges for direction {directions[0]}:")
    for i in range(e_idx.shape[1]):
        src, dst = e_idx[0, i], e_idx[1, i]
        if e_wts is not None:
            w = e_wts[i]
            print(f"Edge from node {src} to node {dst} with weight {w:.3f}")
        else:
            print(f"Edge from node {src} to node {dst}")
