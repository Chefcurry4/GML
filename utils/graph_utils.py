import numpy as np
import argparse
from torch_geometric.data import Data
from utils.dominant_dir import get_dominant_for_window
from utils.static_graphs import build_wake_graphs


def _circular_diff(a: float, b: float) -> float:
    """
    Compute the smallest signed angular difference between angles a and b in degrees.
    """
    return abs((a - b + 180.0) % 360.0 - 180.0)


def select_closest_graph(graphs: dict, dom_dir: float) -> Data:
    """
    Choose the single static graph whose wind direction is closest to the dominant direction.

    Args:
        graphs (dict): Mapping from wind direction (deg) to PyG Data graph.
        dom_dir (float): Dominant wind direction (deg).

    Returns:
        Data: The graph corresponding to the closest static direction.
    """
    closest = min(graphs.keys(), key=lambda d: _circular_diff(d, dom_dir))
    print(f"Dominant direction: {dom_dir:.2f}°. Closest static graph: {closest}°.")
    return graphs[closest]


def interpolate_graphs(graphs: dict, dom_dir: float):
    """
    Select two static graphs bracketing the dominant direction, compute linear interpolation weights.

    Returns:
        dir_low, g_low, w_low, dir_high, g_high, w_high
    """
    dirs = sorted(graphs.keys())
    idx = np.searchsorted(dirs, dom_dir)
    low_idx = (idx - 1) % len(dirs)
    high_idx = idx % len(dirs)
    dir_low = dirs[low_idx]
    dir_high = dirs[high_idx]

    # Compute angular span correctly, considering circular wrap
    span = (_circular_diff(dir_high, dir_low) or 360.0)
    delta_low = _circular_diff(dom_dir, dir_low)

    w_high = delta_low / span
    w_low = 1.0 - w_high

    print(f"Dominant: {dom_dir:.2f}°. Interpolating between {dir_low}° (w={w_low:.2f}) and {dir_high}° (w={w_high:.2f}).")
    return dir_low, graphs[dir_low], w_low, dir_high, graphs[dir_high], w_high


def combine_graphs(graphs: dict, dom_dir: float) -> Data:
    """
    Build a combined graph by linearly interpolating two static graphs to match the dominant direction.

    Args:
        graphs (dict): Static graphs keyed by wind direction.
        dom_dir (float): Dominant wind direction (deg).

    Returns:
        Data: The combined PyG Data graph.
    """
    dir_low, g_low, w_low, dir_high, g_high, w_high = interpolate_graphs(graphs, dom_dir)

    # Initialize combined graph and weight features
    combined = Data()
    # Node features
    combined.x = w_low * g_low.x + w_high * g_high.x
    # Edge connectivity (assumed identical)
    combined.edge_index = g_low.edge_index
    # Edge attributes if present
    if hasattr(g_low, 'edge_attr') and g_low.edge_attr is not None:
        combined.edge_attr = w_low * g_low.edge_attr + w_high * g_high.edge_attr

    print(f"Combined graph: {dir_low}°*{w_low:.2f} + {dir_high}°*{w_high:.2f}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and combine wake graphs based on dominant wind direction.")
    parser.add_argument("--directions", nargs="+", type=float,
                        help="List of static wind directions (deg) to generate graphs, e.g. --directions 0 45 90 ...", required=True)
    parser.add_argument("--start", type=str, help="Start time (ISO format)", required=True)
    parser.add_argument("--end", type=str, help="End time (ISO format)", required=True)
    args = parser.parse_args()

    # Generate static graphs
    graphs = build_wake_graphs(args.directions)
    # Compute dominant direction
    dom_dir = get_dominant_for_window(args.start, args.end)
    # Combine accordingly
    combined_graph = combine_graphs(graphs, dom_dir)

    # The combined_graph is ready for downstream use (saving, visualization, training, etc.)
