from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Args:
    spatial_graph_type: str
    model_type: str
    force_retrain: Optional[bool] = False
    data_subset_turbines: Optional[int] = -1
    data_subset: Optional[float] = 1

def parse_args() -> Args:
    import argparse
    parser = argparse.ArgumentParser()

    # Spatial graph type
    parser.add_argument('spatial_graph_type', type=str, choices=['radius', 'knn'], help='Spatial graph type: radius or knn')

    # Model type
    parser.add_argument('model_type', type=str, choices=['gcn', 'fast-gcn', 'cluster-gcn'], help='Model type: gcn, fast-gcn, cluster-gcn')

    # Data subset for turbines
    parser.add_argument('--data-subset-turbines', type=int, default=-1, help='Number of turbines used from the training set (default: -1 to use all)')

    # Data subset for time days
    parser.add_argument('--data-subset', type=float, help='Percentage of the dataset to use for training and validation (e.g. 0.2 for 20% of the data)', default=1.0)

    # Force retrain flag
    parser.add_argument('--force-retrain', action='store_true', default=False, help='Force retrain the model even if a checkpoint exists')

    parsed = parser.parse_args()

    # Print a summary of the arguments
    print("Settings:")
    print(f"  Spatial graph type: {parsed.spatial_graph_type}")
    print(f"  Model type: {parsed.model_type}")
    print(f"  Force retrain: {parsed.force_retrain}")
    print(f"  Data subset turbines: {parsed.data_subset_turbines}")
    print(f"  Data subset percentage: {parsed.data_subset}")

    return Args(
        spatial_graph_type=parsed.spatial_graph_type,
        model_type=parsed.model_type,
        force_retrain=parsed.force_retrain,
        data_subset_turbines=parsed.data_subset_turbines,
        data_subset=parsed.data_subset
    )