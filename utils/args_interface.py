from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

# Constants and enums
class GRAPH_TYPE(Enum):
    KNN = "knn"
    RADIUS = "radius"
    DOMDIR = "domdir"

@dataclass
class Args:
    spatial_graph_type: str
    model_type: str
    force_retrain: Optional[bool] = False
    data_subset_turbines: Optional[int] = -1
    data_subset: Optional[float] = 1
    epochs: int = 40
    dropout_rate: float = 0.35
    hidden_dimensions: int = 128
    batch_size: int = 32
    patience: float = 10
    learning_rate: float = 0.001
    knn_neighbors: int = 5
    spatial_radius: int = 1500
    plot_images: bool = False

def parse_args() -> Args:
    import argparse
    parser = argparse.ArgumentParser()

    # Spatial graph type
    parser.add_argument('spatial_graph_type', type=str, choices=[g.value for g in GRAPH_TYPE], help='Spatial graph type')

    # Model type
    parser.add_argument('model_type', type=str, choices=['gcn', 'fast-gcn', 'cluster-gcn'], help='Model type: gcn, fast-gcn, cluster-gcn')

    # Data subset for turbines
    parser.add_argument('--data-subset-turbines', type=int, default=-1, help='Number of turbines used from the training set (default: -1 to use all)')

    # Data subset for time days
    parser.add_argument('--data-subset', type=float, help='Percentage of the dataset to use for training and validation (e.g. 0.2 for 20% of the data)', default=1.0)

    # Force retrain flag
    parser.add_argument('--force-retrain', action='store_true', default=False, help='Force retrain the model even if a checkpoint exists')

    # Options for training the model
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train the model (default: 40)')
    parser.add_argument('--dropout-rate', type=float, default=0.35, help='Dropout rate for the model (default: 0.35)')
    parser.add_argument('--hidden-dimensions', type=int, default=128, help='Number of hidden dimensions in the model (default: 128)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--patience', type=int, default=10, help='Patience for the cluster GCN (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001)')

    # knn neighbors
    parser.add_argument('--knn-neighbors', type=int, default=5, help='Number of neighbors for KNN graph; when knn is selected (default: 5)')

    # spatial radius
    parser.add_argument('--spatial-radius', type=int, default=1500, help='Radius for spatial graph; when radius is selected (default: 1500)')

    # Flag to plot images
    parser.add_argument('--plot-images', action='store_true', default=False, help='Plot images during training')

    parsed = parser.parse_args()

    # Set graph type correctly
    parsed.spatial_graph_type = GRAPH_TYPE(parsed.spatial_graph_type)

    # Print a summary of the arguments
    print("Settings:")
    print(f"  Spatial graph type: {parsed.spatial_graph_type.value}")
    print(f"  Model type: {parsed.model_type}")
    print(f"  Force retrain: {parsed.force_retrain}")
    print(f"  Data subset turbines: {parsed.data_subset_turbines}")
    print(f"  Data subset percentage: {parsed.data_subset}")
    print(f"  Epochs: {parsed.epochs}")
    print(f"  Dropout rate: {parsed.dropout_rate}")
    print(f"  Hidden dimensions: {parsed.hidden_dimensions}")
    print(f"  Batch size: {parsed.batch_size}")
    print(f"  Patience: {parsed.patience}")
    print(f"  Learning rate: {parsed.learning_rate}")
    print(f"  KNN neighbors: {parsed.knn_neighbors}")
    print(f"  Spatial radius: {parsed.spatial_radius}")
    print(f"  Plot images: {parsed.plot_images}")

    return Args(
        spatial_graph_type=parsed.spatial_graph_type,
        model_type=parsed.model_type,
        force_retrain=parsed.force_retrain,
        data_subset_turbines=parsed.data_subset_turbines,
        data_subset=parsed.data_subset,
        epochs=parsed.epochs,
        dropout_rate=parsed.dropout_rate,
        hidden_dimensions=parsed.hidden_dimensions,
        batch_size=parsed.batch_size,
        patience=parsed.patience,
        learning_rate=parsed.learning_rate,
        knn_neighbors=parsed.knn_neighbors,
        spatial_radius=parsed.spatial_radius,
        plot_images=parsed.plot_images
    )
