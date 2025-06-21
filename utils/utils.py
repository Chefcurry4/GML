# wind_power_forecasting/utils.py

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
import os
import networkx as nx
import matplotlib.pyplot as plt
import csv
from datetime import datetime

def load_data(scada_path, location_path):
    """Loads SCADA and location data."""
    if not os.path.exists(scada_path):
        raise FileNotFoundError(f"SCADA data not found at {scada_path}")
    if not os.path.exists(location_path):
         raise FileNotFoundError(f"Location data not found at {location_path}")

    scada_df = pd.read_csv(scada_path)
    location_df = pd.read_csv(location_path)
    return scada_df, location_df

def preprocess_scada_data(df, input_features):
    """Cleans and structures SCADA data."""
    # Create Tm column from Day and Tmstamp if it doesn't exist
    if 'Tm' not in df.columns and 'Day' in df.columns and 'Tmstamp' in df.columns:
        # Convert Day to timedelta and add to a base date
        base_date = pd.Timestamp('2020-01-01')  # Using an arbitrary base date
        day_delta = pd.to_timedelta(df['Day'] - 1, unit='D')
        
        # Convert HH:MM to timedelta by adding ':00' for seconds
        time_delta = pd.to_timedelta(df['Tmstamp'] + ':00')
        
        # Combine both into datetime
        df['Tm'] = base_date + day_delta + time_delta
        
        # Drop the original Day and Tmstamp columns
        df = df.drop(['Day', 'Tmstamp'], axis=1)
    elif 'Tm' in df.columns:
        df['Tm'] = pd.to_datetime(df['Tm'])
    else:
        raise ValueError("Data must contain either 'Tm' column or both 'Day' and 'Tmstamp' columns")

    # Ensure all turbines are present if subsetting was applied upstream
    # The pivot operation naturally handles the turbines present in the df
    turbine_ids = sorted(df['TurbID'].unique())

    # Pivot data to have turbines as columns, time as index
    # Using selected input features
    pivot_df = df.pivot_table(index='Tm', columns='TurbID', values=input_features)

    # Handle multi-level columns resulting from pivot
    # Ensure consistent column order like [('Feature1', TurbID1), ('Feature2', TurbID1), ..., ('FeatureN', TurbID_Last), ...]
    # Pivot table might not preserve exact order, explicitly reorder
    expected_cols = pd.MultiIndex.from_product([input_features, turbine_ids], names=['Feature', 'TurbineID'])
    pivot_df = pivot_df[expected_cols]

    return pivot_df, turbine_ids, input_features

def identify_missing_data(df):
    """Creates a boolean mask for missing values (NaN)."""
    return df.isna()

class TimeSeriesSlidingWindowDataset(Dataset):
    """Custom Dataset for sliding window time series data."""
    def __init__(self, data_df, missing_mask_df, input_sequence_length, output_sequence_length):
        """
        Args:
            data_df (pd.DataFrame): DataFrame with time series data (Time index, MultiIndex columns (Feature, TurbineID)).
            missing_mask_df (pd.DataFrame): Boolean mask DataFrame of the same shape.
            input_sequence_length (int): Length of the input window.
            output_sequence_length (int): Length of the output window.
        """
        self.data_df = data_df
        self.missing_mask_df = missing_mask_df
        self.input_seq_len = input_sequence_length
        self.output_seq_len = output_sequence_length

        # Calculate number of possible starting points for windows
        # A window of size input_seq_len starts at index i and ends at i + input_seq_len - 1.
        # The corresponding output window starts at i + input_seq_len and ends at i + input_seq_len + output_seq_len - 1.
        # The last possible starting index 'i' is when i + input_seq_len + output_seq_len - 1 is the last index of the dataframe.
        self.num_samples = len(self.data_df) - self.input_seq_len - self.output_seq_len + 1
        if self.num_samples <= 0:
             raise ValueError(f"Data is too short for the specified window sizes. Need at least {input_sequence_length + output_sequence_length} time steps, got {len(self.data_df)}.")


        # Store turbine IDs and feature names for potential use in collate_fn/evaluation
        self.turbine_ids = sorted(self.data_df.columns.get_level_values(1).unique())
        self.feature_names = list(self.data_df.columns.get_level_values(0).unique())
        self.num_turbines = len(self.turbine_ids)
        self.num_features_per_turbine = len(self.feature_names)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the window slice from the dataframes
        input_slice_df = self.data_df.iloc[idx : idx + self.input_seq_len]
        input_mask_slice_df = self.missing_mask_df.iloc[idx : idx + self.input_seq_len]
        output_slice_df = self.data_df.iloc[idx + self.input_seq_len : idx + self.input_seq_len + self.output_seq_len]
        output_mask_slice_df = self.missing_mask_df.iloc[idx + self.input_seq_len : idx + self.input_seq_len + self.output_seq_len]

        # Convert dataframes to torch tensors.
        # Shape: (SeqLen, Features * Turbines) -> Reshape later in collate or model?
        # Let's keep this shape (SeqLen, N_turbines * N_features) for now.
        # The GNN collate function will reshape to (SeqLen * N_turbines, N_features) per window.

        input_tensor = torch.FloatTensor(input_slice_df.values) # (SeqLen, N_turbines * N_features)
        input_mask_tensor = torch.BoolTensor(input_mask_slice_df.values) # (SeqLen, N_turbines * N_features)
        output_tensor = torch.FloatTensor(output_slice_df.values) # (OutSeqLen, N_turbines * N_features)
        output_mask_tensor = torch.BoolTensor(output_mask_slice_df.values) # (OutSeqLen, N_turbines * N_features)


        return {
            'input': input_tensor,
            'input_mask': input_mask_tensor,
            'output': output_tensor,
            'output_mask': output_mask_tensor,
            'window_start_time_idx': idx # Keep track of the start index of this window
        }


def create_dataloaders(full_data_df, full_missing_mask_df, input_seq_len, output_seq_len, train_ratio, val_ratio, test_ratio, batch_size, collate_fn=None):
    """Creates train, validation, and test DataLoaders."""
    full_dataset = TimeSeriesSlidingWindowDataset(
        full_data_df,
        full_missing_mask_df,
        input_seq_len,
        output_seq_len
    )

    total_samples = len(full_dataset)
    if total_samples == 0:
        raise ValueError("No valid windows can be created from the data.")

    # Ensure split sums to 1 or less, with a buffer
    if train_ratio + val_ratio + test_ratio > 1.001: # Allow small floating point error
         raise ValueError("Train, val, test ratios sum to more than 1.")

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    # Ensure test_size takes remaining, handle potential off-by-one
    test_size = total_samples - train_size - val_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
         print(f"Warning: Train ({train_size}), val ({val_size}), or test ({test_size}) size is zero or less. Adjust split ratios or data size.")


    # Create Subsets for splitting the dataset by window index
    # The indices here refer to the indices returned by the full_dataset __getitem__
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn) # Don't drop_last for test

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset # Return datasets too for scaler fitting


def get_time_steps_for_indices(data_df, indices):
    """Gets the actual time steps from a dataframe for a list of row indices."""
    if not indices:
        return pd.DatetimeIndex([])
    return data_df.index[indices]

def inverse_normalize_target(scaled_data, scaler_dict, original_turbine_ids, target_feature_name):
    """
    Applies inverse normalization to scaled target feature data.
    Args:
        scaled_data (np.ndarray): Scaled target data, shape (TimeSteps * OutSeqLen, num_turbines).
        scaler_dict (dict): Dictionary of scalers, {feature: {col_name: scaler_obj}}.
        original_turbine_ids (list): List of turbine IDs matching columns in scaled_data.
        target_feature_name (str): Name of the target feature.
    Returns:
        np.ndarray: Unscaled data, shape (TimeSteps * OutSeqLen, num_turbines).
    """
    unscaled_data = np.empty_like(scaled_data)
    num_turbines = scaled_data.shape[1]

    if target_feature_name not in scaler_dict:
         print(f"Warning: Scalers not found for target feature '{target_feature_name}'. Returning original scaled data.")
         return scaled_data

    feature_scalers = scaler_dict[target_feature_name]

    for i, turb_id in enumerate(original_turbine_ids):
        col_name = (target_feature_name, turb_id)
        if col_name in feature_scalers:
            s = feature_scalers[col_name]
            # Inverse transform column by column
            unscaled_data[:, i] = s.inverse_transform(scaled_data[:, i].reshape(-1, 1)).flatten()
        else:
            # Scaler not found for this turbine/feature combination, keep original value
            unscaled_data[:, i] = scaled_data[:, i]
            print(f"Warning: Scaler not found for feature '{target_feature_name}' turbine {turb_id}. Column not inverse normalized.")

    return unscaled_data 



def visualize_spatial_graph(
    edge_index: torch.Tensor,
    locations: np.ndarray,
    save_path: str = "graph.png",
    node_size: int = 30,
    node_color: str = "blue",
    edge_color: str = "gray",
    edge_width: float = 0.5,
    dpi: int = 500,
    draw_node_number: bool = False
):
    """
    Visualize and save a 2D plot of the spatial graph.
    Args:
        edge_index: torch.Tensor of shape (2, num_edges)
        locations: np.ndarray of shape (num_nodes, 2) with x,y coords
        save_path: path to save the image
        node_size: size of nodes
        node_color: color of nodes
        edge_color: color of edges
        edge_width: width of edges
        dpi: resolution of saved figure
    """

    # Build NetworkX graph
    G = nx.Graph()
    # Add nodes with position attribute
    for i, (x, y) in enumerate(locations):
        G.add_node(i, pos=(float(x), float(y)))

    # Convert edge_index to numpy
    ei = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    for u, v in zip(ei[0], ei[1]):
        G.add_edge(int(u), int(v))

    # Extract positions
    pos = nx.get_node_attributes(G, 'pos')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, ax=ax)
    if draw_node_number:
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', ax=ax)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Spatial graph saved to {save_path}")



def visualize_temporal_graph(
    temporal_edge_index: torch.Tensor,
    num_time_steps: int,
    save_path: str = "temporal_graph.png",
    node_size: int = 30,
    node_color: str = "blue",
    edge_color: str = "gray",
    edge_width: float = 0.5,
    dpi: int = 500
):
    """
    Visualize and save a 2D plot of the temporal graph.
    Each time step is shown as a column, with edges between time steps.
    Args:
        temporal_edge_index: torch.Tensor of shape (2, num_edges)
        num_time_steps: int, number of time steps (nodes)
        save_path: path to save the image
        node_size: size of nodes
        node_color: color of nodes
        edge_color: color of edges
        edge_width: width of edges
        dpi: resolution of saved figure
    """
    G = nx.DiGraph()
    # Add nodes for each time step
    for t in range(num_time_steps):
        G.add_node(t, pos=(t, 0))  # Place nodes in a row

    # Add edges
    ei = temporal_edge_index.cpu().numpy() if isinstance(temporal_edge_index, torch.Tensor) else temporal_edge_index
    for u, v in zip(ei[0], ei[1]):
        G.add_edge(int(u), int(v))

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(num_time_steps, 2))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, arrows=True, arrowstyle='-|>')
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Temporal Graph")
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Temporal graph saved to {save_path}")



def visualize_spatio_temporal_graph(
    st_edge_index: torch.Tensor,
    locations: np.ndarray,
    N: int,
    T: int,
    time_offset: float = 1.0,
    save_path: str = "spatio_temporal_graph.png",
    node_size: int = 30,
    node_color: str = "blue",
    spatial_edge_color: str = "gray",
    temporal_edge_color: str = "red",
    spatial_edge_width: float = 0.5,
    temporal_edge_width: float = 0.2,
    dpi: int = 500
):
    """
    Visualize and save a 2D plot of the spatio‐temporal product graph.
    Each time‐layer is offset horizontally by `time_offset`. Spatial edges (within the same layer)
    are drawn in `spatial_edge_color`; temporal edges (connecting across layers) in `temporal_edge_color`.

    Args:
        st_edge_index:        torch.Tensor of shape (2, num_edges) for the product graph on N*T nodes.
        locations:            np.ndarray of shape (N, 2) with base x,y coordinates per spatial node.
        N:                    Number of spatial nodes (per time slice).
        T:                    Number of time layers.
        time_offset:          Horizontal shift applied to each time layer.
        save_path:            Path to save the resulting figure.
        node_size:            Size of each plotted node.
        node_color:           Color applied to all nodes.
        spatial_edge_color:   Color for edges within the same time layer.
        temporal_edge_color:  Color for edges between consecutive layers.
        spatial_edge_width:   Line‐width for spatial edges.
        temporal_edge_width:  Line‐width for temporal edges.
        dpi:                  Resolution (dots per inch) for the saved figure.
    """

    G = nx.Graph()
    num_nodes = N * T

    # 1) Add all nodes with 2D "pos" attribute: (x + t*time_offset, y).
    #    Node IDs run from 0 to N*T - 1; for node_id, spatial_idx = node_id % N, time_idx = node_id // N.
    for node_id in range(num_nodes):
        spatial_idx = node_id % N
        time_idx = node_id // N
        x, y = locations[spatial_idx]
        G.add_node(node_id, pos=(float(x) + time_idx * time_offset, float(y)))

    # 2) Separate spatial vs. temporal edges from st_edge_index
    ei = st_edge_index.cpu().numpy() if isinstance(st_edge_index, torch.Tensor) else st_edge_index
    all_edges = list(zip(ei[0].tolist(), ei[1].tolist()))

    spatial_edges = []
    temporal_edges = []
    for u, v in all_edges:
        t_u = u // N
        t_v = v // N
        if t_u == t_v:
            # Same time layer → spatial edge
            spatial_edges.append((int(u), int(v)))
        else:
            # Different time layer → temporal edge
            temporal_edges.append((int(u), int(v)))

    # 3) Extract positions
    pos = nx.get_node_attributes(G, 'pos')

    # 4) Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        ax=ax
    )
    if spatial_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=spatial_edges,
            edge_color=spatial_edge_color,
            width=spatial_edge_width,
            ax=ax
        )
    if temporal_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=temporal_edges,
            edge_color=temporal_edge_color,
            width=temporal_edge_width,
            ax=ax
        )

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Spatio temporal graph saved to {save_path}")

def log_train_results(args, num_epochs, total_time, best_val_loss, log_file_name="results.csv"):
    """
    Log training results to a CSV file. Creates the file if it doesn't exist.
    Appends a new row with all training parameters and results.
    
    Args:
        args: Arguments object containing all training settings
        num_epochs: int, number of epochs trained
        total_time: float, total training time in seconds
        best_val_loss: float, best validation loss achieved
        log_file_path: str, path to the CSV log file
    """
    output_dir = "output"
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file_path = os.path.join(output_dir, log_file_name)

    # Define the header columns
    fieldnames = [
        'timestamp',
        'spatial_graph_type',
        'model_type',
        'force_retrain',
        'data_subset_turbines',
        'data_subset',
        'epochs_requested',
        'epochs_trained',
        'dropout_rate',
        'hidden_dimensions',
        'batch_size',
        'patience',
        'learning_rate',
        'knn_neighbors',
        'spatial_radius',
        'total_time_seconds',
        'best_val_loss'
    ]
    
    # Prepare the row data
    row_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'spatial_graph_type': getattr(args, 'spatial_graph_type', 'N/A'),
        'model_type': getattr(args, 'model_type', 'N/A'),
        'force_retrain': getattr(args, 'force_retrain', False),
        'data_subset_turbines': getattr(args, 'data_subset_turbines', -1),
        'data_subset': getattr(args, 'data_subset', 1.0),
        'epochs_requested': getattr(args, 'epochs', 'N/A'),
        'epochs_trained': num_epochs,
        'dropout_rate': getattr(args, 'dropout_rate', 'N/A'),
        'hidden_dimensions': getattr(args, 'hidden_dimensions', 'N/A'),
        'batch_size': getattr(args, 'batch_size', 'N/A'),
        'patience': getattr(args, 'patience', 'N/A'),
        'learning_rate': getattr(args, 'learning_rate', 'N/A'),
        'knn_neighbors': getattr(args, 'knn_neighbors', 'N/A'),
        'spatial_radius': getattr(args, 'spatial_radius', 'N/A'),
        'total_time_seconds': round(total_time, 2),
        'best_val_loss': round(best_val_loss, 6)
    }
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(log_file_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else '.', exist_ok=True)
    
    # Write to CSV file
    with open(log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow(row_data)
    
    print(f"Training results logged to {log_file_path}")

def plot_training_curves(train_losses, val_losses, model_name, save_dir="plots"):
    """
    Plot and save training and validation loss curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        model_name: Name of the model for the plot title and filename
        save_dir: Directory to save the plot
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, color='orange', label='Validation Loss', linewidth=2)
    
    plt.title(f'{model_name} - Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name.lower().replace(' ', '_')}_loss_curves_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Loss curves saved to: {filepath}")
    
    plt.close()  # Close the figure to free memory

def plot_power_output(X_sample, Y_sample, turbine_ids, image_name="patv_plot.png", save_dir="images", patv_idx=10):
    """
    Plot the Patv (active power) attribute over time windows from X_train and Y_train.
    
    Args:
        X_sample: Single sample from X_train with shape (sliding_window_size * num_turbines, features_per_turbine)
        Y_sample: Single sample from Y_train with shape (num_turbines,) - flattened target
        turbine_ids: List of turbine indices to plot
        image_name: Name of the output image file
        save_dir: Directory to save the plot
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine dimensions from X_sample
    total_nodes = X_sample.shape[0]
    features_per_turbine = X_sample.shape[1]
    
    # Calculate number of turbines and time steps
    # From the flattened shape: total_nodes = sliding_window_size * num_turbines
    # We need to infer these from Y_sample shape and total_nodes
    num_turbines = Y_sample.shape[0]
    sliding_window_size = total_nodes // num_turbines
    
    # Reshape X_sample back to (sliding_window_size, num_turbines, features_per_turbine)
    X_reshaped = X_sample.reshape(sliding_window_size, num_turbines, features_per_turbine)
    
    # Extract Patv values for input sequence
    patv_input = X_reshaped[:, :, patv_idx]  # Shape: (sliding_window_size, num_turbines)
    
    # Create time axis
    input_time = list(range(sliding_window_size))
    output_time = [sliding_window_size]  # Next time step for prediction
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot for each requested turbine
    for turb_id in turbine_ids:
        if turb_id < num_turbines:
            # Plot input sequence
            plt.plot(input_time, patv_input[:, turb_id], 
                    label=f'Turbine {turb_id} (Input)', 
                    marker='o', markersize=4, linewidth=2)
            
            # Plot target value
            plt.plot(output_time, [Y_sample[turb_id]], 
                    marker='s', markersize=8, 
                    label=f'Turbine {turb_id} (Target)', 
                    linestyle='None')
            
            # Connect last input point to target with dashed line
            plt.plot([input_time[-1], output_time[0]], 
                    [patv_input[-1, turb_id], Y_sample[turb_id]], 
                    '--', alpha=0.5, color=plt.gca().lines[-2].get_color())
    
    # Customize plot
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Active Power (Patv)', fontsize=12)
    plt.title(f'Power Output Over Time - Sample Visualization', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line to separate input and prediction
    plt.axvline(x=sliding_window_size-0.5, color='red', linestyle=':', alpha=0.7, 
                label='Input/Prediction Boundary')
    
    # Add annotations
    plt.text(sliding_window_size/2, plt.ylim()[1]*0.95, 'Input Sequence', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(sliding_window_size, plt.ylim()[1]*0.95, 'Target', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, image_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_power_output_and_prediction(X_sample, Y_sample, Y_prediction, turbine_ids, image_name="patv_plot.png", save_dir="images", patv_idx=10):
    """
    Plot the Patv (active power) attribute over time windows from X_sample, Y_sample, and Y_prediction.
    
    Args:
        X_sample: Single sample from X_train with shape (sliding_window_size * num_turbines, features_per_turbine)
        Y_sample: Single sample from Y_train with shape (num_turbines,) - true target
        Y_prediction: Single sample of predictions with shape (num_turbines,)
        turbine_ids: List of turbine indices to plot
        image_name: Name of the output image file
        save_dir: Directory to save the plot
        patv_idx: Index of the Patv feature in the feature dimension
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine dimensions from X_sample
    total_nodes = X_sample.shape[0]
    features_per_turbine = X_sample.shape[1]
    
    # Calculate number of turbines and time steps
    num_turbines = Y_sample.shape[0]
    sliding_window_size = total_nodes // num_turbines
    
    # Reshape X_sample back to (sliding_window_size, num_turbines, features_per_turbine)
    X_reshaped = X_sample.reshape(sliding_window_size, num_turbines, features_per_turbine)
    
    # Extract Patv values for input sequence
    patv_input = X_reshaped[:, :, patv_idx]  # Shape: (sliding_window_size, num_turbines)
    
    # Create time axis
    input_time = list(range(sliding_window_size))
    output_time = [sliding_window_size]  # Next time step for prediction
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot for each requested turbine
    for turb_id in turbine_ids:
        if turb_id < num_turbines:
            # Plot input sequence
            plt.plot(input_time, patv_input[:, turb_id], 
                    label=f'Turbine {turb_id} (Input)', 
                    marker='o', markersize=4, linewidth=2)
            
            # Plot target value
            plt.plot(output_time, [Y_sample[turb_id]], 
                    marker='s', markersize=8, 
                    label=f'Turbine {turb_id} (Target)', 
                    linestyle='None', color='green')
            
            # Plot prediction value
            plt.plot(output_time, [Y_prediction[turb_id]], 
                    marker='^', markersize=8, 
                    label=f'Turbine {turb_id} (Prediction)', 
                    linestyle='None', color='orange')
            
            # Connect last input point to target with dashed line
            plt.plot([input_time[-1], output_time[0]], 
                    [patv_input[-1, turb_id], Y_sample[turb_id]], 
                    '--', alpha=0.5, color='green')
            
            # Connect last input point to prediction with dashed line
            plt.plot([input_time[-1], output_time[0]], 
                    [patv_input[-1, turb_id], Y_prediction[turb_id]], 
                    '--', alpha=0.5, color='orange')
    
    # Customize plot
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Active Power (Patv)', fontsize=12)
    plt.title(f'Power Output Over Time - Sample Visualization', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line to separate input and prediction
    plt.axvline(x=sliding_window_size-0.5, color='red', linestyle=':', alpha=0.7, 
                label='Input/Prediction Boundary')
    
    # Add annotations
    plt.text(sliding_window_size/2, plt.ylim()[1]*0.95, 'Input Sequence', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(sliding_window_size, plt.ylim()[1]*0.95, 'Target/Prediction', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, image_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_histogram(data, feature_idx, image_path="images", image_name="x_scaled_histogram.png", title="Histogram of Data"):
    patv_values = data[..., feature_idx].flatten()
    plt.figure(figsize=(8, 5))
    plt.hist(patv_values, bins=100, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Patv Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    save_path = os.path.join(image_path, image_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()