# wind_power_forecasting/utils.py

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
import os
from config import INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, INPUT_FEATURES, TARGET_FEATURE, FEATURES_TO_NORMALIZE, MISSING_VALUE_PLACEHOLDER_GNN
import networkx as nx
import matplotlib.pyplot as plt

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
    edge_attr: torch.Tensor = None,
    save_path: str = "graph.png",
    node_size: int = 30,
    node_color: str = "blue",
    edge_color: str = "gray",
    edge_width: float = 0.5,
    dpi: int = 500
):
    """
    Visualize and save a 2D plot of the spatial graph.
    Args:
        edge_index: torch.Tensor of shape (2, num_edges)
        locations: np.ndarray of shape (num_nodes, 2) with x,y coords
        edge_attr: torch.Tensor of shape (num_edges, 1) or None
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
    # Convert edge_attr to flat numpy array if provided
    if edge_attr is not None:
        ea = edge_attr.cpu().numpy().flatten() if isinstance(edge_attr, torch.Tensor) else edge_attr.flatten()
        for (u, v), w in zip(zip(ei[0], ei[1]), ea):
            G.add_edge(int(u), int(v), weight=float(w))
    else:
        for u, v in zip(ei[0], ei[1]):
            G.add_edge(int(u), int(v))

    # Extract positions
    pos = nx.get_node_attributes(G, 'pos')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, ax=ax)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Spatial graph saved to {save_path}")