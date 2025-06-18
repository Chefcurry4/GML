# wind_power_forecasting/missing_data_handling.py

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, identity
from scipy.sparse.linalg import spsolve
import torch
from utils.graph_construction import build_spatial_graph, build_temporal_graph, build_product_graph
from config import SPATIAL_GRAPH_TYPE, SPATIAL_RADIUS, K_NEIGHBORS, TEMPORAL_GRAPH_TYPE, INPUT_FEATURES, TARGET_FEATURE, TIKHONOV_LAMBDA, MISSING_VALUE_PLACEHOLDER_GNN
import time

def tikhonov_interpolation_product_graph(data_df, missing_mask_df, location_df, lambda_reg=TIKHONOV_LAMBDA):
    """
    Performs Tikhonov regularization based interpolation on the product graph.
    Interpolates missing values for ALL features simultaneously, feature-by-feature.
    This is a static interpolation method applied BEFORE forecasting.

    Args:
        data_df (pd.DataFrame): DataFrame with time series data (Time index, MultiIndex columns (Feature, TurbineID)).
        missing_mask_df (pd.DataFrame): Boolean DataFrame indicating missing values (NaN).
        location_df (pd.DataFrame): DataFrame with turbine locations (sorted by TurbID).
        lambda_reg (float): Regularization parameter.

    Returns:
        pd.DataFrame: DataFrame with interpolated values. Returns original df if graph cannot be built.
    """
    print(f"Starting Tikhonov interpolation with lambda={lambda_reg}...")
    start_time = time.time()

    num_time_steps = data_df.shape[0]
    turbine_ids_sorted = sorted(data_df.columns.get_level_values(1).unique())
    num_turbines = len(turbine_ids_sorted)
    num_features_per_turbine = len(INPUT_FEATURES)

    if num_time_steps <= 1 or num_turbines == 0:
        print("Cannot perform Tikhonov interpolation: Insufficient time steps or turbines.")
        return data_df.copy()

    # Build the spatial graph once (num_turbines nodes).
    temp_spatial_graph, _ = build_spatial_graph(location_df, SPATIAL_GRAPH_TYPE, SPATIAL_RADIUS, K_NEIGHBORS)
    # Build the temporal graph for the entire time series (num_time_steps nodes).
    temp_temporal_graph = build_temporal_graph(num_time_steps, TEMPORAL_GRAPH_TYPE)

    if temp_spatial_graph.edge_index.numel() == 0 and temp_temporal_graph.edge_index.numel() == 0:
         print("Warning: Both spatial and temporal graphs are empty. Tikhonov regularization requires graph structure. Skipping interpolation.")
         return data_df.copy()

    # Build product graph for (num_turbines * num_time_steps) nodes.
    # Node indexing: time_step * num_turbines + turbine_id
    full_product_edge_index, _ = build_product_graph(
        temp_spatial_graph,
        temp_temporal_graph,
        num_turbines,
        num_time_steps # Window size is the full time series length here
    )

    num_product_nodes = num_turbines * num_time_steps
    print(f"Full product graph built: {num_product_nodes} nodes, {full_product_edge_index.shape[1]} edges.")

    if full_product_edge_index.numel() == 0:
         print("Warning: Full product graph is empty. Tikhonov regularization requires graph structure. Skipping interpolation.")
         return data_df.copy()


    # Build the Laplacian matrix for the product graph
    # Using sparse matrix for efficiency
    adj_matrix_sparse = lil_matrix((num_product_nodes, num_product_nodes), dtype=float)
    # Populate adjacency matrix from edge_index
    edge_list = full_product_edge_index.t().tolist()
    # Ensure no duplicate edges if graph construction added them (shouldn't for simple line/undirected spatial)
    # Convert list of edges to set of tuples then back to list if necessary, but lil_matrix handles duplicates by summing (we want 1)
    adj_matrix_sparse[list(zip(*full_product_edge_index.tolist()))] = 1.0


    degree_matrix_sparse = lil_matrix((num_product_nodes, num_product_nodes), dtype=float)
    # Calculate degrees for the Laplacian (assuming unnormalized Laplacian L = D - A)
    degrees = adj_matrix_sparse.sum(axis=1)
    degree_matrix_sparse.setdiag(degrees.flatten())

    laplacian_sparse = degree_matrix_sparse - adj_matrix_sparse # Combinatorial Laplacian

    # Prepare the known values vector `y` and the masking matrix `M`
    # Reshape the data and mask to (num_time_steps * num_turbines, num_features_per_turbine)
    # Order should match product graph nodes: (t0, u0), (t0, u1), ..., (t1, u0), ...
    # Original data_df index is time, columns are (Feature, TurbID).
    # Pivot table columns are already (Feature, TurbID).
    # Need to reorder columns to match product graph node structure:
    # Time index rows, columns ordered (Feature, TurbID) -> (t0, feat1, turb1), (t0, feat2, turb1)...(t0, feat1, turbN)...
    # No, node is (t, u). Need features per node.
    # We need shape (num_product_nodes, num_features_per_turbine)
    # Rows should be ordered (t0, u0), (t0, u1), ..., (t0, uN-1), (t1, u0), ...
    # Columns are the features for that node.

    # Dataframe columns are (Feature, TurbID). Let's reorder columns to (TurbID, Feature) first.
    reordered_cols_by_turb = [(turb_id, feature) for turb_id in turbine_ids_sorted for feature in INPUT_FEATURES]
    data_reordered_by_turb = data_df[reordered_cols_by_turb] # Shape: (Time, Turbines * Features)
    mask_reordered_by_turb = missing_mask_df[reordered_cols_by_turb] # Shape: (Time, Turbines * Features)

    # Reshape from (Time, Turbines * Features) to (Time * Turbines, Features)
    # The rows will be ordered (t0, (u0 features)), (t0, (u1 features))... (t1, (u0 features))...
    # This matches the t * num_turbines + u node indexing.
    data_flat_nodes = data_reordered_by_turb.values.reshape(num_time_steps * num_turbines, num_features_per_turbine)
    mask_flat_nodes = mask_reordered_by_turb.values.reshape(num_time_steps * num_turbines, num_features_per_turbine)


    interpolated_data_flat_nodes = np.empty_like(data_flat_nodes)

    # Apply Tikhonov interpolation feature-by-feature
    for feature_idx, feature_name in enumerate(INPUT_FEATURES):
        # print(f"  Interpolating feature: {feature_name}...")
        feature_values = data_flat_nodes[:, feature_idx] # Shape (num_product_nodes,)
        feature_mask = mask_flat_nodes[:, feature_idx] # Shape (num_product_nodes,), True for missing

        # Known values vector y_k (only non-missing values)
        y_k = feature_values[~feature_mask] # Shape (num_known_values,)

        if len(y_k) == 0:
             print(f"Warning: Feature '{feature_name}' has no known values. Cannot interpolate. Filling with placeholder.")
             interpolated_data_flat_nodes[:, feature_idx] = MISSING_VALUE_PLACEHOLDER_GNN # Or 0, or mean?
             continue
        if len(y_k) == num_product_nodes:
            print(f"Feature '{feature_name}' has no missing values. Skipping interpolation for this feature.")
            interpolated_data_flat_nodes[:, feature_idx] = feature_values
            continue


        # Masking matrix M (selects known values)
        # M is a sparse matrix of shape (num_known_values, num_product_nodes)
        num_known_values = len(y_k)
        M_lil = lil_matrix((num_known_values, num_product_nodes), dtype=float)
        known_indices = np.where(~feature_mask)[0] # Indices in the flat vector corresponding to known values
        M_lil[np.arange(num_known_values), known_indices] = 1.0 # Create rows [0,0,1,0..], [0,0,0,0,1,0...]

        # Solve (M^T M + lambda * L) x = M^T y
        # M^T M is a diagonal matrix where diagonal is 1 for known indices, 0 for missing
        MTM_lil = lil_matrix((num_product_nodes, num_product_nodes), dtype=float)
        MTM_lil.setdiag((~feature_mask).astype(float)) # Diagonal is 1 where value is known, 0 where missing

        # Right-hand side: M^T y_k
        MT_y_k = M_lil.T @ y_k

        # System matrix A = M^T M + lambda * L
        A_sparse = MTM_lil + lambda_reg * laplacian_sparse

        # Solve for x (the interpolated values)
        # Convert A to CSR or CSC for spsolve
        A_csr = A_sparse.tocsr()
        # Check if A_csr is singular or ill-conditioned (optional but good practice)
        # spsolve might raise SingularMatrixError

        try:
            x_interpolated = spsolve(A_csr, MT_y_k)
            # Replace original known values with themselves (should be close but numerical error)
            # and use interpolated values for the missing points.
            # Or, just trust spsolve output? It's solving for the *entire* vector x.
            # A more robust way: fill missing entries in the original vector
            feature_values_interp = feature_values.copy()
            missing_indices = np.where(feature_mask)[0]
            feature_values_interp[missing_indices] = x_interpolated[missing_indices] # Use the interpolated values for missing spots

            interpolated_data_flat_nodes[:, feature_idx] = feature_values_interp

        except Exception as e: # Catch potential solver errors
            print(f"Error solving Tikhonov system for feature '{feature_name}': {e}")
            print("Interpolating missing values with mean/median instead for this feature.")
            # Fallback: simple interpolation for this feature
            if feature_mask.sum() > 0:
                 # Calculate mean/median from known values
                 fill_value = np.nanmean(feature_values) # Use nanmean to ignore NaNs
                 if np.isnan(fill_value): fill_value = 0.0 # Fallback if all values were NaN
                 interpolated_feature_values = feature_values.copy()
                 interpolated_feature_values[feature_mask] = fill_value
                 interpolated_data_flat_nodes[:, feature_idx] = interpolated_feature_values
            else:
                 interpolated_data_flat_nodes[:, feature_idx] = feature_values # No missing values

    end_time = time.time()
    print(f"Tikhonov interpolation finished in {end_time - start_time:.2f} sec.")

    # Reshape back to original data_df structure
    # From (num_time_steps * num_turbines, num_features)
    # to (num_time_steps, num_turbines * num_features) matching reordered_cols_by_turb
    interpolated_flat_reordered_by_turb = interpolated_data_flat_nodes.reshape(num_time_steps, num_turbines * num_features_per_turbine)

    # Create new DataFrame with original time index and reordered columns
    interpolated_df_reordered = pd.DataFrame(
        interpolated_flat_reordered_by_turb,
        index=data_df.index,
        columns=pd.MultiIndex.from_tuples(reordered_cols_by_turb, names=['TurbineID', 'Feature'])
    )

    # Reorder columns back to original (Feature, TurbineID) structure
    original_col_order = data_df.columns
    interpolated_df = interpolated_df_reordered.stack(level=0).unstack(level=0).swaplevel(axis=1) # Swap (TurbID, Feature) to (Feature, TurbID)
    interpolated_df.columns.names = ['Feature', 'TurbineID'] # Restore column names
    # Ensure final column order matches original
    interpolated_df = interpolated_df[original_col_order]


    # Verify no NaNs remain (unless fallback failed or original data was all NaN)
    if interpolated_df.isna().sum().sum() > 0:
         print("Warning: NaNs still present after Tikhonov interpolation.")

    return interpolated_df


def simple_interpolate(data_df, method='mean'):
    """
    Applies simple interpolation methods to the data.
    Applied column-wise (per-turbine per-feature).

    Args:
        data_df (pd.DataFrame): DataFrame with time series data (Time index, MultiIndex columns (Feature, TurbineID)).
        method (str): 'remove', 'mean', 'median', 'ffill', 'bfill'.

    Returns:
        pd.DataFrame: DataFrame after interpolation or removal.
    """
    print(f"Starting simple interpolation using method: '{method}'...")
    interpolated_df = data_df.copy()

    if method == 'remove':
        # Find rows with any missing values across all columns and drop them
        initial_rows = len(interpolated_df)
        interpolated_df = interpolated_df.dropna(how='any')
        print(f"Removed rows with missing values. Original rows: {initial_rows}, Remaining rows: {len(interpolated_df)}")
        # Note: This changes the time index and may create gaps.
    elif method in ['mean', 'median']:
        # Impute missing values column-wise (per-turbine per-feature)
        # Calculate fill value per column ignoring existing NaNs
        fill_values = interpolated_df.mean() if method == 'mean' else interpolated_df.median()
        interpolated_df = interpolated_df.fillna(fill_values)
        print(f"Filled missing values using column {method}.")
    elif method in ['ffill', 'bfill']:
        # Forward or backward fill column-wise
        interpolated_df = interpolated_df.fillna(method=method)
        # Handle any remaining NaNs (e.g., at the start for ffill or end for bfill, or if a whole column is NaN)
        # Fill remaining with 0 or mean as a final fallback
        if interpolated_df.isna().sum().sum() > 0:
             print(f"Warning: NaNs remaining after {method} and inverse fill. Filling with 0.")
             interpolated_df = interpolated_df.fillna(0) # Fallback fill
        print(f"Filled missing values using {method}.")
    else:
        raise ValueError(f"Unknown simple interpolation method: {method}")

    # Verify no NaNs remain (unless method was 'remove')
    if method != 'remove' and interpolated_df.isna().sum().sum() > 0:
         print(f"Warning: NaNs still present after simple interpolation method '{method}'.")

    return interpolated_df 