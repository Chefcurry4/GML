# wind_power_forecasting/main.py

import torch
import os
import pandas as pd
import numpy as np # Import numpy for nan handling
from utils import load_data, preprocess_scada_data, identify_missing_data, create_dataloaders, TimeSeriesSlidingWindowDataset # Import TimeSeriesSlidingWindowDataset
from graph_construction import build_spatial_graph, build_temporal_graph, build_product_graph
from missing_data_handling import simple_interpolate, tikhonov_interpolation_product_graph
from training import train_gru_model, train_gnn_model
from evaluation import evaluate_model, calculate_scalability_metrics, save_predictions
from config import * # Import all from config
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler
import time # For overall timing
from torch.utils.data import DataLoader, Subset # Import DataLoader and Subset
from torch_geometric.data import Batch # Import Batch from torch_geometric.data


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom collate function factory for PyG Batch (handles normalization and placeholders)
def gnn_collate_fn_factory(product_edge_index_template, product_edge_attr_template, num_turbines_in_batch, scalers_dict, features_to_norm_list, all_input_features_list, placeholder_val, use_mask_feat):
    # Renamed factory arguments for clarity and to avoid clashes if Pylance is confused.
    # p_edge_index -> product_edge_index_template
    # p_edge_attr -> product_edge_attr_template
    # num_t -> num_turbines_in_batch
    # scalers -> scalers_dict
    # features_to_normalize -> features_to_norm_list
    # input_features_list -> all_input_features_list
    # placeholder_value -> placeholder_val
    # use_mask_feature -> use_mask_feat

    def collate_fn_with_info(batch_list):
        # batch_list is a list of dictionaries from TimeSeriesSlidingWindowDataset.__getitem__()
        # Each item's tensors are shape (SeqLen, N_turbines * N_features)

        # Stack input tensors
        batch_input_raw_stacked = torch.stack([item['input'] for item in batch_list], dim=0) # (Batch, Seq, N_turbines * N_features)
        batch_input_mask_raw_stacked = torch.stack([item['input_mask'] for item in batch_list], dim=0) # (Batch, Seq, N_turbines * N_features), True is missing

        # Get other batch info (outputs and masks are for evaluation/loss, not GNN input features)
        batch_output_stacked = torch.stack([item['output'] for item in batch_list], dim=0) # (Batch, OutSeq, N_turbines * N_features)
        batch_output_mask_stacked = torch.stack([item['output_mask'] for item in batch_list], dim=0) # (Batch, OutSeq, N_turbines * N_features)
        # window_start_time_indices = torch.tensor([item['window_start_time_idx'] for item in batch_list], dtype=torch.long) # Not used in PyG Batch construction

        num_samples_in_this_batch = batch_input_raw_stacked.shape[0]
        current_seq_len_in_batch = batch_input_raw_stacked.shape[1] # Should be INPUT_SEQUENCE_LENGTH
        num_turbines_times_features_in_batch = batch_input_raw_stacked.shape[2] # N_turbines * N_features
        current_num_features_per_turbine_in_batch = num_turbines_times_features_in_batch // num_turbines_in_batch

        num_nodes_per_window_instance = num_turbines_in_batch * current_seq_len_in_batch # Nodes per product graph instance

        # --- Normalization and Placeholder Replacement ---
        # Get original dataset object to access turbine_ids and feature_names
        # This relies on the modified __getitem__ that adds '_dataset'
        original_dataset_ref = batch_list[0]['_dataset']
        turbine_ids_for_this_batch = original_dataset_ref.turbine_ids
        feature_names_for_this_batch = original_dataset_ref.feature_names

        # Reshape input data for normalization: (Batch, Seq, Turb, Feat)
        x_temp_reshaped_for_norm = batch_input_raw_stacked.view(
            num_samples_in_this_batch,
            current_seq_len_in_batch,
            num_turbines_in_batch,
            current_num_features_per_turbine_in_batch
        )
        normalized_x_temp_reshaped = torch.empty_like(x_temp_reshaped_for_norm)

        for turb_idx in range(num_turbines_in_batch):
            turb_id_actual = turbine_ids_for_this_batch[turb_idx]
            for feat_idx, feat_name in enumerate(feature_names_for_this_batch):
                col_name = (feat_name, turb_id_actual)
                if feat_name in features_to_norm_list and feat_name in scalers_dict and col_name in scalers_dict[feat_name]:
                    scaler = scalers_dict[feat_name][col_name]
                    col_data = x_temp_reshaped_for_norm[:, :, turb_idx, feat_idx].numpy()
                    scaled_col_data = scaler.transform(col_data.reshape(-1, 1)).reshape(col_data.shape)
                    normalized_x_temp_reshaped[:, :, turb_idx, feat_idx] = torch.FloatTensor(scaled_col_data)
                else:
                    normalized_x_temp_reshaped[:, :, turb_idx, feat_idx] = x_temp_reshaped_for_norm[:, :, turb_idx, feat_idx]

        # Flatten after normalization: (Batch * Seq * Turb, Features) = (TotalProductNodes, NumFeaturesPerTurbine)
        x_normalized_flattened_for_pyg = normalized_x_temp_reshaped.view(-1, current_num_features_per_turbine_in_batch)

        # Also flatten the raw mask for placeholder replacement and optional feature concatenation
        input_mask_flattened_for_pyg = batch_input_mask_raw_stacked.view(-1, current_num_features_per_turbine_in_batch)


        # Replace NaNs/masked values with the placeholder
        # The mask (input_mask_flattened_for_pyg) indicates where original NaNs were.
        # Normalization might turn NaNs into numbers or keep them NaNs.
        # We use the original mask to decide where to put placeholders.
        x_processed_flattened_for_pyg = x_normalized_flattened_for_pyg.clone()
        x_processed_flattened_for_pyg[input_mask_flattened_for_pyg] = placeholder_val


        # If using mask feature, concatenate it
        if use_mask_feat:
            mask_as_feature = input_mask_flattened_for_pyg.float() # True (missing) -> 1.0, False (not missing) -> 0.0
            x_processed_flattened_for_pyg = torch.cat([x_processed_flattened_for_pyg, mask_as_feature], dim=-1)
            # The GNN model's initial_input_dim should account for these extra features.

        # Create batch vector for PyG: [0,0,...,0, 1,1,...,1, ...] where 0, 1 are graph indices
        # Each graph has num_nodes_per_window_instance nodes.
        batch_vector_for_pyg = torch.arange(num_samples_in_this_batch, dtype=torch.long).repeat_interleave(num_nodes_per_window_instance)

        # Create batched edge_index and edge_attr
        # The product_edge_index_template and product_edge_attr_template are for a single window/graph.
        batched_edge_indices_list = []
        batched_edge_attrs_list = []
        for i in range(num_samples_in_this_batch):
            offset = i * num_nodes_per_window_instance # Offset node indices for each graph in the batch
            batched_edge_indices_list.append(product_edge_index_template + offset)
            batched_edge_attrs_list.append(product_edge_attr_template) # Attributes are the same for each window's graph structure

        final_batched_edge_index = torch.cat(batched_edge_indices_list, dim=1)
        final_batched_edge_attr = torch.cat(batched_edge_attrs_list, dim=0)


        # Create PyG Data object representing the batch
        pyg_batch_object = Batch(
            x=x_processed_flattened_for_pyg,    # Node features
            edge_index=final_batched_edge_index, # Graph connectivity
            edge_attr=final_batched_edge_attr,   # Edge features/attributes
            batch=batch_vector_for_pyg           # Batch assignment vector
        )
        # Add output targets and masks (not flattened, shape (Batch, OutSeq, N_turbines * N_features))
        pyg_batch_object.output = batch_output_stacked
        pyg_batch_object.output_mask = batch_output_mask_stacked
        pyg_batch_object.num_turbines = num_turbines_in_batch # Pass to model for reshaping predictions

        # Clean up the temporary _dataset reference from the batch items (optional, good practice)
        for item in batch_list:
            if '_dataset' in item:
                del item['_dataset']

        return pyg_batch_object

    return collate_fn_with_info


def run_experiment(
    model_type,
    interpolation_method, # 'remove', 'mean', 'median', 'ffill', 'bfill', 'tikhonov', 'joint'
    data_subset_turbines=None, # List of turbine IDs or None for all
    data_subset_time_days=None, # Number of days or None for all
    # Add parameter to control scaler fitting for baselines vs GNN
    fit_scaler_on_train_subset=True
):
    """Runs a single experiment with specified configuration."""
    exp_name = f"Model={model_type}_Interp={interpolation_method}"
    if data_subset_turbines is not None: exp_name += f"_Turbines={len(data_subset_turbines)}"
    if data_subset_time_days is not None: exp_name += f"_Time={data_subset_time_days}days"
    print(f"\n--- Running Experiment: {exp_name} ---")

    # --- 1. Load and Preprocess Data ---
    start_time_data = time.time()
    scada_df_orig, location_df_orig = load_data(SCADA_DATA_PATH, LOCATION_DATA_PATH)

    # Apply turbine subsetting
    if data_subset_turbines is not None:
        # Ensure location_df only contains specified turbines, sorted
        location_df = location_df_orig[location_df_orig['TurbID'].isin(data_subset_turbines)].sort_values('TurbID').reset_index(drop=True).copy()
        # Ensure scada_df only contains specified turbines
        scada_df = scada_df_orig[scada_df_orig['TurbID'].isin(data_subset_turbines)].copy()
        # Ensure turbine IDs in scada_df are sorted consistently for pivot
        scada_df['TurbID'] = pd.Categorical(scada_df['TurbID'], categories=sorted(data_subset_turbines), ordered=True)
        scada_df = scada_df.sort_values(['Tm', 'TurbID']).reset_index(drop=True)
        current_turb_ids = sorted(data_subset_turbines)
    else:
        location_df = location_df_orig.copy()
        scada_df = scada_df_orig.copy()
        # Get all turbine IDs from original location data for consistency
        current_turb_ids = sorted(location_df_orig['TurbID'].unique())
        # Filter location_df to just these IDs if needed (it should be all by default)
        location_df = location_df_orig[location_df_orig['TurbID'].isin(current_turb_ids)].sort_values('TurbID').reset_index(drop=True).copy()

    current_num_turbines = len(current_turb_ids)
    if current_num_turbines == 0:
         print("Error: No turbines selected or found after subsetting.")
         return


    # Apply time subsetting
    if data_subset_time_days is not None:
        start_date = scada_df['Tm'].min()
        end_date = start_date + pd.Timedelta(days=data_subset_time_days)
        scada_df = scada_df[scada_df['Tm'] < end_date].copy() # Use < to include start day but exclude end day + 1

    if scada_df.empty:
        print("Error: No SCADA data found after time subsetting.")
        return

    # Preprocess SCADA data (pivot)
    pivot_df_orig, _, _ = preprocess_scada_data(scada_df, INPUT_FEATURES)
    current_time_steps = len(pivot_df_orig)

    if current_time_steps < INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH:
         print(f"Error: Not enough time steps ({current_time_steps}) for window size ({INPUT_SEQUENCE_LENGTH}+{OUTPUT_SEQUENCE_LENGTH}). Skipping experiment.")
         return


    # Identify missing data BEFORE interpolation/handling
    missing_mask_df_orig = identify_missing_data(pivot_df_orig)

    # --- 2. Handle Missing Data based on Interpolation Method ---
    # This step prepares the data_for_model and missing_mask_for_model
    # based on the chosen interpolation method.
    if interpolation_method == 'joint':
        # For joint, the model handles NaNs/placeholders directly.
        # Data keeps original NaNs, missing_mask reflects original NaNs.
        data_for_model = pivot_df_orig.copy() # Contains NaNs
        missing_mask_for_model = missing_mask_df_orig.copy() # True where NaN originally
        print("Using joint missing data handling within the model.")
    elif interpolation_method in SIMPLE_INTERPOLATION_METHODS:
        # Simple interpolation baselines applied BEFORE dataset creation.
        # Returns dataframe with NaNs filled. Mask should be all False (no missing).
        interpolated_df = simple_interpolate(pivot_df_orig.copy(), method=interpolation_method)
        data_for_model = interpolated_df
        missing_mask_for_model = identify_missing_data(data_for_model) # Should be all False
        print(f"Data shape after '{interpolation_method}' interpolation: {data_for_model.shape}")
    elif interpolation_method == 'tikhonov':
        # Tikhonov interpolation on the full available data.
        # Returns dataframe with NaNs filled. Mask should be all False.
        # Need to pass location_df for graph construction within Tikhonov func.
        interpolated_df = tikhonov_interpolation_product_graph(pivot_df_orig.copy(), missing_mask_df_orig.copy(), location_df.copy(), TIKHONOV_LAMBDA)
        data_for_model = interpolated_df
        missing_mask_for_model = identify_missing_data(data_for_model) # Should be all False
        print(f"Data shape after '{interpolation_method}' interpolation: {data_for_model.shape}")
    elif interpolation_method == 'remove':
         # Simple row removal. Returns dataframe with potentially fewer rows and no NaNs.
         # Mask should be all False.
         interpolated_df = simple_interpolate(pivot_df_orig.copy(), method=interpolation_method)
         data_for_model = interpolated_df
         missing_mask_for_model = identify_missing_data(data_for_model) # Should be all False
         # Re-check if enough data remains for windowing
         if len(data_for_model) < INPUT_SEQUENCE_LENGTH + OUTPUT_SEQUENCE_LENGTH:
              print(f"Error: Not enough time steps ({len(data_for_model)}) remaining after removal for window size ({INPUT_SEQUENCE_LENGTH}+{OUTPUT_SEQUENCE_LENGTH}). Skipping experiment.")
              return
         print(f"Data shape after '{interpolation_method}' interpolation: {data_for_model.shape}")
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")


    # --- 3. Create Datasets and DataLoaders ---

    # Create the full dataset *before* splitting window indices
    # This dataset will contain the data_for_model (interpolated or raw with NaNs)
    full_dataset = TimeSeriesSlidingWindowDataset(
         data_for_model, # This df is the source, potentially containing NaNs for 'joint'
         missing_mask_for_model, # Mask indicating original NaNs (or all False)
         INPUT_SEQUENCE_LENGTH,
         OUTPUT_SEQUENCE_LENGTH
    )

    # Split into train, val, test Datasets (these are torch.utils.data.Subset)
    # create_dataloaders returns loaders AND the subset datasets
    # We need the subset datasets to figure out which rows of the original df belong to the training set
    # to fit the scaler correctly.
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
         data_for_model, # Pass the data used by the dataset (interpolated or raw)
         missing_mask_for_model,
         INPUT_SEQUENCE_LENGTH,
         OUTPUT_SEQUENCE_LENGTH,
         TRAIN_SPLIT_RATIO,
         VAL_SPLIT_RATIO,
         TEST_SPLIT_RATIO,
         BATCH_SIZE,
         collate_fn=None # Use default collate initially, will replace for GNN
    )

    # --- 3b. Normalization ---
    # Fit scaler ONLY on the training data rows used by the training windows.
    # Then, the normalization transformation will be applied either:
    # A) Directly to the dataframes/tensors if simple/tikhonov interpolation was used (data_for_model is already NaN-free)
    # B) Within the GNN's custom collate_fn if 'joint' interpolation was used (data_for_model has NaNs)

    scalers = {}
    if fit_scaler_on_train_subset and FEATURES_TO_NORMALIZE:
        print("Fitting scalers on training data subset...")
        # Get the row indices from the original data_for_model that are covered by the training windows
        train_data_row_indices = []
        for window_idx in train_dataset.indices: # Indices of windows in the full_dataset
             # A window starting at index `window_idx` covers rows `window_idx` to `window_idx + INPUT_SEQUENCE_LENGTH - 1`
             start_row = window_idx
             end_row = window_idx + INPUT_SEQUENCE_LENGTH - 1
             train_data_row_indices.extend(range(start_row, end_row + 1))
        train_data_row_indices = sorted(list(set([i for i in train_data_row_indices if i < len(data_for_model)]))) # Unique and sorted valid row indices

        # Get the subset of the data_for_model corresponding to these rows
        train_data_subset_for_scaler_fit = data_for_model.iloc[train_data_row_indices]

        # Fit MinMaxScaler for each feature of each turbine on this subset
        for feature in FEATURES_TO_NORMALIZE:
            feature_scalers = {}
            # Iterate through the turbine IDs that are actually in this experiment subset
            # The columns of train_data_subset_for_scaler_fit reflect these turbines
            turb_ids_in_subset = sorted(train_data_subset_for_scaler_fit.columns.get_level_values(1).unique())

            for turb_id in turb_ids_in_subset:
                col_name = (feature, turb_id)
                if col_name in train_data_subset_for_scaler_fit.columns:
                    scaler = MinMaxScaler()
                    col_data = train_data_subset_for_scaler_fit[col_name].values.reshape(-1, 1)
                    # Fit handles NaNs by ignoring them
                    scaler.fit(col_data)
                    feature_scalers[col_name] = scaler
            scalers[feature] = feature_scalers
        print("Scaler fitting complete.")

        # Apply normalization to the full data_for_model IF interpolation was NOT 'joint'
        # If 'joint', normalization happens in collate_fn.
        if interpolation_method != 'joint':
             print("Applying normalization to the full dataset...")
             normalized_data_full_df = data_for_model.copy() # Start from interpolated data
             for feature in FEATURES_TO_NORMALIZE:
                 if feature in scalers:
                     for turb_id in current_turb_ids: # Iterate through all turbines in this experiment
                         col_name = (feature, turb_id)
                         if col_name in normalized_data_full_df.columns and col_name in scalers[feature]:
                              s = scalers[feature][col_name]
                              # Transform the entire column using the scaler fit on training data
                              normalized_data_full_df[col_name] = s.transform(normalized_data_full_df[col_name].values.reshape(-1, 1)).flatten()
                         # else: print(f"Scaler not found for {col_name}, skipping normalization.")

             # Create new Datasets using the normalized dataframe
             full_dataset = TimeSeriesSlidingWindowDataset(
                 normalized_data_full_df, # Use normalized data
                 missing_mask_for_model, # Mask (should be all False for non-joint)
                 INPUT_SEQUENCE_LENGTH,
                 OUTPUT_SEQUENCE_LENGTH
             )
             # Re-create subsets using the same indices but the new dataset
             train_dataset = Subset(full_dataset, train_dataset.indices)
             val_dataset = Subset(full_dataset, val_dataset.indices)
             test_dataset = Subset(full_dataset, test_dataset.indices)
             print("Normalization applied to datasets.")

    # --- 3c. Set up final DataLoaders (with custom collate for GNN) ---
    collate_fn_to_use = None
    if model_type == 'gnn':
        print("Creating GNN DataLoaders with custom collate_fn...")
        # Build the spatial graph once (needed by the collate_fn)
        spatial_graph, locations = build_spatial_graph(location_df, SPATIAL_GRAPH_TYPE, SPATIAL_RADIUS, K_NEIGHBORS)
        # Build temporal graph template for the window size (needed by collate_fn)
        temporal_graph_template = build_temporal_graph(INPUT_SEQUENCE_LENGTH, TEMPORAL_GRAPH_TYPE)
        # Build product graph template (for the window size) - used by GNN collate_fn
        product_edge_index_template, product_edge_attr_template = build_product_graph(
            spatial_graph,
            temporal_graph_template,
            current_num_turbines, # Use num turbines in the current subset
            INPUT_SEQUENCE_LENGTH
        )
        if product_edge_index_template.numel() == 0 and current_num_turbines > 0 and INPUT_SEQUENCE_LENGTH > 0:
             print("Warning: Product graph template is empty, but data exists. Check graph construction parameters.")

        # Create GNN collate fn using the factory
        collate_fn_to_use = gnn_collate_fn_factory(
            product_edge_index_template, # Pass the template graph structure
            product_edge_attr_template,  # Pass the template graph attributes
            current_num_turbines,    # Num turbines in the subset for this experiment
            scalers,                 # Dictionary of fitted scalers
            FEATURES_TO_NORMALIZE,   # List of feature names to normalize
            INPUT_FEATURES,          # Pass list of all input feature names (for order)
            MISSING_VALUE_PLACEHOLDER_GNN, # Placeholder value for missing data
            USE_MISSING_MASK_GNN     # Boolean flag for using mask as a feature
        )
        # Modify the __getitem__ of the base dataset (TimeSeriesSlidingWindowDataset) to include self reference
        # This allows collate_fn access to turbine_ids and feature_names for this specific dataset instance.
        
        # Check if modification is already done to avoid multiple monkey-patching if run_experiment is called multiple times
        if not hasattr(TimeSeriesSlidingWindowDataset, '_getitem_original_for_gnn_patch'):
            TimeSeriesSlidingWindowDataset._getitem_original_for_gnn_patch = TimeSeriesSlidingWindowDataset.__getitem__
            
            def getitem_with_dataset_ref_for_gnn(self, idx):
                item = self._getitem_original_for_gnn_patch(idx)
                item['_dataset'] = self # Add reference to the dataset instance
                return item
            TimeSeriesSlidingWindowDataset.__getitem__ = getitem_with_dataset_ref_for_gnn
        
    # Create DataLoaders using the Dataset Subsets and the selected collate_fn
    # Ensure that train_dataset, val_dataset, test_dataset are Subsets of the *potentially normalized* full_dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn_to_use)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, collate_fn=collate_fn_to_use)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_fn_to_use) # Don't drop_last for test


    print(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    end_time_data = time.time()
    print(f"Data loading, preprocessing, and setup time: {end_time_data - start_time_data:.2f} sec")


    # --- 4. Train Model ---
    start_time_train = time.time()
    trained_model = None
    scalability_metrics = {}

    if model_type == 'gru':
        # GRU needs number of features per turbine as input dim
        # The dataset object within the loader has the original num_features_per_turbine
        num_features_per_turbine = train_loader.dataset.dataset.num_features_per_turbine
        trained_model, scalability_metrics = train_gru_model(
            train_loader, val_loader, current_num_turbines, device
        )

    elif model_type == 'gnn':
        # GNN needs input feature dimension (original features + mask if used)
        num_features_per_turbine = train_loader.dataset.dataset.num_features_per_turbine
        # The GNN model init needs the correct initial_input_dim calculated based on mask usage
        # This is calculated inside train_gnn_model now.
        trained_model, scalability_metrics = train_gnn_model(
            train_loader, val_loader, # Pass loaders (contain graph structure via collate_fn)
            current_num_turbines, # Pass num_turbines
            num_features_per_turbine, # Pass num features per turbine
            device
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    end_time_train = time.time()
    print(f"Training time: {end_time_train - start_time_train:.2f} sec")


    # --- 5. Evaluate Model ---
    start_time_eval = time.time()
    eval_metrics, predictions_df = evaluate_model(
        trained_model,
        test_loader,
        device,
        model_type,
        scaler=scalers if fit_scaler_on_train_subset else None # Pass scalers for inverse normalization if used
    )
    end_time_eval = time.time()
    print(f"Evaluation time: {end_time_eval - start_time_eval:.2f} sec")


    # --- 6. Save Results ---
    results_summary = {
        'experiment_name': exp_name,
        'model_type': model_type,
        'interpolation_method': interpolation_method,
        'num_turbines_in_experiment': current_num_turbines,
        'total_time_steps_in_experiment_data': current_time_steps, # Time steps in the full df subset used
        'input_seq_len': INPUT_SEQUENCE_LENGTH,
        'output_seq_len': OUTPUT_SEQUENCE_LENGTH,
        'train_split_ratio': TRAIN_SPLIT_RATIO,
        'val_split_ratio': VAL_SPLIT_RATIO,
        'test_split_ratio': TEST_SPLIT_RATIO,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'patience': PATIENCE,
        'MAE': eval_metrics['MAE'],
        'RMSE': eval_metrics['RMSE'],
        'R2': eval_metrics['R2'],
    }
    # Add scalability metrics if available
    results_summary.update(scalability_metrics)

    results_df_summary = pd.DataFrame([results_summary])
    results_file = os.path.join(OUTPUT_DIR, 'experiment_results.csv')
    results_df_summary.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)
    print(f"Results summary saved to {results_file}")

    # Save predictions DataFrame
    save_predictions(predictions_df, model_type, interpolation_method, data_description=exp_name)

    # Print scalability metrics if available
    if scalability_metrics:
         calculate_scalability_metrics(scalability_metrics)


    print(f"--- Experiment Finished: {exp_name} ---\n")


if __name__ == "__main__":
    # Define Scalability Subsets (requires loading data first)
    try:
        _, location_df_orig_all = load_data(SCADA_DATA_PATH, LOCATION_DATA_PATH)
        ALL_TURBINE_IDS = sorted(location_df_orig_all['TurbID'].unique())
        # Define subsets based on the number of available turbines
        SCALABILITY_TURBINE_SUBSETS = [10, 50, len(ALL_TURBINE_IDS)] # Test with varying numbers of turbines
        # Ensure subsets don't exceed available turbines
        SCALABILITY_TURBINE_SUBSETS = [n for n in SCALABILITY_TURBINE_SUBSETS if n <= len(ALL_TURBINE_IDS)]
        if len(ALL_TURBINE_IDS) not in SCALABILITY_TURBINE_SUBSETS:
             SCALABILITY_TURBINE_SUBSETS.append(len(ALL_TURBINE_IDS)) # Always include all turbines
        SCALABILITY_TURBINE_SUBSETS = sorted(list(set(SCALABILITY_TURBINE_SUBSETS)))


        # Define time subsets (in days)
        # Need total time steps available to pick reasonable days
        # A rough estimate: 245 days * 24 hours/day * 6 steps/hour = 35280 steps
        # 30 days ~ 4320 steps, 90 days ~ 12960 steps
        SCALABILITY_TIME_SUBSETS_DAYS = [30, 90, 245] # Test with varying total time periods (approximate days)
        # Ensure subsets don't exceed available time (based on file description)
        # The max time steps is ~245 days. No strict check needed beyond config.
        SCALABILITY_TIME_SUBSETS_DAYS = sorted(list(set(SCALABILITY_TIME_SUBSETS_DAYS)))


    except FileNotFoundError:
        print("Data files not found. Cannot define scalability subsets based on actual data.")
        ALL_TURBINE_IDS = list(range(1, 135)) # Default to 1-134 if data not found
        SCALABILITY_TURBINE_SUBSETS = [10, 50, 134]
        SCALABILITY_TIME_SUBSETS_DAYS = [30, 90, 245]


    print("Ensure data is in the ./data/ directory. Download links:")
    print("[1] SDWPF: https://bj.bcebos.com/v1/ai-studio-online/85b5cb4eea5a4f259766f42a7499c43elae4cc28fbdee8e087e2385.zip")
    print("[2] Location: https://bj.bcebos.com/v1/ai-studio-online/e927ce742c884955bf2a667929d36b2ef41c572cd6e245fa86257ecc2f7be7bc.zip")
    print("Extract both zip files into the ./data/ directory.")
    print("-" * 30)


    # --- Define Experiments ---
    # Configurations: (model_type, interpolation_method, turbine_subset_ids, time_subset_days)
    # None for subset means use all available data after initial load/filter

    experiment_configs = [
        # Baseline 1: GRU per turbine with simple imputation
        ('gru', 'mean', None, None),

        # Baseline 1 (more advanced): GRU with Tikhonov interpolation
        ('gru', 'tikhonov', None, None),

        # Main Approach: Product Graph GNN with joint interpolation
        ('gnn', 'joint', None, None),

        # Scalability Experiments for GNN (vary turbines, full time)
        # Note: Subset turbine IDs by taking the first N from the sorted list
        # Example: ALL_TURBINE_IDS[:num_t]
        # (GNN experiments will disable scaler fitting here as it's handled in collate)
        *[(
            'gnn',
            'joint',
            ALL_TURBINE_IDS[:num_t], # Subset turbine IDs
            None # Use all time steps for this subset
        ) for num_t in SCALABILITY_TURBINE_SUBSETS if num_t > 0], # Ensure num_t is positive

        # Scalability Experiments for GNN (vary time, all turbines)
        # (GNN experiments will disable scaler fitting here as it's handled in collate)
        *[(
            'gnn',
            'joint',
            None, # Use all turbines
            num_days # Subset time in days
        ) for num_days in SCALABILITY_TIME_SUBSETS_DAYS if num_days > 0] # Ensure num_days is positive

    ]

    # Filter out duplicate configurations (e.g., if 134 turbines is in SCALABILITY_TURBINE_SUBSETS and None is used for full)
    # Simple representation for set: tuple of (type, interp, tuple(turb_ids), days)
    seen_configs = set()
    unique_configs = []
    for cfg in experiment_configs:
        cfg_key = (cfg[0], cfg[1], tuple(cfg[2]) if cfg[2] is not None else None, cfg[3])
        if cfg_key not in seen_configs:
            unique_configs.append(cfg)
            seen_configs.add(cfg_key)
        # else:
            # print(f"Skipping duplicate configuration: {cfg}")


    # --- Run Experiments ---
    for cfg in unique_configs:
        model_type, interpolation_method, subset_turb_ids, subset_time_days = cfg
        # Pass fit_scaler=True for baselines, it will be handled within run_experiment
        # For GNN joint, run_experiment will fit scalers but apply them in collate
        # Pass the subset_turb_ids (list) to run_experiment
        run_experiment(
            model_type,
            interpolation_method,
            data_subset_turbines=subset_turb_ids,
            data_subset_time_days=subset_time_days,
            fit_scaler_on_train_subset=True # Always fit scaler if possible
        )

    print("\nAll scheduled experiments finished.")
    print(f"Results summary saved in {OUTPUT_DIR}/experiment_results.csv")
    print(f"Predictions saved in {PREDICTIONS_DIR}") 