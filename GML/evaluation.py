# wind_power_forecasting/evaluation.py

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Subset
# Removed config import, pass necessary params
from models.product_graph_gnn import TARGET_FEATURE_INDEX # Need target feature index
from utils import inverse_normalize_target
import os
from config import PREDICTIONS_DIR, TARGET_FEATURE, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH


def evaluate_model(model, data_loader, device, model_type, scaler=None):
    """
    Evaluates the model on the given data loader.
    Args:
        model: The trained model (either dict of GRUs or ProductGraphGNN).
        data_loader: DataLoader for evaluation data (e.g., test_loader).
        device: Device to use ('cuda' or 'cpu').
        model_type (str): 'gru' or 'gnn'.
        scaler (dict): Dictionary of scalers to inverse transform predictions, if normalization was used.
                       Expected format {feature: {col_name: scaler_obj}}
                       This is needed ONLY if normalization was applied.
    Returns:
        dict: Dictionary of evaluation metrics (MAE, RMSE, R2) averaged across turbines.
        pd.DataFrame: DataFrame containing actual vs predicted values for the target feature.
    """
    print(f"Evaluating {model_type} model...")
    actuals_list = [] # Store actual values for target feature
    predictions_list = [] # Store predicted values for target feature
    masks_list = [] # Store masks for target feature

    # Ensure model(s) are in evaluation mode
    if model_type == 'gru':
        for m in model.values(): m.eval()
        # Get sorted turbine IDs for GRU models
        original_dataset = data_loader.dataset.dataset # Subset -> TimeSeriesSlidingWindowDataset
        original_turbine_ids = original_dataset.turbine_ids

    elif model_type == 'gnn':
        model.eval()
        # Get original turbine IDs from the GNN DataLoader's dataset
        original_dataset = data_loader.dataset.dataset # Subset -> TimeSeriesSlidingWindowDataset
        original_turbine_ids = original_dataset.turbine_ids
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    num_turbines = len(original_turbine_ids)
    num_features_per_turbine = len(original_dataset.feature_names) # Get from dataset


    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # Batch data structure depends on model type (dict for GRU, PyG Batch for GNN)
            if model_type == 'gru':
                # GRU data is raw tensors from TimeSeriesSlidingWindowDataset
                batch_input = batch_data['input'].to(device) # (batch_size, seq_len, N_turbines * N_features)
                batch_output = batch_data['output'].to(device) # (batch_size, out_seq_len, N_turbines * N_features)
                batch_output_mask = batch_data['output_mask'].to(device) # (batch_size, out_seq_len, N_turbines * N_features)

                # Reshape to (batch_size, seq_len, num_turbines, num_features)
                batch_input_reshaped = batch_input.view(batch_input.shape[0], INPUT_SEQUENCE_LENGTH, num_turbines, num_features_per_turbine)
                batch_output_reshaped = batch_output.view(batch_output.shape[0], OUTPUT_SEQUENCE_LENGTH, num_turbines, num_features_per_turbine)
                batch_output_mask_reshaped = batch_output_mask.view(batch_output_mask.shape[0], OUTPUT_SEQUENCE_LENGTH, num_turbines, num_features_per_turbine)


                # Need target values and mask for TARGET_FEATURE only
                target_output = batch_output_reshaped[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1) # (batch_size, out_seq_len, num_turbines, 1)
                target_output_mask = batch_output_mask_reshaped[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1) # (batch_size, out_seq_len, num_turbines, 1)

                batch_predictions = [] # Store predictions for this batch, shaped (batch_size, num_turbines, out_seq_len, 1)
                # Iterate over turbine IDs in the sorted order to match data slicing
                for turb_id in original_turbine_ids:
                    # Find the index of this turbine in the current batch's tensor columns
                    # The data batch is ordered by the sorted unique turbine IDs that were in the original df subset.
                    turb_idx = original_turbine_ids.index(turb_id)
                    gru_model = model[turb_id]
                    turbine_input = batch_input_reshaped[:, :, turb_idx, :] # (batch_size, seq_len, num_features)
                    predictions_turb = gru_model(turbine_input) # (batch_size, out_seq_len, 1)
                    batch_predictions.append(predictions_turb)

                # Stack predictions from all turbines
                batch_predictions = torch.stack(batch_predictions, dim=1) # (batch_size, num_turbines, out_seq_len, 1)

                # Reshape actuals and masks to match prediction shape
                # target_output shape: (batch_size, out_seq_len, num_turbines, 1)
                batch_actuals = target_output.permute(0, 2, 1, 3) # (batch_size, num_turbines, out_seq_len, 1)
                batch_masks = target_output_mask.permute(0, 2, 1, 3) # (batch_size, num_turbines, out_seq_len, 1)


            elif model_type == 'gnn':
                 # GNN data is a PyG Batch object
                 batch_data = batch_data.to(device)
                 batch_predictions = model(batch_data) # (batch_size, num_turbines, out_seq_len, 1)

                 # Extract actuals and masks from the Batch object
                 # Shapes: (batch_size, out_seq_len, num_turbines, num_features_per_turbine)
                 batch_output_all_features = batch_data.output.to(device)
                 batch_output_mask_all_features = batch_data.output_mask.to(device)

                 # Get target values and mask for TARGET_FEATURE only
                 batch_actuals = batch_output_all_features[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1) # (batch_size, out_seq_len, num_turbines, 1)
                 batch_masks = batch_output_mask_all_features[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1) # (batch_size, out_seq_len, num_turbines, 1)

                 # Reorder actuals and masks to match prediction shape (batch, turb, out_seq, 1)
                 batch_actuals = batch_actuals.permute(0, 2, 1, 3)
                 batch_masks = batch_masks.permute(0, 2, 1, 3)
            else:
                raise ValueError(f"Unknown model type: {model_type}")


            # Store predictions, actuals, and masks (on CPU)
            # Shape: (batch_size, num_turbines, out_seq_len, 1)
            predictions_list.append(batch_predictions.cpu().numpy())
            actuals_list.append(batch_actuals.cpu().numpy())
            masks_list.append(batch_masks.cpu().numpy()) # True means missing


    # Concatenate results from all batches
    # Shape: (TotalWindows, num_turbines, out_seq_len, 1)
    actuals = np.concatenate(actuals_list, axis=0)
    predictions = np.concatenate(predictions_list, axis=0)
    masks = np.concatenate(masks_list, axis=0) # True indicates missing (should be excluded)


    # Reshape for metric calculation: (TotalWindows * num_turbines * out_seq_len, 1)
    actuals_flat = actuals.flatten()
    predictions_flat = predictions.flatten()
    masks_flat = masks.flatten() # True indicates missing

    # --- Inverse Normalization ---
    if scaler is not None:
         print("Applying inverse normalization...")
         # Reshape predictions and actuals to (TotalWindows * out_seq_len, num_turbines)
         # for easier column-wise inverse scaling per turbine
         predictions_reshaped_for_scaling = predictions.squeeze(-1).reshape(-1, num_turbines) # (TotalWindows * out_seq_len, num_turbines)
         actuals_reshaped_for_scaling = actuals.squeeze(-1).reshape(-1, num_turbines)

         # Apply inverse normalization using the function in utils
         unscaled_predictions_reshaped = inverse_normalize_target(
             predictions_reshaped_for_scaling,
             scaler,
             original_turbine_ids, # Pass turbine IDs to match columns
             TARGET_FEATURE # Pass target feature name
         )
         unscaled_actuals_reshaped = inverse_normalize_target(
             actuals_reshaped_for_scaling,
             scaler,
             original_turbine_ids,
             TARGET_FEATURE
         )

         # Flatten the unscaled data for metric calculation, applying the mask
         # The mask needs to be reshaped to match the unscaled data shape
         masks_reshaped_for_scaling = masks.squeeze(-1).reshape(-1, num_turbines)
         unscaled_masks_flat = masks_reshaped_for_scaling.flatten() # True indicates missing

         actuals_known = unscaled_actuals_reshaped.flatten()[~unscaled_masks_flat]
         predictions_known = unscaled_predictions_reshaped.flatten()[~unscaled_masks_flat]

    else:
         # If no scaler, use the scaled (or raw if no normalization) flattened data directly
         actuals_known = actuals_flat[~masks_flat]
         predictions_known = predictions_flat[~masks_flat]

    # Check if there are any known values to evaluate
    if len(actuals_known) == 0:
        print("Warning: No known target values found in the evaluation set after applying mask. Cannot compute metrics.")
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan}, pd.DataFrame()


    # Calculate metrics
    mae = mean_absolute_error(actuals_known, predictions_known)
    rmse = np.sqrt(mean_squared_error(actuals_known, predictions_known))
    # R2 score can be negative, which means the model is worse than predicting the mean of actuals_known
    r2 = r2_score(actuals_known, predictions_known)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

    print(f"Evaluation Metrics ({model_type}): MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")


    # Prepare DataFrame for actuals vs predictions
    # The actuals/predictions arrays (before flattening/masking) have shape (TotalWindows, num_turbines, out_seq_len, 1)
    # Need to get the correct time index for these predicted steps.
    # The first prediction timestamp corresponds to the start time of the first window + input_seq_len steps.
    # Subsequent prediction timestamps increment by TIME_STEP_MINUTES.

    # Get the original time index from the DataLoader's dataset
    original_data_df = original_dataset.data_df # Original dataframe used by the dataset
    # The start index of evaluation windows in the original data_df
    # The Subset stores the indices of the full dataset it uses. The full dataset
    # uses row indices from the original data_df directly.
    first_window_idx_in_full_data = data_loader.dataset.indices[0] if isinstance(data_loader.dataset, Subset) else 0

    # The first row index in the original data_df corresponding to the first prediction timestamp
    first_prediction_data_row_idx = first_window_idx_in_full_data + INPUT_SEQUENCE_LENGTH

    # Calculate the actual timestamps for the predictions
    # The total number of prediction time steps is TotalWindows * out_seq_len
    total_predicted_time_steps = actuals.shape[0] * actuals.shape[2] # TotalWindows * out_seq_len
    prediction_data_row_indices = np.arange(
        first_prediction_data_row_idx,
        first_prediction_data_row_idx + total_predicted_time_steps
    )

    # Ensure indices are within the bounds of the original data_df index
    valid_prediction_data_row_indices = [
        idx for idx in prediction_data_row_indices if idx < len(original_data_df.index)
    ]
    if len(valid_prediction_data_row_indices) != total_predicted_time_steps:
         print(f"Warning: Mismatch in predicted time steps ({total_predicted_time_steps}) and available data rows ({len(valid_prediction_data_row_indices)}). Truncating results.")

    prediction_time_index = original_data_df.index[valid_prediction_data_row_indices]

    # Reshape predictions and actuals to (Time, Turbines) for DataFrame
    # Need to select only the valid time steps identified above
    # predictions shape: (TotalWindows, num_turbines, out_seq_len, 1)
    # Reshape to (TotalWindows * out_seq_len, num_turbines)
    predictions_reshaped_time_turb = predictions.squeeze(-1).transpose(0, 2, 1).reshape(-1, num_turbines) # (out_seq_len * TotalWindows, num_turbines) -> time is first dim

    # Select only the rows corresponding to valid prediction timestamps
    predictions_df_values = predictions_reshaped_time_turb[
        (prediction_data_row_indices >= first_prediction_data_row_idx) &
        (prediction_data_row_indices < first_prediction_data_row_idx + len(prediction_time_index))
    ]

    # Repeat for actuals and masks
    actuals_reshaped_time_turb = actuals.squeeze(-1).transpose(0, 2, 1).reshape(-1, num_turbines)
    actuals_df_values = actuals_reshaped_time_turb[
         (prediction_data_row_indices >= first_prediction_data_row_idx) &
         (prediction_data_row_indices < first_prediction_data_row_idx + len(prediction_time_index))
    ]

    masks_reshaped_time_turb = masks.squeeze(-1).transpose(0, 2, 1).reshape(-1, num_turbines)
    masks_df_values = masks_reshaped_time_turb[
         (prediction_data_row_indices >= first_prediction_data_row_idx) &
         (prediction_data_row_indices < first_prediction_data_row_idx + len(prediction_time_index))
    ]

    # Create DataFrames
    actuals_df = pd.DataFrame(actuals_df_values, index=prediction_time_index, columns=pd.MultiIndex.from_product([[f'{TARGET_FEATURE}_actual'], original_turbine_ids], names=['Feature', 'TurbineID']))
    predictions_df = pd.DataFrame(predictions_df_values, index=prediction_time_index, columns=pd.MultiIndex.from_product([[f'{TARGET_FEATURE}_predicted'], original_turbine_ids], names=['Feature', 'TurbineID']))
    masks_df = pd.DataFrame(masks_df_values, index=prediction_time_index, columns=pd.MultiIndex.from_product([[f'{TARGET_FEATURE}_missing_mask'], original_turbine_ids], names=['Feature', 'TurbineID'])) # Mask indicates missing=True

    # Combine into a single DataFrame
    results_df = pd.concat([actuals_df, predictions_df, masks_df], axis=1)
    results_df.sort_index(axis=1, inplace=True) # Sort columns

    print("Evaluation finished.")

    return metrics, results_df


def calculate_scalability_metrics(metrics_dict):
    """Prints or logs scalability metrics."""
    print("\n--- Scalability Metrics ---")
    print(f"  Num Turbines Tested: {metrics_dict.get('num_turbines', 'N/A')}")
    print(f"  Total Time Steps in Data: {metrics_dict.get('total_time_steps_in_data', 'N/A')}")
    print(f"  Input Sequence Length: {metrics_dict.get('input_seq_len', 'N/A')}")
    print(f"  Output Sequence Length: {metrics_dict.get('output_seq_len', 'N/A')}")
    print(f"  Batch Size: {metrics_dict.get('batch_size', 'N/A')}")
    print(f"  Training Time (sec): {metrics_dict.get('training_time_sec', np.nan):.2f}")
    print(f"  Peak Memory Usage (MB): {metrics_dict.get('peak_memory_mb', np.nan):.2f}")
    print("---------------------------\n")

    # Optional: Save scalability metrics to a file
    # pd.DataFrame([metrics_dict]).to_csv(os.path.join(OUTPUT_DIR, f"scalability_metrics_T{metrics_dict.get('num_turbines','All')}_TS{metrics_dict.get('total_time_steps_in_data','All')}.csv"))

def save_predictions(predictions_df, model_name, interpolation_method, data_description="test"):
    """Saves the actual vs predicted DataFrame to a CSV file."""
    # Create a safe filename based on parameters
    filename = f"{model_name}_{interpolation_method}_{data_description}_predictions.csv"
    # Replace problematic characters if any
    filename = filename.replace(' ', '_').replace('=', '-').replace(',', '_')
    filepath = os.path.join(PREDICTIONS_DIR, filename)
    predictions_df.to_csv(filepath)
    print(f"Predictions saved to {filepath}") 