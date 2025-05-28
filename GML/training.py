# wind_power_forecasting/training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch # Import Batch
import torch.cuda.memory as memory # For more granular memory tracking
import resource # For maxrss
import time
import os
import pandas as pd
import numpy as np
from config import * # Import all from config
from models.gru import GRUModel
from models.product_graph_gnn import ProductGraphGNN, TARGET_FEATURE_INDEX # Import TARGET_FEATURE_INDEX
from evaluation import evaluate_model, calculate_scalability_metrics # Import from evaluation


def train_gru_model(train_loader, val_loader, num_turbines, device):
    """Trains independent GRU models for each turbine."""
    print("Training independent GRU models...")

    models = {} # Store models for each turbine
    optimizers = {}
    criterions = {} # Loss function per turbine if needed, or just one MSELoss

    criterion = nn.MSELoss() # Use MSE for training

    # Assuming input_dim for GRU is the number of features for a single turbine
    gru_input_dim = len(INPUT_FEATURES)
    # Assuming output_dim for GRU is OUTPUT_SEQUENCE_LENGTH (for TARGET_FEATURE)
    gru_output_seq_len = OUTPUT_SEQUENCE_LENGTH # Predicts 1 feature (TARGET_FEATURE) for OUTPUT_SEQUENCE_LENGTH steps

    # Get sorted turbine IDs from the dataset
    # Accessing original dataset properties via Subset chain
    original_dataset = train_loader.dataset.dataset # Subset -> TimeSeriesSlidingWindowDataset
    turb_ids = original_dataset.turbine_ids


    # Instantiate a model and optimizer for each turbine
    for turb_id in turb_ids:
        # print(f"  Initializing GRU for Turbine {turb_id}...")
        model = GRUModel(gru_input_dim, GRU_HIDDEN_DIM, gru_output_seq_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        models[turb_id] = model
        optimizers[turb_id] = optimizer
        criterions[turb_id] = criterion # Same criterion for all

    best_val_loss = float('inf') # Track best average validation loss across all turbines
    patience_counter = 0
    best_epoch = 0

    # Timer and Memory Tracker
    start_time = time.time()
    # Note: Tracking memory for multiple independent models in one process is tricky.
    # resource.ru_maxrss tracks the peak for the whole process.
    # For GRU, this might just reflect the largest model/batch loaded at any point.
    # True scalability for GRU is per-model performance.
    peak_memory_mb = 0


    print("Starting GRU training...")
    for epoch in range(NUM_EPOCHS):
        total_train_loss = {turb_id: 0.0 for turb_id in turb_ids}
        num_train_batches = 0

        # Set models to training mode
        for model in models.values():
            model.train()

        # Get batch data (batch_size, seq_len, num_turbines, num_features) etc.
        # The DataLoader provides the raw tensors from the dataset __getitem__
        for batch_idx, batch_data in enumerate(train_loader):
            batch_input = batch_data['input'].to(device) # (batch_size, seq_len, N_turbines * N_features)
            batch_output = batch_data['output'].to(device) # (batch_size, out_seq_len, N_turbines * N_features)
            batch_output_mask = batch_data['output_mask'].to(device) # (batch_size, out_seq_len, N_turbines * N_features)

            # Reshape to (batch_size, seq_len, num_turbines, num_features)
            num_features_per_turbine = len(INPUT_FEATURES)
            batch_input_reshaped = batch_input.view(batch_input.shape[0], INPUT_SEQUENCE_LENGTH, num_turbines, num_features_per_turbine)
            batch_output_reshaped = batch_output.view(batch_output.shape[0], OUTPUT_SEQUENCE_LENGTH, num_turbines, num_features_per_turbine)
            batch_output_mask_reshaped = batch_output_mask.view(batch_output_mask.shape[0], OUTPUT_SEQUENCE_LENGTH, num_turbines, num_features_per_turbine)


            # Need target values for TARGET_FEATURE only
            target_output = batch_output_reshaped[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1) # (batch_size, out_seq_len, num_turbines, 1)
            target_output_mask = batch_output_mask_reshaped[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1) # (batch_size, out_seq_len, num_turbines, 1)

            for turb_idx, turb_id in enumerate(turb_ids):
                model = models[turb_id]
                optimizer = optimizers[turb_id]
                # criterion = criterions[turb_id] # Using same criterion for all

                # Select data for this turbine
                turbine_input = batch_input_reshaped[:, :, turb_idx, :] # (batch_size, seq_len, num_features)
                turbine_target = target_output[:, :, turb_idx, :] # (batch_size, out_seq_len, 1)
                turbine_target_mask = target_output_mask[:, :, turb_idx, :] # (batch_size, out_seq_len, 1)


                optimizer.zero_grad()

                # Forward pass
                predictions = model(turbine_input) # Shape (batch_size, out_seq_len, 1)

                # Calculate loss only on non-missing target values
                # The loss should only consider elements where turbine_target_mask is FALSE (not missing)
                loss = criterion(predictions[~turbine_target_mask], turbine_target[~turbine_target_mask])

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                total_train_loss[turb_id] += loss.item()

            num_train_batches += 1
            # Track memory usage
            peak_memory_mb = max(peak_memory_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024) # on Linux/macOS


        # --- Validation ---
        total_val_loss = {turb_id: 0.0 for turb_id in turb_ids}
        num_val_batches = 0

        # Set models to evaluation mode
        for model in models.values():
            model.eval()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                batch_input = batch_data['input'].to(device)
                batch_output = batch_data['output'].to(device)
                batch_output_mask = batch_data['output_mask'].to(device)

                num_features_per_turbine = len(INPUT_FEATURES)
                batch_input_reshaped = batch_input.view(batch_input.shape[0], INPUT_SEQUENCE_LENGTH, num_turbines, num_features_per_turbine)
                batch_output_reshaped = batch_output.view(batch_output.shape[0], OUTPUT_SEQUENCE_LENGTH, num_turbines, num_features_per_turbine)
                batch_output_mask_reshaped = batch_output_mask.view(batch_output_mask.shape[0], OUTPUT_SEQUENCE_LENGTH, num_turbines, num_features_per_turbine)


                target_output = batch_output_reshaped[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1)
                target_output_mask = batch_output_mask_reshaped[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1)


                for turb_idx, turb_id in enumerate(turb_ids):
                    model = models[turb_id]
                    turbine_input = batch_input_reshaped[:, :, turb_idx, :]
                    turbine_target = target_output[:, :, turb_idx, :]
                    turbine_target_mask = target_output_mask[:, :, turb_idx, :]

                    predictions = model(turbine_input)
                    loss = criterion(predictions[~turbine_target_mask], turbine_target[~turbine_target_mask])
                    total_val_loss[turb_id] += loss.item()

                num_val_batches += 1
                peak_memory_mb = max(peak_memory_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


        # Calculate average losses
        avg_train_loss = {turb_id: total_train_loss[turb_id] / num_train_batches if num_train_batches > 0 else 0 for turb_id in turb_ids}
        avg_val_loss = {turb_id: total_val_loss[turb_id] / num_val_batches if num_val_batches > 0 else 0 for turb_id in turb_ids}

        # Calculate average farm validation loss
        avg_farm_val_loss = sum(avg_val_loss.values()) / num_turbines if num_turbines > 0 else 0

        # Print epoch summary
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Farm Avg Train Loss = {sum(avg_train_loss.values())/num_turbines if num_turbines > 0 else 0:.4f}, Farm Avg Val Loss = {avg_farm_val_loss:.4f}")
        # Optional: Print per-turbine losses if needed

        # --- Check Early Stopping and Save Checkpoints ---
        # Use average farm validation loss for early stopping
        if avg_farm_val_loss < best_val_loss:
            # print("  Farm average validation loss improved.")
            best_val_loss = avg_farm_val_loss
            best_epoch = epoch
            # Note: Not saving individual GRU models here to keep it simple,
            # but in a real scenario, you'd save the state_dict for each model
            # or the state_dict of a ModuleDict containing all models.
            # print("  Checkpoint saving skipped for GRU baseline.")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
             print(f"Early stopping triggered after {PATIENCE} epochs without improvement in farm average validation loss.")
             print(f"Best epoch was {best_epoch+1} with Farm Avg Val Loss: {best_val_loss:.4f}")
             break

    end_time = time.time()
    training_time_sec = end_time - start_time

    print("GRU training finished.")

    # Prepare scalability metrics (basic info)
    scalability_metrics = {
        'training_time_sec': training_time_sec,
        'peak_memory_mb': peak_memory_mb,
        'num_turbines': num_turbines,
        'total_time_steps_trained_on': len(original_dataset.data_df) # Approx total time steps in original data used for train windows
    }

    # Return the trained models (last epoch's state, not necessarily the best)
    # In a real project, load the best state dicts if saving was implemented.
    return models, scalability_metrics


def train_gnn_model(train_loader, val_loader, num_turbines, num_features_per_turbine, device):
    """Trains the Product Graph GNN model."""
    print("Training Product Graph GNN model...")

    # Need input feature dimension (original features + mask if used)
    gnn_initial_input_dim = num_features_per_turbine
    if USE_MISSING_MASK_GNN:
        gnn_initial_input_dim += num_features_per_turbine # Adding a mask feature for each original feature

    model = ProductGraphGNN(
        gnn_initial_input_dim,
        GNN_HIDDEN_DIM,
        GNN_NUM_LAYERS,
        GNN_DROPOUT,
        num_turbines,
        INPUT_SEQUENCE_LENGTH,
        OUTPUT_SEQUENCE_LENGTH
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Use MSE for training

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    # Timer and Memory Tracker
    start_time = time.time()
    peak_memory_mb = 0 # Tracks peak resident set size


    print("Starting GNN training...")

    for epoch in range(NUM_EPOCHS):
        total_train_loss = 0.0
        num_train_batches = 0

        model.train()
        # Batch data is already a PyG Batch object from the custom collate_fn
        for batch_idx, batch_data in enumerate(train_loader):
            # Move the whole Batch object to device
            batch_data = batch_data.to(device)

            optimizer.zero_grad()

            # Forward pass
            # predictions shape: (batch_size, num_turbines, output_sequence_length, 1)
            predictions = model(batch_data)

            # Get target values and mask from the Batch object
            # Shapes: (batch_size, out_seq_len, num_turbines, num_features_per_turbine)
            target_output_all_features = batch_data.output.to(device)
            target_output_mask_all_features = batch_data.output_mask.to(device)

            # Need target values and mask for TARGET_FEATURE only
            target_output_target_feature = target_output_all_features[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1) # (batch_size, out_seq_len, num_turbines, 1)
            target_output_mask_target_feature = target_output_mask_all_features[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1) # (batch_size, out_seq_len, num_turbines, 1)


            # Calculate loss only on non-missing target values
            # Ensure dimensions match between predictions and target slice
            # predictions shape: (batch_size, num_turbines, output_sequence_length, 1)
            # target_output_target_feature shape: (batch_size, output_sequence_length, num_turbines, 1)
            # Need to reorder dimensions of target to match predictions: (batch, turb, out_seq, 1)
            target_output_target_feature = target_output_target_feature.permute(0, 2, 1, 3)
            target_output_mask_target_feature = target_output_mask_target_feature.permute(0, 2, 1, 3)

            loss = criterion(predictions[~target_output_mask_target_feature], target_output_target_feature[~target_output_mask_target_feature])

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

            # Track memory usage during training (approximate peak)
            peak_memory_mb = max(peak_memory_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024) # on Linux/macOS
            # Can also use torch.cuda.max_memory_allocated() for CUDA memory

        # --- Validation ---
        total_val_loss = 0.0
        num_val_batches = 0

        model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                batch_data = batch_data.to(device)

                predictions = model(batch_data)

                target_output_all_features = batch_data.output.to(device)
                target_output_mask_all_features = batch_data.output_mask.to(device)

                target_output_target_feature = target_output_all_features[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1)
                target_output_mask_target_feature = target_output_mask_all_features[:, :, :, TARGET_FEATURE_INDEX].unsqueeze(-1)

                # Reorder target dims
                target_output_target_feature = target_output_target_feature.permute(0, 2, 1, 3)
                target_output_mask_target_feature = target_output_mask_target_feature.permute(0, 2, 1, 3)


                loss = criterion(predictions[~target_output_mask_target_feature], target_output_target_feature[~target_output_mask_target_feature])
                total_val_loss += loss.item()
                num_val_batches += 1
                peak_memory_mb = max(peak_memory_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


        # Calculate average losses
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0

        # Print epoch summary
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Check Early Stopping and Save Checkpoints ---
        if avg_val_loss < best_val_loss:
            print("  Validation loss improved. Saving model checkpoint...")
            best_val_loss = avg_val_loss
            best_epoch = epoch
            # Save model state dict
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'product_graph_gnn_best.pth'))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
             print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
             print(f"Best epoch was {best_epoch+1} with Val Loss: {best_val_loss:.4f}")
             break

    end_time = time.time()
    training_time_sec = end_time - start_time

    print("Product Graph GNN training finished.")

    # Load best model weights before returning
    best_model_path = os.path.join(CHECKPOINT_DIR, 'product_graph_gnn_best.pth')
    if os.path.exists(best_model_path):
         model.load_state_dict(torch.load(best_model_path))
         print("Loaded best model weights.")
    else:
         print("Warning: Best model checkpoint not found. Returning model after last epoch.")


    # Prepare scalability metrics
    # Need total time steps in the original data used for this run's dataset
    original_dataset = train_loader.dataset.dataset # Subset -> TimeSeriesSlidingWindowDataset
    total_time_steps_in_data = len(original_dataset.data_df)

    scalability_metrics = {
        'training_time_sec': training_time_sec,
        'peak_memory_mb': peak_memory_mb,
        'num_turbines': num_turbines,
        'total_time_steps_in_data': total_time_steps_in_data, # Time steps in the full dataframe subset used for this experiment
        'input_seq_len': INPUT_SEQUENCE_LENGTH,
        'output_seq_len': OUTPUT_SEQUENCE_LENGTH,
        'batch_size': BATCH_SIZE
    }


    return model, scalability_metrics 