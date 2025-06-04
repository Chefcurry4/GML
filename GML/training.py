# wind_power_forecasting/training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch # Import Batch
import torch.cuda.memory as memory # For more granular memory tracking
#import resource # For maxrss
import time
import os
import pandas as pd
import numpy as np
from config import * # Import all from config
from models.gru import GRUModel
from models.product_graph_gnn import ProductGraphGNN, TARGET_FEATURE_INDEX # Import TARGET_FEATURE_INDEX
from evaluation import evaluate_model, calculate_scalability_metrics # Import from evaluation

# Define the directory for saving trained models
TRAINED_MODELS_DIR = "GML/trained_models"
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)


def save_individual_model(model, optimizer, epoch, loss, model_name, turb_id=None):
    """Saves a single model (GRU or GNN) and removes previous versions."""
    if turb_id is not None: # For GRU models
        filename = f"{model_name}_turbine_{turb_id}_best.pt"
        # Remove old model files for this turbine
        for f in os.listdir(TRAINED_MODELS_DIR):
            if f.startswith(f"{model_name}_turbine_{turb_id}_") and f != filename:
                os.remove(os.path.join(TRAINED_MODELS_DIR, f))
    else: # For GNN model
        filename = f"{model_name}_best.pt"
        # Remove old model files
        for f in os.listdir(TRAINED_MODELS_DIR):
            if f.startswith(f"{model_name}_") and f != filename:
                os.remove(os.path.join(TRAINED_MODELS_DIR, f))
    
    filepath = os.path.join(TRAINED_MODELS_DIR, filename)
    
    # Create a state dictionary
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(state, filepath)
    print(f"Saved model to {filepath}")


def load_individual_model(model, optimizer, model_name, device, turb_id=None):
    """Loads a single model (GRU or GNN). Searches for the best (latest epoch) model."""
    best_epoch = -1
    best_filepath = None
    
    prefix = f"{model_name}_turbine_{turb_id}_epoch_" if turb_id is not None else f"{model_name}_epoch_"
    
    if not os.path.exists(TRAINED_MODELS_DIR):
        print(f"Trained models directory {TRAINED_MODELS_DIR} does not exist. Cannot load model.")
        return None # Indicate model not loaded

    for f in os.listdir(TRAINED_MODELS_DIR):
        if f.startswith(prefix) and f.endswith(".pt"):
            try:
                epoch_num = int(f.replace(prefix, "").replace(".pt", ""))
                if epoch_num > best_epoch:
                    best_epoch = epoch_num
                    best_filepath = os.path.join(TRAINED_MODELS_DIR, f)
            except ValueError:
                continue # Skip files that don't match the epoch pattern

    if best_filepath:
        print(f"Loading model from {best_filepath}...")
        checkpoint = torch.load(best_filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer: # Optimizer might not be needed for inference-only loading
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Return model, optimizer, and other saved info if needed
        # For simplicity, we'll just signal that loading was successful
        print(f"Loaded model {model_name} (epoch {checkpoint['epoch']+1}) with loss {checkpoint['loss']:.4f}")
        return model, optimizer, checkpoint['epoch'], checkpoint['loss']
    else:
        print(f"No saved model found for {model_name}" + (f" (Turbine {turb_id})" if turb_id is not None else ""))
        return None


def train_gru_model(train_loader, val_loader, num_turbines, device, force_retrain=False):
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
    original_dataset = train_loader.dataset.dataset 
    turb_ids = original_dataset.turbine_ids


    # Instantiate a model and optimizer for each turbine
    for turb_id in turb_ids:
        # print(f\"  Initializing GRU for Turbine {turb_id}...\")
        model = GRUModel(gru_input_dim, GRU_HIDDEN_DIM, gru_output_seq_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        if not force_retrain:
            loaded_data = load_individual_model(model, optimizer, "GRU", device, turb_id=turb_id)
            if loaded_data:
                # Model, optimizer are updated in-place by load_individual_model
                # We could potentially skip training for this turbine if loaded successfully
                # For simplicity, we'll just note it was loaded and continue to train all or none
                # Or, collect loaded models and return if all are loaded.
                print(f"Pre-trained GRU model loaded for turbine {turb_id}.")
                # If we want to skip training if ALL models are loaded, we'd need a flag here.
                # For now, we'll assume if one is loaded, we might still retrain,
                # or we expect all to be loaded to skip.
                # Let's assume we want to load all or train all.
                # This part needs a clearer strategy: load individual and use, or load all and skip?
                # For now, this load attempt is per turbine but training proceeds.
                # A better approach: try to load all, if all succeed, return them.
                pass # Loaded, but training will proceed and overwrite.

        models[turb_id] = model
        optimizers[turb_id] = optimizer
        criterions[turb_id] = criterion # Same criterion for all

    # Check if all models were loaded (requires a different loading strategy)
    # For now, if force_retrain is False, and models *could* be loaded,
    # we would ideally skip training. Let's adjust this:
    # We'll try to load all. If all are loaded, we return them.
    
    all_models_loaded = True
    if not force_retrain:
        print("Attempting to load all pre-trained GRU models...")
        loaded_models_temp = {}
        for turb_id in turb_ids:
            model_temp = GRUModel(gru_input_dim, GRU_HIDDEN_DIM, gru_output_seq_len).to(device)
            # Optimizer state is also important if we want to resume training,
            # but for just returning a trained model, only model state_dict is crucial.
            # For simplicity, we pass None for optimizer here if just checking for existence.
            # However, load_individual_model expects an optimizer.
            optimizer_temp = optim.Adam(model_temp.parameters(), lr=LEARNING_RATE) # Dummy optimizer for loading
            
            loaded_data = load_individual_model(model_temp, optimizer_temp, "GRU", device, turb_id=turb_id)
            if loaded_data:
                loaded_models_temp[turb_id] = loaded_data[0] # Store the model
            else:
                all_models_loaded = False
                print(f"Could not load GRU model for turbine {turb_id}. Will retrain all GRU models.")
                break 
        
        if all_models_loaded and loaded_models_temp:
            print("All GRU models successfully loaded. Skipping training.")
            # Need to prepare scalability metrics if returning early
            # This is tricky as training time would be 0.
            # For now, let's return the models and a minimal metrics dict.
            scalability_metrics = {
                'training_time_sec': 0,
            #    'peak_memory_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, # Current memory
                'num_turbines': num_turbines,
                'total_time_steps_trained_on': 0
            }
            return loaded_models_temp, scalability_metrics # Return the dictionary of loaded models

    # If not all models loaded or force_retrain is True, proceed with training.
    # Re-initialize models and optimizers for a clean training session if some were partially loaded/modified
    print("Proceeding with GRU model training (either forced or some models were not found).")
    models = {}
    optimizers = {}
    for turb_id in turb_ids:
        model = GRUModel(gru_input_dim, GRU_HIDDEN_DIM, gru_output_seq_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        models[turb_id] = model
        optimizers[turb_id] = optimizer
        # criterions already initialized

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
        #    peak_memory_mb = max(peak_memory_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024) # on Linux/macOS


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
            #    peak_memory_mb = max(peak_memory_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


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
            
            # Save the best model for each turbine
            print(f"New best farm validation loss at epoch {epoch+1}. Saving GRU models...")
            for turb_id, model in models.items():
                optimizer = optimizers[turb_id] # Get the corresponding optimizer
                # We save based on the farm's best val loss epoch
                save_individual_model(model, optimizer, epoch, avg_val_loss[turb_id], "GRU", turb_id=turb_id)
            
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
    
    # After training, ensure the latest (best) models are saved if early stopping occurred earlier than NUM_EPOCHS
    # Or, if training completed fully, the last saved models during early stopping check are the best ones.
    # If no improvement was ever made, no models would be saved by the above logic.
    # We should save the models from the best_epoch.
    # The current logic saves when best_val_loss improves. So this is already handled.

    # If we want to save the models at the *very end* of training, regardless of early stopping logic for "best":
    # print(f"Saving final GRU models after {NUM_EPOCHS} epochs (or early stop point)...")
    # for turb_id, model in models.items():
    #     optimizer = optimizers[turb_id]
    #     # Use the final epoch number and the validation loss from that epoch.
    #     # This requires storing the last epoch's val loss if different from best_val_loss due to early stopping.
    #     # For simplicity, we are saving on *best_val_loss* improvement.
    #     # If no early stopping, it implies the last epoch was the best or equal.

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


def train_gnn_model(train_loader, val_loader, num_turbines, num_features_per_turbine, device, force_retrain=False):
    """Trains the ProductGraphGNN model."""
    print("Training GNN model...")

    # Calculate input dimension
    # If using missing mask feature, add one dimension per feature
    initial_input_dim = num_features_per_turbine
    if USE_MISSING_MASK_GNN:
        initial_input_dim = initial_input_dim * 2  # Double the features (original + mask)
    
    model = ProductGraphGNN(
        initial_input_dim=initial_input_dim,
        hidden_dim=GNN_HIDDEN_DIM,
        num_layers=GNN_NUM_LAYERS,
        dropout=GNN_DROPOUT,
        num_turbines=num_turbines,
        input_sequence_length=INPUT_SEQUENCE_LENGTH,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Try to load pre-trained model if not force_retrain
    if not force_retrain:
        loaded_data = load_individual_model(model, optimizer, "GNN", device)
        if loaded_data:
            print("Pre-trained GNN model loaded successfully. Skipping training.")
            model, optimizer, best_epoch, best_val_loss = loaded_data
            # Return early with loaded model and minimal metrics
            return model, {
                'training_time_sec': 0,
            #    'peak_memory_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                'num_turbines': num_turbines,
                'total_time_steps_trained_on': 0
            }

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    start_time = time.time()
    peak_memory_mb = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = batch.to(device)
            
            # Get target values (power output only)
            # Reshape targets to match model output shape
            targets = batch.output.to(device)  # Shape: [batch_size, output_seq_len, num_turbines * num_features]
            target_power = targets.view(targets.size(0), targets.size(1), num_turbines, -1)  # Shape: [batch_size, output_seq_len, num_turbines, num_features]
            target_power = target_power[..., INPUT_FEATURES.index(TARGET_FEATURE)].unsqueeze(-1)  # Shape: [batch_size, output_seq_len, num_turbines, 1]
            target_power = target_power.permute(0, 2, 1, 3)  # Shape: [batch_size, num_turbines, output_seq_len, 1]
            
            # Forward pass
            outputs = model(batch)  # Shape: [batch_size, num_turbines, output_seq_len, 1]
            loss = criterion(outputs, target_power)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        #    peak_memory_mb = max(peak_memory_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
        
        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Get target values (power output only)
                targets = batch.output.to(device)
                target_power = targets.view(targets.size(0), targets.size(1), num_turbines, -1)
                target_power = target_power[..., INPUT_FEATURES.index(TARGET_FEATURE)].unsqueeze(-1)
                target_power = target_power.permute(0, 2, 1, 3)
                
                outputs = model(batch)
                val_loss = criterion(outputs, target_power)
                
                total_val_loss += val_loss.item()
                num_val_batches += 1
            #    peak_memory_mb = max(peak_memory_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            print(f"New best validation loss at epoch {epoch+1}. Saving GNN model...")
            save_individual_model(model, optimizer, epoch, avg_val_loss, "GNN")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best epoch was {best_epoch+1} with validation loss: {best_val_loss:.4f}")
                break
    
    end_time = time.time()
    training_time_sec = end_time - start_time

    # Load the best model state if it exists
    best_model_path = os.path.join(TRAINED_MODELS_DIR, f"GNN_epoch_{best_epoch+1}.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best model from epoch {best_epoch+1}...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Prepare scalability metrics
    scalability_metrics = {
        'training_time_sec': training_time_sec,
        'peak_memory_mb': peak_memory_mb,
        'num_turbines': num_turbines,
        'total_time_steps_trained_on': len(train_loader.dataset.dataset.data_df)
    }
    
    return model, scalability_metrics 