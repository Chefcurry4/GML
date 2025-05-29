import os
import re
from collections import defaultdict

def cleanup_model_files(models_dir):
    """Keeps only the latest epoch model for each turbine and model type."""
    # Group files by turbine and model type
    model_groups = defaultdict(list)
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.pt'):
            if 'turbine' in filename:
                # Extract turbine number and epoch
                match = re.match(r'([A-Za-z]+)_turbine_(\d+)_epoch_(\d+)\.pt', filename)
                if match:
                    model_type, turbine, epoch = match.groups()
                    key = (model_type, turbine)
                    model_groups[key].append((int(epoch), filename))
            else:
                # Handle GNN models without turbine number
                match = re.match(r'([A-Za-z]+)_epoch_(\d+)\.pt', filename)
                if match:
                    model_type, epoch = match.groups()
                    key = (model_type, None)
                    model_groups[key].append((int(epoch), filename))
    
    # For each group, keep only the latest epoch
    for key, files in model_groups.items():
        if files:
            # Sort by epoch number
            files.sort(key=lambda x: x[0])
            # Keep only the latest epoch file
            files_to_remove = files[:-1]  # All but the last one
            
            # Remove older files
            for _, filename in files_to_remove:
                filepath = os.path.join(models_dir, filename)
                try:
                    os.remove(filepath)
                    print(f"Removed old model file: {filename}")
                except Exception as e:
                    print(f"Error removing {filename}: {e}")

if __name__ == "__main__":
    models_dir = "trained_models"
    if os.path.exists(models_dir):
        print("Cleaning up old model files...")
        cleanup_model_files(models_dir)
        print("Cleanup complete!")
    else:
        print(f"Models directory '{models_dir}' not found.") 