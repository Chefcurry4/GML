# wind_power_forecasting/config.py

import os

# Data Paths
DATA_DIR = 'GML/data/'
SCADA_DATA_PATH = os.path.join(DATA_DIR, 'wind_power_sdwpf.csv') # Ensure your downloaded file matches this name
LOCATION_DATA_PATH = os.path.join(DATA_DIR, 'turbine_location.CSV') # Ensure your downloaded file matches this name

# Data Preprocessing
TARGET_FEATURE = 'Patv' # Power output feature
INPUT_FEATURES = ['Patv', 'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv'] # All available features from the dataset
FEATURES_TO_NORMALIZE = ['Patv', 'Wspd', 'Etmp', 'Itmp', 'Prtv'] # Features that represent continuous physical quantities

# Time Series & Windowing
TIME_STEP_MINUTES = 10 # Data recording frequency
INPUT_SEQUENCE_LENGTH = 12 # Number of past time steps to use for prediction (e.g., 12 * 10 min = 2 hours)
OUTPUT_SEQUENCE_LENGTH = 1 # Number of future time steps to predict (e.g., 10 minutes ahead)

# Dataset
TRAIN_VAL_SPLIT_RATIO = 0.8 # Ratio of data used for training divided by whole data
SHUFFLE_TRAIN_VAL_DATASET = True # Whether to shuffle the training and validation data. The sliding window will still work

# Graph Construction
SPATIAL_GRAPH_TYPE = 'radius'  # 'radius' or 'knn'
SPATIAL_RADIUS = 1500  # Radius for radius-based graph
K_NEIGHBORS = 5  # Number of neighbors for kNN graph
TEMPORAL_GRAPH_TYPE = 'sequential'  # Only sequential supported for now
DOMDIR_WIND_DIR = 0
DOMDIR_INCLUDE_WEIGHTS = False
DOMDIR_DECAY_LENGTH = 1000.0  # For calculation of edge weights
DOMDIR_ANGLE_THRESHOLD = 20.0 # For edge creation based on wind direction similarity
DOMDIR_MAX_DISTANCE = 1500.0  # Maximum distance for edge creation based on wind direction similarity

# Missing Data Handling
# Simple baselines: applied BEFORE training/dataset creation
SIMPLE_INTERPOLATION_METHODS = ['mean', 'median', 'ffill', 'bfill'] # 'remove' is less practical for forecasting
TIKHONOV_LAMBDA = 1.0 # Regularization parameter for Tikhonov interpolation

# Joint GNN handling: applied WITHIN model/dataloader
MISSING_VALUE_PLACEHOLDER_GNN = 0.0 # Value to replace NaNs in GNN input features
USE_MISSING_MASK_GNN = True # If True, concatenate a binary mask indicating original missingness

# Training Parameters
TRAIN_SPLIT_RATIO = 0.7 # Ratio of windows for training
VAL_SPLIT_RATIO = 0.15 # Ratio of windows for validation
TEST_SPLIT_RATIO = 0.15 # Ratio of windows for testing (should sum to 1 with buffer)
BATCH_SIZE = 64 # Number of independent sliding windows per batch
LEARNING_RATE = 0.0015
NUM_EPOCHS = 25 # Increased epochs for training stability
PATIENCE = 5 # Early stopping patience (epochs)

# Model Parameters
GRU_HIDDEN_DIM = 32
GNN_HIDDEN_DIM = 32
GNN_NUM_LAYERS = 2 # Increased layers
GNN_DROPOUT = 0.2

# Scalability Testing Parameters
# Define subsets of turbine IDs and time periods to test GNN scalability
# Need to load data first to get available turbine IDs
# SCALABILITY_TURBINE_SUBSETS = [10, 50, 134] # Test with varying numbers of turbines
# SCALABILITY_TIME_SUBSETS_DAYS = [30, 90, 245] # Test with varying total time periods


# Output and Checkpointing
OUTPUT_DIR = 'output/'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints/')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions/')

# Device Selection
DEVICE_SELECTION = "cuda"  # Options: "cuda", "cpu"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True) 