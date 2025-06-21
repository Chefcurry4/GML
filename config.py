import os

# Time Series & Windowing
INPUT_SEQUENCE_LENGTH = 12 # Number of past time steps to use for prediction (e.g., 12 * 10 min = 2 hours)
OUTPUT_SEQUENCE_LENGTH = 1 # Number of future time steps to predict (e.g., 10 minutes ahead)

# Dataset
TRAIN_VAL_SPLIT_RATIO = 0.8 # Ratio of data used for training divided by whole data
SHUFFLE_TRAIN_VAL_DATASET = False # Whether to shuffle the training and validation data. The sliding window will still work

# Graph Construction
TEMPORAL_GRAPH_TYPE = 'sequential'  # Only sequential supported for now
DOMDIR_WIND_DIR = 0
DOMDIR_INCLUDE_WEIGHTS = False
DOMDIR_DECAY_LENGTH = 1000.0  # For calculation of edge weights
DOMDIR_ANGLE_THRESHOLD = 20.0 # For edge creation based on wind direction similarity
DOMDIR_MAX_DISTANCE = 1500.0  # Maximum distance for edge creation based on wind direction similarity

# Output and Checkpointing
OUTPUT_DIR = 'output/'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints/')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions/')

# Data Paths
DATA_DIR = 'data/'
SCADA_DATA_PATH = os.path.join(DATA_DIR, 'wind_power_sdwpf.csv') # Ensure your downloaded file matches this name
LOCATION_DATA_PATH = os.path.join(DATA_DIR, 'turbine_location.CSV') # Ensure your downloaded file matches this name

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True) 