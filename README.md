# Wind Power Forecasting with GNN and GRU Models

This project implements wind power forecasting using both Graph Neural Networks (GNN) and Gated Recurrent Units (GRU). It handles spatial-temporal data from multiple wind turbines, incorporating both temporal dependencies and spatial relationships between turbines.

## 📋 Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Models](#models)

## 🔧 Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd GML
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: PyTorch Geometric installation might require specific CUDA versions. Follow the official [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your setup.

## 🏗️ Project Structure

```
GML/
├── data/                    # Data directory
│   ├── wind_power_sdwpf.csv    # SCADA data
│   └── turbine_location.CSV    # Turbine location data
├── models/                  # Model architectures
│   ├── gru.py                  # GRU model implementation
│   └── product_graph_gnn.py    # GNN model implementation
├── config.py               # Configuration parameters
├── main.py                # Main training script
├── training.py            # Training loop implementations
├── evaluation.py          # Model evaluation functions
├── graph_construction.py  # Graph building utilities
├── utils.py              # Utility functions
├── missing_data_handling.py # Missing data interpolation
├── datasets_stats.py     # Dataset statistics
└── cleanup.py           # Utility for cleaning model files
```

## 🚀 Usage

### Download Data

Before you can start using the model, you need to download the dataset. This can be done using the script `download_data.py` script.

```bash
python GML/download_data.py
```

### Basic Training

Train a model using:
```bash
python GML/main.py [MODEL_TYPE] [INTERPOLATION_METHOD] [OPTIONS]
```

Arguments:
- `MODEL_TYPE`: Choose from ['GRU', 'GNN', 'BOTH']
- `INTERPOLATION_METHOD`: Choose from ['remove', 'mean', 'median', 'ffill', 'bfill', 'linear', 'tikhonov', 'joint']

Options:
- `--force-retrain`: Force retraining even if saved models exist (NOTE THAT TRAINING A MODEL WITH THE WHOLE DATASET REQUIRES HOURS)
- `--data-subset-time-days N`: Use only N days of data (this might still not work perfectly)
- `--data-subset-turbines N`: Use only N turbines (this might still not work perfectly)

Examples:
```bash
# Train GNN model with joint interpolation
python GML/main.py GNN joint

# Train both models with mean interpolation using subset of data
python GML/main.py BOTH mean --data-subset-time-days 30 --data-subset-turbines 5

# Force retrain GRU model
python GML/main.py GRU mean --force-retrain
```

### Cleanup Old Models

To remove old model checkpoints and keep only the latest: (this might still not work perfectly)
```bash
python GML/cleanup.py 
```

## ⚙️ Configuration

Key parameters in `config.py`:

### Data Processing
- `INPUT_FEATURES`: List of features to use
- `TARGET_FEATURE`: Target feature to predict (default: 'Patv')
- `INPUT_SEQUENCE_LENGTH`: Number of past time steps (default: 12)
- `OUTPUT_SEQUENCE_LENGTH`: Number of future steps to predict (default: 1)

### Training Parameters
- `BATCH_SIZE`: Batch size (default: 64)
- `LEARNING_RATE`: Learning rate (default: 0.0015)
- `NUM_EPOCHS`: Maximum epochs (default: 25)
- `PATIENCE`: Early stopping patience (default: 5)

### Model Parameters
- `GRU_HIDDEN_DIM`: GRU hidden dimension (default: 32)
- `GNN_HIDDEN_DIM`: GNN hidden dimension (default: 32)
- `GNN_NUM_LAYERS`: Number of GNN layers (default: 2)
- `GNN_DROPOUT`: Dropout rate (default: 0.2)

## 📊 Data Format

Required data files:
1. `wind_power_sdwpf.csv`: SCADA data with columns:
   - Day: Day number
   - Tmstamp: Time stamp
   - TurbID: Turbine ID
   - Patv: Active power
   - [Other features...]

2. `turbine_location.CSV`: Turbine locations with columns:
   - TurbID: Turbine ID
   - X: X coordinate
   - Y: Y coordinate

## 🤖 Models

### GRU Model
- Independent GRU for each turbine
- Predicts future power output based on temporal patterns
- Handles missing data through interpolation
  (NOTE: this is not a GNN model but I choose it cause I know that Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) that's designed to handle sequential data more effectively than 
   traditional RNNs. It's particularly useful for tasks like natural language processing, speech recognition, and time series prediction. THIS MODEL REACHED A R^2 score >= 0.95 on the whole dataset (very high)).

### GNN Model ((this still does not work perfectly: must be refined also looking at professor suggestions. We might even opt for more than 1 GNN model.)
- Single model for all turbines
- Captures both spatial and temporal dependencies
- Uses product graph structure
- Handles missing data through joint learning

## 📋 Outputs

The models produce:
1. Trained model checkpoints in `GML/trained_models/`
2. Predictions in `output/predictions/`
3. Evaluation metrics in `output/experiment_results.csv`
4. Optional statistics plots in `output/stats_plots/`

## 📈 Next steps

1. Modify the training pipeline to make it faster (maybe craft a smaller dataset) otherwise we'll take ages to test and train the models.
2. Work extensively on the GNN model introducing all professor suggestions. For now, there's basically no progress in that direction (see point 3.).
3. For now I did NOT address scalability, did NOT try suggested architectures like MPNNs / Attention / others, ecc.


   
