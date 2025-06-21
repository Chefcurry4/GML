# 🌤️💨 Wind Power Forecasting with GNN and GRU Models 

Accurate wind power forecasting is essential for reliable integration of renewable energy into modern power markets. This project tackles the dual challenge of modeling both temporal patterns and spatial dependencies between turbines—without relying on expensive physical simulations or sacrificing scalability as wind farms grow.

We investigate the practical scalability and forecasting performance of various Graph Convolutional Network (GCN) architectures for short-term, turbine-level wind power prediction. Our methodology constructs a spatio-temporal product graph using multiple spatial graph formulations to capture both spatial and temporal dependencies.

This repository provides a comprehensive pipeline for implementing, benchmarking, and experimenting with these GNN models on the [SDWPF dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC11187227/).

## 📋 Table of Contents
- [🌤️💨 Wind Power Forecasting with GNN and GRU Models](#️-wind-power-forecasting-with-gnn-and-gru-models)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔧 Installation](#-installation)
  - [🏗️ Project Structure](#️-project-structure)
  - [🚀 Usage](#-usage)
    - [Download Data](#download-data)
    - [Running experiments](#running-experiments)
  - [🏃 Model Runner](#-model-runner)
  - [⚙️ Configuration](#️-configuration)
  - [📊 Data Description](#-data-description)
  - [📈 Future Work](#-future-work)

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
├── data/                        # Downloaded data will be stored here
│   ├── wind_power_sdwpf.csv     # SCADA data
│   └── turbine_location.CSV     # Turbine location data
├── models/                      # Model architectures
├── images/                      # Output images will be stored here
├── jupyter_notebooks/           # Jupyter notebooks for dataset exploration and graph construction
├── output/                      # Outputs of the experiment runs
├── config.py                    # Configuration parameters
├── download_data.py             # Script used to download dataset
├── main.py                      # Main script of pipeline
├── model-runner.py              # Model runner to automatically run experiments
```

## 🚀 Usage

### Download Data

Before you can start using the model, you need to download the dataset. This can be done using the script `download_data.py` script.

```bash
python download_data.py
```

### Running experiments

Train a model using:
```bash
python main.py [SPATIAL_GRAPH_TYPE] [MODEL_TYPE] [OPTIONS]
```

You can customize your experiment by passing the following arguments to `main.py`:

**Positional Arguments**
- **`spatial_graph_type`**: Type of spatial graph to use. Choices:
  - `knn` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (K-Nearest Neighbors)
  - `radius` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Epsilon-ball / radius graph)
  - `domdir` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (Dominant direction graph)
- **`model_type`**: Model architecture to use. Choices:
  - `gcn`
  - `fast-gcn`
  - `cluster-gcn`

**Optional Arguments**
- **`--data-subset-turbines`**: Number of turbines to use from the training set (default: -1, use all).
- **`--data-subset`**: Percentage of the dataset to use for training/validation (e.g. `0.2` for 20%, default: `1.0`).
- **`--epochs`**: Number of epochs to train the model (default: `40`).
- **`--dropout-rate`**: Dropout rate for the model (default: `0.35`).
- **`--hidden-dimensions`**: Number of hidden dimensions in the model (default: `128`).
- **`--batch-size`**: Batch size for training (default: `32`).
- **`--patience`**: Early stopping patience (default: `10`).
- **`--learning-rate`**: Learning rate for the optimizer (default: `0.001`).
- **`--knn-neighbors`**: Number of neighbors for KNN graph (when `knn` is selected, default: `5`).
- **`--spatial-radius`**: Radius for spatial graph (when `radius` is selected, default: `1500`).
- **`--plot-images`**: Plot images during training (flag, default: `False`).
- **`--image-path`**: Path to save images during training (default: `images/<datetime>`).

**Example**
Use epsilon-ball (radius) for spatial graph, train a GCN, only use 20% of the dataset and plot images
```bash
python main.py radius gcn --data-subset 0.2 --plot-images
```

## 🏃 Model Runner

If you want to run several models after each other, you can use the model runner.

```bash
python model-runner.py [SCHEDULE_FILE]
```

This will run all the variations defined in the provided schedule file and store the output to `output/model_runs`. Some examples of schedules can be found in the `schedules/` folder.

## ⚙️ Configuration

Additional parameters can be customized in `config.py`. 

- **Time Series & Windowing**
  - `INPUT_SEQUENCE_LENGTH`: Number of past time steps used as input for prediction (e.g., 12 for 2 hours if each step is 10 minutes).
  - `OUTPUT_SEQUENCE_LENGTH`: Number of future time steps to predict (e.g., 1 for 10 minutes ahead).

- **Dataset Splitting**
  - `TRAIN_VAL_SPLIT_RATIO`: Fraction of data used for training (rest for validation).
  - `SHUFFLE_TRAIN_VAL_DATASET`: Whether to shuffle the dataset before splitting.

- **Graph Construction**
  - Parameters for spatial graph construction, such as dominant wind direction, edge weights, angle thresholds, and maximum distance for edge creation.

## 📊 Data Description

The following files are used in this project:
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

## 📈 Future Work

While this repository provides a solid foundation for wind power forecasting—including a flexible pipeline and tools for spatio-temporal graph exploration—the current GNN-based models struggle with accurate forecasting due to a lack of explicit temporal awareness. Although the framework is in place, the models have difficulty capturing temporal dependencies within the data, which limits their predictive performance compared to dedicated temporal models like GRUs.

Future work could focus on:
- Integrating more advanced spatio-temporal architectures (e.g., combining GNNs with temporal modules such as GRU/LSTM or Temporal Convolutional Networks).
- Exploring attention mechanisms or transformer-based models for improved temporal and spatial reasoning.
- Developing better strategies for handling missing data and improving data interpolation.
- Benchmarking additional scalable GNN variants and hybrid approaches.
- Enhancing the interpretability and robustness of the models.

We welcome and encourage contributions from the community! If you have ideas for improving temporal modeling, novel graph constructions, or other enhancements, feel free to open an issue or submit a pull request.

   
