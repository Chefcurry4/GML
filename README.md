# 🌪️ Wind Power Forecasting with Product Graph Neural Networks

## 🎯 Project Overview

This project implements a sophisticated approach to wind power forecasting using Product Graph Neural Networks (GNNs). The system uniquely combines **joint forecasting and missing data interpolation** through a product graph structure that captures both spatial relationships between wind turbines and temporal dependencies in the data.

## 🔑 Key Features

### 1. 📊 Joint Interpolation & Forecasting
- Product Graph GNN architecture that handles missing data and forecasting simultaneously
- Innovative placeholder system with optional masking for missing values
- Real-time interpolation during the forecasting process

### 2. 📈 Scalability Analysis
- Comprehensive testing on varying subsets of turbines and time periods
- Performance metrics tracking (training time, memory usage)
- Scalability evaluation with different dataset sizes

### 3. 🕸️ Product Graph Architecture
- Combines spatial (turbine locations) and temporal dimensions
- Smart node indexing: `time_step_in_window * num_turbines + turbine_id`
- Flexible graph construction with e-ball or k-NN approaches

### 4. 🔄 Data Processing
- Tikhonov regularization for baseline interpolation
- MinMaxScaler normalization with training-data-only fitting
- Custom collate function for GNN batching with placeholder handling

## 🏗️ Project Structure

```
├── config.py                    # Configuration parameters and settings
├── utils.py                     # Data loading, preprocessing, and utility functions
├── graph_construction.py        # Spatial, temporal, and product graph builders
├── missing_data_handling.py     # Interpolation methods including Tikhonov
├── models/
│   ├── gru.py                   # GRU baseline model
│   └── product_graph_gnn.py     # Main Product Graph GNN implementation
├── data/
│   ├── turbine_location.csv     # Turbines x-y location
│   └── wind_power_sdwpf.csv     # Main dataset 
├── requirements.txt             # Install necessary dependencies
├── training.py                  # Training loops and procedures
├── evaluation.py                # Metrics calculation and evaluation
└── main.py                      # Experiment orchestration
```

## 🛠️ Technical Components

### Data Processing
- SCADA data preprocessing
- Missing data identification
- Custom TimeSeriesSlidingWindowDataset
- Specialized dataloaders with custom collate functions

### Models
1. **Product Graph GNN**
   - Joint interpolation-forecasting architecture
   - Batched product graph handling
   - Dynamic node embedding extraction

2. **GRU Baseline**
   - Traditional sequence modeling
   - Pre-interpolated data processing

### Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² Score
- Training time and memory usage metrics

## 🚀 Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your experiment in `config.py`:
   - Set data paths
   - Adjust preprocessing parameters
   - Configure model hyperparameters
   - Set scalability testing parameters

3. Run experiments:
```bash
python main.py
```

## 📊 Evaluation

The system evaluates:
- Forecast accuracy on non-missing target values
- Interpolation quality
- Computational efficiency
- Memory usage across different dataset sizes

## 🔧 Implementation Details

### Graph Construction
- Spatial graphs: e-ball or k-NN based on turbine locations
- Temporal graphs: sequential connections with optional skip connections
- Product graphs: combined spatial-temporal representation

### Training Process
- Early stopping
- Model checkpointing
- Scalability metrics collection
- Custom loss functions for joint optimization

## 📝 Notes

- GPU acceleration supported (automatically detected)
- Modular design for easy extension
- Comprehensive logging and metric tracking
- Flexible configuration system

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

## 📚 References

- Product Graph Neural Networks
- Tikhonov Regularization
- Time Series Forecasting with GNNs
- Missing Data Interpolation Techniques 
