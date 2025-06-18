# wind_power_forecasting/models/gru.py

import torch
import torch.nn as nn
import torch.optim as optim
# Removed config imports, pass necessary params directly

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_seq_len, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len

        # GRU layer: takes input_dim features per time step
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # FC layer: maps hidden state to the output sequence of target feature values
        # Predicts `output_seq_len` values (one for each future step)
        self.fc = nn.Linear(hidden_dim, output_seq_len)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0) # out: (batch_size, seq_len, hidden_dim)

        # Use the hidden state from the last input time step for prediction
        last_step_output = out[:, -1, :] # Shape (batch_size, hidden_dim)

        # Predict the next `output_seq_len` values using the FC layer
        predictions = self.fc(last_step_output) # Shape (batch_size, output_seq_len)

        # Add a dimension for the single target feature
        predictions = predictions.unsqueeze(-1) # Shape (batch_size, output_seq_len, 1)

        return predictions 