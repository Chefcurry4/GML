# wind_power_forecasting/models/product_graph_gnn.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv # Example GNN layers
import torch.nn.functional as F
# Removed config imports, pass necessary params or get from batch
from config import INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, TARGET_FEATURE, INPUT_FEATURES # Need these for indexing/shapes

# Need to map target feature name to index in the feature vector
# This mapping depends on the order of features in INPUT_FEATURES
# The data loading and collate_fn must maintain this order.
# Assuming INPUT_FEATURES list defines the order:
try:
    TARGET_FEATURE_INDEX = INPUT_FEATURES.index(TARGET_FEATURE)
except ValueError:
    raise ValueError(f"Target feature '{TARGET_FEATURE}' not found in INPUT_FEATURES list.")


class ProductGraphGNN(nn.Module):
    def __init__(self, initial_input_dim, hidden_dim, num_layers, dropout, num_turbines, input_sequence_length, output_sequence_length):
        super(ProductGraphGNN, self).__init__()
        self.initial_input_dim = initial_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_turbines = num_turbines # Stored for prediction head logic
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        # GNN layers
        self.convs = nn.ModuleList()
        # Standard GCNConv doesn't use edge_attr directly, but it can be extended or replaced
        # with layers like GCNConv which support edge_weight (if edge_attr is scalar weight)
        # or custom message passing layers. For simplicity, starting with basic GCNConv
        # ignoring the edge_attr indicator for now. A more advanced version could use it.
        self.convs.append(GCNConv(initial_input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer: Predicts OUTPUT_SEQUENCE_LENGTH steps of the TARGET_FEATURE
        # Input to prediction head: hidden state from the LAST input time step for EACH turbine
        # Shape (batch_size * num_turbines, hidden_dim)
        # Output needed: (batch_size * num_turbines, OUTPUT_SEQUENCE_LENGTH) - flat output for FC
        self.prediction_head = nn.Linear(hidden_dim, output_sequence_length)


    def forward(self, data):
        # data is a PyG Batch object created by the custom collate_fn
        # It contains:
        # data.x: Node features (batch_size * num_turbines * input_sequence_length, initial_input_dim)
        # data.edge_index: Edge index for the batched product graph (2, num_edges_total)
        # data.edge_attr: Edge attributes (num_edges_total, dim) - indicator in our case
        # data.batch: Batch assignment vector (batch_size * num_turbines * input_sequence_length)
        # data.input_mask: Original missing masks for input features (batch_size * num_turbines * input_sequence_length, num_features_per_turbine)
        # data.output: Target values for the future (batch_size, out_seq_len, num_turbines, num_features_per_turbine)
        # data.output_mask: Missing masks for target values (batch_size, out_seq_len, num_turbines, num_features_per_turbine)
        # data.num_turbines: Number of turbines in this batch (redundant if batch size fixed num_turbines)

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply GCN layers
        h = x
        for i, conv in enumerate(self.convs):
            # Pass edge_attr if the GCNConv variant supports it and it's meaningful (our indicator attr might not be directly useful for standard GCNConv)
            # For standard GCNConv(in_channels, out_channels, improved=False, cached=False, normalize=True, bias=True)
            # it doesn't take edge_attr unless modified or using a different layer type.
            # Let's pass None for edge_attr to use standard GCNConv.
            h = conv(h, edge_index, edge_weight=None) # Use edge_weight if edge_attr is a scalar weight, otherwise None
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)

        # h shape: (TotalNodesInBatch, hidden_dim) = (batch_size * num_turbines * input_sequence_length, hidden_dim)

        # --- Prediction Head ---
        # Extract node embeddings corresponding to the last input time step for each turbine.
        # Node index `i` corresponds to (time_step_in_window, turbine_id) where
        # time_step_in_window = i // self.num_turbines
        # turbine_id = i % self.num_turbines
        # We want nodes where time_step_in_window == self.input_sequence_length - 1.
        # So, indices i such that `i // self.num_turbines == self.input_sequence_length - 1`.

        # This logic needs to be applied correctly across the batch.
        # For a batch of size B, total nodes N_p_total = B * self.num_turbines * self.input_sequence_length.
        # The first B nodes are (t=0, u=0) for each batch item.
        # The nodes for the last input time step (T_in_L-1) are located at indices:
        # (T_in_L-1 * self.num_turbines + u) for u in 0...self.num_turbines-1, for each batch item.

        # Example indices for batch_item_k:
        # start_idx_of_graph_k = k * (self.num_turbines * self.input_sequence_length)
        # indices_in_graph_k_for_last_step = [ (self.input_sequence_length - 1) * self.num_turbines + u for u in range(self.num_turbines) ]
        # actual_indices_in_batch_k = [ start_idx_of_graph_k + idx_in_graph for idx_in_graph in indices_in_graph_k_for_last_step ]
        # We can build a tensor of these indices.

        last_step_node_indices_in_batch = []
        num_nodes_per_window = self.num_turbines * self.input_sequence_length
        last_step_start_idx_in_window = (self.input_sequence_length - 1) * self.num_turbines

        # Iterate through batch indices (0 to batch.max())
        for graph_idx in range(batch.max().item() + 1):
            # Find the starting node index for this graph in the flattened batch tensor
            # PyG batches nodes by concatenating graphs: [Graph0_nodes, Graph1_nodes, ...]
            # The start index of Graph k in the flattened batch tensor is k * (nodes per graph)
            # This assumes all graphs in the batch have the same number of nodes, which is true here.
            start_idx_in_batch = graph_idx * num_nodes_per_window
            indices_for_this_graph = torch.arange(
                start_idx_in_batch + last_step_start_idx_in_window,
                start_idx_in_batch + last_step_start_idx_in_window + self.num_turbines,
                device=x.device # Ensure indices are on the same device as x
            )
            last_step_node_indices_in_batch.append(indices_for_this_graph)

        last_step_node_indices_in_batch = torch.cat(last_step_node_indices_in_batch) # Shape (batch_size * num_turbines,)

        # Extract embeddings for these nodes
        last_step_embeddings = h[last_step_node_indices_in_batch] # Shape (batch_size * num_turbines, hidden_dim)

        # Apply prediction head
        predictions_flat = self.prediction_head(last_step_embeddings) # Shape (batch_size * num_turbines, OUTPUT_SEQUENCE_LENGTH)

        # Reshape to (batch_size, num_turbines, OUTPUT_SEQUENCE_LENGTH, 1) to match evaluation/loss format
        predictions = predictions_flat.view(-1, self.num_turbines, self.output_sequence_length, 1)


        return predictions 