import pandas as pd
from config import INPUT_SEQUENCE_LENGTH, LOCATION_DATA_PATH, OUTPUT_SEQUENCE_LENGTH, SCADA_DATA_PATH, SHUFFLE_TRAIN_VAL_DATASET, TRAIN_VAL_SPLIT_RATIO
from models.fastGCN import train_fastgcn_from_arrays
from models.GCN import train_gcn
from models.clusterGCN import train_clustergcn_from_arrays
from utils.new_data_preprocessing import load_and_preprocess_data
from utils.graph_construction import build_graph
from utils.args_interface import parse_args
import os

def main():
    # Load the arguments passed from the command line
    args = parse_args()

    # ==========================================================
    # Step 0: Data Preprocessing
    X_train, Y_train, X_val, Y_val, locations_df = load_and_preprocess_data(
        csv_path=SCADA_DATA_PATH,
        input_len=INPUT_SEQUENCE_LENGTH,
        output_len=OUTPUT_SEQUENCE_LENGTH,
        train_val_ratio=TRAIN_VAL_SPLIT_RATIO,
        data_subset=args.data_subset,
        data_subset_turbines=args.data_subset_turbines,
        shuffle_train_val_dataset=SHUFFLE_TRAIN_VAL_DATASET
    )
    # ===========================================================

    # ===========================================================
    # Step 1: Build Spatio-Temporal Graph
    edge_index = build_graph(locations_df, args)
    # ===========================================================

    # ===========================================================
    # Step 3: Choose the model and train it
    if args.model_type == 'fast-gcn':
        model = train_fastgcn_from_arrays(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            edge_index=edge_index,
            hidden=args.hidden_dimensions,
            dropout=args.dropout_rate,
            epochs=args.epochs,
            lr=args.learning_rate,
            patience=args.patience,
            batch_size=args.batch_size
        )

    elif args.model_type == 'cluster-gcn':
        model = train_clustergcn_from_arrays(
            X_train, Y_train, X_val, Y_val, edge_index,
            hidden=args.hidden_dimensions,
            dropout=args.dropout_rate,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.learning_rate
        )

    elif args.model_type == 'gcn':
        model = train_gcn(
            X_train, Y_train, X_val, Y_val, edge_index,
            hidden=args.hidden_dimensions,
            samples=0,  # not used for standard GCN
            dropout=args.dropout_rate,
            epochs=args.epochs,
            lr=args.learning_rate,
            patience=args.patience,
            batch_size=args.batch_size
        )

    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    # ===========================================================

    # Step 4: Evaluate the model
    # evaluate_model(model, data, args)


if __name__ == "__main__":
    main()
