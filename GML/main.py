import pandas as pd
from config import INPUT_SEQUENCE_LENGTH, LOCATION_DATA_PATH, OUTPUT_SEQUENCE_LENGTH, SCADA_DATA_PATH, SHUFFLE_TRAIN_VAL_DATASET, TRAIN_VAL_SPLIT_RATIO
from models.fastGCN import train_fastgcn
from data_preprocessing import load_and_preprocess_data
from graph_construction import build_graph
from args_interface import parse_args
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
    edge_index, edge_attr = build_graph(locations_df, args)
    # ===========================================================

    exit()

    # Step 3: Choose the model and train it
    # TODO: use preprocessed data
    scada_path = os.path.join("GML", "data", "wind_power_sdwpf.csv") 
    train_fastgcn(scada_path, edge_index)  

    # Step 4: Evaluate the model
    # evaluate_model(model, data, args)


if __name__ == "__main__":
    main()