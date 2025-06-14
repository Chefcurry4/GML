from config import INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, SCADA_DATA_PATH, SHUFFLE_TRAIN_VAL_DATASET, TRAIN_VAL_SPLIT_RATIO
from models.fastGCN import train_fastgcn
from data_preprocessing import load_and_preprocess_data
from graph_construction import build_graph, build_spatial_graph
# from graph_temporal import build_temporal_graph
# from graph_product import build_product_graph
# from models.gcn import GCNModel
# from models.graphsage import GraphSageModel
# from training import train_model
# from evaluation import evaluate_model
from args_interface import parse_args
from utils import visualize_spatial_graph, get_image_path
import os

def main():
    # Load the arguments passed from the command line
    args = parse_args()

    # ==========================================================
    # Step 0: Data Preprocessing
    X_train, Y_train, X_val, Y_val = load_and_preprocess_data(
        csv_path=SCADA_DATA_PATH,
        input_len=INPUT_SEQUENCE_LENGTH,
        output_len=OUTPUT_SEQUENCE_LENGTH,
        train_val_ratio=TRAIN_VAL_SPLIT_RATIO,
        data_subset=args.data_subset,
        data_subset_turbines=args.data_subset_turbines,
        shuffle_train_val_dataset=SHUFFLE_TRAIN_VAL_DATASET
    )
    # ===========================================================

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_val shape:", X_val.shape)
    print("Y_val shape:", Y_val.shape)
    exit()
    # ===========================================================
    # Step 1: Build Spatial-Temporal Graph
    # edge_index, edge_attr, locations = build_graph(data, args)
    # ===========================================================

    # TODO: remove
    print("Done")

    # Step 3: Choose the model and train it
    # TODO: use preprocessed data
    scada_path = os.path.join("GML", "data", "wind_power_sdwpf.csv") 
    train_fastgcn(scada_path, edge_index)  

    # Step 4: Evaluate the model
    # evaluate_model(model, data, args)


if __name__ == "__main__":
    main()