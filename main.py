import pandas as pd
from config import INPUT_SEQUENCE_LENGTH, LOCATION_DATA_PATH, OUTPUT_SEQUENCE_LENGTH, SCADA_DATA_PATH, SHUFFLE_TRAIN_VAL_DATASET, TRAIN_VAL_SPLIT_RATIO
from models.fastGCN import train_fastgcn_from_arrays
from models.GCN import forecast, train_gcn
from models.clusterGCN import train_clustergcn_from_arrays
from utils.new_data_preprocessing import get_patv_feature_idx, load_and_preprocess_data
from utils.graph_construction import build_graph
from utils.args_interface import parse_args
import os
from utils.utils import plot_power_output_and_prediction

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
        shuffle_train_val_dataset=SHUFFLE_TRAIN_VAL_DATASET,
        args=args
    )
    # ===========================================================

    # ===========================================================
    # Step 1: Build Spatio-Temporal Graph
    edge_index = build_graph(locations_df, args)
    # ===========================================================

    # ===========================================================
    # Step 3: Choose the model and train it
    model = None
    if args.model_type == 'fast-gcn':
        model, train_losses, val_losses = train_fastgcn_from_arrays(
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
            batch_size=args.batch_size,
            args=args
        )

    elif args.model_type == 'cluster-gcn':
        model, train_losses, val_losses = train_clustergcn_from_arrays(
            X_train, Y_train, X_val, Y_val, edge_index,
            hidden=args.hidden_dimensions,
            dropout=args.dropout_rate,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            args=args
        )

    elif args.model_type == 'gcn':
        model, train_losses, val_losses = train_gcn(
            X_train, Y_train, X_val, Y_val, edge_index,
            hidden=args.hidden_dimensions,
            dropout=args.dropout_rate,
            epochs=args.epochs,
            lr=args.learning_rate,
            patience=args.patience,
            batch_size=args.batch_size,
            args=args
        )

    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    # ===========================================================

    # ===========================================================
    # Step 4: Evaluate the model
    # Plot graphs showing the forecasting
    if not args.plot_images:
        return

    num_plots = 5
    patv_idx = get_patv_feature_idx()
    turbine_ids = [0, 12, 34, 42, 69, 120]
    save_dir = os.path.join(args.image_path, 'patv_prediction_plots')
    os.makedirs(save_dir, exist_ok=True)

    # Do for training data
    for i in range(num_plots):
        num_turbines = Y_train[i].shape[0]
        y_pred_train = forecast(model, X_train[i], edge_index, num_turbines)
        plot_power_output_and_prediction(
            X_sample=X_train[i],
            Y_sample=Y_train[i],
            Y_prediction=y_pred_train,
            turbine_ids=turbine_ids,
            image_name=f"train_pred_{i}.png",
            save_dir=save_dir,
            patv_idx=patv_idx
        )
        y_pred_val = forecast(model, X_val[i], edge_index, num_turbines)
        plot_power_output_and_prediction(
            X_sample=X_train[i],
            Y_sample=Y_train[i],
            Y_prediction=y_pred_val,
            turbine_ids=turbine_ids,
            image_name=f"val_pred_{i}.png",
            save_dir=save_dir,
            patv_idx=patv_idx
        )


if __name__ == "__main__":
    main()
