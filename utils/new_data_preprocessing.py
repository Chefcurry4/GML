import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# ADDED: Import StandardScaler for feature and target scaling
from sklearn.preprocessing import StandardScaler
from config import INPUT_SEQUENCE_LENGTH, LOCATION_DATA_PATH, OUTPUT_SEQUENCE_LENGTH, SCADA_DATA_PATH, SHUFFLE_TRAIN_VAL_DATASET, TRAIN_VAL_SPLIT_RATIO
from utils.utils import plot_data_histogram, plot_power_output

# Do not use TurbID, Day and Tmstamp as features, since they are not physical quantities and do not provide useful information for the model.
feature_names = [
    'Wspd',        # Wind speed
    'Wdir',        # Wind direction
    'Etmp',        # Temperature of the surrounding environment
    'Itmp',        # Temperature of the turbine's internal components
    'Ndir',        # Direction of the nacelle
    'Pab1',        # Pitch angle of blade 1
    'Pab2',        # Pitch angle of blade 2
    'Pab3',        # Pitch angle of blade 3
    'Prtv',        # Reactive power
    'Patv',        # Active power output
]

# MODIFIED: The valid_indicator_feature is no longer needed, as we will use np.nan to mark invalid data.
# valid_indicator_feature = "data_valid"


# MODIFIED: This function has been completely rewritten to follow best practices.
def preprocess_data(df):
    """
    Identifies invalid or abnormal data points based on rules from the SDWPF paper
    and marks them as NaN to prepare for interpolation. This function modifies
    the DataFrame in place.
    """
    print("Identifying invalid data points and marking them as NaN for interpolation...")
    
    # Clip Patv values at 0, since negative output power is not possible.
    df['Patv'] = df['Patv'].clip(lower=0)

    # Rule 1: Turbine is stopped for external reasons.
    # When Patv <= 0 and Wspd > 2.5, the reading is invalid.
    # When any blade angle > 89 degrees, the turbine is stopped.
    # We mark the 'Patv' feature as NaN in these cases.
    unknown_mask = ((df['Patv'] <= 0) & (df['Wspd'] > 2.5)) | \
                   (df[['Pab1', 'Pab2', 'Pab3']].gt(89).any(axis=1))
    df.loc[unknown_mask, 'Patv'] = np.nan

    # Rule 2: Abnormal sensor readings due to system errors.
    # Ndir should be within [-720, 720] and Wdir within [-180, 180].
    # When outside this range, the entire row of data for that turbine is suspect.
    abnormal_mask = (df['Ndir'].abs() > 720) | (df['Wdir'].abs() > 180)
    # We mark all features as NaN for these rows, as the entire reading is abnormal.
    df.loc[abnormal_mask, feature_names] = np.nan
    
    # MODIFIED: Instead of calculating based on a flag, we now calculate the percentage
    # of actual NaN values in the target column after marking. This gives a better
    # sense of how much data needs to be filled by interpolation.
    invalid_percentage = df['Patv'].isna().mean() * 100
    print(f"Percentage of 'Patv' values marked as invalid/missing: {invalid_percentage:.2f}%")

    return df

def load_sdwpf_data(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)

    # Combine day and timestamp into a single colum 'TimeIndex' that counts the measurements since day 0 time 0
    # Measurements are done every 10 minutes, convert the Tmstamp to a number (for one specific day)
    def tmstamp_to_index(tmstamp_str):
        # tmstamp_str is like "HH:MM"
        h, m = map(int, tmstamp_str.split(":"))
        return h * 6 + m // 10
    df['TimeIndex'] = df['Tmstamp'].apply(tmstamp_to_index)
    # Now also consider the day of the measurement
    df['TimeIndex'] = (df['Day'] - 1) * 144 + df['TimeIndex']

    # Sort the DataFrame by 'TimeIndex' and 'TurbID'
    df = df.sort_values(['TimeIndex', 'TurbID']).reset_index(drop=True)
    
    return df

# MODIFIED: Removed the 'valid_indicator_feature' from the list of features to pivot.
def pivot_sdwpf_multi(df):
    """
    Transforms a DataFrame containing time series data for multiple turbines and features into a 3D NumPy array.
    """
    pivots = []
    # MODIFIED: We no longer need to pivot the valid_indicator_feature
    for feat in feature_names:
        piv = df.pivot(index='TimeIndex', columns='TurbID', values=feat)
        pivots.append(piv)
    # Stack to shape (num_timesteps, num_turbines, num_features)
    data = np.stack([p.values for p in pivots], axis=-1)
    return data  # shape: (timesteps, turbines, features)

def sliding_window(data: np.ndarray, input_len: int, output_len: int):
    """
    data: shape (timesteps, turbines, features)
    Returns:
        X: (num_samples, input_len, turbines, features)
        Y: (num_samples, output_len, turbines, 1)
        Y only considers the Patv feature (active power output)
    """
    patv_feature_idx = feature_names.index('Patv')  # Index of the Patv feature in the features list

    X, Y = [], []
    total_steps = data.shape[0] - input_len - output_len + 1
    for i in range(total_steps):
        X.append(data[i:i+input_len])
        y_window = data[i+input_len:i+input_len+output_len, :, patv_feature_idx]
        Y.append(y_window)
    X = np.stack(X)
    Y = np.stack(Y)

    if output_len == 1:
        Y = Y[:, 0, :]

    return X, Y


def get_patv_feature_idx():
    return feature_names.index('Patv')

def load_and_preprocess_data(csv_path, input_len=12, output_len=1, train_val_ratio=0.8, data_subset=1, data_subset_turbines=-1, shuffle_train_val_dataset = True, random_state=42, args=None):
    """
    MODIFIED: This function now includes interpolation and scaling for both features (X) and targets (Y).
    Returns:
        X_train_scaled (np.ndarray): Scaled training input data.
        Y_train_scaled (np.ndarray): Scaled training output data.
        X_val_scaled (np.ndarray): Scaled validation input data.
        Y_val_scaled (np.ndarray): Scaled validation output data.
        locations_df (pd.DataFrame): DataFrame with turbine locations.
        y_scaler (StandardScaler): The scaler fitted on Y_train, needed for inverse transform.
    """
    print("\n==========================================================")
    print("Starting data preprocessing...")

    # Load the whole dataset    
    df = load_sdwpf_data(csv_path)

    # Subset the data if needed
    if data_subset < 1.0:
        print("Subsetting data to", data_subset * 100, "%")
        num_rows = int(len(df) * data_subset)
        cur_day = df.iloc[num_rows - 1]['Day']
        cur_tmstamp = df.iloc[num_rows - 1]['Tmstamp']
        while num_rows < len(df) and (df.iloc[num_rows]['Day'] == cur_day and df.iloc[num_rows]['Tmstamp'] == cur_tmstamp):
            num_rows += 1
        df_subset = df.iloc[:num_rows].reset_index(drop=True)
    else:
        df_subset = df


    # Do subsetting for turbines
    if data_subset_turbines > 0:
        print(f"Subsetting data to first {data_subset_turbines} turbines")
        df_subset = df_subset[df_subset['TurbID'] <= data_subset_turbines].reset_index(drop=True)
    
    print("Loaded data shape:", df.shape)

    # Preprocess the data (this now marks invalid data as NaN)
    preprocess_data(df_subset)

    # Pivot the data for easier computation with sliding window
    data = pivot_sdwpf_multi(df_subset) # shape: (timesteps, turbines, features)
    print("Pivoted data shape:", data.shape)
    
    # ADDED: Interpolation step to fill the NaN values.
    # This is done on the 3D numpy array by iterating through turbines and features.
    print("Interpolating data to fill NaN values...")
    for i in range(data.shape[1]): # Iterate over turbines
        for j in range(data.shape[2]): # Iterate over features
            s = pd.Series(data[:, i, j])
            data[:, i, j] = s.interpolate(method='linear', limit_direction='both').to_numpy()

    # ADDED: Check for NaNs *after* interpolation to ensure all gaps were filled.
    if np.isnan(data).any():
        raise ValueError("Data still contains NaN values after interpolation. This might happen if an entire column is NaN.")

    # Create feature and target sequences
    X, Y = sliding_window(data, input_len, output_len)
    print(f"Sliding window created: X shape {X.shape}, Y shape {Y.shape}")

    # Flatten X and Y
    num_samples, sliding_window_size, num_turbines, features_per_turbine = X.shape
    X = X.reshape(num_samples, sliding_window_size * num_turbines, features_per_turbine)
    Y = Y.reshape(num_samples, -1) # Flatten Y to 2D for scaler
    print(f"Flattened X shape: {X.shape}")
    print(f"Flattened Y shape: {Y.shape}")

    # Scale data
    print("Scaling features and target variable...")
    x_scaler = StandardScaler()
    # Reshape to 2D for scaler, fit on training data, then transform and reshape back to 3D
    # Scale X
    nsamples, nsteps, nfeatures = X.shape
    X_scaled = x_scaler.fit_transform(X.reshape(-1, nfeatures)).reshape(nsamples, nsteps, nfeatures)
    # Scale Y
    y_scaler = StandardScaler()
    Y_scaled = y_scaler.fit_transform(Y)

    # Plot histograms of the Patv values to see their distribution
    if args.plot_images:
        plot_data_histogram(X, get_patv_feature_idx(), image_path=args.image_path, filename="patv_histogram_x.png", title="Histogram of Patv Values (X)")
        plot_data_histogram(X_scaled, get_patv_feature_idx(), image_path=args.image_path, filename="patv_histogram_x_scaled.png", title="Histogram of Scaled Patv Values (X_scaled)")

    # Train/val split
    X_train_scaled, X_val_scaled, Y_train_scaled, Y_val_scaled = train_test_split(
        X_scaled, Y_scaled, train_size=train_val_ratio, random_state=random_state, shuffle=shuffle_train_val_dataset
    )

    # Print the sizes of the resulting datasets
    print(f"Training set size: {X_train_scaled.shape}, {Y_train_scaled.shape}")
    print(f"Validation set size: {X_val_scaled.shape}, {Y_val_scaled.shape}")

    # Load locations DataFrame
    locations_df = pd.read_csv(LOCATION_DATA_PATH)
    if data_subset_turbines > 0:
        locations_df = locations_df[locations_df['TurbID'] <= data_subset_turbines].reset_index(drop=True)

    # Plot some power output graphs
    if args.plot_images:
        num_plots = 10
        for i in range(num_plots):
            patv_idx = get_patv_feature_idx()
            turbine_ids = [0, 12, 34, 42, 69, 120]
            save_dir = os.path.join(args.image_path, 'patv_plots_scaled')

            # Plot for training data
            plot_power_output(X_train_scaled[i], Y_train_scaled[i], turbine_ids=turbine_ids, image_name=f"patv_train_{i}.png", patv_idx=patv_idx, save_dir=save_dir)

            # Plot for validation data
            plot_power_output(X_val_scaled[i], Y_val_scaled[i], turbine_ids=turbine_ids, image_name=f"patv_val_{i}.png", patv_idx=patv_idx, save_dir=save_dir)

    print("Preprocessing, interpolation, and scaling complete")
    print("==========================================================\n")

    # Return the scaled data split into training and validation and locations DataFrame
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, locations_df

