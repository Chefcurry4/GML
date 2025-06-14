import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import INPUT_SEQUENCE_LENGTH, LOCATION_DATA_PATH, OUTPUT_SEQUENCE_LENGTH, SCADA_DATA_PATH, SHUFFLE_TRAIN_VAL_DATASET, TRAIN_VAL_SPLIT_RATIO

# Do not use TurbID, Day and Tmstamp as features, since they are not physical quantities and do not provide useful information for the model.
feature_names = [
    # 'TurbID',    # Turbine ID
    # 'Day',       # Day of the measurement
    # 'Tmstamp',   # Timestamp of the measurement
    # 'TimeIndex', # Combines day and timestamp to get an index for the time of the measurement
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

# Having only one feature per time step, since analysis show, that when one feature is NaN, the others are also NaN
valid_indicator_feature = "data_valid"



def preprocess_data(df):
    """
    We introduce a few caveats about when to use this data to train and evaluate the models. Attention needs to be paid to these caveats since there are always some outliers and missing values in the data due to data collection, system maintenance, and equipment failures.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11187227/
    This function preprocesses the SDWPF data to be used for our models
    """  

    # Clip Patv values at 0, since negative output power is not possible and only caused by some components consuming electricity when no power is produced
    df['Patv'] = df['Patv'].clip(lower=0)

    # Initialise the valid bit to all True
    df[valid_indicator_feature] = True

    # Check for missing values
    for feature in feature_names:
        if not feature in df.columns:
            raise ValueError(f"Feature '{feature}' not found in the DataFrame.")
        
        # Set the valid bit to False where feature is NaN
        # Use bitwise AND to keep the valid indicator True only where the feature is not NaN
        df[valid_indicator_feature] &= ~df[feature].isna()

        # Replace NaN values with 0 in the feature column
        df[feature] = df[feature].fillna(0)

    # Sometimes, the wind turbines are stopped from generating power by external reasons such as wind turbine renovation and/or actively scheduling the powering to avoid overloading the grid.
    # When Patv≤0, and Wspd > 2.5 at time t, the actual active power Patv of this wind turbine at time t is unknown (since the wind speed is large enough to generate the power, the only reason that Patv≤0 is this turbine is stopped);
    # When Pab1 > 89° or Pab2 > 89° or Pab3 > 89° (Pab1, Pab2, and Pab3 always have the same values) at time t, the actual active power Patv of this wind turbine at time t should be unknown (since no matter at then how large the wind speed is, the wind turbine is at rest in this situation).
    unknown_mask = ((df['Patv'] <= 0) & (df['Wspd'] > 2.5)) | \
               (df[['Pab1', 'Pab2', 'Pab3']].gt(89).any(axis=1))
    # Set the valid bit to False where the unknown conditions are met
    df[valid_indicator_feature] &= ~unknown_mask

    # There are some abnormal values collected from the SCADA system.
    # The reasonable range for Ndir is [−720°, 720°], as the turbine system allows the nacelle to turn at most two rounds in one direction and would force the nacelle to return to the original position otherwise.
    # The reasonable range for Wdir is [−180°, 180°]. Records beyond this range can be seen as outliers caused by the recording system. When there are Widr > 180° or Widr < −180° at time t, then the recorded values of this wind turbine at time t is abnormal.
    abnormal_mask = (df['Ndir'].abs() > 720) | (df['Wdir'].abs() > 180)
    # Set the valid bit to False where the abnormal conditions are met
    df[valid_indicator_feature] &= ~abnormal_mask

    # Print the percentage of invalid data
    invalid_percentage = 100 * (1 - df[valid_indicator_feature].mean())
    print(f"Percentage of invalid data: {invalid_percentage:.2f}%")

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

def pivot_sdwpf_multi(df):
    """
    Transforms a DataFrame containing time series data for multiple turbines and features into a 3D NumPy array.
    The function pivots the DataFrame for each feature (including a validity indicator feature), stacking the resulting 2D arrays along a new axis to produce a 3D array with dimensions corresponding to timesteps, turbines, and features.
    """

    # Pivot for each feature, then stack along the last axis
    pivots = []
    for feat in feature_names + [valid_indicator_feature]:
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
        # Append features for X
        X.append(data[i:i+input_len])  # (input_len, turbines, features)
        
        # For Y, we only want to consider the Patv feature (active power output)
        y_window = data[i+input_len:i+input_len+output_len, :, patv_feature_idx]  # (output_len, turbines)
        Y.append(y_window)
    X = np.stack(X)
    Y = np.stack(Y)

    # Squeeze Y if output_len is 1, to have shape (samples, turbines)
    if output_len == 1:
        Y = Y[:, 0, :]  # (samples, turbines)

    return X, Y


def load_and_preprocess_data(csv_path, input_len=12, output_len=1, train_val_ratio=0.8, data_subset=1, data_subset_turbines=-1, shuffle_train_val_dataset = True, random_state=42):
    """
    Preprocess SDWPF data using sliding window and split into training/validation sets.
    Args:
        csv_path (str): Path to the SDWPF CSV file.
        input_len (int): Length of the input sequence for a time window.
        output_len (int): Length of the output sequence for a time window.
        train_val_ratio (float): Ratio of training data to total data.
        data_subset (float): Fraction of the dataset to use (1.0 means all).
        random_state (int): Random seed for reproducibility.
    Returns:
        X_train (np.ndarray): Training input data.
        Y_train (np.ndarray): Training output data.
        X_val (np.ndarray): Validation input data.
        Y_val (np.ndarray): Validation output data.
        locations_df (pd.DataFrame): DataFrame with turbine locations.
    """
    print("\n==========================================================")
    print("Starting data preprocessing...")

    # Load the whole dataset    
    df = load_sdwpf_data(csv_path)

    # Subset the data if needed
    if data_subset < 1.0:
        print("Subsetting data to", data_subset * 100, "%")
        num_rows = int(len(df) * data_subset)
        # Increase the num_rows to ensure, that a measurement at the same Day and Tmstamp is not cut off
        cur_day = df.iloc[num_rows - 1]['Day']
        cur_tmstamp = df.iloc[num_rows - 1]['Tmstamp']
        while num_rows < len(df) and (df.iloc[num_rows]['Day'] == cur_day and df.iloc[num_rows]['Tmstamp'] == cur_tmstamp):
            num_rows += 1

        df_subset = df.iloc[:num_rows].reset_index(drop=True)
    else:
        df_subset = df


    # Do subsetting for turbines by dropping all turbines with TurbID greater than data_subset_turbines
    if data_subset_turbines > 0:
        print(f"Subsetting data to first {data_subset_turbines} turbines")
        df_subset = df_subset[df_subset['TurbID'] <= data_subset_turbines].reset_index(drop=True)
    
    print("Loaded data shape:", df.shape)

    # Preprocess the data
    preprocess_data(df_subset)

    # Pivot the data for easier computation with sliding window
    data = pivot_sdwpf_multi(df_subset) # shape: (timesteps, turbines, features)
    print("Pivoted data shape:", data.shape)

    # Check if some values in data are NaN
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values after preprocessing. Please check the preprocessing steps.")

    # Create feature and target sequences for training the model using sliding windows
    X, Y = sliding_window(data, input_len, output_len)

    print(f"Sliding window created: X shape {X.shape}, Y shape {Y.shape}")

    # Train/val split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, train_size=train_val_ratio, random_state=random_state, shuffle=shuffle_train_val_dataset
    )

    # Print the sizes of the resulting datasets
    print(f"Training set size: {X_train.shape}, {Y_train.shape}")
    print(f"Validation set size: {X_val.shape}, {Y_val.shape}")

    # Also load the locations DataFrame for turbine locations
    locations_df = pd.read_csv(LOCATION_DATA_PATH)
    # Check if subsetting turbines is applied, if so, filter the locations DataFrame
    if data_subset_turbines > 0:
        locations_df = locations_df[locations_df['TurbID'] <= data_subset_turbines].reset_index(drop=True)

    print("Preprocessing complete")
    print("==========================================================\n")

    return X_train, Y_train, X_val, Y_val, locations_df

# Example usage:
if __name__ == "__main__":
    # Only consider 20% of the data
    data_subset = 0.2
    data_subset_turbines = -1  # Use all turbines

    X_train, Y_train, X_val, Y_val, locations_df = load_and_preprocess_data(
        csv_path=SCADA_DATA_PATH,
        input_len=INPUT_SEQUENCE_LENGTH,
        output_len=OUTPUT_SEQUENCE_LENGTH,
        train_val_ratio=TRAIN_VAL_SPLIT_RATIO,
        data_subset=data_subset,
        data_subset_turbines=data_subset_turbines,
        shuffle_train_val_dataset=SHUFFLE_TRAIN_VAL_DATASET
    )
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_val shape:", X_val.shape)
    print("Y_val shape:", Y_val.shape)