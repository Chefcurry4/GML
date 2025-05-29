import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

# Attempt to import configurations from the GML package/directory
# This assumes the script might be run from outside GML directory in some contexts
# or as part of a larger project structure.
try:
    from config import SCADA_DATA_PATH, LOCATION_DATA_PATH, TARGET_FEATURE, INPUT_FEATURES, OUTPUT_DIR, DEVICE_SELECTION
except ModuleNotFoundError:
    print("Warning: Could not import from GML.config. Attempting relative import for config.")
    # This relative import works if the script is run from the GML directory directly (e.g., python datasets_stats.py)
    try:
        from .config import SCADA_DATA_PATH, LOCATION_DATA_PATH, TARGET_FEATURE, INPUT_FEATURES, OUTPUT_DIR, DEVICE_SELECTION
    except ImportError:
        print("Error: Failed to import config.py. Ensure GML is in PYTHONPATH or run from GML directory.")
        print("Using default paths and settings for critical parts if possible.")
        # Define critical paths manually if import fails, for basic script functionality
        SCADA_DATA_PATH = 'GML/data/wind_power_sdwpf.csv'
        LOCATION_DATA_PATH = 'GML/data/turbine_location.csv'
        TARGET_FEATURE = 'Patv'
        INPUT_FEATURES = ['Patv', 'Wspd', 'Wdir', 'Tempr', 'Pr', 'Den', 'Rhum', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3']
        OUTPUT_DIR = 'output/'
        DEVICE_SELECTION = "cuda" # Default (else cpu)

# Define the directory for saving plots, ensure it's correctly formed
STATS_PLOT_DIR = os.path.join(OUTPUT_DIR, 'stats_plots/')
os.makedirs(STATS_PLOT_DIR, exist_ok=True)

def main():
    """Main function to perform dataset statistical analysis."""
    
    # --- 1. Device Setup ---
    if DEVICE_SELECTION == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if DEVICE_SELECTION == "cuda":
            print("CUDA selected but not available. Using CPU.")
        else:
            print("Using CPU.")
    
    # --- 2. Data Loading ---
    print(f"\nLoading SCADA data from: {SCADA_DATA_PATH}")
    try:
        # Load the data without parsing dates first
        scada_df = pd.read_csv(SCADA_DATA_PATH)
        
        # Convert Day to timedelta and add to a base date
        base_date = pd.Timestamp('2020-01-01')  # Using an arbitrary base date
        day_delta = pd.to_timedelta(scada_df['Day'] - 1, unit='D')
        
        # Convert HH:MM to timedelta by adding ':00' for seconds
        time_delta = pd.to_timedelta(scada_df['Tmstamp'] + ':00')
        
        # Combine both into datetime
        scada_df['Tm'] = base_date + day_delta + time_delta
        
        # Drop the original Day and Tmstamp columns
        scada_df = scada_df.drop(['Day', 'Tmstamp'], axis=1)
        
    except FileNotFoundError:
        print(f"Error: SCADA data file not found at {SCADA_DATA_PATH}. Please check the path in config.py.")
        return
    except Exception as e:
        print(f"Error loading SCADA data: {e}")
        return

    print(f"Loading Location data from: {LOCATION_DATA_PATH}")
    try:
        location_df = pd.read_csv(LOCATION_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Location data file not found at {LOCATION_DATA_PATH}. Please check the path in config.py.")
        # Continue without location data if not found, some stats might be skipped
        location_df = None 
    except Exception as e:
        print(f"Error loading Location data: {e}")
        location_df = None

    print("\n--- 3. Initial Data Overview ---")
    print("SCADA Data Shape:", scada_df.shape)
    print("\nSCADA Data Info:")
    scada_df.info(verbose=True, show_counts=True)
    print("\nSCADA Data Head:")
    print(scada_df.head())
    
    print("\nSCADA Data Descriptive Statistics (Overall):")
    # Include datetime features if you want to see their min/max, otherwise select_dtypes
    print(scada_df.describe(include='all'))

    if location_df is not None:
        print("\nLocation Data Shape:", location_df.shape)
        print("\nLocation Data Info:")
        location_df.info()
        print("\nLocation Data Head:")
        print(location_df.head())
        print("\nLocation Data Descriptive Statistics:")
        print(location_df.describe(include='all'))

    # --- 4. Missing Data Analysis ---
    print("\n--- 4. Missing Data Analysis ---")
    missing_values = scada_df.isnull().sum()
    missing_percentage = (missing_values / len(scada_df)) * 100
    missing_summary = pd.DataFrame({'Missing Values': missing_values, 'Percentage (%)': missing_percentage})
    print("Missing Data Summary (Overall):")
    print(missing_summary[missing_summary['Missing Values'] > 0])

    plt.figure(figsize=(15, 8))
    sns.heatmap(scada_df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Data Heatmap (Overall SCADA)')
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_PLOT_DIR, 'missing_data_heatmap_overall.png'))
    plt.close()
    print(f"Saved missing data heatmap to {os.path.join(STATS_PLOT_DIR, 'missing_data_heatmap_overall.png')}")

    # Missing Patv per turbine
    if 'TurbID' in scada_df.columns and TARGET_FEATURE in scada_df.columns:
        missing_patv_per_turbine = scada_df.groupby('TurbID')[TARGET_FEATURE].apply(lambda x: x.isnull().sum())
        total_per_turbine = scada_df.groupby('TurbID')[TARGET_FEATURE].apply(lambda x: len(x))
        percentage_missing_patv = (missing_patv_per_turbine / total_per_turbine) * 100
        missing_patv_summary = pd.DataFrame({
            'Missing Patv': missing_patv_per_turbine,
            'Total Records': total_per_turbine,
            '% Missing Patv': percentage_missing_patv
        }).sort_values(by='% Missing Patv', ascending=False)
        print(f"\nMissing '{TARGET_FEATURE}' Summary per Turbine:")
        print(missing_patv_summary.head(10)) # Print top 10
        
        if not missing_patv_summary.empty:
            plt.figure(figsize=(12, 6))
            missing_patv_summary['% Missing Patv'].plot(kind='bar')
            plt.title(f'% Missing {TARGET_FEATURE} per Turbine')
            plt.ylabel('% Missing Values')
            plt.xlabel('Turbine ID')
            plt.tight_layout()
            plt.savefig(os.path.join(STATS_PLOT_DIR, 'missing_patv_per_turbine.png'))
            plt.close()
            print(f"Saved missing {TARGET_FEATURE} per turbine plot to {os.path.join(STATS_PLOT_DIR, 'missing_patv_per_turbine.png')}")


    # --- 5. Time-based Analysis ---
    print("\n--- 5. Time-based Analysis ---")
    if 'Tm' in scada_df.columns:
        print(f"Overall Time Range: {scada_df['Tm'].min()} to {scada_df['Tm'].max()}")
        
        # Daily Average Power Output (Overall)
        daily_avg_power = scada_df.set_index('Tm').resample('D')[TARGET_FEATURE].mean()
        if not daily_avg_power.empty:
            plt.figure(figsize=(15, 7))
            daily_avg_power.plot()
            plt.title(f'Daily Average {TARGET_FEATURE} (Overall Farm)')
            plt.xlabel('Date')
            plt.ylabel(f'Average {TARGET_FEATURE}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(STATS_PLOT_DIR, 'daily_avg_power_overall.png'))
            plt.close()
            print(f"Saved daily average power plot to {os.path.join(STATS_PLOT_DIR, 'daily_avg_power_overall.png')}")

        # Monthly Average Power Output (Overall)
        monthly_avg_power = scada_df.set_index('Tm').resample('ME')[TARGET_FEATURE].mean() # ME for Month End
        if not monthly_avg_power.empty:
            plt.figure(figsize=(15, 7))
            monthly_avg_power.plot(kind='bar', colormap='viridis')
            plt.title(f'Monthly Average {TARGET_FEATURE} (Overall Farm)')
            plt.xlabel('Month')
            plt.ylabel(f'Average {TARGET_FEATURE}')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(STATS_PLOT_DIR, 'monthly_avg_power_overall.png'))
            plt.close()
            print(f"Saved monthly average power plot to {os.path.join(STATS_PLOT_DIR, 'monthly_avg_power_overall.png')}")
    
    # --- 6. Per-Turbine Analysis ---
    print("\n--- 6. Per-Turbine Analysis ---")
    if 'TurbID' in scada_df.columns:
        print(f"Descriptive statistics for {TARGET_FEATURE} per turbine:")
        print(scada_df.groupby('TurbID')[TARGET_FEATURE].describe().head())

        # Box plot of Patv per Turbine (sample if too many turbines)
        num_turbines = scada_df['TurbID'].nunique()
        sample_n_turbines = min(num_turbines, 20) # Sample if more than 20 turbines
        
        if num_turbines > 0:
            sampled_turb_ids = np.random.choice(scada_df['TurbID'].unique(), size=sample_n_turbines, replace=False)
            sampled_df_for_plot = scada_df[scada_df['TurbID'].isin(sampled_turb_ids)]

            plt.figure(figsize=(15, 8))
            sns.boxplot(x='TurbID', y=TARGET_FEATURE, data=sampled_df_for_plot, order=sorted(sampled_turb_ids))
            plt.title(f'{TARGET_FEATURE} Distribution per Turbine (Sample of {sample_n_turbines})')
            plt.xlabel('Turbine ID')
            plt.ylabel(TARGET_FEATURE)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(STATS_PLOT_DIR, 'patv_boxplot_per_turbine_sample.png'))
            plt.close()
            print(f"Saved {TARGET_FEATURE} boxplot to {os.path.join(STATS_PLOT_DIR, 'patv_boxplot_per_turbine_sample.png')}")

            if 'Wspd' in scada_df.columns:
                plt.figure(figsize=(15, 8))
                sns.boxplot(x='TurbID', y='Wspd', data=sampled_df_for_plot, order=sorted(sampled_turb_ids))
                plt.title(f'Wspd Distribution per Turbine (Sample of {sample_n_turbines})')
                plt.xlabel('Turbine ID')
                plt.ylabel('Wspd')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(STATS_PLOT_DIR, 'wspd_boxplot_per_turbine_sample.png'))
                plt.close()
                print(f"Saved Wspd boxplot to {os.path.join(STATS_PLOT_DIR, 'wspd_boxplot_per_turbine_sample.png')}")

    # --- 7. Feature Distributions ---
    print("\n--- 7. Feature Distributions ---")
    numerical_features_for_dist = [TARGET_FEATURE, 'Wspd', 'Wdir', 'Tempr', 'Pr', 'Den', 'Rhum'] # Select a subset
    
    for feature in numerical_features_for_dist:
        if feature in scada_df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(scada_df[feature].dropna(), kde=True, bins=50)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(STATS_PLOT_DIR, f'distribution_{feature.lower()}.png'))
            plt.close()
            print(f"Saved distribution plot for {feature} to {os.path.join(STATS_PLOT_DIR, f'distribution_{feature.lower()}.png')}")

    # --- 8. Correlation Analysis ---
    print("\n--- 8. Correlation Analysis ---")
    # Select numerical features for correlation. Exclude TurbID, Ndir, Pab1/2/3 for now as they might be categorical/flags.
    correlation_features = [f for f in INPUT_FEATURES if f in scada_df.columns and scada_df[f].dtype in [np.float64, np.int64]]
    
    if correlation_features:
        correlation_matrix = scada_df[correlation_features].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(os.path.join(STATS_PLOT_DIR, 'correlation_matrix.png'))
        plt.close()
        print(f"Saved correlation matrix heatmap to {os.path.join(STATS_PLOT_DIR, 'correlation_matrix.png')}")

        if TARGET_FEATURE in correlation_matrix:
            print(f"\nCorrelations with {TARGET_FEATURE}:")
            print(correlation_matrix[TARGET_FEATURE].sort_values(ascending=False))
    else:
        print("No numerical features found for correlation analysis.")

    # --- 9. Power Curve (Wspd vs Patv) ---
    print("\n--- 9. Power Curve ---")
    if 'Wspd' in scada_df.columns and TARGET_FEATURE in scada_df.columns:
        # Using a sample for clarity if the dataset is very large
        sample_df_power_curve = scada_df.sample(n=min(5000, len(scada_df)), random_state=42)
        plt.figure(figsize=(10, 6))
        plt.scatter(sample_df_power_curve['Wspd'], sample_df_power_curve[TARGET_FEATURE], alpha=0.3, s=10)
        plt.title(f'Power Curve ({TARGET_FEATURE} vs. Wspd) - Sampled')
        plt.xlabel('Wspd (Wind Speed)')
        plt.ylabel(TARGET_FEATURE)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(STATS_PLOT_DIR, 'power_curve_sample.png'))
        plt.close()
        print(f"Saved power curve plot to {os.path.join(STATS_PLOT_DIR, 'power_curve_sample.png')}")

    # --- 10. Wind Direction Analysis (Example: Bar plot of Wdir categories) ---
    print("\n--- 10. Wind Direction Analysis ---")
    if 'Wdir' in scada_df.columns:
        # Simple bar plot of wind direction. For a true wind rose, a specialized library might be better.
        # Create bins for wind direction if it's continuous (0-360 degrees)
        # Assuming Wdir is in degrees.
        try:
            # Ensure Wdir is numeric and not all NaN
            wdir_numeric = pd.to_numeric(scada_df['Wdir'], errors='coerce').dropna()
            if not wdir_numeric.empty:
                bins = np.arange(0, 361, 45) # 45-degree bins
                labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
                wdir_binned = pd.cut(wdir_numeric, bins=bins, labels=labels, right=False, include_lowest=True)
                
                plt.figure(figsize=(12, 7))
                wdir_binned.value_counts().sort_index().plot(kind='bar', colormap='viridis')
                plt.title('Wind Direction Distribution (Binned)')
                plt.xlabel('Wind Direction (Degrees)')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(STATS_PLOT_DIR, 'wind_direction_distribution.png'))
                plt.close()
                print(f"Saved wind direction distribution plot to {os.path.join(STATS_PLOT_DIR, 'wind_direction_distribution.png')}")
            else:
                print("Wind direction ('Wdir') column is all NaN or non-numeric after coercion. Skipping plot.")
        except Exception as e:
            print(f"Could not plot wind direction: {e}")


    # --- 11. Insights for ML Modeling ---
    print("\n\n--- 12. Potential Insights for ML Modeling ---")
    print("1. Missing Data: Significant missing data in features like 'Patv', 'Wspd' etc. needs careful handling.")
    print("   - Consider imputation strategies (mean, median, model-based like KNNImputer, or advanced like in your 'joint' GNN method).")
    print(f"   - Missing {TARGET_FEATURE} per turbine varies; some turbines might have too much missing data to be reliable without robust imputation.")
    print("2. Feature Distributions & Scaling:")
    print("   - Features like 'Patv' and 'Wspd' are often skewed. Normalization/scaling (e.g., MinMaxScaler, StandardScaler) is crucial for many ML models.")
    print("   - Power ('Patv') is typically zero-inflated and bounded; consider transformations if needed (e.g., log for positive values, or models that handle bounds).")
    print("   - Wind Direction ('Wdir', 'Ndir') is cyclical (0-360 degrees). Convert to sin/cos components for ML models to capture circular nature.")
    print("3. Correlation & Feature Selection:")
    print(f"   - '{TARGET_FEATURE}' shows strong positive correlation with 'Wspd', as expected (power curve).")
    print("   - Some features show high collinearity (e.g., 'Etmp' and 'Itmp'). This might be an issue for some linear models (multicollinearity). Consider removing one or using regularization.")
    print("   - Identify features with low or no correlation with the target if dimensionality reduction is needed, but be cautious as non-linear relationships might exist.")
    print("4. Time Series Nature:")
    print("   - Daily and monthly plots show potential seasonality or trends in power output, which time series models or feature engineering (e.g., time-based features) should capture.")
    print("5. Outliers:")
    print("   - Box plots per turbine for 'Patv' and 'Wspd' can help identify turbines with unusual behavior or potential outliers in data.")
    print("   - The power curve plot can also reveal outliers (e.g., power generation at zero wind speed, or unexpectedly low power at high wind speeds).")
    print("6. Turbine Variability:")
    print("   - Statistics and box plots show variability in performance and data quality across turbines. Model per turbine (like your GRU approach) or graph-based models that can learn turbine-specific embeddings might be beneficial.")
    print("7. Categorical/Flag Features:")
    print("   - Features like 'Pab1', 'Pab2', 'Pab3' (pitch angles/flags) might be better treated as categorical or require specific encoding if they represent states rather than continuous values.")

    print("\nDataset statistics script finished.")

if __name__ == '__main__':
    main() 