import pandas as pd
import numpy as np
from utils.new_data_preprocessing import load_sdwpf_data  # Provides preprocessing and loading of SDWPF data fileciteturn4file0
# --- USER CONFIGURATION ---
# paths to your data
WIND_CSV        = "GML/data/wind_power_sdwpf.csv"         # CSV of wind directions; index=timestamp, columns=turbine IDs
POS_CSV         = "GML/data/turbine_location.csv"        # turbine positions, if needed
OUTPUT_TIMESTAMP = 4                                       # which window index to inspect
WINDOW_SIZE     = 10                                      # timestamps per batch (for grouping)
BIN_COUNT       = 36                                      # number of directional bins (e.g., 10° each)

# Auto-generated module to compute dominant wind directions via binned histogram + circular mean


# --- Helper Functions ---
def normalize_angle(angle_deg: np.ndarray) -> np.ndarray:
    """Normalize angles to [0, 360)."""
    return angle_deg % 360


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Compute circular mean of angles in degrees."""
    radians = np.deg2rad(angles_deg)
    sin_sum = np.sum(np.sin(radians))
    cos_sum = np.sum(np.cos(radians))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    return float(np.rad2deg(mean_rad) % 360)

# --- Absolute Wind Direction ---
def compute_absolute_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate absolute wind direction in degrees relative to true north (0-360°).
    Uses nacelle direction (Ndir) and sensor wind direction (Wdir).
    """
    df = df.copy()
    df['Wdir'] = df['Wdir'].fillna(0)
    df['Ndir'] = df['Ndir'].fillna(0)
    df['AbsWdir'] = normalize_angle(df['Ndir'] + df['Wdir'])
    return df

# --- Dominant Direction Computation ---
def dominant_direction(angles: np.ndarray, bins: int = BIN_COUNT) -> float:
    """
    Bin angles into equal-width bins, find the bin with highest count,
    and return the circular mean of angles in that bin.
    """
    dirs = normalize_angle(angles[~np.isnan(angles)])
    if dirs.size == 0:
        return np.nan
    counts, edges = np.histogram(dirs, bins=bins, range=(0, 360))
    idx = np.argmax(counts)
    low, high = edges[idx], edges[idx+1]
    spike = dirs[(dirs >= low) & (dirs < high)]
    return circular_mean_deg(spike) if spike.size > 0 else circular_mean_deg(dirs)

# --- Main Computation ---

def compute_timestamp_dominant(window_size: int = None) -> pd.Series:
    """
    Compute farm-wide dominant absolute wind direction per timestamp or window.

    Args:
        window_size (int or None): None for per timestamp; int for fixed-window grouping.
    Returns:
        pd.Series: Index = TimeIndex or WindowIndex; values = dominant abs wind direction (deg).
    """
    # Load raw data
    df = load_sdwpf_data(WIND_CSV)
    # Compute absolute wind direction
    df = compute_absolute_wind_direction(df)

    # Pivot by TimeIndex, Turbine ID
    wdir_df = df.pivot(index='TimeIndex', columns='TurbID', values='AbsWdir')

    if window_size is None:
        series = wdir_df.apply(lambda row: dominant_direction(row.values), axis=1)
        series.index.name = 'TimeIndex'
    else:
        grp = wdir_df.groupby(np.arange(len(wdir_df)) // window_size)
        series = grp.apply(lambda dfw: dominant_direction(dfw.values.flatten()))
        series.index.name = 'WindowIndex'

    series.name = 'FarmDomAbsDir'
    return series


def get_dominant_for_window(window_index: int,
                             window_size: int = WINDOW_SIZE) -> float:
    """
    Retrieve dominant absolute wind direction for a specific window.
    """
    series = compute_timestamp_dominant(window_size=window_size)
    try:
        return float(series.loc[window_index])
    except KeyError:
        raise IndexError(f"Window index {window_index} not found. Valid: 0 to {len(series)-1}")

# --- Visualization ---

def plot_dominant_histogram(series: pd.Series,
                            bins: int = BIN_COUNT,
                            show: bool = True) -> None:
    """
    Plot a standard histogram of dominant absolute directions.
    """
    plt.figure()
    plt.hist(series.dropna(), bins=bins, range=(0, 360))
    plt.xlabel('Dominant Abs Direction (deg)')
    plt.ylabel('Count')
    plt.title('Histogram of Farm Dominant Absolute Wind Directions')
    if show:
        plt.show()


def plot_dominant_rose(series: pd.Series,
                       bins: int = BIN_COUNT,
                       show: bool = True) -> None:
    """
    Plot a circular rose diagram of dominant absolute directions.
    """
    angles = series.dropna().values
    theta = np.deg2rad(angles)
    counts, bin_edges = np.histogram(theta, bins=bins, range=(0, 2*np.pi))
    widths = np.diff(bin_edges)
    ax = plt.subplot(projection='polar')
    ax.bar(bin_edges[:-1], counts, width=widths, edgecolor='black', align='edge')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Rose Diagram of Farm Dominant Absolute Directions')
    if show:
        plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    # Per timestamp
    ts_dom = compute_timestamp_dominant()
    print(ts_dom.head())

    # Per window
    win_dom = compute_timestamp_dominant(window_size=WINDOW_SIZE)
    print(f"\nWindows (size={WINDOW_SIZE}):", win_dom.head())

    # Query specific window
    idx = 500
    try:
        print(f"Window {idx} DomAbsDir: {get_dominant_for_window(idx):.2f}°")
    except IndexError as e:
        print(e)

    # Plots
    plot_dominant_histogram(win_dom)
    plot_dominant_rose(win_dom)
