import requests
from tqdm import tqdm
import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure the data directory is created in the script's directory
data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# Output paths for data files (always in the script's directory)
scada_output_path = os.path.join(data_dir, 'wind_power_sdwpf.csv')
turbine_output_path = os.path.join(data_dir, 'turbine_location.csv')

def download_with_progress(url, output_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


# URLs of the datasets
scada_url = "https://bj.bcebos.com/v1/ai-studio-online/85b5cb4eea5a4f259766f42a448e2c04a7499c43e1ae4cc28fbdee8e087e2385?responseContentDisposition=attachment%253B%2520filename%253Dwtbdata_245days.csv"
turbine_url = "https://bj.bcebos.com/v1/ai-studio-online/e927ce742c884955bf2a667929d36b2ef41c572cd6e245fa86257ecc2f7be7bc?responseContentDisposition=attachment%253B%2520filename%253Dsdwpf_baidukddcup2022_turb_location.CSV"

# Download with progress bar
download_with_progress(scada_url, scada_output_path)
download_with_progress(turbine_url, turbine_output_path)

# Load into pandas DataFrame if needed
scada_df = pd.read_csv(scada_output_path)
turbine_df = pd.read_csv(turbine_output_path)

print("Done")