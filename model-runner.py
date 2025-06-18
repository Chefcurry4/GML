import subprocess
import os
import sys
from datetime import datetime

output_dir = "output/model_runs/"
pipeline_file = "main.py"

def main():
    if len(sys.argv) != 2:
        print("Usage: py model-runner.py schedule.txt")
        sys.exit(1)

    schedule_file = sys.argv[1]
    if not os.path.isfile(schedule_file):
        print(f"Schedule file '{schedule_file}' not found.")
        sys.exit(1)

    # Create output folder with date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_output_dir, exist_ok=True)

    # Read schedule
    with open(schedule_file, "r") as f:
        lines = f.readlines()

    run_num = 1
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # Skip empty lines and comments

        print(f"\n=== Running experiment #{run_num} ===")
        # Call main.py with the arguments
        cmd = f"py {pipeline_file} {line}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Save stdout and stderr
        with open(os.path.join(run_output_dir, f"run{run_num}_stdout.txt"), "w") as out_f:
            out_f.write(result.stdout)
        with open(os.path.join(run_output_dir, f"run{run_num}_stderr.txt"), "w") as err_f:
            err_f.write(result.stderr)

        print(f"Run #{run_num} finished.")
        run_num += 1

if __name__ == "__main__":
    main()