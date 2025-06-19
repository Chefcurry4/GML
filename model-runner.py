import subprocess
import os
import sys
from datetime import datetime

output_dir = "output/model_runs/"
pipeline_file = "main.py"

def main():
    if len(sys.argv) != 2:
        print("Usage: python model-runner.py schedule.txt")
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
        print(f"Command: {sys.executable} {pipeline_file} {line}")
        
        # Use the same Python interpreter that's running this script
        cmd = [sys.executable, pipeline_file] + line.split()
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Save stdout and stderr
        with open(os.path.join(run_output_dir, f"run{run_num}_stdout.txt"), "w", encoding='utf-8') as out_f:
            out_f.write(result.stdout)
        with open(os.path.join(run_output_dir, f"run{run_num}_stderr.txt"), "w", encoding='utf-8') as err_f:
            err_f.write(result.stderr)

        # Print status
        if result.returncode == 0:
            print(f"Run #{run_num} completed successfully.")
        else:
            print(f"Run #{run_num} failed with return code {result.returncode}")
            print(f"Error: {result.stderr[:200]}...")  # Show first 200 chars of error

        run_num += 1

    print(f"\nAll experiments completed. Results saved in: {run_output_dir}")

if __name__ == "__main__":
    main()