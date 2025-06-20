import subprocess
import os
import sys
from datetime import datetime
import threading

output_dir = "output/model_runs/"
pipeline_file = "main.py"

def stream_output(pipe, file_handle, prefix=""):
    """Stream subprocess output to both console and file"""
    for line in iter(pipe.readline, ''):
        if line:
            # Print to console with optional prefix
            print(f"{prefix}{line}", end='')
            # Write to file
            file_handle.write(line)
            file_handle.flush()  # Ensure immediate write

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
        
        # Open output files
        stdout_file = os.path.join(run_output_dir, f"run{run_num}_stdout.txt")
        stderr_file = os.path.join(run_output_dir, f"run{run_num}_stderr.txt")
        
        with open(stdout_file, "w", encoding='utf-8') as out_f, \
            open(stderr_file, "w", encoding='utf-8') as err_f:
            
            # Set environment variables to directly print output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Start subprocess with pipes
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                universal_newlines=True,
                env=env
            )
            
            # Create threads to handle stdout and stderr streams
            stdout_thread = threading.Thread(
                target=stream_output,
                args=(process.stdout, out_f, "")  # Removed prefix for cleaner output
            )
            stderr_thread = threading.Thread(
                target=stream_output,
                args=(process.stderr, err_f, "[ERR] ")
            )
            
            # Start threads
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Wait for threads to finish
            stdout_thread.join()
            stderr_thread.join()

        # Print status
        if return_code == 0:
            print(f"\nRun #{run_num} completed successfully.")
        else:
            print(f"\nRun #{run_num} failed with return code {return_code}")
            # Show error from file
            try:
                with open(stderr_file, 'r') as f:
                    error_content = f.read()
                    if error_content.strip():
                        print(f"Error: {error_content[:200]}...")
            except:
                pass

        run_num += 1

    print(f"\nAll experiments completed. Results saved in: {run_output_dir}")

if __name__ == "__main__":
    main()