#!/bin/bash

# Activate the Conda environment
source /research2/abdfahad/anaconda3/bin/activate gatne  # Adjust the path as needed

# Define the error log file
ERROR_LOG="error_log_gatne.txt"
> "$ERROR_LOG"  # Clear previous log content

# Run scripts in parallel and capture errors
./run_gatne_moreno.sh 2>>"$ERROR_LOG" &
./run_gatne_mayaan.sh 2>>"$ERROR_LOG" &
./run_gatne_power.sh 2>>"$ERROR_LOG" &
./run_gatne_pgp.sh 2>>"$ERROR_LOG" &
./run_gatne_tntp.sh 2>>"$ERROR_LOG" &
./run_gatne_cora.sh 2>>"$ERROR_LOG" &

# Wait for all background processes to finish
wait

echo "All scripts have completed. Check $ERROR_LOG for errors."
