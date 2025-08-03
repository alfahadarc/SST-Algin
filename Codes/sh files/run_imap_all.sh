#!/bin/bash

# Activate the Conda environment
source /research2/abdfahad/anaconda3/bin/activate sst  # Adjust the path as needed

# Define the error log file
ERROR_LOG="error_log.txt"
> "$ERROR_LOG"  # Clear previous log content

# Run scripts in parallel and capture errors
./run_imap_moreno.sh 2>>"$ERROR_LOG" &
./run_imap_mayaan.sh 2>>"$ERROR_LOG" &
./run_imap_power.sh 2>>"$ERROR_LOG" &
./run_imap_pgp.sh 2>>"$ERROR_LOG" &
./run_imap_tntp.sh 2>>"$ERROR_LOG" &
./run_imap_cora.sh 2>>"$ERROR_LOG" &

# Wait for all background processes to finish
wait

echo "All scripts have completed. Check $ERROR_LOG for errors."
