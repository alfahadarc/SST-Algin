#!/bin/bash

# List of folders
folders=(
    "../../data/moreno_propro"
    "../../data/maayan-vidal"
    "../../data/powergrid"
    "../../data/tntp-Chicago"
    "../../data/pgp"
    "../../data/cora"
)

logfile="orca_log.txt"
> "$logfile"

orca_path="orca.exe"

# Number of nodes for orca
nodes=4

for f in "${folders[@]}"
do
    # Loop through each file in the folder
    for file in "$f"/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "Processing $filename" >> "$logfile"
            output="$f/graphlet4_$filename"
            "$orca_path" node "$nodes" "$file" "$output" >> "$logfile"
        fi
    done
    echo -e "-------\n" >> "$logfile"
done
