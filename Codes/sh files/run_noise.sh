#!/bin/bash

file="noise_graph.log"
> $file

#list of files
files=(
    "../../data/moreno_propro/moreno_propro-ConCompo_duplicate.txt"
    "../../data/maayan-vidal/maayan-vidal-ConCompo_duplicate.txt"
    "../../data/pgp/pgp-ConCompo_duplicate.txt"
    "../../data/powergrid/powergrid-ConCompo_duplicate.txt"
    "../../data/tntp-Chicago/tntp-Chicago-ConCompo_duplicate.txt"
    "../../data/cora/cora-ConCompo_duplicate.txt"
)


for f in "${files[@]}"
do
    # Log the command for each file
    echo "Running command: python Noisy_graph.py $f" >> "$file"
    
    # Loop over the parameters 5, 10, and 20
    for p in 2 5 10
    do
        # Run the Python command with the file and parameter p
        python Noisy_graph.py "$f" $p >> "$file"
    done
    echo -e "Done with file: $f\n\n" >> "$file"
done

## saved file in 30_more