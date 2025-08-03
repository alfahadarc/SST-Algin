#!/bin/bash

# Define folder path
folder="../../data/pgp"

# Define the base filename
base_file="pgp-ConCompo"

# Define the map file (this does not have a suffix)
map_file="$folder/$base_file"_map.txt

# Define log file for appending output
log_file="gatne_pgp.log"
> $log_file  # Clear the log file at the start

# Arrays for hyperparameters with multiple values
learning_rates=("0.0001" "0.001" "0.01")   # Learning rate options
epochs=("35" "50" "100")                    # Epochs options
# learning_rates="0.0001"   # Learning rate options
# epochs="35" 
batch_size="32"     # Fixed batch size
dim="128"           # Fixed dimensionality of the embeddings
topk="10"           # Fixed topk value

# Function to run the GATNE script
run_gatne() {
    echo "Running command: python3 GATNE.py $folder/$base_file.txt $folder/$base_file$duplicate_suffix.txt $folder/graphlet4_$base_file.txt $folder/graphlet4_$base_file$duplicate_suffix.txt $init_map_file $map_file $1 $2 $3 $4 $5" >> "$log_file"
    python3 ../GATNE.py \
        "$folder/$base_file.txt" \
        "$folder/$base_file$duplicate_suffix.txt" \
        "$folder/graphlet4_$base_file.txt" \
        "$folder/graphlet4_$base_file$duplicate_suffix.txt" \
        "$init_map_file" \
        "$map_file" \
        $1 $2 $3 $4 $5 >> "$log_file" 2>&1
}

# Run the function for each duplicate file (with different suffixes)
for duplicate_suffix in "_duplicate" "_duplicate_2.0" "_duplicate_5.0" "_duplicate_10.0"
do
    # Define the init_map_file inside the loop
    init_map_file="init_map/pgp/init_map_$base_file"_"$base_file$duplicate_suffix.csv"

    # Log which file is being processed
    echo "Processing: $base_file$duplicate_suffix" >> "$log_file"

    # Iterate over learning rates and epochs
    for lr in "${learning_rates[@]}"; do
        for epc in "${epochs[@]}"; do
            # Log the current testing values
            echo "Testing with: lr=$lr, epc=$epc, bs=$batch_size, dim=$dim, topk=$topk" >> "$log_file"
            
            # Call the function to run GATNE with the current hyperparameter values
            run_gatne $lr $epc $batch_size $dim $topk
        done
    done

    echo "Finished: $base_file$duplicate_suffix" >> "$log_file"
    echo -e "---------------------------------------------\n\n" >> "$log_file"
done

echo "All processes completed." >> "$log_file"
