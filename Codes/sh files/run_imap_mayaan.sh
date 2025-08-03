#!/bin/bash

# Define folder path
name="maayan-vidal"
folder="../../data/$name"

# Define the base filename
base_file="$name-ConCompo"

# Define the map file (this does not have a suffix)
map_file="$folder/${base_file}_map.txt"

# Define log file for appending output
log_file="imap_$name.log"
> $log_file
# List of specific values for each duplicate file
values=(
    150  # Value for first file (no suffix)
    150  # Value for second file 
    300  # Value for third file 
    700 # Value for fourth file 
)

# Function to run the Python script
run_graphlet_imap() {
    echo "Running command: python3 Graphlet_imap.py $folder/$base_file.txt $folder/$base_file$duplicate_suffix.txt $folder/graphlet4_$base_file.txt $folder/graphlet4_$base_file$duplicate_suffix.txt $map_file $1" >> "$log_file"
    python3 ../Graphlet_imap.py \
        "$folder/$base_file.txt" \
        "$folder/$base_file$duplicate_suffix.txt" \
        "$folder/graphlet4_$base_file.txt" \
        "$folder/graphlet4_$base_file$duplicate_suffix.txt" \
        "$map_file" $1 >> "$log_file" 2>&1
}

# Run the function for each duplicate file (with different suffixes)
counter=0
for duplicate_suffix in "_duplicate" "_duplicate_2.0" "_duplicate_5.0" "_duplicate_10.0"
do
    # Get the corresponding value from the values list
    value=${values[$counter]}
    
    # Log which file is being processed
    echo "Processing: $base_file$duplicate_suffix with value: $value" >> "$log_file"
    
    # Call the function to run the Python script with the corresponding value
    run_graphlet_imap $value
    
    # Check if the Python script ran successfully
    if [ $? -eq 0 ]; then
        echo "Successfully ran with value: $value" >> "$log_file"
    else
        echo -e "Error with value: $value, moving to next file...\n\n" >> "$log_file"
    fi
    
    counter=$((counter + 1))  # Increment counter to pick the next value
    echo -e "---------------------------------------------\n\n" >> "$log_file"
done

echo "All processes completed." >> "$log_file"
