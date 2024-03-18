#!/bin/bash

# Define the lists of arguments
args_list_1=(
    "euclidean"
    "resampling"
    "noise"
    "truncated_urn"
)

# Function to execute Python script with given arguments
run_python_process() {
    arg1=$1

    python3 generate_reference.py  --file experiments/30x60/"$arg1"/elections.txt
}

# Iterate through both lists of arguments and run processes in parallel
for arg1 in "${args_list_1[@]}"; do
    run_python_process "$arg1" &
done

# Wait for all background processes to finish
wait

