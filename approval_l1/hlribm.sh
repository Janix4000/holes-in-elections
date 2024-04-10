#!/bin/bash

# Define the lists of arguments
args_list_1=(
    "euclidean"
    "resampling"
    "noise"
    "truncated_urn"
    # Add more arguments as needed
)

# Function to execute Python script with given arguments
run_python_process() {
    arg1=$1
    i=$2
    dir=experiments/30x60/"$arg1"

    python3 how_long_reference_is_better_metric.py --save_results --load_from_file "$dir"/elections.txt --reference_load_from_file "$dir"/reference.txt --num_reference_instances "$i" --family "$arg1" 
}

# Iterate through both lists of arguments and run processes in parallel
for arg1 in "${args_list_1[@]}"; do
    for ((i=1; i<12; i++)); do
        run_python_process "$arg1" "$i" &
    done
done

# Wait for all background processes to finish
wait

