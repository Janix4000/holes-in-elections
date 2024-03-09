#!/bin/bash

# Define the lists of arguments
args_list_1=(
    "euclidean"
    "resampling"
    "noise"
    "truncated_urn"
    # Add more arguments as needed
)

args_list_2=(
    "basin_hopping"
    "greedy_dp"
    "pairs"
    # Add more arguments as needed
)

# Function to execute Python script with given arguments
run_python_process() {
    arg1=$1
    arg2=$2

    python3 heuristic_to_change_next_reference_metric.py --save_results --family "$arg1" --algorithm "$arg2"
}

# Iterate through both lists of arguments and run processes in parallel
for arg1 in "${args_list_1[@]}"; do
    for arg2 in "${args_list_2[@]}"; do
        run_python_process "$arg1" "$arg2" &
    done
done

# Wait for all background processes to finish
wait

