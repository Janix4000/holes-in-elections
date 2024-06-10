#!/bin/bash
for fam in noise truncated_urn euclidean resampling; do
    for algo in basin_hopping basin_hopping_random pair greedy_dp; do
        python3 space_filling_big_elections.py --family $fam --algorithm $algo &
    done
done

wait