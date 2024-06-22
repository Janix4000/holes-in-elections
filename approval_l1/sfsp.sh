#!/bin/bash
for fam in noise truncated_urn euclidean resampling; do
    python3 space_filling_starting_point.py --family $fam &
done

wait