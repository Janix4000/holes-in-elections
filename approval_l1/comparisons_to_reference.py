import os
import time
from typing import Callable

import numpy as np

import scripts.approvalwise_vector as approvalwise_vector
from scripts.approvalwise_vector import ApprovalwiseVector

from scripts.basin_hopping import basin_hopping

Algorithm = Callable[[list[ApprovalwiseVector]],
                     tuple[ApprovalwiseVector, int]]


def run_experiment(
        approvalwise_vectors: list[ApprovalwiseVector],
        reference_approvalwise_vectors: list[ApprovalwiseVector],
    algorithm: Algorithm, report_out,
        output_dir: str | None = None):

    num_elections_reference = len(reference_approvalwise_vectors)

    report_out.write(
        "experiment_size,distance,execution_time,num_starting_elections\n")
    for num_starting_elections in range(num_elections_reference):
        starting_approvalwise_vectors = approvalwise_vectors.copy()
        new_approvalwise_vectors = []
        for i in range(num_starting_elections):
            starting_approvalwise_vectors.append(
                reference_approvalwise_vectors[i])
        for idx in range(num_elections_reference - num_starting_elections):
            start_time = time.time()
            farthest_approvalwise_vectors, distance = algorithm(
                starting_approvalwise_vectors)
            execution_time_s = time.time() - start_time

            report_out.write(
                f"{len(starting_approvalwise_vectors)},{distance},{execution_time_s},{num_starting_elections}\n")

            starting_approvalwise_vectors.append(farthest_approvalwise_vectors)
            new_approvalwise_vectors.append(farthest_approvalwise_vectors)

        if output_dir:
            with open(os.path.join(output_dir, f"new_approvalwise_vectors_{num_starting_elections}.txt"), 'w') as out:
                approvalwise_vector.dump_to_text_file(
                    new_approvalwise_vectors, out)


def main():
    import sys
    if len(sys.argv) < 4:
        print(
            f"Usage: {sys.argv[0]} <input_file> <reference_input_file> <algorithm> [output_dir]")
        return 1

    with open(sys.argv[1], 'r') as in_file:
        approvalwise_vectors = approvalwise_vector.load_from_text_file(
            in_file)
        approvalwise_vectors = list(approvalwise_vectors.values())

    with open(sys.argv[2], 'r') as in_ref:
        reference_approvalwise_vectors = approvalwise_vector.load_from_text_file(
            in_ref)
        reference_approvalwise_vectors = list(
            reference_approvalwise_vectors.values())

    algorithm_name = sys.argv[3]
    match algorithm_name:
        case "basin_hopping":
            def algorithm(approvalwise_vectors): return basin_hopping(
                approvalwise_vectors=approvalwise_vectors,
                step_size=7,
                seed=2137,
                big_step_chance=0.2,
                x0='step_vector'
            )
        case "basin_hopping_random":
            def algorithm(approvalwise_vectors): return basin_hopping(
                approvalwise_vectors=approvalwise_vectors,
                step_size=7,
                seed=2137,
                big_step_chance=0.2,
                x0='random'
            )
        case _:
            print(f"Unknown algorithm: {algorithm_name}")
            return 1

    if len(sys.argv) == 5:
        output_dir = os.path.join(sys.argv[4], algorithm_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "report.csv"), 'w') as report_out:
            run_experiment(approvalwise_vectors, reference_approvalwise_vectors,
                           algorithm, report_out, output_dir)
    else:
        run_experiment(approvalwise_vectors, reference_approvalwise_vectors,
                       algorithm, sys.stdout)


if __name__ == "__main__":
    main()
