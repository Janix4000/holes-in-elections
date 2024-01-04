import os
import sys
import time

import generate_elections

import scripts.approvalwise_vector as approvalwise_vector
from scripts.approvalwise_vector import ApprovalwiseVector, get_approvalwise_vectors

from scripts.algorithms import Algorithm, algorithms


def measure_iteration(approvalwise_vectors: list[ApprovalwiseVector], algorithm: Algorithm, **kwargs):
    start_time = time.time()
    farthest_approvalwise_vectors, distance = algorithm(
        approvalwise_vectors, **kwargs)
    execution_time_s = time.time() - start_time

    return farthest_approvalwise_vectors, distance, execution_time_s


def generate_reference(approvalwise_vectors: list[ApprovalwiseVector],
                       num_new_instances: int,
                       algorithm: Algorithm,
                       idx: int,
                       report_out,
                       output_dir: str | None = None,
                       **kwargs):
    if num_new_instances < 1:
        raise ValueError("num_new_instances must be greater than 0")

    approvalwise_vectors = approvalwise_vectors.copy()
    new_approvalwise_vectors = []

    if idx == 0:
        report_out.write(
            "experiment_size,distance,execution_time,idx\n")

    for _ in range(num_new_instances):
        farthest_approvalwise_vectors, distance, execution_time_s = measure_iteration(
            approvalwise_vectors, algorithm, **kwargs)

        new_approvalwise_vectors.append(farthest_approvalwise_vectors)

        report_out.write(
            f"{len(approvalwise_vectors)},{distance},{execution_time_s},{idx}\n")

        approvalwise_vectors.append(farthest_approvalwise_vectors)

        if output_dir:
            filepath = os.path.join(
                output_dir, f"new_approvalwise_vectors.txt")
            with open(filepath, 'w') as out:
                approvalwise_vector.dump_to_text_file(
                    new_approvalwise_vectors, out)
            print(f'New approvalwise vectors saved to {filepath}')

    return new_approvalwise_vectors


def run_experiment(
        approvalwise_vectors: list[ApprovalwiseVector],
        reference_approvalwise_vectors: list[ApprovalwiseVector],
        num_new_instances: int,
        algorithm: Algorithm,
        idx: int,
        report_out,
        output_dir: str | None = None,
):

    if num_new_instances < 1:
        raise ValueError("num_new_instances must be greater than 0")

    approvalwise_vectors = approvalwise_vectors.copy()
    new_approvalwise_vectors = []
    reference_approvalwise_vectors = list(
        reversed(reference_approvalwise_vectors))

    if idx == 0:
        report_out.write(
            "experiment_size,distance,execution_time,idx\n")

    max_dist = None
    for _ in range(len(reference_approvalwise_vectors) + 1):
        farthest_approvalwise_vectors, distance, execution_time_s = measure_iteration(
            approvalwise_vectors, algorithm, max_dist=max_dist)

        new_approvalwise_vectors.append(farthest_approvalwise_vectors)
        max_dist = distance

        report_out.write(
            f"{len(approvalwise_vectors)},{distance},{execution_time_s},{idx}\n")

        if output_dir:
            filepath = os.path.join(
                output_dir, f"new_approvalwise_vectors.txt")
            with open(filepath, 'w') as out:
                approvalwise_vector.dump_to_text_file(
                    new_approvalwise_vectors, out)
            print(f'New approvalwise vectors saved to {filepath}')

        if reference_approvalwise_vectors:
            approvalwise_vectors.append(reference_approvalwise_vectors.pop())


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_candidates", type=int, default=30)
    parser.add_argument("--num_voters", type=int, default=100)
    parser.add_argument("--num_instances", type=int, default=6*6)
    parser.add_argument("--num_new_instances", type=int, default=12)
    parser.add_argument("--family", type=str, default="euclidean")
    parser.add_argument("--algorithm", type=str, default="basin_hopping")
    parser.add_argument("--reference_algorithm", type=str, default=None)
    parser.add_argument("--load_from_file", type=str, default=None)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--save_results", action="store_true")

    args = parser.parse_args()

    num_candidates = args.num_candidates
    num_voters = args.num_voters
    num_instances = args.num_instances
    num_new_instances = args.num_new_instances
    family = args.family
    load_from_file = args.load_from_file
    algorithm_name = args.algorithm
    reference_algorithm_name = args.reference_algorithm
    save_results = args.save_results
    num_trials = args.num_trials

    algorithm = algorithms[algorithm_name]
    reference_algorithm = algorithms[reference_algorithm_name] if reference_algorithm is not None else None
    experiment_id = os.path.join(
        f'{num_candidates}x{num_voters}', family)

    for trial_idx in range(num_trials):

        if load_from_file:
            load_dir = os.path.join(
                'experiments', experiment_id, load_from_file)
            with open(os.path.join(load_dir, "approvalwise_vectors.txt"), 'r') as in_file:
                approvalwise_vectors = approvalwise_vector.load_from_text_file(
                    in_file)
                approvalwise_vectors = list(approvalwise_vectors.values())
        else:
            experiment = generate_elections.generate(
                num_candidates, num_voters, num_instances, family)
            approvalwise_vectors = get_approvalwise_vectors(
                experiment.elections)

        if save_results:
            results_dir = os.path.join(
                'results', experiment_id, 'one_step_metric', algorithm_name)
            os.makedirs(results_dir, exist_ok=True)

            with open(os.path.join(results_dir, "report.csv"), 'a') as report_out:

                reference_approvalwise_vectors = generate_reference(
                    approvalwise_vectors, num_new_instances, algorithm, trial_idx, report_out, results_dir)

            if reference_algorithm is not None:
                with open(os.path.join(results_dir, "reference-report.csv"), 'a') as report_out:
                    run_experiment(approvalwise_vectors, reference_approvalwise_vectors, num_new_instances,
                                   reference_algorithm, trial_idx, report_out, results_dir)
        else:
            reference_approvalwise_vectors = generate_reference(
                approvalwise_vectors, num_new_instances, algorithm, sys.stdout)
            if reference_algorithm is not None:
                run_experiment(approvalwise_vectors, reference_approvalwise_vectors, num_new_instances,
                               reference_algorithm, trial_idx, sys.stdout, None)


if __name__ == "__main__":
    main()
