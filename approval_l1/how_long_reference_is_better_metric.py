import os
import sys
from source.how_long_reference_is_better import how_long_reference_is_better, measure_iteration
from scripts.algorithms import algorithms
from scripts.approvalwise_vector import dump_to_text_file, get_approvalwise_vectors, load_from_text_file
import generate_elections


def generate_reference_approvalwise_vectors(approvalwise_vectors, reference_algorithm, num_reference_instances, csv_report_out):
    reference_approvalwise_vectors = []
    approvalwise_vectors = approvalwise_vectors[:]
    for i in range(num_reference_instances):
        reference_approvalwise_vector, reference_distance, reference_dt = measure_iteration(
            approvalwise_vectors, reference_algorithm)
        reference_approvalwise_vectors.append(
            reference_approvalwise_vector)
        approvalwise_vectors.append(reference_approvalwise_vector)
        csv_report_out.write(
            f'-1,{i},{reference_distance},{reference_dt},reference_init\n')
    return reference_approvalwise_vectors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--num_voters", type=int, default=20)
    parser.add_argument("--num_instances", type=int, default=5*5)
    parser.add_argument("--num_reference_instances", type=int, default=7)

    parser.add_argument("--family", type=str, default="euclidean")
    parser.add_argument("--algorithm", type=str, default="basin_hopping")
    parser.add_argument("--reference_algorithm", type=str, default='gurobi')
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--load_from_file", type=str, default=None)
    parser.add_argument("--reference_load_from_file", type=str, default=None)

    args = parser.parse_args()

    num_candidates = args.num_candidates
    num_voters = args.num_voters
    num_instances = args.num_instances
    num_reference_instances = args.num_reference_instances

    load_from_file = args.load_from_file
    reference_load_from_file = args.reference_load_from_file

    num_trials = args.num_trials
    family = args.family
    algorithm_name = args.algorithm
    reference_algorithm_name = args.reference_algorithm
    save_results = args.save_results

    if load_from_file:
        with open(load_from_file, 'r') as in_file:
            approvalwise_vectors = load_from_text_file(
                in_file)
        approvalwise_vectors = list(approvalwise_vectors.values())
        num_candidates = approvalwise_vectors[0].num_candidates
        num_voters = approvalwise_vectors[0].num_voters
        num_instances = len(approvalwise_vectors)

    else:
        print("Generating elections")
        experiment = generate_elections.generate(
            num_candidates, num_voters, num_instances, family)
        approvalwise_vectors = get_approvalwise_vectors(
            experiment.elections)
        num_candidates = approvalwise_vectors[0].num_candidates
        print("Generated elections")

    if save_results:
        results_dir = os.path.join(
            'results', 'how_long_heuristic_is_better', f'{num_candidates}x{num_voters}', algorithm_name, family)
        os.makedirs(results_dir, exist_ok=True)
        csv_report_out = open(os.path.join(results_dir, "report.csv"), 'a')
    else:
        print("Using stdout as output")
        csv_report_out = sys.stdout

    csv_report_columns = ['trial', 'iteration', 'distance',
                          'execution_time_s', 'algorithm']
    csv_report_out.write(','.join(csv_report_columns) + '\n')

    heuristic_algorithm = algorithms[algorithm_name]
    reference_algorithm = algorithms[reference_algorithm_name] if reference_algorithm_name is not None else None

    if reference_load_from_file:
        with open(reference_load_from_file, 'r') as in_file:
            reference_approvalwise_vectors = load_from_text_file(
                in_file)
            reference_approvalwise_vectors = list(
                reference_approvalwise_vectors.values())
        reference_approvalwise_vectors = reference_approvalwise_vectors[:num_reference_instances]
    else:
        print("Generating reference approvalwise vectors")
        reference_approvalwise_vectors = generate_reference_approvalwise_vectors(
            approvalwise_vectors, reference_algorithm, num_reference_instances, csv_report_out)
        print("Reference:", reference_approvalwise_vectors)

    heuristic_kwargs = {}

    for i_trial in range(num_trials):
        if save_results:
            trial_results_dir = os.path.join(
                results_dir, f'trial_{i_trial}') if num_trials > 1 else results_dir
            os.makedirs(trial_results_dir, exist_ok=True)

        experiment_results = how_long_reference_is_better(
            approvalwise_vectors, reference_approvalwise_vectors, reference_algorithm, heuristic_algorithm, csv_report_out, i_trial=i_trial, **heuristic_kwargs)

        for new_reference_approvalwise_vectors, new_heuristics_approvalwise_vectors in experiment_results:
            if save_results:
                if new_reference_approvalwise_vectors:
                    with open(os.path.join(trial_results_dir, "new_reference_approvalwise_vectors.txt"), 'w') as out_file:
                        dump_to_text_file(
                            new_reference_approvalwise_vectors, out_file)
                if new_heuristics_approvalwise_vectors:
                    with open(os.path.join(trial_results_dir, "new_heuristics_approvalwise_vectors.txt"), 'w') as out_file:
                        dump_to_text_file(
                            new_heuristics_approvalwise_vectors, out_file)

        if not save_results:
            print("Reference:", reference_approvalwise_vectors)
            print("New reference:", new_reference_approvalwise_vectors)
            print("New heuristic:", new_heuristics_approvalwise_vectors)

    if save_results:
        csv_report_out.close()


if __name__ == "__main__":
    main()
