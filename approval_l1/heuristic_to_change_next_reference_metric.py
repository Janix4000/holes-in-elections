import os
import sys
from source.heuristic_to_change_next_reference import heuristic_to_change_next_reference
from scripts.algorithms import Algorithm, algorithms
from scripts.approvalwise_vector import ApprovalwiseVector, dump_to_text_file, get_approvalwise_vectors
import generate_elections


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_candidates", type=int, default=20)
    parser.add_argument("--num_voters", type=int, default=50)
    parser.add_argument("--num_instances", type=int, default=5*5)
    parser.add_argument("--num_new_instances", type=int, default=3)
    parser.add_argument("--family", type=str, default="euclidean")
    parser.add_argument("--algorithm", type=str, default="basin_hopping")
    parser.add_argument("--reference_algorithm", type=str, default='gurobi')
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--save_results", action="store_true")

    args = parser.parse_args()

    num_candidates = args.num_candidates
    num_voters = args.num_voters
    num_instances = args.num_instances
    num_new_instances = args.num_new_instances
    family = args.family
    algorithm_name = args.algorithm
    reference_algorithm_name = args.reference_algorithm
    num_trials = args.num_trials
    save_results = args.save_results

    experiment_id = os.path.join(
        f'{num_candidates}x{num_voters}', family)

    if save_results:
        results_dir = os.path.join(
            'results', 'heuristic_to_change_next_reference_metric', experiment_id, algorithm_name)
        os.makedirs(results_dir, exist_ok=True)
        csv_report_out = open(os.path.join(results_dir, "report.csv"), 'a')
    else:
        csv_report_out = sys.stdout

    heuristic_algorithm = algorithms[algorithm_name]
    reference_algorithm = algorithms[reference_algorithm_name] if reference_algorithm_name is not None else None

    experiment = generate_elections.generate(
        num_candidates, num_voters, num_instances, family)
    approvalwise_vectors = get_approvalwise_vectors(
        experiment.elections)

    for new_reference_approvalwise_vectors, new_heuristics_approvalwise_vectors in heuristic_to_change_next_reference(
            approvalwise_vectors, reference_algorithm, heuristic_algorithm, num_new_instances,  csv_report_out=csv_report_out, num_trials=num_trials):
        if save_results:
            if new_reference_approvalwise_vectors:
                with open(os.path.join(results_dir, "new_reference_approvalwise_vectors.txt"), 'w') as out_file:
                    dump_to_text_file(
                        new_reference_approvalwise_vectors, out_file)
            if new_heuristics_approvalwise_vectors:
                with open(os.path.join(results_dir, "new_heuristics_approvalwise_vectors.txt"), 'w') as out_file:
                    dump_to_text_file(
                        new_heuristics_approvalwise_vectors, out_file)

    if save_results:
        csv_report_out.close()


if __name__ == "__main__":
    main()
