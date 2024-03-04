import os
from source.heuristic_to_change_next_reference import heuristic_to_change_next_reference
from scripts.algorithms import Algorithm, algorithms
from scripts.approvalwise_vector import ApprovalwiseVector, get_approvalwise_vectors
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

    args = parser.parse_args()

    num_candidates = args.num_candidates
    num_voters = args.num_voters
    num_instances = args.num_instances
    num_new_instances = args.num_new_instances
    family = args.family
    algorithm_name = args.algorithm
    reference_algorithm_name = args.reference_algorithm
    num_trials = args.num_trials

    heuristic_algorithm = algorithms[algorithm_name]
    reference_algorithm = algorithms[reference_algorithm_name] if reference_algorithm_name is not None else None
    experiment_id = os.path.join(
        f'{num_candidates}x{num_voters}', family)

    experiment = generate_elections.generate(
        num_candidates, num_voters, num_instances, family)
    approvalwise_vectors = get_approvalwise_vectors(
        experiment.elections)

    res = heuristic_to_change_next_reference(
        approvalwise_vectors, reference_algorithm, heuristic_algorithm, num_new_instances, num_trials, verbose=True)
    print(res)


if __name__ == "__main__":
    main()
