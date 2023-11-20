import os
import pickle
import matplotlib.pyplot as plt
import mapel.elections as mapel
from approval_l1.scripts.basin_hopping import basin_hopping
from scripts.approvalwise_vector import get_approvalwise_vectors
from scripts.gurobi import gurobi_ilp
import argparse
import scripts.experiments as experiments


def get_algorithm(algorithm_id: str):
    match algorithm_id:
        case "basin_hopping":
            return lambda approvalwise_vectors: basin_hopping(
                approvalwise_vectors=approvalwise_vectors,
                step_size=7,
                seed=2137,
                big_step_chance=0.2,
                x0='step_vector'
            )
        case "basin_hopping_random":
            return lambda approvalwise_vectors: basin_hopping(
                approvalwise_vectors=approvalwise_vectors,
                step_size=7,
                seed=2137,
                big_step_chance=0.2,
                x0='random'
            )
        case "gurobi":
            return gurobi_ilp
        case _:
            raise ValueError(f"Unknown algorithm: {algorithm_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_instances', type=int,
                        help='The number of elections to generate', default=36)
    parser.add_argument('--num_candidates', type=int,
                        help='The number of candidates', default=30)
    parser.add_argument('--num_voters', type=int,
                        help='The number of voters', default=100)
    parser.add_argument('--family', type=str,
                        help='The family of elections to generate', default='euclidean')
    parser.add_argument('--load_pickle', type=bool,
                        help='Whether to load experiment from the pickle file', default=True)
    parser.add_argument('--algorithm', type=str,
                        default='gurobi', help='The algorithm to use')

    args = parser.parse_args()

    num_instances = args.num_instances
    num_candidates = args.num_candidates
    num_voters = args.num_voters
    family_id = args.family
    load_pickle = args.load_pickle
    algorithm_id = args.algorithm

    run_experiment(family_id, algorithm_id, num_instances,
                   num_candidates, num_voters, load_pickle)


def run_experiment(family_id: str, algorithm_id: str, num_instances: int, num_candidates: int, num_voters: int, load_pickle: bool):
    experiment_id = f'{num_candidates}x{num_voters}/{family_id}'

    generator = get_algorithm(algorithm_id)

    if load_pickle:
        with open(os.path.join('experiments', experiment_id, 'elections.pkl'), 'rb') as f:
            elections = pickle.load(f)
    else:
        experiment = mapel.prepare_offline_approval_experiment(
            experiment_id=experiment_id,
            distance_id="l1-approvalwise",
            embedding_id="fr"
        )
        experiment.prepare_elections()
        elections = experiment.elections

    family_dirpath = os.path.join('results', algorithm_id, experiment_id)
    os.makedirs(family_dirpath, exist_ok=True)

    meaningful_elections = experiments.get_meaningful_elections(elections)
    approvalwise_vectors = get_approvalwise_vectors(meaningful_elections)

    _new_approvalwise_vectors, report = experiments.generate_farthest_elections_l1_approvalwise(
        approvalwise_vectors, num_instances, generator, experiment_id, save_snapshots=family_dirpath)

    report.to_csv(os.path.join(family_dirpath, 'report.csv'))
