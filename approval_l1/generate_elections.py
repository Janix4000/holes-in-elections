import argparse
import os
import pickle
import mapel.elections as mapel
import numpy as np


def generate(num_candidates: int, num_voters: int, num_instances: int, family_id: str) -> mapel.ApprovalElectionExperiment:
    size_id = f'{num_candidates}x{num_voters}'

    experiment_id = f'{size_id}/{family_id}'

    experiment = mapel.prepare_online_approval_experiment(
        distance_id="l1-approvalwise",
        embedding_id="fr"
    )

    experiment.set_default_num_candidates(num_candidates)
    experiment.set_default_num_voters(num_voters)

    match family_id:
        case "euclidean":
            experiment.add_empty_family(culture_id=family_id, num_candidates=num_candidates,
                                        num_voters=num_voters, family_id='euclidean_1d', color='blue')
            experiment.add_empty_family(culture_id=family_id, num_candidates=num_candidates,
                                        num_voters=num_voters, family_id='euclidean_2d', color='green')
            for radius in np.linspace(0, 0.5, num_instances // 2):
                d1 = mapel.generate_approval_election(
                    election_id=f'1d_r={radius:0.2}',
                    num_candidates=num_candidates,
                    num_voters=num_voters,
                    culture_id='euclidean',
                    params={'radius': radius}
                )
                d1.instance_id = f'1d_r={radius:0.2}'
                d2 = mapel.generate_approval_election(
                    election_id=f'1d_r={radius:0.2}',
                    num_candidates=num_candidates,
                    num_voters=num_voters,
                    culture_id='euclidean',
                    params={'radius': radius}
                )
                d2.instance_id = f'2d_r={radius:0.2}'
                experiment.add_election_to_family(d1, family_id='euclidean_1d')
                experiment.add_election_to_family(d2, family_id='euclidean_2d')
        case "resampling" | "noise":
            experiment.add_empty_family(culture_id=family_id, num_candidates=num_candidates,
                                        num_voters=num_voters, family_id=family_id, color='green')
            t = int(np.round(np.sqrt(num_instances)))
            k = num_instances // t
            for p in np.linspace(0, 1, t):
                for phi in np.linspace(0, 1, k):
                    election = mapel.generate_approval_election(
                        election_id=f'p={p:.2f},phi={phi:.2}',
                        num_candidates=num_candidates,
                        num_voters=num_voters,
                        culture_id=family_id,
                        params={'p': p, 'phi': phi}
                    )
                    election.instance_id = f'p={p:.2f},phi={phi:.2}'
                    experiment.add_election_to_family(
                        election, family_id=family_id)
        case "truncated_urn":
            experiment.add_empty_family(culture_id=family_id, num_candidates=num_candidates,
                                        num_voters=num_voters, family_id=family_id, color='green')
            t = int(np.round(np.sqrt(num_instances)))
            k = num_instances // t
            for p in np.linspace(0, 1, t):
                for alpha in np.linspace(0, 1, k):
                    election = mapel.generate_approval_election(
                        election_id=f'p={p:.2f},alpha={alpha:.2}',
                        num_candidates=num_candidates,
                        num_voters=num_voters,
                        culture_id=family_id,
                        params={'p': p, 'alpha': alpha}
                    )
                    election.instance_id = f'p={p:.2f},alpha={alpha:.2}'
                    experiment.add_election_to_family(
                        election, family_id=family_id)

    experiment.compute_distances(distance_id='l1-approvalwise')
    experiment.embed_2d(embedding_id="fr")
    return experiment


def main():

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument('--num_candidates', type=int,
                        help='Number of candidates', default=30)
    parser.add_argument('--num_voters', type=int,
                        help='Number of voters', default=100)
    parser.add_argument('--num_instances', type=int,
                        help='Number of elections to generate', default=36)
    parser.add_argument('--family_id', type=str,
                        help='Family ID', default='euclidean')

    # Parse the command line arguments
    args = parser.parse_args()

    # Get the values from the command line arguments
    num_candidates = args.num_candidates
    num_voters = args.num_voters
    num_instances = args.num_instances
    family_id = args.family_id

    experiment = generate(
        num_candidates, num_voters, num_instances, family_id)

    experiments_path = os.path.join(
        'experiments', f'{num_candidates}x{num_voters}', family_id)
    os.makedirs(experiments_path, exist_ok=True)
    with open(os.path.join(experiments_path, 'elections.pkl'), 'wb') as file:
        pickle.dump(experiment.elections, file)


if __name__ == "__main__":
    main()
