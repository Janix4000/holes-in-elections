from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import time

import mapel.elections as mapel
from scripts.approvalwise_vector import ApprovalwiseVector
from tqdm import tqdm
import os
import pickle


to_purge_elections_ids_regexes = [
    '.*empty.*',
    '.*full.*',
    '.*IC.*',
    '.*ID.*',
    '.*background_.*',
    '.*FES.*',
    '.*compass.*',
]


def add_grid(experiment: mapel.ApprovalElectionExperiment, num_voters: int, num_candidates: int) -> None:
    for p in np.linspace(0, 1, 11):
        for phi in np.linspace(0, 1, 11):
            election = mapel.generate_approval_election(
                election_id=f'p={p:.1f}, phi={phi:.1}',
                num_candidates=num_candidates,
                num_voters=num_voters,
                culture_id='resampling',
                params={'p': p, 'phi': phi}
            )
            experiment.add_election_to_family(
                election, family_id='compass')


def add_compass(experiment: mapel.ApprovalElectionExperiment, with_grid: bool = False):
    experiment.add_election(culture_id='full', election_id='FULL', color='red')
    experiment.add_election(
        culture_id='empty', election_id='EMPTY', color='blue')
    experiment.add_election(culture_id='ic', params={
                            'p': 0.5}, election_id='IC 0.5', color='green')
    experiment.add_election(culture_id='id', params={
                            'p': 0.5}, election_id='ID 0.5', color='orange')

    experiment.add_empty_family(family_id='compass', color='#CCCCCC')

    sample_election = next(iter(experiment.elections.values()))
    num_candidates = sample_election.num_candidates
    num_voters = sample_election.num_voters

    if with_grid:
        add_grid(experiment, num_voters, num_candidates)


def show_2d_map_with_compass(experiment: mapel.ApprovalElectionExperiment, with_grid: bool = False, show_legend: bool = True):
    add_compass(experiment, with_grid=with_grid)
    experiment.compute_distances(distance_id='l1-approvalwise')
    experiment.embed_2d(embedding_id="fr")
    experiment.print_map_2d(legend=show_legend)


def generate_farthest_elections_l1_approvalwise(
    approvalwise_vectors: list[ApprovalwiseVector],
    num_generated: int,
    generator: Callable[[list[ApprovalwiseVector]], tuple[ApprovalwiseVector, int]],
    family_id: str,
    save_snapshots: str | None = None,
):
    new_approvalwise_vectors = {}
    new_distances = []
    execution_times = []
    current_experiment_size = len(approvalwise_vectors)
    experiment_sizes = list(
        range(current_experiment_size, current_experiment_size + num_generated))

    family_id = f'NFE-{family_id}'

    for idx in tqdm(range(num_generated), desc='Finding farthest elections'):
        start = time.time()
        approvalwise_vector, distance = generator(
            approvalwise_vectors)
        execution_times.append(time.time() - start)

        new_approvalwise_vectors[f'{family_id}-{idx}'] = approvalwise_vector
        approvalwise_vectors.append(approvalwise_vector)
        new_distances.append(distance)

        report = pd.DataFrame({
            'experiment_size': experiment_sizes[:idx + 1],
            'execution_time': execution_times,
            'distance': new_distances
        })

        if save_snapshots is not None:
            snapshot_filepath = os.path.join(
                save_snapshots, 'new-approvalwise-vectors.pkl')
            with open(snapshot_filepath, 'wb') as f:
                pickle.dump(new_approvalwise_vectors, f)
            report.to_csv(os.path.join(save_snapshots, 'report.csv'))
            print(f'Snapshot {idx} saved to {snapshot_filepath}')

    report = pd.DataFrame({
        'experiment_size': experiment_sizes,
        'execution_time': execution_times,
        'distance': new_distances
    })

    return new_approvalwise_vectors, report


def plot_report(report: pd.DataFrame):
    plt.scatter(report.experiment_size, report.distance)
    plt.plot(report.experiment_size, report.distance, '--')
    plt.title('Distance to the closest election')
    plt.xlabel('Experiment size')
    plt.ylabel('Distance to the closest election')
    plt.ylim(0, None)
    plt.show()

    plt.scatter(report.experiment_size, report.execution_time)
    plt.plot(report.experiment_size, report.execution_time, '--')
    plt.title('Execution time')
    plt.xlabel('Experiment size')
    plt.ylabel('Execution time [s]')
    plt.show()


def get_meaningful_elections(elections: dict[str, mapel.ApprovalElection]):
    to_purge_regex = '|'.join(to_purge_elections_ids_regexes)
    return {instance_id: election for instance_id, election in elections.items() if not re.match(to_purge_regex, instance_id)}
