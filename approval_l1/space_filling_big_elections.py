import argparse
import os
import time

import pandas as pd
from scripts.algorithms import algorithms
from scripts.approvalwise_vector import dump_to_text_file, load_from_text_file

parser = argparse.ArgumentParser(
    description='Generate farthest elections for space filling using only heuristic')
parser.add_argument('--num_candidates', type=int,
                    default=50, help='Number of candidates')
parser.add_argument('--num_voters', type=int,
                    default=500, help='Number of voters')
parser.add_argument('--algorithm', type=str,
                    default='basin_hopping', help='Algorithm ID')
parser.add_argument('--family', type=str,
                    default='euclidean', help='Family ID')
parser.add_argument('--trials', type=int, default=10, help='Number of trials')
parser.add_argument('--steps', type=int, default=20, help='Number of steps')

args = parser.parse_args()

num_candidates = args.num_candidates
num_voters = args.num_voters
algorithm_id = args.algorithm
family_id = args.family
trials = args.trials
steps = args.steps


experiment_id = f'{num_candidates}x{num_voters}'
csv_report_path = os.path.join(
    'results', experiment_id, family_id, algorithm_id, f'space_filling_report.csv')

i_trials = range(trials)
report_rows = []

print(
    f'Generating farthest elections for {experiment_id} {family_id} using {algorithm_id} algorithm')

for i_trial in i_trials:
    results_dir = os.path.join('results', experiment_id, family_id)
    res_elections_path = os.path.join(
        results_dir, algorithm_id, f'start_0',  f'new_approvalwise_vectors_trial_{i_trial}.txt')
    os.makedirs(os.path.dirname(res_elections_path), exist_ok=True)

    with open(os.path.join('experiments', experiment_id, family_id, 'elections.txt'), 'r') as file:
        approvalwise_vectors = load_from_text_file(file)
        approvalwise_vectors = list(approvalwise_vectors.values())

    algorithm = algorithms[algorithm_id]
    starting_approval_vectors = approvalwise_vectors
    new_approvalwise_vectors = []

    for i in range(steps):
        time_start = time.time()
        farthest_vector, distance = algorithm(starting_approval_vectors)
        dt = time.time() - time_start

        starting_approval_vectors.append(farthest_vector)
        new_approvalwise_vectors.append(farthest_vector)
        report_rows.append(
            [family_id, algorithm_id, 0, i_trial, i, distance, dt])

    with open(res_elections_path, 'w') as file:
        dump_to_text_file(new_approvalwise_vectors, file)

    csv_report = pd.DataFrame(report_rows, columns=[
                              'family', 'algorithm', 'i_start', 'i_trial', 'iteration', 'distance', 'dt'])
    csv_report.to_csv(csv_report_path, index=False)

print(
    f'Farthest elections for {experiment_id} {family_id} using {algorithm_id} algorithm generated')
