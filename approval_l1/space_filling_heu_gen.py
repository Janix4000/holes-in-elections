import time
import pandas as pd
import os
import numpy as np
import pickle
from scripts.approvalwise_vector import ApprovalwiseVector, dump_to_text_file, load_from_text_file
from scripts.algorithms import algorithms

from itertools import product, chain

num_candidates = 20
num_voters = 50
reference_algorithm_id = 'gurobi'

family_ids = ['noise', 'resampling', 'truncated_urn']
i_starts = range(0, 12)
i_trials = range(10)

experiment_id = f'{num_candidates}x{num_voters}'
report_rows = []
csv_report_path = os.path.join(
    'results', experiment_id, f'space_filling_report.csv')

experiment_configurations = chain(
    product(family_ids[2:], ['basin_hopping'], i_starts, i_trials),
    product(family_ids, ['basin_hopping_random'], i_starts, i_trials),
    product(family_ids, ['pairs', 'greedy_dp'],
            i_starts, range(1)),
)

for family_id, algorithm_id, i_start, i_trial in experiment_configurations:
    print(f'Start {family_id} {algorithm_id} {i_start} {i_trial}')

    results_dir = os.path.join('results', experiment_id, family_id)
    res_elections_path = os.path.join(
        results_dir, algorithm_id, f'start_{i_start}',  f'new_approvalwise_vectors_trial_{i_trial}.txt')
    os.makedirs(os.path.dirname(res_elections_path), exist_ok=True)

    with open(os.path.join('experiments', experiment_id, family_id, 'elections.txt'), 'r') as file:
        approvalwise_vectors = load_from_text_file(file)
        approvalwise_vectors = list(approvalwise_vectors.values())

    with open(os.path.join(results_dir, reference_algorithm_id, 'new-approvalwise-vectors.txt'), 'r') as f:
        reference_new_approvalwise_vectors = load_from_text_file(f)
        reference_new_approvalwise_vectors = list(map(lambda x: ApprovalwiseVector(list(sorted(x, reverse=True)), num_voters),
                                                      reference_new_approvalwise_vectors.values()))

    algorithm = algorithms[algorithm_id]
    starting_approval_vectors = approvalwise_vectors + \
        reference_new_approvalwise_vectors[:i_start]
    new_approvalwise_vectors = []

    for i in range(i_start, len(reference_new_approvalwise_vectors)):
        time_start = time.time()
        farthest_vector, distance = algorithm(starting_approval_vectors)
        dt = time.time() - time_start

        starting_approval_vectors.append(farthest_vector)
        new_approvalwise_vectors.append(farthest_vector)
        report_rows.append(
            [family_id, algorithm_id, i_start, i_trial, i, distance, dt])

    print(f'Done {family_id} {algorithm_id} {i_start} {i_trial}')

    with open(res_elections_path, 'w') as file:
        dump_to_text_file(new_approvalwise_vectors, file)

    csv_report = pd.DataFrame(report_rows, columns=[
                              'family', 'algorithm', 'i_start', 'i_trial', 'iteration', 'distance', 'dt'])
    csv_report.to_csv(csv_report_path, index=False)
