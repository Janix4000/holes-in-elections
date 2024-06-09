import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scripts.approvalwise_vector import load_from_text_file

plt.rcParams['figure.dpi'] = 300


parser = argparse.ArgumentParser()

parser.add_argument('--reference_solver_id', type=str,
                    help='The ID of the solver used for reference', default='gurobi')
parser.add_argument('--num_candidates', type=int,
                    help='The number of candidates', default=20)
parser.add_argument('--num_voters', type=int,
                    help='The number of voters', default=50)
parser.add_argument('--family', type=str,
                    help='The family of elections', default='euclidean')

args = parser.parse_args()

reference_algorithm = args.reference_solver_id
num_candidates = args.num_candidates
num_voters = args.num_voters
family_id = args.family


experiment_id = f'{num_candidates}x{num_voters}/{family_id}'

results_dir = os.path.join('results', experiment_id)
table_dir = os.path.join('tables', experiment_id)

with open(os.path.join(results_dir, reference_algorithm, 'new-approvalwise-vectors.txt'), 'r') as file:
    reference_new_approvalwise_vectors = load_from_text_file(file)
reference_new_approvalwise_vectors = list(
    reference_new_approvalwise_vectors.values())

with open(os.path.join('experiments', experiment_id, 'elections.txt'), 'r') as file:
    approvalwise_vectors = load_from_text_file(file)
approvalwise_vectors = list(approvalwise_vectors.values())


def l1_distance(av, other_avs) -> int:
    # print(av, other_avs)
    av = np.array(av).reshape(1, -1)
    other_avs = np.array(other_avs)
    return np.min(np.sum(np.abs(av - other_avs), axis=1))


def calculate_space_filling_metric_reference(algorithm: str, i_start: int):
    with open(os.path.join(results_dir, algorithm, 'new-approvalwise-vectors.txt'), 'r') as file:
        new_approvalwise_vectors = load_from_text_file(file)

    vectors = list(new_approvalwise_vectors.values())
    vectors = vectors[i_start:]
    metrics = []

    for i in range(len(vectors)):
        metric = np.mean(
            [l1_distance(vector, approvalwise_vectors + vectors[:j] + vectors[j+1:i+1])
             for j, vector in enumerate(vectors[:i+1])]
        )
        metrics.append(metric)

    return metrics


def calculate_space_filling_metric_heuristic(algorithm: str, i_start: int, trials: int = 1):
    trial_metrics = []
    for i_trial in range(trials):
        dirpath = os.path.join(results_dir, algorithm, f'start_{i_start}')
        with open(os.path.join(dirpath, f'new_approvalwise_vectors_trial_{i_trial}.txt'), 'r') as f:
            new_approvalwise_vectors = load_from_text_file(f)

        vectors = list(new_approvalwise_vectors.values())
        reference_vectors = reference_new_approvalwise_vectors[:i_start]
        metrics = []

        for i in range(len(vectors)):
            metric = np.mean(
                [l1_distance(vector, approvalwise_vectors + reference_vectors + vectors[:j] + vectors[j+1:i+1])
                 for j, vector in enumerate(vectors[:i+1])]
            )
            metrics.append(metric)
        trial_metrics.append(metrics)
    return trial_metrics


last_iteration = 9

heuristics_trials = [
    ('basin_hopping', 10),
    ('basin_hopping_random', 10),
    ('pairs', 1),
    ('greedy_dp', 1)
]
algorithm_labels = {
    'gurobi': 'gurobi',
    'basin_hopping': 'basinhopping',
    'basin_hopping_random': 'basinhoppingrandom',
    'pairs': 'Pairs',
    'greedy_dp': 'greedydp'
}

i_starts = range(10)

results = []

for i_start in i_starts:
    metrics_rows = []
    for algorithm, trials in heuristics_trials:
        trial_metrics = calculate_space_filling_metric_heuristic(
            algorithm, i_start, trials)
        algorithm_label = algorithm_labels[algorithm]
        metrics_flatten = [[algorithm_label, i + i_start, metric, trial]
                           for trial, metrics in enumerate(trial_metrics) for i, metric in enumerate(metrics)]
        metrics_rows.extend(metrics_flatten)

    rows = calculate_space_filling_metric_reference(
        reference_algorithm, i_start)
    metrics_rows.extend(
        [[reference_algorithm, i + i_start, metric, 1]
            for i, metric in enumerate(rows)]
    )

    metrics_df = pd.DataFrame(metrics_rows, columns=[
        'Algorithm', 'i', 'metric', 'trial'])

    last_iteration_mask = metrics_df['i'] == last_iteration
    grouped_df = metrics_df[last_iteration_mask].groupby('Algorithm')
    means = grouped_df['metric'].mean().reset_index(
    ).transpose().reset_index(drop=True)
    means.columns = means.iloc[0]
    means = means.drop(0).add_suffix('mean')

    stds = grouped_df['metric'].std().reset_index(
    ).transpose().reset_index(drop=True)
    stds.columns = stds.iloc[0]
    stds = stds.drop(0).add_suffix('std')

    summary_df = pd.concat([means, stds], axis=1)
    summary_df['istart'] = i_start
    results.append(summary_df)

results_df = pd.concat(results)

os.makedirs(table_dir, exist_ok=True)
results_df.round(2).to_csv(os.path.join(
    table_dir, 'space_filling_summary.csv'), index=False)
