import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from scripts.approvalwise_vector import get_approvalwise_vectors, load_from_text_file

from seaborn import barplot

plt.rcParams['figure.dpi'] = 300


parser = argparse.ArgumentParser()

parser.add_argument('--reference_solver_id', type=str,
                    help='The ID of the solver used for reference', default='gurobi')
# parser.add_argument('--solver_id', type=str,
#                     help='The ID of the solver used', required=True)
parser.add_argument('--num_candidates', type=int,
                    help='The number of candidates', default=30)
parser.add_argument('--num_voters', type=int,
                    help='The number of voters', default=100)
parser.add_argument('--family', type=str,
                    help='The family of elections', default='euclidean')
parser.add_argument('--save', type=bool, default=True, help='Save the plot')
parser.add_argument('--lb', type=int, help='Plot y-axis lower bound')

args = parser.parse_args()

reference_algorithm = args.reference_solver_id
# algorithm = args.solver_id
num_candidates = args.num_candidates
num_voters = args.num_voters
family_id = args.family
save = args.save
plot_lower_bound = args.lb


experiment_id = f'{num_candidates}x{num_voters}/{family_id}'

results_dir = os.path.join('results', experiment_id)
plot_dir = os.path.join('plots', experiment_id)

with open(os.path.join(results_dir, reference_algorithm, 'new-approvalwise-vectors.pkl'), 'rb') as f:
    reference_new_approvalwise_vectors = pickle.load(f)
    reference_new_approvalwise_vectors = list(
        reference_new_approvalwise_vectors.values())

with open(os.path.join('experiments', experiment_id, 'elections.pkl'), 'rb') as file:
    meaningful_elections = pickle.load(file)
approvalwise_vectors = get_approvalwise_vectors(meaningful_elections)


def l1_distance(av, other_avs) -> int:
    # print(av, other_avs)
    av = np.array(av).reshape(1, -1)
    other_avs = np.array(other_avs)
    return np.min(np.sum(np.abs(av - other_avs), axis=1))


def calculate_space_filling_metric_reference(algorithm: str, i_start: int):
    with open(os.path.join(results_dir, algorithm, 'new-approvalwise-vectors.pkl'), 'rb') as f:
        new_approvalwise_vectors = pickle.load(f)

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


heuristics_trials = [
    ('basin_hopping', 10),
    ('basin_hopping_random', 10),
    ('pairs', 1),
    ('greedy_dp', 1)
]
algorithm_labels = {
    'gurobi': 'Gurobi ILP',
    'basin_hopping': 'Basin Hopping',
    'basin_hopping_random': 'Basin Hopping Random',
    'pairs': 'Pairs',
    'greedy_dp': 'Greedy DP'
}

i_starts = [0, 3, 5, 7]
draw_legend = [False, False, False, True]
draw_xlabel = [False, False, True, True]
draw_ylabel = [True, False, True, False]

for i_start, legend, xlabel, ylabel in zip(i_starts, draw_legend, draw_xlabel, draw_ylabel):
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
        [[algorithm_labels[reference_algorithm], i + i_start, metric, 1]
            for i, metric in enumerate(rows)]
    )

    metrics_df = pd.DataFrame(metrics_rows, columns=[
        'Algorithm', 'i', 'metric', 'trial'])

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    ax = barplot(data=metrics_df, x='i', y='metric',
                 hue='Algorithm', ax=ax, legend=legend)
    if xlabel:
        ax.set_xlabel("Next farthest vector's index", fontsize=20)
    else:
        ax.set_xlabel("")

    if ylabel:
        ax.set_ylabel("Average Space Filling Metric", fontsize=20)
    else:
        ax.set_ylabel("")

    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.tight_layout()

    if save:
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(
            plot_dir, f'space_filling_{reference_algorithm}_{i_start}.png'))
        print(
            f"Saved space filling plot for {reference_algorithm} at {i_start} to {plot_dir}")
    else:
        plt.show()
