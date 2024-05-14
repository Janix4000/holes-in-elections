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

i_start = 7

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

with open(os.path.join('experiments', experiment_id, 'elections.pkl'), 'rb') as file:
    meaningful_elections = pickle.load(file)
approvalwise_vectors = get_approvalwise_vectors(meaningful_elections)


def l1_distance(av, other_avs) -> int:
    # print(av, other_avs)
    av = np.array(av).reshape(1, -1)
    other_avs = np.array(other_avs)
    return np.min(np.sum(np.abs(av - other_avs), axis=1))


def calculate_space_filling_metric(algorithm: str):
    if os.path.exists(os.path.join(results_dir, algorithm, 'new-approvalwise-vectors.pkl')):
        with open(os.path.join(results_dir, algorithm, 'new-approvalwise-vectors.pkl'), 'rb') as f:
            new_approvalwise_vectors = pickle.load(f)
    else:
        with open(os.path.join(results_dir, algorithm, f'new_approvalwise_vectors_{i_start}.txt'), 'r') as f:
            new_approvalwise_vectors = load_from_text_file(f)

    vectors = list(new_approvalwise_vectors.values() if isinstance(
        new_approvalwise_vectors, dict) else new_approvalwise_vectors)
    vectors = vectors[-i_start:]

    metrics = []

    for i in range(len(vectors)):
        metric = np.mean(
            [l1_distance(vector, approvalwise_vectors + vectors[:j] + vectors[j+1:i+1])
             for j, vector in enumerate(vectors[:i+1])]
        )
        metrics.append(metric)
        # break

    return metrics


metrics_rows = []

for algorithm in ['gurobi', 'basin_hopping', 'basin_hopping_random', 'pairs', 'greedy_dp']:
    metrics = calculate_space_filling_metric(algorithm)
    metrics_rows.extend(
        [[algorithm, i + i_start, metric] for i, metric in enumerate(metrics)]
    )

metrics_df = pd.DataFrame(metrics_rows, columns=['algorithm', 'i', 'metric'])
fig, ax = plt.subplots()
ax = barplot(data=metrics_df, x='i', y='metric',
             hue='algorithm', ax=ax)
# ax.set_yscale("log")

if save:
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(
        plot_dir, f'space_filling_{reference_algorithm}_{i_start}.png'))
    print(
        f"Saved space filling plot for {reference_algorithm} at {i_start} to {plot_dir}")
else:
    plt.show()
