# %%
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import mapel.elections as mapel
from scripts.approvalwise_vector import add_sampled_elections_to_experiment, load_from_text_file

import seaborn as sns

plt.rcParams['figure.dpi'] = 300


parser = argparse.ArgumentParser()

parser.add_argument('--num_candidates', type=int,
                    help='The number of candidates', default=20)
parser.add_argument('--num_voters', type=int,
                    help='The number of voters', default=50)
parser.add_argument('--num_generated', type=int,
                    help='The number of generated elections', default=20)
parser.add_argument('--family', type=str,
                    help='The family of elections', default='euclidean')

args = parser.parse_args()
num_candidates = args.num_candidates
num_voters = args.num_voters
family_id = args.family
num_generated = args.num_generated

experiment_id = f'{num_candidates}x{num_voters}/{family_id}'
experiment_dir = os.path.join('experiments', experiment_id)
results_dir = os.path.join('results', experiment_id)
plot_dir = os.path.join('plots', experiment_id.replace('_', '-'))
os.makedirs(plot_dir, exist_ok=True)

with open(os.path.join('experiments', experiment_id, 'elections.txt'), 'r') as file:
    approvalwise_vectors = load_from_text_file(file)
approvalwise_vectors = list(approvalwise_vectors.values())


def l1_distance(av, other_avs) -> int:
    av = np.array(av).reshape(1, -1)
    other_avs = np.array(other_avs)
    return np.min(np.sum(np.abs(av - other_avs), axis=1))


def add_compass(experiment: mapel.ApprovalElectionExperiment):
    experiment.add_election(culture_id='full', election_id='FULL', color='red')
    experiment.add_election(
        culture_id='empty', election_id='EMPTY', color='blue')
    experiment.add_election(culture_id='ic', params={
                            'p': 0.5}, election_id='IC 0.5', color='green')
    experiment.add_election(culture_id='id', params={
                            'p': 0.5}, election_id='ID 0.5', color='orange')


def calculate_space_filling_metric_heuristic(algorithm: str, i_start: int, trials: int = 1):
    trial_metrics = []
    for i_trial in range(trials):
        dirpath = os.path.join(results_dir, algorithm, f'start_{i_start}')
        with open(os.path.join(dirpath, f'new_approvalwise_vectors_trial_{i_trial}.txt'), 'r') as f:
            new_approvalwise_vectors = load_from_text_file(f)

        vectors = list(new_approvalwise_vectors.values())
        metrics = []

        for i in range(len(vectors)):
            metric = np.mean(
                [l1_distance(vector, approvalwise_vectors + vectors[:j] + vectors[j+1:i+1])
                 for j, vector in enumerate(vectors[:i+1])]
            )
            metrics.append(metric)
        trial_metrics.append(metrics)
    return trial_metrics


def prepare_experiment(num_voters: int, num_candidates: int) -> mapel.ApprovalElectionExperiment:
    experiment = mapel.prepare_online_approval_experiment(
        distance_id="l1-approvalwise",
        embedding_id="fr"
    )

    experiment.set_default_num_candidates(num_candidates)
    experiment.set_default_num_voters(num_voters)

    return experiment


plt.rcParams['figure.dpi'] = 300

heuristics_trials = [
    ('basin_hopping', 10),
    ('basin_hopping_random', 10),
    ('greedy_dp', 1)
]

algorithm_labels = {
    'gurobi': 'Gurobi ILP',
    'basin_hopping': 'Basin Hopping',
    'basin_hopping_random': 'Basin Hopping Random',
    'pairs': 'Pairs',
    'greedy_dp': 'Greedy DP'
}
algorithm_colors = {
    'gurobi': 'purple',
    'basin_hopping': 'red',
    'basin_hopping_random': 'green',
    'pairs': 'blue',
    'greedy_dp': 'orange'
}
family_labels = {
    'euclidean': 'Euclidean',
    'truncated_urn': 'Truncated Urn',
    'noise': 'Noise',
    'resampling': 'Resampling'
}
palette = {label: algorithm_colors.get(
    algo_id) for algo_id, label in algorithm_labels.items()}


space_filling_report_df = pd.read_csv(os.path.join(
    results_dir, '..', 'space_filling_report.csv'))


if 'iteration' not in space_filling_report_df.columns:
    iterations = []
    start, trial = None, None
    iteration = None
    for i, row in space_filling_report_df.iterrows():
        if start != row['i_start'] or trial != row['i_trial']:
            iteration = row['i_start']
            start = row['i_start']
            trial = row['i_trial']
        else:
            iteration += 1
        iterations.append(iteration)
    space_filling_report_df['iteration'] = iterations

# %% Plot the plain map
experiment = prepare_experiment(num_voters, num_candidates)
add_compass(experiment)
add_sampled_elections_to_experiment(
    approvalwise_vectors, experiment, family_labels[family_id], color='cyan', seed=0)
experiment.compute_distances(distance_id='l1-approvalwise')
experiment.embed_2d(embedding_id="fr")
map_filepath = os.path.join(
    '..', plot_dir, f'space-filling-map.png')
plt.rcParams.update({'font.size': 10})
experiment.print_map_2d(legend=True, saveas=map_filepath, show=False)


# %% Plot the space filling metric
i_start = 0
metrics_rows = []
for algorithm, trials in heuristics_trials:
    trial_metrics = calculate_space_filling_metric_heuristic(
        algorithm, i_start, trials)
    algorithm_label = algorithm_labels[algorithm]
    metrics_flatten = [[algorithm_label, i + i_start, metric, trial]
                       for trial, metrics in enumerate(trial_metrics) for i, metric in enumerate(metrics)]
    metrics_rows.extend(metrics_flatten)
metrics_df = pd.DataFrame(metrics_rows, columns=[
    'Algorithm', 'i', 'metric', 'trial'])
fig, ax = plt.subplots()
ax = sns.lineplot(data=metrics_df, x='i', y='metric',
                  hue='Algorithm', ax=ax, legend=False, markers=True, dashes=True, style='Algorithm', palette=palette)
ax.set_xlabel("Next farthest vector's index", fontsize=20)
ax.set_ylabel("Average Space Filling Metric", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xticks(
    range(i_start + 1, num_generated + 1))
plt.grid(True)
fig.tight_layout()
os.makedirs(plot_dir, exist_ok=True)
fig.savefig(os.path.join(
    plot_dir, f'space-filling-{i_start}-metric.png'))
print(
    f"Saved space filling plot for {family_id} to {plot_dir}")

# %% Plot the time
i_start = 0
fig, ax = plt.subplots()
i_start_mask = space_filling_report_df['i_start'] == i_start
family_mask = space_filling_report_df['family'] == family_id
report_df = space_filling_report_df[i_start_mask & family_mask]
report_df['Algorithm'] = report_df['algorithm'].map(algorithm_labels)
plt.rcParams.update({'font.size': 12})
ax = sns.lineplot(data=report_df, x='iteration', y='dt',
                  hue='Algorithm', style='Algorithm', markers=True, dashes=True, palette=palette, ax=ax)
ax.set(yscale='log')
ax.set_ylabel("Time (s)\n(logarithmic scale)", fontsize=20)
ax.set_xlabel("Next farthest vector's index", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xticks(
    range(i_start + 1, num_generated + 1))
ax.legend(loc='center', ncol=2)
plt.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, f'space-filling-{i_start}-time.png'))
print(
    f"Saved time plot for for {family_id} to {plot_dir}")

# %% Plot the map
i_start = 0
experiment = prepare_experiment(num_voters, num_candidates)
add_compass(experiment)
family_label = family_labels[family_id]
add_sampled_elections_to_experiment(
    approvalwise_vectors, experiment, family_label, color='cyan', seed=0)
algorithms = ['basin_hopping', 'basin_hopping_random', 'greedy_dp']
for algorithm in algorithms:
    dirpath = os.path.join(results_dir, algorithm, f'start_{i_start}')
    with open(os.path.join(dirpath, f'new_approvalwise_vectors_trial_0.txt'), 'r') as f:
        new_approvalwise_vectors = load_from_text_file(f)
    color = algorithm_colors[algorithm]
    label = algorithm_labels[algorithm]
    add_sampled_elections_to_experiment(
        new_approvalwise_vectors, experiment, label, color=color, seed=0)
experiment.compute_distances(distance_id='l1-approvalwise')
experiment.embed_2d(embedding_id="fr")
map_filepath = os.path.join(
    '..', plot_dir, f'space-filling-{i_start}-map.png')
plt.rcParams.update({'font.size': 8})
experiment.print_map_2d(legend=True, saveas=map_filepath, show=False)
