import argparse
import pandas as pd
import matplotlib.pyplot as plt
import mapel.elections as mapel
import scripts.experiments as experiments
import os
import numpy as np
import pickle
from scripts.approvalwise_vector import add_sampled_elections_to_experiment, get_approvalwise_vectors, load_from_text_file, sample_election_from_approvalwise_vector
from matplotlib import ticker

plt.rcParams['figure.dpi'] = 300


def calculate_cumulative_proportional_vector(reference_distances: list[int], distancess: list[list[int]]) -> np.ndarray:
    distances_sums = [np.sum(distances) for distances in distancess]
    reference_distances = np.array(reference_distances)
    reference_distances_sums = np.cumsum(reference_distances[::-1])[::-1]
    return distances_sums / reference_distances_sums


def split_reports_if_necessary(reports) -> list[pd.DataFrame]:
    if isinstance(reports, list):
        return reports
    all_num_starting_elections = reports.num_starting_elections.unique()
    return [
        reports[reports.num_starting_elections == num_starting_elections] for num_starting_elections in all_num_starting_elections
    ]


def plot_trajectories(reference_report, reports):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(reference_report.experiment_size,
               reference_report.distance, label="Gurobi")
    ax.plot(reference_report.experiment_size, reference_report.distance, '--')

    for idx, report in enumerate(reports):
        ax.scatter(report.experiment_size, report.distance,
                   label=f'From {idx} gurobi elections')
        ax.plot(report.experiment_size, report.distance, '--')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig, ax


def get_metric(reference_report, reports):
    return calculate_cumulative_proportional_vector(reference_report.distance, [report.distance for report in reports])


def plot_metric(metric):
    fig, ax = plt.subplots()

    ax.bar(np.arange(len(metric)), metric)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    ax.set_ylim(0, 1)
    return fig, ax


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

reference_report = pd.read_csv(os.path.join(
    results_dir, reference_algorithm, 'report.csv'))

id_scheme = f'NFE-{experiment_id}-%d'
with open(os.path.join('experiments', experiment_id, 'elections.pkl'), 'rb') as file:
    meaningful_elections = pickle.load(file)
approvalwise_vectors = get_approvalwise_vectors(meaningful_elections)


def prepare_experiment() -> mapel.ApprovalElectionExperiment:
    experiment = mapel.prepare_online_approval_experiment(
        distance_id="l1-approvalwise",
        embedding_id="fr"
    )

    experiment.set_default_num_candidates(num_candidates)
    experiment.set_default_num_voters(num_voters)

    return experiment


def plot_maps(algorithm: str, algorithm_plots_dir, save: bool):
    experiment = prepare_experiment()
    add_sampled_elections_to_experiment(
        approvalwise_vectors, experiment, family_id, color='green', seed=0)
    if os.path.exists(os.path.join(results_dir, algorithm, 'new-approvalwise-vectors.pkl')):
        with open(os.path.join(results_dir, algorithm, 'new-approvalwise-vectors.pkl'), 'rb') as f:
            new_approvalwise_vectors = pickle.load(f)
    else:
        with open(os.path.join(results_dir, algorithm, 'new_approvalwise_vectors_0.txt'), 'r') as f:
            new_approvalwise_vectors = load_from_text_file(f)
    add_sampled_elections_to_experiment(
        new_approvalwise_vectors, experiment, algorithm, color='blue', seed=0)
    experiments.add_compass(experiment, with_grid=True)

    experiment.compute_distances(distance_id='l1-approvalwise')
    experiment.embed_2d(embedding_id="fr")
    experiment.print_map_2d(legend=True, show=not save,
                            saveas=os.path.join('..', algorithm_plots_dir,  'map.png') if save else None)


def plot_algorithm(algorithm: str):
    report = pd.read_csv(os.path.join(results_dir, algorithm, 'report.csv'))
    reports = split_reports_if_necessary(report)
    algorithm_plots_dir = os.path.join(plot_dir, algorithm)
    os.makedirs(algorithm_plots_dir, exist_ok=True)

    fig, ax = plot_trajectories(reference_report, reports)
    ax.set_xlabel('Number of elections')
    ax.set_ylabel('New farthest distance')
    ax.set_title(f'Farthest elections for {algorithm}')
    ax.set_ylim(plot_lower_bound, None)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(algorithm_plots_dir, 'trajectories.png'))
        print(
            f"Saved trajectories plot for {algorithm} to {algorithm_plots_dir}")
    else:
        plt.show()

    metric = get_metric(reference_report, reports)
    fig, ax = plot_metric(metric)
    ax.set_xlabel('Number of starting elections from the reference')
    ax.set_ylabel('Percentage of summed distances to the reference')
    ax.set_title(f'Farthest elections for {algorithm}')
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(algorithm_plots_dir, 'metric.png'))
        print(f"Saved metric plot for {algorithm} to {algorithm_plots_dir}")
    else:
        plt.show()

    plot_maps(algorithm, algorithm_plots_dir, save)


algorithm_plots_dir = os.path.join(plot_dir, 'gurobi')
os.makedirs(algorithm_plots_dir, exist_ok=True)
plot_maps('gurobi', algorithm_plots_dir, save)

plot_algorithm('basin_hopping')
plot_algorithm('basin_hopping_random')
plot_algorithm('pairs')
plot_algorithm('greedy_dp')
