import os
import pickle
import matplotlib.pyplot as plt
import mapel.elections as mapel
from scripts.approvalwise_vector import get_approvalwise_vectors
from scripts.gurobi import gurobi_ilp
import argparse
import scripts.experiments as experiments

experiments_results = {}
reports = {}
new_approval_vectors_per_experiment = {}


generator = gurobi_ilp
num_generated = 36
num_candidates = 30
num_voters = 100

parser = argparse.ArgumentParser()
parser.add_argument('--num_generated', type=int,
                    help='The number of elections to generate')
parser.add_argument('--num_candidates', type=int,
                    help='The number of candidates')
parser.add_argument('--num_voters', type=int, help='The number of voters')
parser.add_argument('--family', type=int,
                    help='The family of elections to generate')
parser.add_argument('--load_pickle', type=bool,
                    help='Whether to load experiment from the pickle file')

args = parser.parse_args()

num_generated = args.num_generated or num_generated
num_candidates = args.num_candidates or num_candidates
num_voters = args.num_voters or num_voters
family_id = args.family or 'euclidean'
load_pickle = args.load_pickle or False


def run_experiment(experiment_id: str):
    global generator, num_generated, num_candidates, num_voters, load_pickle

    experiment_id = f'{num_candidates}x{num_voters}/{experiment_id}'

    experiment = mapel.prepare_offline_approval_experiment(
        experiment_id=experiment_id,
        distance_id="l1-approvalwise",
        embedding_id="fr"
    )
    if load_pickle:
        with open(os.path.join('experiments', experiment_id, 'experiment.pkl'), 'rb') as f:
            experiment = pickle.load(f)
    experiment.prepare_elections()

    family_dirpath = os.path.join('results', 'gurobi', experiment_id)
    if not os.path.exists(family_dirpath):
        os.makedirs(family_dirpath)

    meaningful_elections = experiments.get_meaningful_elections_from_experiment(
        experiment)
    approvalwise_vectors = get_approvalwise_vectors(meaningful_elections)

    _new_approvalwise_vectors, report = experiments.generate_farthest_elections_l1_approvalwise(
        approvalwise_vectors, num_voters, num_generated, generator, experiment_id, save_snapshots=family_dirpath)

    report.to_csv(os.path.join(family_dirpath, 'report.csv'))


run_experiment(family_id)
