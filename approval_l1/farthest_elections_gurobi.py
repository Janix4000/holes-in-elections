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
num_instances = 36
num_candidates = 30
num_voters = 100

parser = argparse.ArgumentParser()
parser.add_argument('--num_instances', type=int,
                    help='The number of elections to generate', default=36)
parser.add_argument('--num_candidates', type=int,
                    help='The number of candidates', default=30)
parser.add_argument('--num_voters', type=int,
                    help='The number of voters', default=100)
parser.add_argument('--family', type=str,
                    help='The family of elections to generate', default='euclidean')
parser.add_argument('--load_pickle', type=bool,
                    help='Whether to load experiment from the pickle file', default=True)

args = parser.parse_args()

num_instances = args.num_instances
num_candidates = args.num_candidates
num_voters = args.num_voters
family_id = args.family
load_pickle = args.load_pickle


def run_experiment(experiment_id: str):
    global generator, num_instances, num_candidates, num_voters, load_pickle

    experiment_id = f'{num_candidates}x{num_voters}/{experiment_id}'

    if load_pickle:
        with open(os.path.join('experiments', experiment_id, 'elections.pkl'), 'rb') as f:
            elections = pickle.load(f)
    else:
        experiment = mapel.prepare_offline_approval_experiment(
            experiment_id=experiment_id,
            distance_id="l1-approvalwise",
            embedding_id="fr"
        )
        experiment.prepare_elections()
        elections = experiment.elections

    family_dirpath = os.path.join('results', 'gurobi', experiment_id)
    if not os.path.exists(family_dirpath):
        os.makedirs(family_dirpath)

    meaningful_elections = experiments.get_meaningful_elections(elections)
    approvalwise_vectors = get_approvalwise_vectors(meaningful_elections)

    _new_approvalwise_vectors, report = experiments.generate_farthest_elections_l1_approvalwise(
        approvalwise_vectors, num_voters, num_instances, generator, experiment_id, save_snapshots=family_dirpath)

    report.to_csv(os.path.join(family_dirpath, 'report.csv'))


run_experiment(family_id)
