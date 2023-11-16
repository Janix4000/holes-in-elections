import os
import pickle
import matplotlib.pyplot as plt
import mapel.elections as mapel
from scripts.gurobi import gurobi_ilp
import scripts.experiments as experiments

experiments_results = {}
reports = {}
new_approval_vectors_per_experiment = {}


generator = gurobi_ilp
num_generated = 12
num_candidates = 50
num_voters = 100


def run_experiment(experiment_id: str):
    experiment_id = f'{num_candidates}x{num_voters}/{experiment_id}'

    experiment = mapel.prepare_offline_approval_experiment(
        experiment_id=experiment_id,
        distance_id="l1-approvalwise",
        embedding_id="fr"
    )
    experiment.prepare_elections()
    experiment.print_map_2d()

    family_dirpath = os.path.join('results', 'gurobi', experiment_id)
    if not os.path.exists(family_dirpath):
        os.makedirs(family_dirpath)

    _new_approvalwise_vectors, report = experiments.generate_farthest_elections_l1_approvalwise(
        experiment, num_generated, generator, experiment_id, save_snapshots=family_dirpath)

    with open(os.path.join(family_dirpath, 'report.pkl'), 'wb') as f:
        pickle.dump(report, f)


run_experiment('euclidean')
run_experiment('truncated_urn')
run_experiment('noise')
