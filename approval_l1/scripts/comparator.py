import json
import gurobi
import app_format
import basin_hopping as bh
import time
import numpy as np
import os

import argparse

VotingHist = np.ndarray


def load_mapel(source: str) -> list[list[int]]:
    if source.endswith(".app"):
        files_paths = [source]
    else:
        files_paths = list(os.listdir(source))
    elections_histograms = []
    for path in files_paths:
        if not path.endswith(".app"):
            continue
        election, _N = app_format.load(os.path.join(source, path))
        elections_histograms.append(app_format.to_histograms(election, N))

    return elections_histograms


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Yield distances for next R farthest votings.')

    parser.add_argument('N', type=int, help='Number of voters')
    parser.add_argument('M', type=int, help='Number of candidates')
    parser.add_argument('R', type=int, help='Number of new votings to yield')
    parser.add_argument('--algorithm', type=str, help='Algorithm to use',
                        default='bh', choices=['bh', 'gurobi'])
    parser.add_argument('--json_input', type=str,
                        help='Load votings from the json file')
    parser.add_argument('--mapel_input', type=str,
                        help='Load votings from the mapel file')

    args = parser.parse_args()

    N = args.N
    M = args.M
    R = args.R

    if args.mapel_input and args.json_input:
        raise ValueError('Only one input file can be specified')

    if args.json_input:
        with open(args.json_input) as f:
            votings_hists_loaded = json.loads(f.read())["hists"]
    elif args.mapel_input:
        votings_hists_loaded = app_format.load_mapel(args.mapel_input)

    R = len(votings_hists_loaded)

    print('r,dist,dist_prop,time,from')
    for R_Init in range(2, len(votings_hists_loaded)):
        votings_hists = votings_hists_loaded[:R_Init]
        max_expected_dist = N * M // 2
        for i in range(R_Init + 1, R + 1):
            start = time.time()
            if args.algorithm == 'bh':
                x, dist = bh.basing_hopping(
                    votings_hists, N, niter=N*M*2, seed=2137)
            elif args.algorithm == 'gurobi':
                x, dist = gurobi.gurobi_ilp(
                    votings_hists, N, max_dist=max_expected_dist)
            dt = time.time() - start

            print(f'{i},{dist},{dist/(N*M):.4f},{dt:.4f},{R_Init}')

            max_expected_dist = dist
            votings_hists.append(x)
