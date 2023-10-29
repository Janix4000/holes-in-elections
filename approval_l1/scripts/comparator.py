import json
import gurobi
import basin_hopping as bh
import time
import numpy as np

import argparse

VotingHist = np.ndarray


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Yield distances for next R farthest votings.')

    parser.add_argument('N', type=int, help='Number of voters')
    parser.add_argument('M', type=int, help='Number of candidates')
    parser.add_argument('R', type=int, help='Number of new votings to yield')
    parser.add_argument('--algorithm', type=str, help='Algorithm to use',
                        default='bh', choices=['bh', 'gurobi'])
    parser.add_argument('--json_input', type=str,
                        help='Load votings from file')

    args = parser.parse_args()

    N = args.N
    M = args.M
    R = args.R
    with open(args.json_input) as f:
        votings_hists_loaded = json.loads(f.read())["hists"]
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
