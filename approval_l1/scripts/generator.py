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
    parser.add_argument('--verbose', type=int,
                        help='Verbose level', default=1)
    parser.add_argument('--algorithm', type=str, help='Algorithm to use',
                        default='bh', choices=['bh', 'gurobi'])
    parser.add_argument('--json_output', type=bool,
                        help='Output as JSON', default=False)
    parser.add_argument('--load_hists', type=str,
                        help='Load votings from file')

    args = parser.parse_args()

    N = args.N
    M = args.M
    R = args.R
    votings_hists = [np.array([0] * M), np.array([N] * M)]
    R_Init = len(votings_hists)

    if args.verbose >= 1:
        print('r,dist,dist_prop,time')
    if args.json_output:
        print('{"hists":[')
        for voting in votings_hists:
            print(f'\t{list(voting)},')
    max_expected_dist = N * M // 2
    for i in range(R_Init + 1, R + R_Init + 1):
        start = time.time()
        if args.algorithm == 'bh':
            x, dist = bh.basin_hopping(
                votings_hists, N, niter=N*M*2, seed=2137)
        elif args.algorithm == 'gurobi':
            x, dist = gurobi.gurobi_ilp(votings_hists, N)
        dt = time.time() - start

        if args.verbose >= 1:
            print(f'{i},{dist},{dist/(N*M):.4f},{dt:.4f}')
        if args.json_output:
            print(f'\t{x}', end=',\n' if i < R + R_Init else '\n')

        max_expected_dist = dist
        votings_hists.append(x)
    if args.json_output:
        print(']}')
