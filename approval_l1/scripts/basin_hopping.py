import time
from typing import Optional
import numpy as np
from scipy.optimize import basinhopping

import math

import argparse

VotingHist = np.ndarray


# def dist(x, y): return distance.minkowski(x, y, 1).astype(np.int32)

def dist(x, y):
    d = np.sum(np.abs(x - y)).astype(np.int32)
    # print(f'{x=}, {y=}, {d=}')
    return d


def distance_across(votings_hists: list[VotingHist], x: VotingHist) -> int:
    return min(dist(x[:-2], v) for v in votings_hists)


def to_int(x):
    return np.round_(x).astype(np.int32)


def basing_hopping(votings_hists: list[VotingHist], N: int, niter: int = 1000, step_size: int = None, seed: Optional[int] = None) -> VotingHist:
    rng = np.random.default_rng(seed)
    M = len(votings_hists[0])
    R = len(votings_hists)
    # x0 = np.sort(rng.integers(0, N, endpoint=True, size=M+2))
    x0 = np.array([N//2] * (M + 2))
    x0[-1] = N
    x0[M] = 0
    step_size = step_size or round(math.pow(N * M * R, 1/3))

    def f(x):
        nonlocal votings_hists
        x = to_int(x)
        d = -distance_across(votings_hists, x)
        return d

    def step_function(x):
        nonlocal rng
        x = to_int(x)
        steps = rng.integers(1, step_size + 1)
        for _ in range(steps):
            while True:
                idx = rng.integers(0, M)
                dx = rng.choice([-1, 1])
                if x[idx - 1] >= x[idx] + dx >= x[idx + 1]:
                    x[idx] += dx
                    break
        return x

    res = basinhopping(f, x0, niter=niter,
                       take_step=step_function, seed=seed)
    x = np.round_(res.x[:-2]).astype(np.int32)
    # print(f'{f_time:.2f} sec, {time.time() - start_time:.2f} sec')
    return list(x), -int(res.fun)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Yield distances for next R farthest votings.')

    parser.add_argument('N', type=int, help='Number of voters')
    parser.add_argument('M', type=int, help='Number of candidates')
    parser.add_argument('R', type=int, help='Number of new votings to yield')

    args = parser.parse_args()

    N = args.N
    M = args.M
    R = args.R
    votings_hists = [np.array([0] * M), np.array([N] * M)]

    print('r,dist,dist_prop,time')
    step_size = round(math.sqrt(N))
    for i in range(3, R + 3):
        start = time.time()
        x, score = basing_hopping(
            votings_hists, N, niter=N*M*10, step_size=step_size)
        dt = time.time() - start
        print(f'{i},{score},{score/(N*M):.4f},{dt:.4f}')
        votings_hists.append(x)
