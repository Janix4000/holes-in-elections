import time
from typing import Optional
import numpy as np
from scipy.optimize import basinhopping
from scipy.spatial import distance

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


def basing_hopping(votings_hists: list[VotingHist], N: int, niter: int = 1000, seed: Optional[int] = None) -> VotingHist:
    rng = np.random.default_rng(seed)
    M = len(votings_hists[0])
    x0 = np.sort(rng.integers(0, N, endpoint=True, size=M+2))
    x0[-1] = 0
    x0[M] = N
    f_time = 0
    start_time = time.time()

    def f(x):
        nonlocal votings_hists, f_time
        start = time.time()
        d = -distance_across(votings_hists, to_int(x))
        f_time += time.time() - start
        return d

    def step_function(x):
        nonlocal rng
        x = to_int(x)
        while True:
            idx = rng.integers(0, M)
            dx = rng.choice([-1, 1])
            if x[idx - 1] <= x[idx] + dx <= x[idx + 1]:
                x[idx] += dx
                break
        return x

    res = basinhopping(f, x0, stepsize=1, niter=niter, take_step=step_function)
    x = np.round_(res.x[:-2]).astype(np.int32)
    print(f'{f_time:.2f} sec, {time.time() - start_time:.2f} sec')
    return x, -int(res.fun)


if __name__ == '__main__':
    N = 20
    M = 20
    R = 60
    votings_hists = [np.array([0] * M), np.array([N] * M)]
    for _ in range(R):
        x, score = basing_hopping(votings_hists, N, niter=N*M*10)
        print(f'{score}: {x}')
        votings_hists.append(x)
