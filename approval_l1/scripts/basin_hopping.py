import argparse
import math
import time
from typing import Optional

import numpy as np
from scipy.optimize import basinhopping
from scripts.approvalwise_vector import ApprovalwiseVector, uniform_approvalwise_vector
from scripts.bindings import greedy_dp
from scripts.distances import l1_across
from scripts.sampling_methods import (
    find_best_starting_step_vector,
    find_best_vector,
    random_approvalwise_vectors,
    sample_approvalwise_vector_with_resampling,
)


def __to_int(x):
    return np.round_(x).astype(np.int32)


def __select_x0(x0, approvalwise_vectors, num_voters, num_candidates, rng):
    match x0:
        case (x0, num_start):
            x0, num_start = x0, num_start
        case _:
            x0, num_start = x0, 100

    if x0 == 'random':
        random_candidates = random_approvalwise_vectors(
            num_voters, num_candidates, tries=num_start, rng=rng)
        return find_best_vector(approvalwise_vectors, random_candidates)
    elif x0 == 'resampling':
        random_candidates = [sample_approvalwise_vector_with_resampling(
            num_voters, num_candidates, rng=rng) for _ in range(num_start)]
        return find_best_vector(approvalwise_vectors, random_candidates)
    elif x0 == 'uniform':
        random_candidates = [uniform_approvalwise_vector(
            num_voters, num_candidates, rng=rng) for _ in range(num_start)]
        return find_best_vector(approvalwise_vectors, random_candidates)
    elif x0 == 'step_vector':
        return find_best_starting_step_vector(approvalwise_vectors)
    elif x0 == 'mix':
        candidates = random_approvalwise_vectors(
            num_voters, num_candidates, tries=num_start, rng=rng)
        return find_best_starting_step_vector(approvalwise_vectors, candidates)
    elif x0 == 'greedy_dp':
        return greedy_dp(approvalwise_vectors)[0]
    else:
        return np.array(x0)


def basin_hopping(
    approvalwise_vectors: list[ApprovalwiseVector],
    niter: int | None = None,
    step_size: int = 1,
    seed: Optional[int] = None,
    big_step_chance: float = 0.0,
    x0: str | ApprovalwiseVector = 'random'
) -> tuple[ApprovalwiseVector, int]:
    """# Summary
    Basin hopping algorithm for finding farthest approvalwise vector.

    For approvalwise vectors, unit step is defined as a difference of +1 or -1 in one coordinate, such that monotonicity is preserved.

    Big step is defined as a difference of +1 or -1 on a prefix/suffix of the vector, such that monotonicity is preserved.

    ## Args:
        `approvalwise_vectors` (list[VotingHist]): list of approvalwise vectors/elections, where each approvalwise vector is a list of non decreasing integers in range [0, `num_voters`].
        `niter` (int, optional): Number of iterations for Basinhopping algorithm. Defaults to at least `1000`.
        `step_size` (int, optional): For every iteration algorithm can make from 1 to `step_size` unit steps (at random). Defaults to `1`.
        `seed` (Optional[int], optional): Seed of random engine. Defaults to `None`.
        `big_step_chance` (float, optional): Chance of making a big step instead of unit steps. Defaults to `0.0`.
        `x0` (str | VotingHist, optional): Initial point for Basinhopping algorithm. Defaults to `'random'` Possible: 'random', 'random_resampling'.

    ## Returns:
        -> (VotingHist, int): Farthest approvalwise vector and its distance from given approvalwise vectors.

    ## Examples
    """
    num_voters = approvalwise_vectors[0].num_voters
    num_candidates = approvalwise_vectors[0].num_candidates

    rng = np.random.default_rng(seed)
    x0_vector: np.ndarray = __select_x0(
        x0, approvalwise_vectors, num_voters, num_candidates, rng)
    x0_vector = np.concatenate([x0_vector, np.array([0, num_voters])])

    approvalwise_vectors = np.array(approvalwise_vectors)
    if niter is None:
        niter = num_candidates * num_voters / \
            (2 * 0.05 * (step_size / 2 * (1 - big_step_chance) +
             num_candidates * big_step_chance))
        niter = max(round(niter), 1000)

    def f(x):
        nonlocal approvalwise_vectors
        x = __to_int(x)
        d = -l1_across(x[:-2], approvalwise_vectors)
        return d

    def unit_step(x):
        nonlocal rng, num_candidates
        while True:
            idx = rng.integers(0, num_candidates)
            dx = rng.choice([-1, 1])
            if x[idx - 1] >= x[idx] + dx >= x[idx + 1]:
                x[idx] += dx
                break
        return x

    def big_step(x):
        nonlocal rng, num_candidates
        while True:
            idx = rng.integers(0, num_candidates)
            dx = rng.choice([-1, 1])
            if dx > 0 and x[idx - 1] >= x[idx] + dx:
                x[idx:-2] += dx
                break
            elif dx < 0 and x[idx] + dx >= x[idx + 1]:
                x[:idx + 1] += dx
                break
        return x

    def step_function(x):
        nonlocal rng
        x = __to_int(x)

        if rng.random() < big_step_chance:
            return big_step(x)

        steps = rng.integers(1, step_size + 1)
        for _ in range(steps):
            x = unit_step(x)
        return x

    res = basinhopping(f, x0_vector, niter=niter,
                       take_step=step_function, seed=seed)
    x = __to_int(res.x[:-2])
    return ApprovalwiseVector(np.array(x, dtype=int), num_voters), -int(res.fun)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Yield distances for next R farthest votings.')

    parser.add_argument('N', type=int, help='Number of voters')
    parser.add_argument('M', type=int, help='Number of candidates')
    parser.add_argument('R', type=int, help='Number of new votings to yield')
    parser.add_argument('niter', type=int, help='Number of iterations')
    parser.add_argument('big_step_chance', type=float,
                        help='Chance of big step')

    args = parser.parse_args()

    num_voters = args.N
    num_candidates = args.M
    num_elections = args.R
    niter = args.niter or num_voters*num_candidates*10
    big_step_chance = args.big_step_chance or 0.0
    approvalwise_vectors = [
        ApprovalwiseVector([0] * num_candidates, num_voters),
        ApprovalwiseVector([num_voters] * num_candidates, num_voters)
    ]

    print('r,dist,dist_prop,time')
    step_size = round(math.sqrt(num_voters))
    for i in range(3, num_elections + 3):
        start = time.time()
        x, score = basin_hopping(
            approvalwise_vectors, niter=niter, step_size=step_size, big_step_chance=big_step_chance)
        dt = time.time() - start
        print(f'{i},{score},{score/(num_voters*num_candidates):.4f},{dt:.4f}')
        approvalwise_vectors.append(x)
