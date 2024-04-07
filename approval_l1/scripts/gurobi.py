import argparse
import time
from typing import Optional

import numpy as np

from scripts.sampling_methods import find_best_starting_step_vector
from scripts.approvalwise_vector import ApprovalwiseVector

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise ImportError(
        "Failed to import Gurobi, please make sure that you have Gurobi installed"
    )


def gurobi_ilp(approvalwise_vectors: list[ApprovalwiseVector], max_dist: int = None, x0: Optional[ApprovalwiseVector] = None, seed: Optional[int] = None) -> tuple[ApprovalwiseVector, int]:
    """# Summary
    Generates farthest approvalwise vector from the given approvalwise vectors.

    Number of gurobi variables: `num_candidates` + 2 * `num_elections` * `num_candidates` + `num_elections`

    Number of gurobi constraints: `num_candidates` + 2 * `num_elections` * `num_candidates` + `num_elections` + 2

    ## Args:
        `approvalwise_vectors` (list[ApprovalwiseVector]): List of approvalwise vectors/elections, where each
        approvalwise vector is a list of non decreasing integers in range [0, `num_voters`].

        `max_dist` (int, optional): Maximum possible distance that can be obtained. Usually set to the previous farthest
        distance. Defaults to `None`.

        `x0` (Optional[ApprovalwiseVector], optional): Initial point for Gurobi algorithm. Defaults to `None`.

    ## Returns:
        -> tuple[ApprovalwiseVector, int]: Farthest approvalwise vector and its distance from given approvalwise
        vectors.

    ## Examples
    """
    num_candidates = len(approvalwise_vectors[0])
    num_elections = len(approvalwise_vectors)
    num_voters = approvalwise_vectors[0].num_voters
    max_dist = max_dist or num_voters * num_candidates

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(env=env)

    # farthest approvalwise vector
    fav = model.addVars(num_candidates, vtype=GRB.INTEGER,
                        name="fav", lb=0, ub=num_voters)
    model.addConstr(num_voters >= fav[0], "N>=fav[0]")
    model.addConstrs((fav[i] >= fav[i + 1]
                     for i in range(num_candidates - 1)), "fav_monotonicity")
    model.addConstr(fav[num_candidates - 1] >= 0, "fav[M-1]>=0")

    diffs = model.addVars(num_elections, num_candidates,
                          vtype=GRB.INTEGER, name="diffs", lb=-num_voters, ub=num_voters)
    model.addConstrs((diffs[r, i] == fav[i] - approvalwise_vectors[r][i]
                     for i in range(num_candidates) for r in range(num_elections)), "diffs")

    diffs_abs = model.addVars(num_elections, num_candidates, vtype=GRB.INTEGER,
                              name="diffs_abs", lb=0, ub=num_voters)
    model.addConstrs((diffs_abs[r, i] == gp.abs_(diffs[r, i])
                      for r in range(num_elections) for i in range(num_candidates)), "diffs_abs")

    dists = model.addVars(num_elections, vtype=GRB.INTEGER,
                          name="dists", lb=0, ub=num_voters*num_candidates)
    model.addConstrs((dists[r] == gp.quicksum(diffs_abs[r, i]
                                              for i in range(num_candidates)) for r in range(num_elections)), "dists")

    min_constr = model.addVar(
        vtype=GRB.INTEGER, name="min_constr", lb=0, ub=max_dist)
    model.addGenConstrMin(min_constr, dists, name="min_constr")

    model.setObjective(min_constr, GRB.MAXIMIZE)

    if x0 is None:
        rng = np.random.default_rng(seed)
        candidates = rng.integers(
            0, num_voters, size=(20, num_candidates))
        candidates[:, ::-1].sort(axis=1)
        x0 = find_best_starting_step_vector(
            approvalwise_vectors, candidates)
    for i in range(num_candidates):
        fav[i].Start = x0[i]
    model.optimize()

    vector = [int(fav[i].X) for i in range(num_candidates)]
    dist = int(min_constr.X)

    return ApprovalwiseVector(vector, num_voters), dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Yield distances for next R farthest votings.')

    parser.add_argument('N', type=int, help='Number of voters')
    parser.add_argument('M', type=int, help='Number of candidates')
    parser.add_argument('R', type=int, help='Number of new votings to yield')

    args = parser.parse_args()

    num_voters = args.N
    num_candidates = args.M
    num_voters = args.R
    votings_hists = [
        ApprovalwiseVector([0] * num_candidates, num_voters),
        ApprovalwiseVector([num_voters] * num_candidates, num_voters)]
    max_expected_dist = num_voters * num_candidates // 2
    print('r,dist,dist_prop,time')
    for i in range(3, num_voters + 3):
        start = time.time()
        x, dist = gurobi_ilp(votings_hists,
                             max_dist=max_expected_dist)
        dt = time.time() - start
        print(f'{i},{dist},{dist/(num_voters*num_candidates):.4f},{dt:.4f}')
        print(x)
        votings_hists.append(x)
        max_expected_dist = dist
