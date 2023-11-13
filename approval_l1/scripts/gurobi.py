import argparse
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise ImportError(
        "Failed to import Gurobi, please make sure that you have Gurobi installed"
    )

ApprovalwiseVector = list[int]


def gurobi_ilp(approvalwise_vectors: list[ApprovalwiseVector], num_voters: int, max_dist: int = None) -> tuple[ApprovalwiseVector, int]:
    """# Summary
    Generates farthest approvalwise vector from the given approvalwise vectors.

    ## Args:
        `approvalwise_vectors` (list[ApprovalwiseVector]): List of approvalwise vectors/elections, where each approvalwise vector is a list of non decreasing integers in range [0, `num_voters`].

        `num_voters` (int): The maximum number of possible votes.

        `max_dist` (int, optional): Maximum possible distance that can be obtained. Usually set to the previous farthest distance. Defaults to `None`.

    ## Returns:
        -> tuple[ApprovalwiseVector, int]: Farthest approvalwise vector and its distance from given approvalwise vectors.

    ## Examples
    """
    num_candidates = len(approvalwise_vectors[0])
    num_elections = len(approvalwise_vectors)
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

    model.optimize()

    votings_hist = [int(fav[i].X) for i in range(num_candidates)]
    dist = int(min_constr.X)

    return votings_hist, dist


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
    votings_hists = [[0] * num_candidates, [num_voters] * num_candidates]
    max_expected_dist = num_voters * num_candidates // 2
    print('r,dist,dist_prop,time')
    for i in range(3, num_voters + 3):
        start = time.time()
        x, dist = gurobi_ilp(votings_hists, num_voters,
                             max_dist=max_expected_dist)
        dt = time.time() - start
        print(f'{i},{dist},{dist/(num_voters*num_candidates):.4f},{dt:.4f}')
        print(x)
        votings_hists.append(x)
        max_expected_dist = dist
