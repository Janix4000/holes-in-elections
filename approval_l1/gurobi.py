import argparse

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise ImportError(
        "Failed to import Gurobi, please make sure that you have Gurobi installed"
    )

VotingHist = list[int]


def gurobi_ilp(votings_hists: list[VotingHist], N: int) -> VotingHist:
    M = len(votings_hists[0])
    R = len(votings_hists)

    env = gp.Env(empty=True)
    # env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(env=env)


    ys = model.addVars(M, vtype=GRB.INTEGER, name="ys")
    model.addConstr(N >= ys[0], "N>=ys[0]")
    model.addConstrs((ys[i] >= ys[i + 1] for i in range(M - 1)), "ys_monotonicity")
    model.addConstr(ys[M - 1] == 0, "ys[M-1]==0")
    
    vh = model.addVars(R, M, vtype=GRB.INTEGER, name="vh")
    model.addConstrs((vh[r, i] == votings_hists[r][i] for i in range(M) for r in range(R)), "vh")
    
    diffs = model.addVars(R, M, vtype=GRB.INTEGER, name="diffs")
    model.addConstrs((diffs[r, i] == ys[i] - vh[r, i] for i in range(M) for r in range(R)), "diffs")

    diffs_abs = model.addVars(R, M, vtype=GRB.INTEGER, name="diffs_abs")
    model.addConstrs((diffs_abs[r, i] == gp.abs_(diffs[r, i])
                    for r in range(R) for i in range(M)), "diffs_abs")

    dists = model.addVars(R, vtype=GRB.INTEGER, name="dists")
    model.addConstrs((dists[r] == gp.quicksum(diffs_abs[r, i]
                    for i in range(M)) for r in range(R)), "dists")

    min_constr = model.addVar(vtype=GRB.INTEGER, name="min_constr")
    model.addGenConstrMin(min_constr, dists, name="min_constr")

    model.setObjective(min_constr, GRB.MAXIMIZE)
    

    model.optimize()
    
    votings_hist = [int(ys[i].X) for i in range(M)]
    dist = int(min_constr.X)
    
    for v in model.getVars():
        print('%s %g' % (v.VarName, v.X))

    print('Obj: %g' % model.ObjVal)

    return votings_hist, dist
    # except gp.GurobiError as e:
    #     print('Error code ' + str(e.errno) + ': ' + str(e))

    # except AttributeError:
    #     print('Encountered an attribute error')


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
    votings_hists = [[0] * M, [N] * M]
    for i in range(3, R + 3):
        x, score = gurobi_ilp(votings_hists, N)
        print(f'{i},{score},{score/(N*M):.4f}')
        votings_hists.append(x)
        