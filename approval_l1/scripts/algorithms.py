

from typing import Callable

from scripts.approvalwise_vector import ApprovalwiseVector
from scripts.basin_hopping import basin_hopping
from scripts.gurobi import gurobi_ilp


Algorithm = Callable[[list[ApprovalwiseVector]],
                     tuple[ApprovalwiseVector, int]]


def _basin_hopping_step(approvalwise_vectors, **kwargs):
    return basin_hopping(
        approvalwise_vectors=approvalwise_vectors,
        step_size=7,
        big_step_chance=0.2,
        x0='mix',
        **kwargs
    )


def _basin_hopping_random(approvalwise_vectors, **kwargs):
    return basin_hopping(
        approvalwise_vectors=approvalwise_vectors,
        step_size=7,
        big_step_chance=0.2,
        x0='random',
        **kwargs
    )


try:
    from scripts.bindings import greedy_dp, pairs

    algorithms = {
        'basin_hopping': _basin_hopping_step,
        'basin_hopping_random': _basin_hopping_random,
        'gurobi': gurobi_ilp,
        'greedy_dp': greedy_dp,
        'pairs': pairs
    }
except Exception as e:
    algorithms = {
        'basin_hopping': _basin_hopping_step,
        'basin_hopping_random': _basin_hopping_random,
        'gurobi': gurobi_ilp,
    }
