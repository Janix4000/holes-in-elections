

from typing import Callable

from scripts.approvalwise_vector import ApprovalwiseVector
from scripts.basin_hopping import basin_hopping
from scripts.gurobi import gurobi_ilp


Algorithm = Callable[[list[ApprovalwiseVector]],
                     tuple[ApprovalwiseVector, int]]

def _basin_hopping_step(approvalwise_vectors, seed = None):
    return basin_hopping(
        approvalwise_vectors=approvalwise_vectors,
        step_size=7,
        seed=seed,
        big_step_chance=0.2,
        x0='step_vector'
    )

def _basin_hopping_random(approvalwise_vectors, seed = None):
    return basin_hopping(
        approvalwise_vectors=approvalwise_vectors,
        step_size=7,
        seed=seed,
        big_step_chance=0.2,
        x0='random'
    )

algorithms = {
    'basin_hopping': _basin_hopping_step,
    'basin_hopping_random': _basin_hopping_random,
    'gurobi': gurobi_ilp
}
