import time
import typing
import numpy as np
import pandas as pd
from scripts.approvalwise_vector import ApprovalwiseVector
from scripts.algorithms import Algorithm


def __distance_across(approvalwise_vectors: np.ndarray, x: ApprovalwiseVector) -> int:
    return int(np.sum(np.abs(approvalwise_vectors - x), axis=1).min())


def __distance_across_from_set(approvalwise_vectors: np.ndarray, other_approvalwise_vectors: np.ndarray) -> int:
    return max(__distance_across(approvalwise_vectors, x) for x in other_approvalwise_vectors)


def measure_iteration(approvalwise_vectors: list[ApprovalwiseVector], algorithm: Algorithm, **kwargs):
    start_time = time.process_time()
    farthest_approvalwise_vectors, distance = algorithm(
        approvalwise_vectors, **kwargs)
    execution_time_s = time.process_time() - start_time

    return farthest_approvalwise_vectors, distance, execution_time_s


def how_long_reference_is_better(initial_approvalwise_vectors: list[ApprovalwiseVector], reference_farthest_approvalwise_vectors: list[ApprovalwiseVector],
                                 reference_algorithm: Algorithm, heuristic_algorithm: Algorithm, csv_report_out: typing.TextIO, i_trial: int, max_generated: int = 1000, **heuristic_kwargs):

    approvalwise_vectors = initial_approvalwise_vectors[:]
    new_heuristic_approvalwise_vectors = []
    new_reference_approvalwise_vectors = []

    all_initial_approvalwise_vectors = initial_approvalwise_vectors + \
        reference_farthest_approvalwise_vectors
    reference_distance = __distance_across(
        all_initial_approvalwise_vectors[:-1], all_initial_approvalwise_vectors[-1])

    for i_generated in range(max_generated):
        heuristic_approvalwise_vector, heuristic_distance, heuristic_dt = measure_iteration(
            approvalwise_vectors, heuristic_algorithm, **heuristic_kwargs)
        approvalwise_vectors.append(heuristic_approvalwise_vector)
        new_heuristic_approvalwise_vectors.append(
            heuristic_approvalwise_vector)

        csv_report_out.write(
            f'{i_trial},{i_generated},{heuristic_distance},{heuristic_dt},heuristic\n')

        yield new_reference_approvalwise_vectors, new_heuristic_approvalwise_vectors

        # if heuristic_distance >= reference_distance:
        #     continue

        if __distance_across_from_set(approvalwise_vectors, reference_farthest_approvalwise_vectors) >= reference_distance:
            continue

        new_reference_approvalwise_vector, new_reference_distance, new_reference_dt = measure_iteration(
            approvalwise_vectors, reference_algorithm
        )
        new_reference_approvalwise_vectors.append(
            new_reference_approvalwise_vector)
        csv_report_out.write(
            f'{i_trial},{i_generated},{new_reference_distance},{new_reference_dt},reference\n')
        yield new_reference_approvalwise_vectors, new_heuristic_approvalwise_vectors

        if new_reference_distance <= reference_distance:
            break
