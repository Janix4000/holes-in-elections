import time
import typing
import numpy as np
import pandas as pd
from scripts.approvalwise_vector import ApprovalwiseVector
from scripts.algorithms import Algorithm


def __distance_across(approvalwise_vectors: np.ndarray, x: ApprovalwiseVector) -> int:
    return np.sum(np.abs(approvalwise_vectors - x), axis=1).min()


def measure_iteration(approvalwise_vectors: list[ApprovalwiseVector], algorithm: Algorithm, **kwargs):
    start_time = time.process_time()
    farthest_approvalwise_vectors, distance = algorithm(
        approvalwise_vectors, **kwargs)
    execution_time_s = time.process_time() - start_time

    return farthest_approvalwise_vectors, distance, execution_time_s


def heuristic_to_change_next_reference(initial_approvalwise_vectors: list[ApprovalwiseVector],
                                       reference_algorithm: Algorithm, heuristic_algorithm: Algorithm,
                                       num_generated: int, csv_report_out: typing.TextIO, num_trials: int = 1):

    approvalwise_vectors = initial_approvalwise_vectors[:]
    new_reference_approvalwise_vectors = []
    new_heuristic_approvalwise_vectors = []

    csv_report_columns = [
        'i_generated', 'distance', 'reference_distance', 'i_heuristic', 'i_trial', 'dt', 'algorithm']
    csv_report_out.write(','.join(csv_report_columns) + '\n')

    for i_generated in range(num_generated):
        new_reference_approvalwise_vector, reference_distance, dt = measure_iteration(
            approvalwise_vectors, reference_algorithm)

        new_reference_approvalwise_vectors.append(
            new_reference_approvalwise_vector)
        yield new_reference_approvalwise_vectors, new_heuristic_approvalwise_vectors

        report_row = (i_generated, reference_distance,
                      reference_distance, -1, -1, dt, 'reference')
        csv_report_out.write(','.join(map(str, report_row)) + '\n')

        new_approvalwise_vectors = None

        for i_trial in range(num_trials):
            i_heuristic = 0
            trial_approvalwise_vectors = approvalwise_vectors[:]
            while True:
                new_heuristic_approvalwise_vector, heuristic_distance, dt = measure_iteration(
                    trial_approvalwise_vectors, heuristic_algorithm)

                if i_trial == num_trials - 1:
                    new_heuristic_approvalwise_vectors.append(
                        new_heuristic_approvalwise_vector)
                    yield new_reference_approvalwise_vectors, new_heuristic_approvalwise_vectors

                trial_approvalwise_vectors.append(
                    new_heuristic_approvalwise_vector)

                report_row = (i_generated, heuristic_distance, reference_distance,
                              i_heuristic, i_trial, dt, 'heuristic')
                csv_report_out.write(','.join(map(str, report_row)) + '\n')

                i_heuristic += 1
                if __distance_across(trial_approvalwise_vectors, new_reference_approvalwise_vector) < reference_distance:
                    break
            new_approvalwise_vectors = trial_approvalwise_vectors
        approvalwise_vectors = new_approvalwise_vectors
