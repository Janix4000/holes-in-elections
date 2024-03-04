import time
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
                                       num_generated: int, num_trials: int = 1, verbose: bool = False):

    approvalwise_vectors = initial_approvalwise_vectors[:]

    report = []

    for i_generated in range(num_generated):
        new_reference_approvalwise_vector, reference_distance, dt = measure_iteration(
            approvalwise_vectors, reference_algorithm)
        row = (i_generated, reference_distance,
               reference_distance, None, None, dt, 'reference')
        report.append(row)
        if verbose:
            print(row)

        new_approvalwise_vectors = None

        for i_trial in range(num_trials):
            i_heuristic = 0
            trial_approvalwise_vectors = approvalwise_vectors[:]
            while True:
                new_heuristic_approvalwise_vector, heuristic_distance, dt = measure_iteration(
                    trial_approvalwise_vectors, heuristic_algorithm)
                trial_approvalwise_vectors.append(
                    new_heuristic_approvalwise_vector)
                row = (i_generated, heuristic_distance, reference_distance,
                       i_heuristic, i_trial, dt, 'heuristic')
                report.append(row)
                if verbose:
                    print(row)
                i_heuristic += 1
                if __distance_across(trial_approvalwise_vectors, new_reference_approvalwise_vector) < reference_distance:
                    break
            new_approvalwise_vectors = trial_approvalwise_vectors
        approvalwise_vectors = new_approvalwise_vectors

    report = pd.DataFrame(report, columns=[
                          'i_generated', 'distance', 'reference_distance', 'i_heuristic', 'i_trial', 'dt', 'algorithm'])

    return report
