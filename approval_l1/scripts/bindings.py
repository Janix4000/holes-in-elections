import numpy as np
import ctypes

from scripts.approvalwise_vector import ApprovalwiseVector

import os
cwd = os.getcwd()
print(cwd)

# Load the shared library
my_functions = ctypes.CDLL('./build/libmy_bindings.dylib')

# Provide the necessary information about the function to call
my_functions.greedy_dp_binding.restype = ctypes.c_int
my_functions.greedy_dp_binding.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32), ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.int32)]

my_functions.pairs_binding.restype = ctypes.c_int
my_functions.pairs_binding.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int32), ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.int32)]


def __create_binding(approvalwise_vectors: list[ApprovalwiseVector], binding) -> tuple[ApprovalwiseVector, int]:
    # Create the output arrays
    num_instances = len(approvalwise_vectors)
    num_candidates = approvalwise_vectors[0].num_candidates
    num_voters = approvalwise_vectors[0].num_voters

    data = np.ascontiguousarray(
        np.stack([av.data for av in approvalwise_vectors], axis=0).flatten(), dtype=np.int32)

    output = np.zeros((num_candidates,), dtype=np.int32)

    # Call the function
    distance = binding(
        data, num_voters, num_candidates, num_instances, output)
    new_approvalwise_vector = ApprovalwiseVector(output, num_voters)
    return new_approvalwise_vector, distance


def greedy_dp(approvalwise_vectors: list[ApprovalwiseVector]) -> tuple[ApprovalwiseVector, int]:
    return __create_binding(approvalwise_vectors, my_functions.greedy_dp_binding)


def pairs(approvalwise_vectors: list[ApprovalwiseVector]) -> tuple[ApprovalwiseVector, int]:
    return __create_binding(approvalwise_vectors, my_functions.pairs_binding)
