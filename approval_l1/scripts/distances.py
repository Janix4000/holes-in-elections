import numpy as np
from scripts.approvalwise_vector import ApprovalwiseVector


def l1(l: ApprovalwiseVector, r: ApprovalwiseVector):
    l = np.array(l)
    r = np.array(r)

    return np.sum(np.abs(l - r))


def l1_across(av: ApprovalwiseVector, other_avs: list[ApprovalwiseVector]) -> int:
    av = np.array(av).reshape(1, -1)
    other_avs = np.array(other_avs)
    return np.min(np.sum(np.abs(av - other_avs), axis=1))
