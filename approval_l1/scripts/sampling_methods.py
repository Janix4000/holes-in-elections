import mapel.elections as mapel
import numpy as np
from scripts.approvalwise_vector import ApprovalwiseVector, get_approvalwise_vector
from scripts.distances import l1_across


def find_best_starting_step_vector(approvalwise_vectors: list[ApprovalwiseVector],
                                   first_candidates: list[ApprovalwiseVector] | None = None) -> ApprovalwiseVector:
    first_candidates = first_candidates if first_candidates is not None else []
    num_candidates = approvalwise_vectors[0].num_candidates
    num_voters = approvalwise_vectors[0].num_voters
    candidates = np.ones((num_candidates + 1, num_candidates)) * num_voters
    for i in range(num_candidates):
        candidates[i, i:] = 0
    candidates = list(candidates) + list(first_candidates)
    return find_best_vector(approvalwise_vectors, candidates)


def find_best_vector(approvalwise_vectors: list[ApprovalwiseVector], candidates: list[ApprovalwiseVector]) -> ApprovalwiseVector:
    distances = np.array(
        [l1_across(x, np.array(approvalwise_vectors)) for x in candidates])
    return candidates[distances.argmax()]


def sample_approvalwise_vector_with_resampling(num_voters: int, num_candidates: int, rng):
    p, phi = rng.random(2)
    election = mapel.generate_approval_election(
        num_candidates=num_candidates,
        num_voters=num_voters,
        culture_id='resampling',
        params={'p': p, 'phi': phi}
    )
    return get_approvalwise_vector(election)


def random_approvalwise_vectors(num_voters: int, num_candidates: int, rng=None, tries=10):
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    x0_vector = rng.integers(
        0, num_voters + 1, size=(tries, num_candidates))
    x0_vector[:, ::-1].sort(axis=1)
    return x0_vector
