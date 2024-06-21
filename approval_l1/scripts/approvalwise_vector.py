from typing import Iterable

import mapel.elections as mapel
import numpy as np


class ApprovalwiseVector(np.ndarray):
    def __new__(cls, approvalwise_vector: np.ndarray | Iterable[int], num_voters: int):
        obj = np.asarray(approvalwise_vector).view(cls)
        if any(obj[1:] > obj[:-1]):
            raise ValueError(
                f'Approvalwise vector must be sorted in non increasing order {obj}')
        obj.num_voters = num_voters
        return obj

    @property
    def num_candidates(self):
        return len(self)


def get_approvalwise_vector(election: mapel.ApprovalElection) -> ApprovalwiseVector:
    vector = np.zeros(election.num_candidates, dtype=int)
    for vote in election.votes:
        vector[list(vote)] += 1
    vector[::-1].sort()
    return ApprovalwiseVector(vector, election.num_voters)


def get_approvalwise_vectors(elections) -> list[ApprovalwiseVector]:
    return [get_approvalwise_vector(election) for election in elections.values()]


def sample_election_from_approvalwise_vector(approvalwise_vector: ApprovalwiseVector, seed: int | None = None) -> mapel.ApprovalElection:
    votes: list[set[int]] = [set()
                             for _ in range(approvalwise_vector.num_voters)]
    rng = np.random.default_rng(seed)
    for candidate_idx, candidate_score in enumerate(approvalwise_vector):
        candidates_votes_indices = rng.choice(
            approvalwise_vector.num_voters, size=int(candidate_score), replace=False)
        for voter_idx in candidates_votes_indices:
            votes[voter_idx].add(candidate_idx)

    election = mapel.generate_approval_election_from_votes(votes)
    election.num_candidates = approvalwise_vector.num_candidates
    return election


def add_sampled_elections_to_experiment(approvalwise_vectors: dir | Iterable[ApprovalwiseVector], experiment: mapel.ApprovalElectionExperiment, family_id: str, color: str, seed: int | None) -> None:
    experiment.add_empty_family(family_id=family_id, color=color)
    size = sum(map(len, experiment.families))
    vectors = approvalwise_vectors.items() if isinstance(
        approvalwise_vectors, dict) else enumerate(approvalwise_vectors)
    for instance_id, approvalwise_vector in vectors:
        election = sample_election_from_approvalwise_vector(
            approvalwise_vector, seed=seed)
        election.instance_id = f'{instance_id}_{size}'
        experiment.add_election_to_family(election, family_id=family_id)


def dump_to_text_file(approvalwise_vectors: dict[str, ApprovalwiseVector] | list[ApprovalwiseVector], file) -> None:
    sample_election_from_approvalwise_vector = next(
        iter(approvalwise_vectors.values() if isinstance(approvalwise_vectors, dict) else approvalwise_vectors))
    num_candidates = sample_election_from_approvalwise_vector.num_candidates
    num_voters = sample_election_from_approvalwise_vector.num_voters
    num_elections = len(approvalwise_vectors)
    file.write(f'{num_elections} {num_voters} {num_candidates}\n')

    iterator = approvalwise_vectors.items() if isinstance(
        approvalwise_vectors, dict) else enumerate(approvalwise_vectors)

    for instance_id, approvalwise_vector in iterator:
        stringified_vector = [str(int(x)) for x in approvalwise_vector]
        file.write(f'{instance_id} {" ".join(stringified_vector)}\n')


def load_from_text_file(file) -> dict[str, ApprovalwiseVector]:
    header = file.readline().strip().split()
    num_elections = int(header[0])
    num_voters = int(header[1])
    _num_candidates = int(header[2])

    approvalwise_vectors = {}
    for _ in range(num_elections):
        line = file.readline().strip().split()
        instance_id = int(line[0])
        vector = np.array([int(x) for x in line[1:]])
        approvalwise_vectors[instance_id] = ApprovalwiseVector(
            vector, num_voters)

    return approvalwise_vectors


def _get_prob_distr_next_height(max_val: int, M: int, t: int) -> np.ndarray:
    max_val_prob = (M - t + 2) / (M - t + max_val + 2)
    multipliers = np.array([k / (M - t + k + 1)
                            for k in range(1, max_val + 1)])
    multipliers = multipliers[::-1]
    np.multiply.accumulate(multipliers, out=multipliers)
    return np.concatenate((multipliers[::-1], [1.0])) * max_val_prob


def uniform_approvalwise_vector(num_voters: int, num_candidates: int, rng=None) -> ApprovalwiseVector:
    M = num_candidates
    N = num_voters
    rng = rng or np.random.default_rng()
    approvalwise_vector = np.array([-1] * num_candidates, dtype=int)
    approvalwise_vector[-1] = N
    for t in range(1, num_candidates + 1):
        i = t - 1
        max_val = approvalwise_vector[i - 1]
        probabilities = _get_prob_distr_next_height(max_val, M, t)
        approvalwise_vector[i] = rng.choice(max_val + 1, p=probabilities)
    return ApprovalwiseVector(approvalwise_vector, num_voters)


def not_uniform_sample_approvalwise_vector(num_voters: int, num_candidates: int, seed: int | None = None) -> ApprovalwiseVector:
    rng = np.random.default_rng(seed)
    M = num_candidates
    N = num_voters + 1
    rng = np.random.default_rng(seed)
    approvalwise_vector = np.array([None] * num_candidates)
    approvalwise_vector[-1] = N
    for t in range(1, num_candidates + 1):
        i = t - 1
        max_val = approvalwise_vector[i - 1]
        approvalwise_vector[i] = rng.integers(max_val + 1)
    return ApprovalwiseVector(approvalwise_vector, num_voters)
