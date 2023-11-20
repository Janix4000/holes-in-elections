from typing import Any, Iterable
import mapel.elections as mapel
import numpy as np


def get_approvalwise_vector(election: mapel.ApprovalElection) -> np.ndarray:
    vector = np.zeros(election.num_candidates)
    for vote in election.votes:
        vector[list(vote)] += 1
    vector[::-1].sort()
    return vector


def get_approvalwise_vectors(elections) -> list[np.ndarray]:
    return [get_approvalwise_vector(election) for election in elections.values()]


def sample_election_from_approvalwise_vector(approvalwise_vector: np.ndarray, num_voters: int, seed: int | None = None) -> mapel.ApprovalElection:
    num_candidates = len(approvalwise_vector)
    votes = [set() for _ in range(num_voters)]
    rng = np.random.default_rng(seed)
    for candidate_idx, candidate_score in enumerate(approvalwise_vector):
        candidates_votes_indices = rng.choice(
            num_voters, size=int(candidate_score), replace=False)
        for voter_idx in candidates_votes_indices:
            votes[voter_idx].add(candidate_idx)

    election = mapel.generate_approval_election_from_votes(votes)
    election.num_candidates = num_candidates
    return election


def add_sampled_elections_to_experiment(approvalwise_vectors, experiment: mapel.ApprovalElectionExperiment, family_id: str, num_voters: int, color: str, seed: int | None) -> None:
    experiment.add_empty_family(family_id=family_id, color=color)
    vectors = approvalwise_vectors.items() if isinstance(
        approvalwise_vectors, dict) else enumerate(approvalwise_vectors)
    for instance_id, approvalwise_vector in vectors:
        election = sample_election_from_approvalwise_vector(
            approvalwise_vector, num_voters, seed=seed)
        election.instance_id = str(instance_id)
        experiment.add_election_to_family(election, family_id=family_id)


def dump_to_text_file(approvalwise_vectors: dict[str, np.ndarray] | Iterable[np.ndarray], num_voters: int, file) -> None:
    sample_election_from_approvalwise_vector = next(
        iter(approvalwise_vectors.values() if isinstance(approvalwise_vectors, dict) else approvalwise_vectors))
    num_candidates = len(sample_election_from_approvalwise_vector)
    num_elections = len(approvalwise_vectors)
    file.write(f'{num_elections} {num_voters} {num_candidates}\n')

    iterator = approvalwise_vectors.items() if isinstance(
        approvalwise_vectors, dict) else enumerate(approvalwise_vectors)

    for instance_id, approvalwise_vector in iterator:
        stringified_vector = [str(int(x)) for x in approvalwise_vector]
        file.write(f'{instance_id} {" ".join(stringified_vector)}\n')


def load_from_text_file(file) -> tuple[dict[str, np.ndarray], int]:
    header = file.readline().strip().split()
    num_elections = int(header[0])
    num_voters = int(header[1])
    _num_candidates = int(header[2])

    approvalwise_vectors = {}
    for _ in range(num_elections):
        line = file.readline().strip().split()
        instance_id = line[0]
        vector = np.array([int(x) for x in line[1:]])
        approvalwise_vectors[instance_id] = np.array(vector)

    return approvalwise_vectors, num_voters
