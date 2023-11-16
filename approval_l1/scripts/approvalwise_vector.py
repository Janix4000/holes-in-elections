from typing import Any
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


def add_sampled_elections_to_experiment(approvalwise_vectors: dir, experiment: mapel.ApprovalElectionExperiment, family_id: str, num_voters: int) -> None:
    experiment.add_empty_family(family_id=family_id)
    for instance_id, approvalwise_vector in approvalwise_vectors.items():
        election = sample_election_from_approvalwise_vector(
            approvalwise_vector, num_voters)
        election.instance_id = instance_id
        experiment.add_election_to_family(election, family_id=family_id)