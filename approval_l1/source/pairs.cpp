#pragma once

#include "pairs.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include "approvalwise_vector.hpp"
#include "greedy_dp.hpp"

namespace pairs {
pair<approvalwise_vector_t, int> farthest_approvalwise_vector(
    const vector<approvalwise_vector_t>& approvalwise_vectors,
    const int num_voters) {
    const int num_elections = approvalwise_vectors.size();
    const int num_candidates = approvalwise_vectors.front().size();

    vector<approvalwise_vector_t> vector_candidates;

    for (size_t first_idx = 0; first_idx < num_elections; ++first_idx) {
        for (size_t second_idx = first_idx + 1; second_idx < num_elections;
             second_idx++) {
            vector<approvalwise_vector_t> two_vectors = {
                approvalwise_vectors[first_idx],
                approvalwise_vectors[second_idx]};
            vector_candidates.push_back(
                greedy_dp::farthest_approvalwise_vector(two_vectors, num_voters)
                    .first);
        }
    }

    int best_distance = -1;
    approvalwise_vector_t best_vector(num_candidates);
    for (const auto& candidate : vector_candidates) {
        int distance =
            score_across(approvalwise_vectors, candidate, num_voters);
        if (distance > best_distance) {
            best_distance = distance;
            best_vector = candidate;
        }
    }

    return {best_vector, best_distance};
}
}  // namespace pairs