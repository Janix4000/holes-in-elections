#pragma once

#include "definitions.hpp"
#include "utils.hpp"

namespace brute {
pair<approvalwise_vector_t, int> farthest_approvalwise_vector(
    const vector<approvalwise_vector_t>& votings_hist, const int N);
}  // namespace brute
