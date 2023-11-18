#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include "definitions.hpp"

namespace greedy_dp {
pair<approvalwise_vector_t, int> farthest_approvalwise_vector(
    const vector<approvalwise_vector_t>& votings_hist, const int N);
}