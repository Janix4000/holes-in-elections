#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include "definitions.hpp"

namespace pairs {
pair<approvalwise_vector_t, int> farthest_approvalwise_vector(
    const vector<approvalwise_vector_t>& approvalwise_vectors,
    const int num_voters);
}