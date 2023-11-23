#include <algorithm>
#include <vector>

#include "definitions.hpp"
#include "greedy_dp.hpp"
#include "pairs.hpp"

using algorithm_t = std::function<std::pair<approvalwise_vector_t, int>(
    const std::vector<approvalwise_vector_t>&, const int)>;

int32_t __create_biding(int32_t* approvalwise_vectors, int32_t num_voters,
                        int32_t num_candidates, int32_t num_instances,
                        int32_t* new_approvalwise_vector,
                        algorithm_t algorithm) {
    std::vector<approvalwise_vector_t> approvalwise_vectors_vec;
    for (int32_t i = 0; i < num_instances; i++) {
        approvalwise_vector_t approvalwise_vector(
            approvalwise_vectors + i * num_candidates,
            approvalwise_vectors + (i + 1) * num_candidates);
        approvalwise_vectors_vec.push_back(approvalwise_vector);
    }
    auto [result_vec, result_dist] =
        algorithm(approvalwise_vectors_vec, num_voters);
    std::copy(result_vec.begin(), result_vec.end(), new_approvalwise_vector);
    return result_dist;
}
extern "C" {

int32_t greedy_dp_binding(int32_t* approvalwise_vectors, int32_t num_voters,
                          int32_t num_candidates, int32_t num_instances,
                          int32_t* new_approvalwise_vector) {
    return __create_biding(approvalwise_vectors, num_voters, num_candidates,
                           num_instances, new_approvalwise_vector,
                           greedy_dp::farthest_approvalwise_vector);
}

int32_t pairs_binding(int32_t* approvalwise_vectors, int32_t num_voters,
                      int32_t num_candidates, int32_t num_instances,
                      int32_t* new_approvalwise_vector) {
    return __create_biding(approvalwise_vectors, num_voters, num_candidates,
                           num_instances, new_approvalwise_vector,
                           pairs::farthest_approvalwise_vector);
}
}