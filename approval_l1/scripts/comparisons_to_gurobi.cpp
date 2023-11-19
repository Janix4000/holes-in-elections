#include <fstream>
#include <functional>
#include <iostream>

#include "approvalwise_vector.hpp"
#include "definitions.hpp"
#include "greedy_dp.hpp"
#include "pairs.hpp"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <reference_input_file> [algorithm]\n";
        return 1;
    }
    std::ifstream in(argv[1]);
    auto [approvalwise_vectors, num_voters] = load_approvalwise_vectors(in);
    size_t num_candidates = approvalwise_vectors.front().size();
    size_t num_elections = approvalwise_vectors.size();

    std::ifstream in_ref(argv[2]);
    auto [reference_approvalwise_vectors, _num_voters] =
        load_approvalwise_vectors(in_ref);
    size_t num_elections_reference = reference_approvalwise_vectors.size();

    std::function<std::pair<approvalwise_vector_t, int>(
        const std::vector<approvalwise_vector_t> &, const int)>
        algorithm;

    if (argc == 4) {
        std::string algorithm_name = argv[4 - 1];
        if (algorithm_name == "greedy_dp") {
            algorithm = greedy_dp::farthest_approvalwise_vector;
        } else if (algorithm_name == "pairs") {
            algorithm = pairs::farthest_approvalwise_vector;
        } else {
            std::cerr << "Unknown algorithm: " << algorithm_name << "\n";
            return 1;
        }
    } else {
        algorithm = greedy_dp::farthest_approvalwise_vector;
    }

    std::cout << num_elections_reference << "\n";
    for (int num_starting_elections = 0;
         num_starting_elections < num_elections_reference;
         num_starting_elections++) {
        std::vector<approvalwise_vector_t> starting_approvalwise_vectors =
            approvalwise_vectors;
        for (int i = 0; i < num_starting_elections; i++) {
            starting_approvalwise_vectors.push_back(
                reference_approvalwise_vectors[i]);
        }
        std::cout << num_elections_reference - num_starting_elections << " ";
        for (size_t idx = 0;
             idx < num_elections_reference - num_starting_elections; idx++) {
            auto [farthest_approvalwise_vectors, distance] =
                algorithm(starting_approvalwise_vectors, num_voters);
            std::cout << distance << " ";
            starting_approvalwise_vectors.push_back(
                farthest_approvalwise_vectors);
        }
        std::cout << endl;
    }
    return 0;
}